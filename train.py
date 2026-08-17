"""Training loop with augmentation, AMP, patient-wise early stopping."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import common as C


class WindowDataset(Dataset):
    """Training windows: base starts on a non-overlapping grid; each access
    applies a random time offset (the label is recomputed for the shifted window)
    plus amplitude / noise / channel-dropout augmentation.

    tag="S" -> soft multi-annotator label: per-second fraction of experts (A,B,C)
    that marked seizure, averaged over the window (target in [0,1]).
    chanind=True -> channel-independent mode (Hogan et al.): each item is a single
    channel (B, 1, T) carrying the window label (weak labelling)."""

    def __init__(self, data, pids, tag="M", augment=True, chanind=False,
                 stride_s=None):
        self.data, self.tag, self.augment, self.chanind = data, tag, augment, chanind
        self.soft = tag == "S"
        self.multi = tag == "MULTI"
        self.index = []
        for p in pids:
            d = data[p]
            step = stride_s or C.WIN_S
            for s in C.window_starts(d["n_sec"], stride_s=step):
                if chanind:
                    # one entry per window; channel sampled at visit time
                    # (same expected coverage as enumerating channels, 18x faster)
                    self.index.append((p, s, -1))
                else:
                    self.index.append((p, s, None))

    def __len__(self):
        return len(self.index)

    def window_label(self, d, s):
        if self.multi:
            return np.array([float(d[f"y{t}"][s:s + C.WIN_S].mean() >= C.LAB_FRAC)
                             for t in "ABC"], dtype=np.float32)
        if self.soft:
            frac = (d["yA"][s:s + C.WIN_S].astype(np.float32)
                    + d["yB"][s:s + C.WIN_S] + d["yC"][s:s + C.WIN_S]) / 3.0
            return float(frac.mean())
        return float(d[f"y{self.tag}"][s:s + C.WIN_S].mean() >= C.LAB_FRAC)

    def __getitem__(self, i):
        pid, s0, ch = self.index[i]
        d = self.data[pid]
        if self.augment:
            off = np.random.randint(0, C.WIN_S)
            if self.chanind and ch < 0:
                ch = np.random.randint(0, d["x"].shape[0])
        else:
            off = 0
            if self.chanind and ch < 0:
                ch = (i * 7) % d["x"].shape[0]   # deterministic coverage in eval
        s = s0 + off
        if s + C.WIN_S > d["n_sec"]:
            s = d["n_sec"] - C.WIN_S
        x = d["x"][:, s * C.SFREQ:(s + C.WIN_S) * C.SFREQ].copy()
        y = self.window_label(d, s)
        if self.chanind:
            x = x[ch:ch + 1]
        if not d.get("prenorm", False):
            x = C.normalize(x)
        if self.augment:
            # global + per-channel amplitude jitter (z-units)
            x *= np.random.uniform(0.8, 1.2)
            x *= np.random.uniform(0.9, 1.1, size=(x.shape[0], 1)).astype(np.float32)
            x += (np.random.randn(*x.shape) * 0.05).astype(np.float32)
            ch2 = np.random.rand(x.shape[0]) < 0.05
            x[ch2] = 0.0
        if self.multi:
            return (torch.from_numpy(x.astype(np.float32)),
                    torch.from_numpy(y))
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.float32)


def pos_weight_of(loader):
    n = p = 0
    for _, y in loader:
        yb = (y.float() >= 0.5).float()
        if yb.ndim == 2:      # multi-head: count each head target
            n += yb.numel()
            p += yb.sum().item()
        else:
            n += yb.numel()
            p += yb.sum().item()
    return (n - p) / max(p, 1)


def evaluate_loader(model, loader, device):
    model.eval()
    P, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for xb, yb in loader:
            out = model(xb.to(device)).float()
            p = torch.sigmoid(out).mean(dim=1).cpu().numpy() if out.ndim == 2 \
                else torch.sigmoid(out).cpu().numpy()
            y = yb.float().mean(dim=1).numpy() if yb.ndim == 2 else yb.numpy()
            P.append(p); Y.append((y >= 0.5).astype(int))
    from sklearn.metrics import roc_auc_score
    P, Y = np.concatenate(P), np.concatenate(Y)
    if Y.min() == Y.max():
        return 0.5
    return roc_auc_score(Y, P)


def train_model(data, train_pids, val_pids, model, tag="M", epochs=60, lr=1.5e-3,
                batch=64, patience=8, device=None, log=print, seed=0, mixup=0.0,
                chanind=False, stride=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if chanind:
        batch = max(batch * 4, 256)
    tr = WindowDataset(data, train_pids, tag, augment=True, chanind=chanind,
                       stride_s=stride)
    va = WindowDataset(data, val_pids, tag, augment=False, chanind=chanind)
    # faster val: subsample windows to stride 8 s
    va.index = [ix for ix in va.index if ix[1] % 8 == 0]
    tl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=True)
    vl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=0)
    pw = torch.tensor(min(pos_weight_of(DataLoader(tr, batch_size=256)), 4.0), device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * max(len(tl), 1), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tl:
            if mixup > 0:
                lam = np.random.beta(mixup, mixup)
                lam = max(lam, 1 - lam)          # keep target near dominant label
                idx = torch.randperm(len(xb))
                xb = lam * xb + (1 - lam) * xb[idx]
                yb = lam * yb + (1 - lam) * yb[idx]
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(xb)
                loss = lossf(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(xb)
        auc = evaluate_loader(model, vl, device)
        hist.append((ep, tot / len(tr), auc))
        if auc > best_auc + 1e-4:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"  epoch {ep:3d}  loss {tot/len(tr):.4f}  valAUC {auc:.4f}  best {best_auc:.4f}  bad {bad}")
        if bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, best_auc, hist
