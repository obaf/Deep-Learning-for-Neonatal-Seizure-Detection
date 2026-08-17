"""Shared utilities: data loading, patient splits, windowing, evaluation.

Data layout: data/proc/eegN.npz with x (18, T) float32 microvolts @ 32 Hz and
per-second labels yA/yB/yC (experts), yM (majority consensus >=2), yU (union >=1).
"""
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROC = HERE / "data" / "proc"

SFREQ = 32
import os as _os
WIN_S = int(_os.environ.get("WIN_S", 16))   # window length in seconds (16 default)
WIN = WIN_S * SFREQ
LAB_FRAC = 0.5      # window is seizure if >= 50% of seconds are seizure (Hogan: >=8/16 s)
LABELS = ["A", "B", "C", "M", "U"]


def load_all(patients=None, norm=None):
    """Return dict pid -> dict(x, yA..yU, n_sec). Loads everything into RAM (~1 GB).

    norm="patient": robust per-channel z-score over the whole recording
    (median / MAD), preserving within-recording amplitude dynamics.
    """
    out = {}
    files = sorted(PROC.glob("eeg*.npz"))
    for f in files:
        pid = int(f.stem.replace("eeg", ""))
        if patients is not None and pid not in patients:
            continue
        d = np.load(f)
        x = d["x"]
        if norm == "patient":
            med = np.median(x, axis=1, keepdims=True)
            mad = np.median(np.abs(x - med), axis=1, keepdims=True) * 1.4826 + 1e-1
            x = ((x - med) / mad).astype(np.float32)
            prenorm = True
        else:
            prenorm = False
        out[pid] = {"x": x, "n_sec": x.shape[1] // SFREQ, "prenorm": prenorm,
                    **{f"y{t}": d[f"y{t}"] for t in LABELS}}
    return out


def patient_seizure_status(data, tag="M"):
    return {pid: int(d[f"y{tag}"].any()) for pid, d in data.items()}


def make_splits(data, seed=42, n_folds=5, val_frac=0.12):
    """Stratified patient-wise k-fold: list of (train, val, test) pid lists.
    Stratification by consensus seizure status."""
    rng = np.random.RandomState(seed)
    status = patient_seizure_status(data, "M")
    pos = [p for p, s in status.items() if s == 1]
    neg = [p for p, s in status.items() if s == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(n_folds)]
    for i, p in enumerate(pos):   # round-robin pos then neg -> stratified folds
        folds[i % n_folds].append(p)
    for i, p in enumerate(neg):
        folds[i % n_folds].append(p)
    splits = []
    for k in range(n_folds):
        test = sorted(folds[k])
        rest = sorted([p for j in range(n_folds) if j != k for p in folds[j]])
        # stratified val carve-out from rest
        rest_pos = [p for p in rest if status[p] == 1]
        rest_neg = [p for p in rest if status[p] == 0]
        n_val = max(2, int(round(val_frac * len(rest))))
        n_val_pos = max(1, int(round(n_val * len(rest_pos) / len(rest))))
        val = sorted(rest_pos[:n_val_pos] + rest_neg[:n_val - n_val_pos])
        train = sorted([p for p in rest if p not in set(val)])
        splits.append((train, val, test))
    return splits


def window_starts(n_sec, stride_s=WIN_S, win_s=WIN_S):
    """Starts (in seconds) of windows fully inside the recording."""
    return np.arange(0, n_sec - win_s + 1, stride_s)


def window_label(y_sec, start_s, win_s=WIN_S, frac=LAB_FRAC):
    return float(y_sec[start_s:start_s + win_s].mean() >= frac)


def normalize(x):
    """Per-channel z-score of a window; x is (C, T) float32."""
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True) + 1e-4
    return (x - m) / s


# ------------------------------------------------------------------ evaluation
def predict_seconds_chanind(model, d, device, stride_s=1, batch=256, topk=3):
    """Channel-independent inference (Hogan-style): per-channel per-second
    probabilities aggregated as the mean of the top-k channels each second."""
    model.eval()
    n_sec = d["n_sec"]
    n_ch = d["x"].shape[0]
    starts = np.arange(0, n_sec - WIN_S + 1, stride_s)
    P = np.full((n_ch, n_sec), np.nan, dtype=np.float32)
    if len(starts) == 0:
        return np.zeros(n_sec, dtype=np.float32)
    prenorm = d.get("prenorm", False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for c in range(n_ch):
            probs = np.empty(len(starts), dtype=np.float32)
            for i in range(0, len(starts), batch):
                idx = starts[i:i + batch]
                w = np.stack([d["x"][c, s * SFREQ:(s + WIN_S) * SFREQ] for s in idx])
                if not prenorm:
                    m = w.mean(axis=1, keepdims=True)
                    sd = w.std(axis=1, keepdims=True) + 1e-4
                    w = (w - m) / sd
                xb = torch.from_numpy(w[:, None, :].astype(np.float32)).to(device)
                out = model(xb).float()
                probs[i:i + len(idx)] = (torch.sigmoid(out).mean(dim=1) if out.ndim == 2
                                         else torch.sigmoid(out)).cpu().numpy()
            P[c, starts + WIN_S // 2] = probs
    idx0 = np.where(~np.isnan(P[0]))[0]
    P[:, :idx0[0]] = P[:, idx0[0]:idx0[0] + 1]
    P[:, idx0[-1] + 1:] = P[:, idx0[-1]:idx0[-1] + 1]
    Pt = np.sort(P, axis=0)
    k = min(topk, n_ch)
    return Pt[-k:].mean(axis=0)
def predict_seconds(model, d, device, stride_s=1, batch=64):
    """Sliding-window inference -> per-second probability series (length n_sec)."""
    model.eval()
    n_sec = d["n_sec"]
    starts = np.arange(0, n_sec - WIN_S + 1, stride_s)
    if len(starts) == 0:
        return np.full(n_sec, 0.0, dtype=np.float32)
    probs = np.empty(len(starts), dtype=np.float32)
    prenorm = d.get("prenorm", False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for i in range(0, len(starts), batch):
            idx = starts[i:i + batch]
            if prenorm:
                xb = np.stack([d["x"][:, s * SFREQ:(s + WIN_S) * SFREQ] for s in idx])
            else:
                xb = np.stack([normalize(d["x"][:, s * SFREQ:(s + WIN_S) * SFREQ]) for s in idx])
            xb = torch.from_numpy(xb.astype(np.float32)).to(device)
            out = model(xb).float()
            probs[i:i + len(idx)] = (torch.sigmoid(out).mean(dim=1) if out.ndim == 2
                                     else torch.sigmoid(out)).cpu().numpy()
    # map window probability to its centre second; fill edges with nearest
    p = np.full(n_sec, np.nan, dtype=np.float32)
    cent = starts + WIN_S // 2
    p[cent] = probs
    idx0 = np.where(~np.isnan(p))[0]
    if len(idx0) == 0:
        return np.zeros(n_sec, dtype=np.float32)
    p[:idx0[0]] = probs[0]
    p[idx0[-1] + 1:] = probs[-1]
    return p


def segment_metrics(p_sec, y_sec):
    """AUC + AP for one patient's per-second probability vs binary labels."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y = y_sec.astype(int)
    if y.min() == y.max():
        return None
    return roc_auc_score(y, p_sec), average_precision_score(y, p_sec)


def pooled_auc(pat_probs, data, tag):
    """Concatenate all patients for a pooled per-second AUC/AP."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    P, Y = [], []
    for pid, p in pat_probs.items():
        y = data[pid][f"y{tag}"]
        n = min(len(p), len(y))
        P.append(p[:n]); Y.append(y[:n])
    P, Y = np.concatenate(P), np.concatenate(Y)
    if Y.min() == Y.max():
        return float("nan"), float("nan")
    return roc_auc_score(Y, P), average_precision_score(Y, P)


def events_from_binary(x, merge_gap=10, min_dur=10):
    x = x.copy()
    if merge_gap > 0 and x.any():
        d = np.diff(x)
        gs = np.where(d == -1)[0]
        ge = np.where(d == 1)[0]
        for g in gs:
            e = ge[ge > g]
            if len(e) and (e[0] - g) <= merge_gap:
                x[g + 1:e[0] + 1] = 1
    runs, s = [], None
    for i, v in enumerate(x):
        if v == 1 and s is None:
            s = i
        elif v == 0 and s is not None:
            runs.append((s, i)); s = None
    if s is not None:
        runs.append((s, len(x)))
    return [(a, b) for a, b in runs if b - a >= min_dur]


def detect_events(p_sec, thresh, smooth_s=16, merge_gap=10, min_dur=10):
    """Post-processing: moving-average smoothing -> threshold -> event extraction."""
    k = smooth_s
    if k > 1:
        cs = np.cumsum(np.pad(p_sec, (k // 2, k // 2), mode="edge"))
        p = (cs[k:] - cs[:-k]) / k
    else:
        p = p_sec
    return events_from_binary((p >= thresh).astype(int), merge_gap, min_dur)


def event_metrics(detected, reference):
    """Any-overlap GDR + false detections per hour reference hours."""
    fd = [d for d in detected if not any(overlap(d, r) > 0 for r in reference)]
    gd = sum(1 for r in reference if any(overlap(d, r) > 0 for d in detected))
    return gd, len(fd), len(reference)


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def tune_postproc(pat_probs, data, tag="M", target_fd_h=0.5, smooths=(32, 48, 69, 96)):
    """Joint tuning of smoothing width + threshold on validation patients:
    highest GDR subject to FD/h <= target; graceful fallback to lowest FD/h."""
    ref = {p: events_from_binary(data[p][f"y{tag}"]) for p in pat_probs}
    hours = sum(data[p]["n_sec"] for p in pat_probs) / 3600
    best_ok = (0.0, 16, 0.5)          # (GDR, smooth, th)
    best_any = (np.inf, 0.0, 16, 0.5)  # (FD/h, -GDR, smooth, th)
    for sm in smooths:
        for th in np.arange(0.30, 0.96, 0.05):
            tp = fd = nref = 0
            for p in pat_probs:
                det = detect_events(pat_probs[p], th, smooth_s=sm)
                gd, f, nr = event_metrics(det, ref[p])
                tp += gd; fd += f; nref += nr
            fdh = fd / hours
            gdr = tp / max(nref, 1)
            if fdh <= target_fd_h and gdr > best_ok[0]:
                best_ok = (gdr, sm, float(th))
            if (fdh, -gdr) < (best_any[0], best_any[1]):
                best_any = (fdh, -gdr, sm, float(th))
    if best_ok[0] > 0:
        return best_ok[1], best_ok[2]
    return best_any[2], best_any[3]
