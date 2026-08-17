"""Classic ML baseline: handcrafted spectral/complexity features per 16-s window
+ LightGBM and RBF-SVM, same fold-0 split and evaluation as the deep models."""
import json
import time
from pathlib import Path

import numpy as np

import common as C

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
BANDS = [(0.5, 2), (2, 4), (4, 6), (6, 9), (9, 12.8)]


def window_features(x):
    """x: (18, 512) microvolts -> feature vector (per-channel stats pooled)."""
    feats = []
    powers = []
    for ch in x:
        ch = ch - ch.mean()
        spec = np.abs(np.fft.rfft(ch * np.hanning(len(ch)))) ** 2
        freqs = np.fft.rfftfreq(len(ch), 1 / C.SFREQ)
        tot = spec[(freqs >= 0.5) & (freqs <= 12.8)].sum() + 1e-12
        bp = []
        for lo, hi in BANDS:
            p = spec[(freqs >= lo) & (freqs < hi)].sum()
            bp.append(p / tot)
        powers.append(bp)
        # complexity features
        zc = np.sum(np.diff(np.sign(ch)) != 0)
        ll = np.mean(np.abs(np.diff(ch)))
        rms = np.sqrt(np.mean(ch ** 2))
        d1, d2 = np.diff(ch), np.diff(ch, 2)
        var = np.var(ch) + 1e-12
        hj_m = np.sqrt(np.var(d1) / var)
        hj_c = np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (hj_m + 1e-12)
        ent = -np.nansum((np.array(bp) + 1e-12) * np.log2(np.array(bp) + 1e-12))
        feats.append([np.log10(tot), *bp, ent, zc, ll, rms, hj_m, hj_c])
    powers = np.array(powers)                       # (18, 5)
    f = np.array(feats)                             # (18, 9)
    pooled = np.concatenate([f.mean(0), f.std(0), powers.max(0), powers.min(0)])
    return pooled


def build_xy(data, pids, tag):
    X, Y, P = [], [], []
    for p in pids:
        d = data[p]
        for s in C.window_starts(d["n_sec"]):
            x = d["x"][:, s * C.SFREQ:(s + C.WIN_S) * C.SFREQ]
            X.append(window_features(x))
            Y.append(float(d[f"y{tag}"][s:s + C.WIN_S].mean() >= C.LAB_FRAC))
            P.append(p)
    return np.array(X, dtype=np.float32), np.array(Y), np.array(P)


def main():
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    t0 = time.time()
    data = C.load_all()
    tr, va, te = C.make_splits(data)[0]
    tag = "M"
    Xtr, ytr, _ = build_xy(data, tr, tag)
    Xva, yva, _ = build_xy(data, va, tag)
    Xte, yte, _ = build_xy(data, te, tag)
    print(f"features {Xtr.shape[1]}d; train {Xtr.shape[0]} windows "
          f"({ytr.mean()*100:.1f}% sz) in {time.time()-t0:.0f}s", flush=True)

    results = {}
    for name, clf in [
        ("lightgbm", make_pipeline(StandardScaler(), LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=63,
            scale_pos_weight=(1 - ytr.mean()) / ytr.mean(), verbose=-1))),
        ("svm-rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0,
                                                        probability=True))),
    ]:
        t1 = time.time()
        clf.fit(Xtr, ytr)
        # per-second test probabilities: predict window probs then map to centre
        probs = {}
        for p in te:
            d = data[p]
            starts = np.arange(0, d["n_sec"] - C.WIN_S + 1, 1)
            Xw = np.array([window_features(d["x"][:, s * C.SFREQ:(s + C.WIN_S) * C.SFREQ])
                           for s in starts], dtype=np.float32)
            pw = clf.predict_proba(Xw)[:, 1]
            ps = np.full(d["n_sec"], np.nan, dtype=np.float32)
            ps[starts + C.WIN_S // 2] = pw
            idx = np.where(~np.isnan(ps))[0]
            ps[:idx[0]] = pw[0]; ps[idx[-1] + 1:] = pw[-1]
            probs[p] = ps
        np.savez_compressed(OUT / f"test_probs_{name}.npz",
                            **{str(p): probs[p] for p in te})
        res = {}
        for tg in ["A", "B", "C", "M", "U"]:
            auc, ap = C.pooled_auc(probs, data, tg)
            res[f"auc_{tg}"] = round(float(auc), 4)
        hours = sum(data[p]["n_sec"] for p in te) / 3600
        # threshold tuned on val
        val_probs = {}
        for p in va:
            d = data[p]
            starts = np.arange(0, d["n_sec"] - C.WIN_S + 1, 8)
            Xw = np.array([window_features(d["x"][:, s * C.SFREQ:(s + C.WIN_S) * C.SFREQ])
                           for s in starts], dtype=np.float32)
            pw = clf.predict_proba(Xw)[:, 1]
            ps = np.full(d["n_sec"], np.nan, dtype=np.float32)
            cs = starts + C.WIN_S // 2
            ps[cs] = pw
            idx = np.where(~np.isnan(ps))[0]
            ps[:idx[0]] = pw[0]; ps[idx[-1] + 1:] = pw[-1]
            val_probs[p] = ps
        th = C.tune_threshold(val_probs, data, "M", target_fd_h=0.5)
        res["threshold"] = round(float(th), 2)
        for tg in ["A", "B", "C", "M"]:
            det = {p: C.detect_events(probs[p], th) for p in te}
            ref = {p: C.events_from_binary(data[p][f"y{tg}"]) for p in te}
            gd = fd = nr = 0
            for p in te:
                g, f, n = C.event_metrics(det[p], ref[p])
                gd += g; fd += f; nr += n
            res[f"gdr_{tg}"] = round(gd / max(nr, 1), 4)
            res[f"fdh_{tg}"] = round(fd / hours, 3)
            res[f"nref_{tg}"] = nr
        results[name] = res
        print(f"{name}: AUC(M)={res['auc_M']} GDR(M)={res['gdr_M']} @ {res['fdh_M']} FD/h "
              f"[{(time.time()-t1)/60:.1f} min]", flush=True)
    json.dump(results, open(OUT / "classic_results.json", "w"), indent=1)
    print("saved classic_results.json")


if __name__ == "__main__":
    main()
