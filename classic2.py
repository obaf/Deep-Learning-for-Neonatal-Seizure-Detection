"""Tapani-inspired engineered-feature baseline: spectral + complexity + time-varying
cross-channel correlation/coherence features per 16-s window with an RBF-SVM.

Reference: Tapani et al. 2019 (Int J Neural Syst) achieved median per-patient
AUC 0.988 on this dataset with SVM + time-varying correlation features.
"""
import json
import time
from pathlib import Path

import numpy as np

import common as C

OUT = Path(__file__).parent / "results"
BANDS = [(0.5, 2), (2, 4), (4, 6), (6, 9), (9, 12.8)]


def channel_feats(ch, freqs, spec):
    tot = spec[(freqs >= 0.5) & (freqs <= 12.8)].sum() + 1e-12
    bp = np.array([spec[(freqs >= lo) & (freqs < hi)].sum() / tot for lo, hi in BANDS])
    cum = np.cumsum(spec[(freqs >= 0.5) & (freqs <= 12.8)]) / (tot + 1e-12)
    fb = freqs[(freqs >= 0.5) & (freqs <= 12.8)]
    if len(cum):
        sef90 = fb[min(np.searchsorted(cum, 0.90), len(fb) - 1)]
    else:
        sef90 = 0.0
    ent = -np.sum((bp + 1e-12) * np.log2(bp + 1e-12))
    zc = np.sum(np.diff(np.sign(ch - ch.mean())) != 0)
    ll = np.mean(np.abs(np.diff(ch)))
    rms = np.sqrt(np.mean(ch ** 2))
    d1, d2 = np.diff(ch), np.diff(ch, 2)
    var = np.var(ch) + 1e-12
    hm = np.sqrt(np.var(d1) / var)
    hc = np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (hm + 1e-12)
    return [np.log10(tot), *bp, ent, sef90, zc / len(ch), ll, rms, hm, hc]


def window_features(x):
    """x: (18, 512) microvolts -> feature vector."""
    n_ch, n_t = x.shape
    xz = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-4)
    win = np.hanning(n_t)
    spec = np.abs(np.fft.rfft(xz * win, axis=1)) ** 2
    freqs = np.fft.rfftfreq(n_t, 1 / C.SFREQ)
    F = np.array([channel_feats(xz[c], freqs, spec[c]) for c in range(n_ch)])  # (18, 13)
    # time-varying cross-channel correlation (4-s subwindows, Tapani-style)
    sub = C.SFREQ * 4
    n_sub = n_t // sub
    corrs = []
    for k in range(n_sub):
        seg = xz[:, k * sub:(k + 1) * sub]
        cm = np.corrcoef(seg)
        iu = np.triu_indices(n_ch, 1)
        corrs.append(cm[iu])
    corrs = np.stack(corrs)                       # (n_sub, 153)
    tv = corrs.std(axis=0)                        # time-variability of each pair corr
    # spectral coherence in 1.5-4 Hz band for all pairs (once per window)
    band = (freqs >= 1.5) & (freqs < 4)
    Xf = np.fft.rfft(xz * win, axis=1)                      # (n_ch, n_freq)
    cross = np.einsum("cf,df->cdf", Xf, Xf.conj())          # per-freq cross-spectra
    p = np.abs(Xf) ** 2
    den = np.sqrt(np.einsum("cf,df->cdf", p, p)) + 1e-12
    coh = np.abs(cross) / den                               # (n_ch, n_ch, n_freq)
    iu = np.triu_indices(n_ch, 1)
    coh_band = coh[iu][:, band].mean(axis=1)
    feats = np.concatenate([
        F.mean(0), F.std(0), F.max(0), F.min(0),
        [corrs.mean(), corrs.mean(axis=1).max(), tv.mean(), tv.max(),
         np.percentile(tv, 90), coh_band.mean(), coh_band.max(),
         np.percentile(coh_band, 90), np.mean(np.abs(corrs).mean(axis=1))],
    ])
    return feats.astype(np.float32)


def probs_for_patients(clf, data, pids, stride):
    probs = {}
    for p in pids:
        d = data[p]
        starts = np.arange(0, d["n_sec"] - C.WIN_S + 1, stride)
        X = np.array([window_features(d["x"][:, s * C.SFREQ:(s + C.WIN_S) * C.SFREQ])
                      for s in starts], dtype=np.float32)
        pw = clf.predict_proba(X)[:, 1]
        ps = np.full(d["n_sec"], np.nan, dtype=np.float32)
        cs = starts + C.WIN_S // 2
        ps[cs] = pw
        idx = np.where(~np.isnan(ps))[0]
        ps[:idx[0]] = pw[0]; ps[idx[-1] + 1:] = pw[-1]
        probs[p] = ps
    return probs


def main():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    t0 = time.time()
    data = C.load_all()
    tr, va, te = C.make_splits(data, val_frac=0.18)[0]
    Xtr, ytr = [], []
    for p in tr:
        d = data[p]
        for s in C.window_starts(d["n_sec"]):
            Xtr.append(window_features(d["x"][:, s * C.SFREQ:(s + C.WIN_S) * C.SFREQ]))
            ytr.append(float(d["yM"][s:s + C.WIN_S].mean() >= C.LAB_FRAC))
    Xtr, ytr = np.array(Xtr), np.array(ytr)
    print(f"features {Xtr.shape[1]}d, {Xtr.shape[0]} windows ({ytr.mean()*100:.1f}% sz) "
          f"built in {(time.time()-t0)/60:.1f} min", flush=True)

    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="rbf", C=2.0, gamma="scale",
                                   class_weight="balanced"), ensemble=False))
    clf.fit(Xtr, ytr)
    print(f"SVM fitted in {(time.time()-t0)/60:.1f} min", flush=True)

    val_probs = probs_for_patients(clf, data, va, stride=8)
    smooth, th = C.tune_postproc(val_probs, data, "M", target_fd_h=0.5)
    probs = probs_for_patients(clf, data, te, stride=1)
    np.savez_compressed(OUT / "ablate_test_probs_featsvm.npz",
                        **{str(p): probs[p] for p in te})
    from run_annot import eval_probs
    r = eval_probs(probs, data, te, th, smooth)
    r.update(smooth=int(smooth), threshold=float(th))
    # per-patient median AUC (Tapani-comparable metric)
    from sklearn.metrics import roc_auc_score
    paucs = [roc_auc_score(data[p]["yM"], probs[p]) for p in te
             if data[p]["yM"].min() != data[p]["yM"].max()]
    r["median_patient_auc_M"] = round(float(np.median(paucs)), 4)
    json.dump(r, open(OUT / "featsvm_results.json", "w"), indent=1)
    print("featsvm:", json.dumps(r, indent=1), flush=True)


if __name__ == "__main__":
    main()
