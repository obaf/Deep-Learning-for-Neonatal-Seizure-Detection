"""Fast fold-0 ablations to select the final training recipe.

Configs (resnet-s unless noted; val 18% for steadier early stopping):
  wnorm         : per-window z-score (original recipe)
  patnorm       : patient-wise robust z-score
  patnorm_mix   : + mixup 0.2
  patnorm_soft  : soft multi-annotator labels (tag S)
  patnorm_chanind: channel-independent resnet (Hogan-style, top-3 aggregation)
Each evaluated with joint (smoothing, threshold) tuning on val.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C
import models
import train as T
from run_annot import eval_probs

OUT = Path(__file__).parent / "results"
CONFIGS = [
    ("wnorm", dict(arch="resnet-s", norm=None, tag="M", mixup=0.0, chanind=False,
                   epochs=45)),
    ("patnorm", dict(arch="resnet-s", norm="patient", tag="M", mixup=0.0, chanind=False,
                     epochs=45)),
    ("patnorm_mix", dict(arch="resnet-s", norm="patient", tag="M", mixup=0.2, chanind=False,
                         epochs=45)),
    ("patnorm_soft", dict(arch="resnet-s", norm="patient", tag="S", mixup=0.0, chanind=False,
                          epochs=45)),
    ("patnorm_chanind", dict(arch="chanindep", norm="patient", tag="M", mixup=0.0,
                             chanind=True, epochs=25)),
]


def predict(model, data, pids, device, chanind):
    fn = C.predict_seconds_chanind if chanind else C.predict_seconds
    return {p: fn(model, data[p], device) for p in pids}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = json.load(open(OUT / "ablate_results.json")) if (OUT / "ablate_results.json").exists() else {}
    for name, cfg in CONFIGS:
        if name in results:
            print(f"skip {name} (already done)", flush=True)
            continue
        t0 = time.time()
        print(f"\n=== {name}: {cfg} ===", flush=True)
        data = C.load_all(norm=cfg["norm"])
        tr, va, te = C.make_splits(data, val_frac=0.18)[0]
        model, best_val, _ = T.train_model(
            data, tr, va, models.build(cfg["arch"]), tag=cfg["tag"],
            mixup=cfg["mixup"], chanind=cfg["chanind"], epochs=cfg["epochs"],
            patience=12, lr=1.2e-3, log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), OUT / f"ablate_{name}.pt")
        val_probs = predict(model, data, va, device, cfg["chanind"])
        smooth, th = C.tune_postproc(val_probs, data, "M", target_fd_h=0.5)
        probs = predict(model, data, te, device, cfg["chanind"])
        np.savez_compressed(OUT / f"ablate_test_probs_{name}.npz",
                            **{str(p): probs[p] for p in te})
        r = eval_probs(probs, data, te, th, smooth)
        r.update(best_val_auc=round(float(best_val), 4), smooth=int(smooth),
                 threshold=float(th), arch=cfg["arch"], params=sum(
                     p.numel() for p in model.parameters()))
        results[name] = r
        print(f"{name}: AUC(M)={r['auc_M']} A/B/C={r['auc_A']}/{r['auc_B']}/{r['auc_C']} "
              f"GDR(M)={r['gdr_M']} @{r['fdh_M']} FD/h (smooth {smooth}, th {th:.2f}) "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)
        json.dump(results, open(OUT / "ablate_results.json", "w"), indent=1)
    print("\nDone.")


if __name__ == "__main__":
    main()
