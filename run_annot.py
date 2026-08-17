"""Inter-annotator label-noise experiments on fold-0 with the chosen architecture.

For each training label source in {A, B, C, M(consensus)}: train on fold-0 train
patients, evaluate pooled per-second AUC on the test patients against every
annotator. Produces the 4x5 transfer matrix in results/annot_results.json.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C
import models
import train as T

OUT = Path(__file__).parent / "results"
# single-head architecture so the label-source effect is isolated from the
# multi-head architecture (multih is evaluated separately in the final CV)
ARCH = "resnet-s"
NORM = "patient"
MIXUP = 0.2


def eval_probs(probs, data, te, th=None, smooth=16, tag="M"):
    res = {}
    for tg in ["A", "B", "C", "M", "U"]:
        auc, ap = C.pooled_auc(probs, data, tg)
        res[f"auc_{tg}"] = round(float(auc), 4)
    if th is not None:
        hours = sum(data[p]["n_sec"] for p in te) / 3600
        for tg in ["A", "B", "C", "M"]:
            det = {p: C.detect_events(probs[p], th, smooth_s=smooth) for p in te}
            ref = {p: C.events_from_binary(data[p][f"y{tg}"]) for p in te}
            gd = fd = nr = 0
            for p in te:
                g, f, n = C.event_metrics(det[p], ref[p])
                gd += g; fd += f; nr += n
            res[f"gdr_{tg}"] = round(gd / max(nr, 1), 4)
            res[f"fdh_{tg}"] = round(fd / hours, 3)
    return res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = C.load_all(norm=NORM)
    tr, va, te = C.make_splits(data, val_frac=0.18)[0]
    results = {"arch": ARCH, "norm": NORM, "mixup": MIXUP,
               "train": tr, "val": va, "test": te}
    for tag in ["A", "B", "C", "M"]:
        t0 = time.time()
        print(f"\n=== train on labels of {tag} ===", flush=True)
        model, best_val, _ = T.train_model(
            data, tr, va, models.build(ARCH), tag=tag, mixup=MIXUP,
            log=lambda s: print(s, flush=True))
        torch.save(model.state_dict(), OUT / f"annot_{tag}.pt")
        val_probs = {p: C.predict_seconds(model, data[p], device) for p in va}
        smooth, th = C.tune_postproc(val_probs, data, "M", target_fd_h=0.5)
        probs = {p: C.predict_seconds(model, data[p], device) for p in te}
        np.savez_compressed(OUT / f"test_probs_train{tag}_{ARCH}.npz",
                            **{str(p): probs[p] for p in te})
        r = eval_probs(probs, data, te, th, smooth)
        r["best_val_auc"] = round(float(best_val), 4)
        r["smooth"] = int(smooth)
        r["threshold"] = round(float(th), 2)
        results[f"train_{tag}"] = r
        print(f"train {tag}: " + " ".join(f"AUC({t})={r[f'auc_{t}']}" for t in "ABCM")
              + f" [{(time.time()-t0)/60:.1f} min]", flush=True)
        json.dump(results, open(OUT / "annot_results.json", "w"), indent=1)
    print("\nDone.")


if __name__ == "__main__":
    main()
