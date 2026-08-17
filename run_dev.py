"""Development run: train all model variants on fold-0 (consensus labels),
evaluate on the held-out test patients against every annotator.

Saves results/dev_results.json, checkpoints, and per-second test probabilities.
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
OUT.mkdir(exist_ok=True)
MODELS = ["shallow", "resnet-s", "resnet", "resnet-l", "crnn"]


def evaluate_on(model, data, pids, device, tag_ref="M"):
    """Per-second probs on patients; segment AUC vs all tags; event metrics at
    val-tuned threshold; returns dict."""
    probs = {p: C.predict_seconds(model, data[p], device) for p in pids}
    res = {}
    for tag in ["A", "B", "C", "M", "U"]:
        auc, ap = C.pooled_auc(probs, data, tag)
        res[f"auc_{tag}"] = round(float(auc), 4)
        res[f"ap_{tag}"] = round(float(ap), 4)
    # event-based vs consensus at tuned threshold
    val_probs = res.pop("_val_probs", None)
    return probs, res


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = C.load_all()
    tr, va, te = C.make_splits(data)[0]
    log = lambda s: print(s, flush=True)
    log(f"split: train {len(tr)} / val {len(va)} / test {len(te)} patients")

    # shared val probs for threshold tuning will be per-model; tune on val per model
    results = {"split": {"train": tr, "val": va, "test": te}}
    for name in MODELS:
        t0 = time.time()
        log(f"\n=== {name} ===")
        model = models.build(name)
        n_par = sum(p.numel() for p in model.parameters())
        model, best_val_auc, hist = T.train_model(data, tr, va, model, tag="M", log=log)
        torch.save(model.state_dict(), OUT / f"{name}.pt")

        # threshold tuning on val (per-second probs, consensus events)
        val_probs = {p: C.predict_seconds(model, data[p], device) for p in va}
        th = C.tune_threshold(val_probs, data, "M", target_fd_h=0.5)

        # test-time inference + metrics
        probs = {p: C.predict_seconds(model, data[p], device) for p in te}
        np.savez_compressed(OUT / f"test_probs_{name}.npz",
                            **{str(p): probs[p] for p in te})
        res = {"params": n_par, "best_val_auc": round(float(best_val_auc), 4),
               "threshold": round(float(th), 2)}
        for tag in ["A", "B", "C", "M", "U"]:
            auc, ap = C.pooled_auc(probs, data, tag)
            res[f"auc_{tag}"] = round(float(auc), 4)
            res[f"ap_{tag}"] = round(float(ap), 4)
        hours = sum(data[p]["n_sec"] for p in te) / 3600
        for tag in ["A", "B", "C", "M"]:
            det = {p: C.detect_events(probs[p], th) for p in te}
            ref = {p: C.events_from_binary(data[p][f"y{tag}"]) for p in te}
            gd = fd = nr = 0
            for p in te:
                g, f, n = C.event_metrics(det[p], ref[p])
                gd += g; fd += f; nr += n
            res[f"gdr_{tag}"] = round(gd / max(nr, 1), 4)
            res[f"fdh_{tag}"] = round(fd / hours, 3)
            res[f"nref_{tag}"] = nr
        # seizure-burden correlation vs consensus
        pb, rb = [], []
        for p in te:
            det = C.detect_events(probs[p], th)
            pb.append(sum(b - a for a, b in det) / 60)
            y = data[p]["yM"]
            ev = C.events_from_binary(y)
            rb.append(sum(b - a for a, b in ev) / 60)
        res["burden_r_M"] = round(float(np.corrcoef(pb, rb)[0, 1]), 4)
        results[name] = res
        log(f"{name}: test AUC(M)={res['auc_M']}  AUC(A/B/C)={res['auc_A']}/{res['auc_B']}/{res['auc_C']}"
            f"  GDR(M)={res['gdr_M']} @ {res['fdh_M']} FD/h  burden_r={res['burden_r_M']}"
            f"  [{(time.time()-t0)/60:.1f} min]")
        json.dump(results, open(OUT / "dev_results.json", "w"), indent=1)
    log("\nDone.")


if __name__ == "__main__":
    main()
