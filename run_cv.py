"""Final 5-fold patient-wise cross-validation with the winning recipe and a
seed-ensemble per fold. Usage:
    python run_cv.py recipe.json
recipe.json example:
  {"arch": "resnet-s", "norm": "patient", "tag": "M", "mixup": 0.2,
   "chanind": false, "epochs": 45, "seeds": [0, 1, 2], "val_frac": 0.18,
   "lr": 0.0012, "name": "final"}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import common as C
import models
import train as T
from run_annot import eval_probs

OUT = Path(__file__).parent / "results"


def predict(model, data, pids, device, chanind):
    fn = C.predict_seconds_chanind if chanind else C.predict_seconds
    return {p: fn(model, data[p], device) for p in pids}


def main(recipe_path):
    R = json.load(open(recipe_path))
    name = R.get("name", Path(recipe_path).stem)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = C.load_all(norm=R.get("norm"))
    splits = C.make_splits(data, val_frac=R.get("val_frac", 0.18))
    seeds = R.get("seeds", [0, 1, 2])
    results = {"recipe": R, "folds": []}
    agg = {}

    def add(what, r):
        for key, v in r.items():
            if isinstance(v, (int, float)):
                agg.setdefault((what, key), []).append(v)

    for k, (tr, va, te) in enumerate(splits):
        t_fold = time.time()
        fold = {"k": k, "train": tr, "val": va, "test": te}
        val_seed_probs, test_seed_probs = {}, {}
        best_vals = []
        for seed in seeds:
            t0 = time.time()
            print(f"\n=== fold {k} seed {seed} ===", flush=True)
            model, best_val, _ = T.train_model(
                data, tr, va, models.build(R["arch"]), tag=R.get("tag", "M"),
                mixup=R.get("mixup", 0.0), chanind=R.get("chanind", False),
                epochs=R.get("epochs", 45), patience=12, lr=R.get("lr", 1.2e-3),
                seed=seed, log=lambda s: print(s, flush=True))
            torch.save(model.state_dict(), OUT / f"{name}_f{k}_s{seed}.pt")
            best_vals.append(best_val)
            val_seed_probs[seed] = predict(model, data, va, device, R.get("chanind", False))
            test_seed_probs[seed] = predict(model, data, te, device, R.get("chanind", False))
            print(f"fold {k} seed {seed}: valAUC {best_val:.4f} "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
        # ensemble (mean of seeds)
        ens_val = {p: np.mean([val_seed_probs[s][p] for s in seeds], axis=0) for p in va}
        ens = {p: np.mean([test_seed_probs[s][p] for s in seeds], axis=0) for p in te}
        np.savez_compressed(OUT / f"{name}_probs_f{k}.npz",
                            **{str(p): ens[p] for p in te})
        smooth, th = C.tune_postproc(ens_val, data, "M", target_fd_h=0.5)
        r_ens = eval_probs(ens, data, te, th, smooth)
        r_ens.update(smooth=int(smooth), threshold=float(th),
                     best_val_auc=round(float(np.mean(best_vals)), 4))
        fold["ensemble"] = r_ens
        add("ensemble", r_ens)
        # single-seed (seed 0) for the ensemble-gain comparison
        r_s0 = eval_probs(test_seed_probs[seeds[0]], data, te, th, smooth)
        fold["single_seed0"] = r_s0
        add("single", r_s0)
        results["folds"].append(fold)
        print(f"fold {k}: ENS AUC(M)={r_ens['auc_M']} GDR(M)={r_ens['gdr_M']} "
              f"@{r_ens['fdh_M']} FD/h | single AUC(M)={r_s0['auc_M']} "
              f"[{(time.time()-t_fold)/60:.1f} min total]", flush=True)
        json.dump(results, open(OUT / f"{name}_results.json", "w"), indent=1)

    summary = {}
    for (what, key), vals in sorted(agg.items()):
        summary.setdefault(what, {})[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "sd": round(float(np.std(vals)), 4)}
    results["summary"] = summary
    json.dump(results, open(OUT / f"{name}_results.json", "w"), indent=1)
    print("\n=== 5-fold summary (mean +/- sd) ===")
    for what in ("ensemble", "single"):
        d = summary.get(what, {})
        print(f"{what}: AUC(M)={d.get('auc_M')} AUC(A)={d.get('auc_A')} "
              f"AUC(B)={d.get('auc_B')} AUC(C)={d.get('auc_C')} "
              f"GDR(M)={d.get('gdr_M')} FD/h(M)={d.get('fdh_M')}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "recipe.json")
