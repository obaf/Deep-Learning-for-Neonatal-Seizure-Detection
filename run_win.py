"""Longer-window ablations (32-s / 60-s), Tapani-style analysis context.
Usage: python run_win.py <win_s> <name>   (e.g. python run_win.py 60 multih60)"""
import json
import os
import sys
import time
from pathlib import Path

WIN = int(sys.argv[1]) if len(sys.argv) > 1 else 60
NAME = sys.argv[2] if len(sys.argv) > 2 else f"win{WIN}"
os.environ["WIN_S"] = str(WIN)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402
import models  # noqa: E402
import train as T  # noqa: E402
from run_ablate import predict  # noqa: E402
from run_annot import eval_probs  # noqa: E402

OUT = Path(__file__).parent / "results"

CONFIGS = {
    32: dict(arch="multih", norm="patient", tag="MULTI", mixup=0.2, epochs=35,
             stride=16, batch=48),
    60: dict(arch="multih", norm="patient", tag="MULTI", mixup=0.2, epochs=30,
             stride=30, batch=24),
}[WIN]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== {NAME}: WIN_S={WIN}s {CONFIGS} ===", flush=True)
    data = C.load_all(norm=CONFIGS["norm"])
    tr, va, te = C.make_splits(data, val_frac=0.18)[0]
    model, best_val, _ = T.train_model(
        data, tr, va, models.build(CONFIGS["arch"]), tag=CONFIGS["tag"],
        mixup=CONFIGS["mixup"], epochs=CONFIGS["epochs"], stride=CONFIGS["stride"],
        batch=CONFIGS["batch"], patience=10, lr=1.2e-3,
        log=lambda s: print(s, flush=True))
    torch.save(model.state_dict(), OUT / f"ablate_{NAME}.pt")
    val_probs = predict(model, data, va, device, False)
    smooth, th = C.tune_postproc(val_probs, data, "M", target_fd_h=0.5)
    probs = predict(model, data, te, device, False)
    np.savez_compressed(OUT / f"ablate_test_probs_{NAME}.npz",
                        **{str(p): probs[p] for p in te})
    r = eval_probs(probs, data, te, th, smooth)
    r.update(best_val_auc=round(float(best_val), 4), smooth=int(smooth),
             threshold=float(th), win_s=WIN)
    res = json.load(open(OUT / "ablate_results.json")) if (OUT / "ablate_results.json").exists() else {}
    res[NAME] = r
    json.dump(res, open(OUT / "ablate_results.json", "w"), indent=1)
    print(f"{NAME}: AUC(M)={r['auc_M']} A/B/C={r['auc_A']}/{r['auc_B']}/{r['auc_C']} "
          f"GDR(M)={r['gdr_M']} @{r['fdh_M']} FD/h", flush=True)


if __name__ == "__main__":
    main()
