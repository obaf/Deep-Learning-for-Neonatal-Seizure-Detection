"""Collect every number quoted in the paper into results/paper_numbers.json,
computed directly from the raw result files (no hand-typed numbers)."""
import json
from pathlib import Path

import numpy as np

import annotations as A
import common as C
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DATA = HERE / "data"
OUT = {}

# ---------------- dataset + inter-rater ----------------
experts = {e: A.load_expert(DATA / f"annotations_2017_{e}.csv") for e in "ABC"}
pids = sorted(experts["A"].keys())
data = C.load_all()
n_pat = len(pids)
total_h = sum(data[p]["n_sec"] for p in data) / 3600
n_cons = sum(1 for p in pids if data[p]["yM"].any())
per = {}
for e in "ABC":
    ev = {p: A.events_from_binary(experts[e][p]) for p in pids}
    dur = np.array([b - a for v in ev.values() for a, b in v])
    per[e] = dict(patients=sum(1 for p in pids if ev[p]), events=len(dur),
                  hours=round(dur.sum() / 3600, 1), med_dur_s=float(np.median(dur)))
OUT["dataset"] = dict(patients=n_pat, hours=round(total_h, 1),
                      consensus_patients=n_cons, per_expert=per)

y = {t: np.concatenate([data[p][f"y{t}"] for p in pids]) for t in "ABCMU"}
OUT["interrater"] = {
    "fleiss": round(float(A.fleiss_kappa(np.column_stack([y[t] for t in "ABC"]))), 3),
    "kappa_AB": round(float(A.cohen_kappa(y["A"], y["B"])), 3),
    "kappa_AC": round(float(A.cohen_kappa(y["A"], y["C"])), 3),
    "kappa_BC": round(float(A.cohen_kappa(y["B"], y["C"])), 3),
}
match = {}
for e1, e2 in [("A", "B"), ("A", "C"), ("B", "C")]:
    tot = ok = 0
    ious = []
    for p in pids:
        ev1 = A.events_from_binary(experts[e1][p])
        ev2 = A.events_from_binary(experts[e2][p])
        tot += len(ev1)
        for a in ev1:
            hits = [A.iou(a, b) for b in ev2 if A.overlap(a, b) > 0]
            if hits:
                ok += 1
                ious.append(max(hits))
    match[f"{e1}{e2}"] = dict(rate=round(ok / tot, 3), median_iou=round(float(np.median(ious)), 2))
OUT["interrater"]["match"] = match
# consensus events
n3 = n2 = n1 = 0
for p in pids:
    marks = []
    for e in "ABC":
        marks += [(a, b, e) for a, b in A.events_from_binary(experts[e][p])]
    used, clusters = set(), []
    marks.sort(key=lambda m: m[0])
    for i, m in enumerate(marks):
        if i in used:
            continue
        cl, stack = {m[2]}, [i]
        used.add(i)
        while stack:
            j = stack.pop()
            for k2, m2 in enumerate(marks):
                if k2 not in used and A.overlap(marks[j][:2], m2[:2]) > 0:
                    used.add(k2); cl.add(m2[2]); stack.append(k2)
        clusters.append(cl)
    n3 += sum(1 for c in clusters if len(c) == 3)
    n2 += sum(1 for c in clusters if len(c) == 2)
    n1 += sum(1 for c in clusters if len(c) == 1)
OUT["interrater"]["clusters"] = dict(all3=n3, two=n2, one=n1, total=n1 + n2 + n3)

# ---------------- baselines + ablations ----------------
def load(r):
    f = RES / r
    return json.load(open(f)) if f.exists() else None

OUT["classic"] = load("classic_results.json") or {}
OUT["featsvm"] = load("featsvm_results.json") or {}
ab = load("ablate_results.json") or {}
OUT["ablation"] = ab
OUT["dev"] = load("dev_results.json") or {}
OUT["annot"] = load("annot_results.json") or {}
cv = load("final_results.json") or load("cv_results.json")
OUT["cv"] = cv

# AP + seizure-burden correlation per fold from saved CV probabilities
if cv and (RES / "final_results.json").exists():
    from sklearn.metrics import average_precision_score
    folds = cv.get("folds", [])
    for tag in ["A", "B", "C", "M", "U"]:
        aps = []
        for f in folds:
            fp = RES / f"final_probs_f{f['k']}.npz"
            if not fp.exists():
                continue
            probs = {int(k): v for k, v in np.load(fp).items()}
            P, Y = [], []
            for p, pv in probs.items():
                yy = data[p][f"y{tag}"]
                n = min(len(pv), len(yy))
                P.append(pv[:n]); Y.append(yy[:n])
            P, Y = np.concatenate(P), np.concatenate(Y)
            if Y.min() != Y.max():
                aps.append(average_precision_score(Y, P))
        if aps:
            cv.setdefault("summary", {}).setdefault("ensemble", {})[f"ap_{tag}"] = {
                "mean": round(float(np.mean(aps)), 4), "sd": round(float(np.std(aps)), 4)}
    # burden correlation per fold (threshold-free: top-decile seconds as detected)
    rs = []
    for f in folds:
        fp = RES / f"final_probs_f{f['k']}.npz"
        if not fp.exists():
            continue
        probs = {int(k): v for k, v in np.load(fp).items()}
        pb, rb = [], []
        for p, pv in probs.items():
            y = data[p]["yM"]
            n = min(len(pv), len(y))
            pv2 = pv[:n]
            det = pv2 >= np.quantile(pv2, 0.95)
            pb.append(det.sum() / 60)
            rb.append(y[:n].sum() / 60)
        if len(pb) > 2 and np.std(rb) > 0 and np.std(pb) > 0:
            rs.append(float(np.corrcoef(pb, rb)[0, 1]))
    if rs:
        cv["summary"]["ensemble"]["burden_r_M"] = {
            "mean": round(float(np.mean(rs)), 4), "sd": round(float(np.std(rs)), 4)}

    # post-hoc operating curve: GDR at matched test FD/h + burden correlation
    # at the 0.5-FD/h operating point (labelled as post-hoc)
    gdr_at = {0.5: [], 1.0: [], 2.0: []}
    rs_op = []
    for f in folds:
        fp = RES / f"final_probs_f{f['k']}.npz"
        if not fp.exists():
            continue
        probs = {int(k): v for k, v in np.load(fp).items()}
        hours = sum(data[p]["n_sec"] for p in probs) / 3600
        curve = []          # (FD/h, GDR, threshold)
        for th in np.arange(0.30, 0.96, 0.025):
            gd = fd = nr = 0
            for p, pv in probs.items():
                det = C.detect_events(pv, th)
                ref = C.events_from_binary(data[p]["yM"])
                g, dd, n = C.event_metrics(det, ref)
                gd += g; fd += dd; nr += n
            curve.append((fd / hours, gd / max(nr, 1), float(th)))
        for tgt in gdr_at:
            ok = [g for fdh, g, _ in curve if fdh <= tgt]
            gdr_at[tgt].append(max(ok) if ok else 0.0)
        # burden correlation at the tightest threshold meeting <=0.5 FD/h
        ok = [(fdh, t) for fdh, g, t in curve if fdh <= 0.5]
        if ok:
            th_op = min(ok)[1]
            pb, rb = [], []
            for p, pv in probs.items():
                det = C.detect_events(pv, th_op)
                pb.append(sum(b - a for a, b in det) / 60)
                ev = C.events_from_binary(data[p]["yM"])
                rb.append(sum(b - a for a, b in ev) / 60)
            if np.std(pb) > 0 and np.std(rb) > 0:
                rs_op.append(float(np.corrcoef(pb, rb)[0, 1]))
    cv["summary"]["ensemble"]["gdr_posthoc_M"] = {
        f"at_{t}": {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4)}
        for t, v in gdr_at.items() if v}
    if rs_op:
        cv["summary"]["ensemble"]["burden_r_M"] = {
            "mean": round(float(np.mean(rs_op)), 4), "sd": round(float(np.std(rs_op)), 4)}

# median per-patient AUC for ablation models (Tapani-comparable)
te18 = set(C.make_splits(data, val_frac=0.18)[0][2])
te0 = set(C.make_splits(data)[0][2])
med = {}
for f in sorted(RES.glob("ablate_test_probs_*.npz")):
    name = f.stem.replace("ablate_test_probs_", "")
    probs = {int(k): v for k, v in np.load(f).items()}
    te = te18 if set(map(int, probs)) == te18 else te0
    paucs = [roc_auc_score(data[p]["yM"], probs[p]) for p in te
             if data[p]["yM"].min() != data[p]["yM"].max()]
    if paucs:
        med[name] = dict(median=round(float(np.median(paucs)), 3),
                         mean=round(float(np.mean(paucs)), 3), n=len(paucs))
OUT["ablation_median_auc"] = med

# ---------------- literature comparison ----------------
OUT["literature"] = {
    "tapani2019": "median per-patient AUC 0.988 (IQR 0.931-0.998), SVM + time-varying correlations, patient-wise CV",
    "daly2024_smallsdb": "AUC 0.963 (O'Shea-style baseline 0.926); trained on all of Helsinki, tested on Cork",
    "hogan2024": "AUC 0.982 pooled on Helsinki; trained on 202 private Cork neonates; expert-equivalent",
    "frassineti2020": "AUC ~0.81 wavelet method",
}

# ---------------- annotator transfer spread ----------------
if annot_files := load("annot_results.json"):
    tags = ["A", "B", "C", "M"]
    spreads = []
    for t in tags:
        r = annot_files.get(f"train_{t}", {})
        vals = [r.get(f"auc_{e}") for e in tags if r.get(f"auc_{e}") is not None]
        if len(vals) >= 2:
            spreads.append(max(vals) - min(vals))
    if spreads:
        annot_files["spread"] = round(max(spreads), 3)
        OUT["annot"] = annot_files

# ---------------- aetiology medians (from CV probs) ----------------
import pandas as pd
pat = sorted(RES.glob("*_probs_f*.npz"))
if pat:
    probs = {}
    for f in pat:
        for k, v in np.load(f).items():
            probs.setdefault(int(k), []).append(v)
    probs = {p: np.mean(v, 0) for p, v in probs.items()}
    clin = pd.read_csv(DATA / "clinical_information.csv").set_index("ID")

    def group(pid):
        dx = str(clin.loc[pid, "Diagnosis"]).lower()
        loc = str(clin.loc[pid, "Primary Localisation"]).lower()
        if "bilateral" in loc or "both hemispheres" in loc:
            return "bilateral"
        if "asphyxia" in dx or "ischaemia" in dx or "hie" in dx:
            return "hie"
        if "infarction" in dx:
            return "infarction"
        return "other"
    auc_by = {}
    for p, pv in probs.items():
        y = data[p]["yM"]
        if y.min() == y.max():
            continue
        n = min(len(pv), len(y))
        auc_by.setdefault(group(p), []).append(roc_auc_score(y[:n], pv[:n]))
    OUT["aetiology"] = {g: round(float(np.median(v)), 3) for g, v in auc_by.items()}
    OUT["aetiology_n"] = {g: len(v) for g, v in auc_by.items()}

json.dump(OUT, open(RES / "paper_numbers.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) else list(v)[:6]) for k, v in OUT.items()}, indent=1)[:1500])
print("saved paper_numbers.json")
