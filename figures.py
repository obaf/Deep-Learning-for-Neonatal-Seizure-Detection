"""Generate all paper figures from saved results. Skips figures whose inputs
are missing. Output: figs/*.png at 300 dpi."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

import common as C

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIGS = HERE / "figs"
FIGS.mkdir(exist_ok=True)
DATA = HERE / "data"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans",
})
BLUE, ORANGE, GREEN, RED, GRAY = "#1A5276", "#E67E22", "#27AE60", "#C0392B", "#7F8C8D"


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- fig 1: inter-rater
def fig_interrater():
    import annotations as A
    experts = {e: A.load_expert(DATA / f"annotations_2017_{e}.csv") for e in "ABC"}
    pids = sorted(experts["A"].keys())
    stats = {}
    for e in "ABC":
        ev = {p: A.events_from_binary(experts[e][p]) for p in pids}
        dur = np.array([b - a for v in ev.values() for a, b in v])
        stats[e] = dict(n_pat=sum(1 for p in pids if ev[p]), n_ev=len(dur),
                        hours=dur.sum() / 3600, dur=dur / 60)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
    e = list("ABC")
    ax[0].bar(e, [stats[x]["n_ev"] for x in e], color=BLUE, alpha=.85)
    for i, x in enumerate(e):
        ax[0].text(i, stats[x]["n_ev"] + 5, str(stats[x]["n_ev"]), ha="center", fontsize=9)
    ax[0].set_ylabel("Seizure events (>10 s)")
    ax[0].set_title("(a) Events marked per expert")
    parts = ax[1].violinplot([stats[x]["dur"] for x in e], positions=range(3),
                             showmedians=True, widths=.8)
    for pc in parts["bodies"]:
        pc.set_facecolor(ORANGE); pc.set_alpha(.6)
    for k in ("cmins", "cmaxes", "cbars", "cmedians"):
        parts[k].set_color(BLUE)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(e)
    ax[1].set_ylabel("Event duration (min)")
    ax[1].set_title("(b) Event duration distributions")

    K = np.zeros((3, 3))
    labels = ["A", "B", "C"]
    for i, e1 in enumerate(labels):
        for j, e2 in enumerate(labels):
            y1 = np.concatenate([experts[e1][p] for p in pids])
            y2 = np.concatenate([experts[e2][p] for p in pids])
            K[i, j] = A.cohen_kappa(y1, y2) if i != j else 1.0
    im = ax[2].imshow(K, cmap="Blues", vmin=.5, vmax=1)
    for i in range(3):
        for j in range(3):
            ax[2].text(j, i, f"{K[i, j]:.2f}" if i != j else "1", ha="center",
                       va="center", fontsize=10,
                       color="white" if K[i, j] > .8 else "black")
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(labels)
    ax[2].set_yticks(range(3)); ax[2].set_yticklabels(labels)
    ax[2].set_title("(c) Per-second Cohen's kappa")
    ax[2].spines[:].set_visible(False)
    fig.colorbar(im, ax=ax[2], fraction=.046)
    _save(fig, "fig1_interrater.png")


# ---------------------------------------------------------------- fig 2: example trace
def fig_example(probs_file=None):
    data = C.load_all()
    cands = sorted(RES.glob("ablate_test_probs_*.npz"))
    if probs_file:
        cands = [RES / probs_file]
    if not cands:
        print("no probs for example figure")
        return
    probs = {int(k): v for k, v in np.load(cands[-1]).items()}
    te = C.make_splits(data, val_frac=0.18)[0][2]
    # pick the seizure patient with the best AUC for a clean illustration
    from sklearn.metrics import roc_auc_score as auc
    best, best_a = None, -1
    for p in te:
        y = data[p]["yM"]
        if y.min() == y.max() or p not in probs:
            continue
        a = auc(y, probs[p])
        if a > best_a:
            best, best_a = p, a
    p = best
    d = data[p]
    yA, yB, yC = d["yA"], d["yB"], d["yC"]
    yM = d["yM"]
    # find first consensus seizure
    on = np.where(np.diff(yM) == 1)[0]
    t0 = max(on[0] - 120, 0) if len(on) else 0
    t1 = min(t0 + 720, d["n_sec"])           # 12-minute window
    secs = np.arange(t0, t1)
    fig, axes = plt.subplots(4, 1, figsize=(10.5, 6.4), sharex=True,
                             gridspec_kw={"height_ratios": [2.6, .5, .9, .9]})
    ch_show = [1, 7, 13]                      # three bipolar derivations
    for k, ch in enumerate(ch_show[:1]):
        seg = d["x"][ch, t0 * C.SFREQ:t1 * C.SFREQ]
        axes[0].plot(secs, seg[:len(secs)] / 50 + k, lw=.4, color="black")
    axes[0].set_ylabel("EEG (a.u.)")
    axes[0].set_title(f"Patient eeg{p}: raw EEG (one bipolar derivation), expert labels, and model output")
    for y, lab, col in [(yA, "Expert A", BLUE), (yB, "Expert B", ORANGE), (yC, "Expert C", GREEN)]:
        axes[1].plot(secs, y[t0:t1] + 0.02 * (0 if lab == "Expert A" else 0), lw=.8, color=col, alpha=.7 if lab != "Expert A" else 1, label=lab)
    axes[1].legend(ncol=3, fontsize=8, loc="upper right", frameon=False)
    axes[1].set_ylabel("Expert\nlabels")
    axes[2].imshow(yM[t0:t1][None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1.5,
                   extent=[t0, t1, 0, 1])
    axes[2].set_ylabel("Consensus\nlabel")
    ps = probs[p][t0:t1]
    axes[3].plot(secs[:len(ps)], ps, color=RED, lw=1.1)
    axes[3].axhline(0.5, color=GRAY, ls="--", lw=.8)
    axes[3].set_ylabel("Model\nP(seizure)")
    axes[3].set_xlabel("Time (s from recording start)")
    axes[3].set_ylim(-.02, 1.02)
    _save(fig, "fig2_example.png")


# ---------------------------------------------------------------- fig 3: model comparison
def fig_models():
    data = C.load_all()
    te18 = set(C.make_splits(data, val_frac=0.18)[0][2])
    te0 = set(C.make_splits(data)[0][2])
    rows = []
    for f in sorted(RES.glob("ablate_test_probs_*.npz")):
        name = f.stem.replace("ablate_test_probs_", "")
        probs = {int(k): v for k, v in np.load(f).items()}
        te = te18 if set(map(int, probs)) == te18 else te0
        auc = C.pooled_auc({p: probs[p] for p in te}, data, "M")[0]
        sm = {p: C.detect_events(probs[p], 0.0, smooth_s=69) for p in te}  # placeholder
        rows.append((name, auc))
    for f in sorted(RES.glob("test_probs_*.npz")):
        if "train" in f.stem:
            continue
        name = f.stem.replace("test_probs_", "")
        probs = {int(k): v for k, v in np.load(f).items()}
        auc = C.pooled_auc(probs, data, "M")[0]
        rows.append((name, auc))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(rows) + 1.4))
    colors = [RED if n in ("multih", "multih_s8", "multih32", "multih60") else BLUE
              for n in names]
    ax.barh(names, vals, color=colors, alpha=.88)
    for i, v in enumerate(vals):
        ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.6, max(vals) + 0.06)
    ax.set_xlabel("Pooled per-second AUC vs consensus (fold-0 held-out patients)")
    _save(fig, "fig3_models.png")


# ---------------------------------------------------------------- fig 4: transfer matrix
def fig_transfer():
    f = RES / "annot_results.json"
    if not f.exists():
        print("skip transfer")
        return
    R = json.load(open(f))
    tags = ["A", "B", "C", "M"]
    M = np.zeros((4, 4))
    for i, tr in enumerate(tags):
        r = R.get(f"train_{tr}", {})
        for j, ev in enumerate(tags):
            M[i, j] = r.get(f"auc_{ev}", np.nan)
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1.0)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > .85 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(["A", "B", "C", "Consensus"])
    ax.set_yticks(range(4)); ax.set_yticklabels(["A", "B", "C", "Consensus"])
    ax.set_xlabel("Evaluated against")
    ax.set_ylabel("Trained on labels of")
    ax.set_title("Annotator transfer: pooled AUC")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=.046)
    _save(fig, "fig4_transfer.png")


# ---------------------------------------------------------------- fig 5: ROC/PR of final CV
def fig_roc():
    pat = sorted(RES.glob("*_probs_f*.npz"))
    if not pat:
        print("skip roc")
        return
    data = C.load_all()
    probs = {}
    for f in pat:
        for k, v in np.load(f).items():
            probs.setdefault(int(k), []).append(v)
    probs = {p: np.mean(v, 0) for p, v in probs.items()}
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    colors = {"A": BLUE, "B": ORANGE, "C": GREEN, "M": RED}
    names = {"A": "Expert A", "B": "Expert B", "C": "Expert C", "M": "Consensus"}
    for tag in ["A", "B", "C", "M"]:
        P, Y = [], []
        for p, pv in probs.items():
            y = data[p][f"y{tag}"]
            n = min(len(pv), len(y))
            P.append(pv[:n]); Y.append(y[:n])
        P, Y = np.concatenate(P), np.concatenate(Y)
        fpr, tpr, _ = roc_curve(Y, P)
        ax[0].plot(fpr, tpr, color=colors[tag], lw=1.4,
                   label=f"{names[tag]} (AUC {roc_auc_score(Y, P):.3f})")
        from sklearn.metrics import precision_recall_curve
        pr, rc, _ = precision_recall_curve(Y, P)
        ax[1].plot(rc, pr, color=colors[tag], lw=1.4,
                   label=f"{names[tag]} (AP {average_precision_score(Y, P):.3f})")
    ax[0].plot([0, 1], [0, 1], color=GRAY, ls=":", lw=.8)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) ROC, pooled test seconds"); ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("(b) Precision-recall"); ax[1].legend(fontsize=8)
    _save(fig, "fig5_roc.png")


# ---------------------------------------------------------------- fig 6: aetiology
def fig_aetiology():
    import pandas as pd
    pat = sorted(RES.glob("*_probs_f*.npz"))
    if not pat:
        print("skip aetiology")
        return
    data = C.load_all()
    probs = {}
    for f in pat:
        for k, v in np.load(f).items():
            probs.setdefault(int(k), []).append(v)
    probs = {p: np.mean(v, 0) for p, v in probs.items()}
    clin = pd.read_csv(DATA / "clinical_information.csv").set_index("ID")

    def group(pid):
        dx = str(clin.loc[pid, "Diagnosis"]).lower()
        loc = str(clin.loc[pid, "Primary Localisation"]).lower()
        if "bilateral" in loc or "both" in loc:
            return "Bilateral"
        if "asphyxia" in dx or "ischaemia" in dx or "hie" in dx:
            return "HIE / asphyxia"
        if "infarction" in dx or "stroke" in dx:
            return "Focal infarction"
        return "Other"
    rows = []
    for p, pv in probs.items():
        y = data[p]["yM"]
        if y.min() == y.max():
            continue
        n = min(len(pv), len(y))
        rows.append((group(p), roc_auc_score(y[:n], pv[:n])))
    if not rows:
        return
    df = pd.DataFrame(rows, columns=["group", "auc"])
    order = ["Bilateral", "HIE / asphyxia", "Focal infarction", "Other"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, g in enumerate(order):
        v = df[df.group == g]["auc"].values
        if not len(v):
            continue
        ax.scatter(np.full(len(v), i) + np.random.uniform(-.09, .09, len(v)), v,
                   s=26, color=BLUE, alpha=.7, zorder=3)
        ax.hlines(np.median(v), i - .2, i + .2, color=RED, lw=2, zorder=4)
        ax.text(i, .30, f"n={len(v)}", ha="center", fontsize=8, color=GRAY)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, fontsize=8)
    ax.set_ylabel("Per-patient AUC (consensus)")
    ax.set_ylim(0.2, 1.02)
    ax.axhline(0.5, color=GRAY, ls=":", lw=.8)
    _save(fig, "fig6_aetiology.png")


if __name__ == "__main__":
    fig_interrater()
    fig_example()
    fig_models()
    fig_transfer()
    fig_roc()
    fig_aetiology()
