"""Parse 3-expert per-second annotations from the Helsinki neonatal EEG dataset and
compute inter-expert variability statistics.

Annotation CSVs: rows = seconds (1 Hz), columns = patients 1..79, values 0/1.
Outputs: data/annotations.npz with per-patient binary series per expert + consensus.
Prints an inter-expert variability report (validated against Stevenson et al. 2019).
"""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
N_EXP = 3
MERGE_GAP = 10   # merge runs separated by < MERGE_GAP s (detection tolerance)
MIN_DUR = 10     # seizures defined as events > 10 s (Stevenson et al. 2019)


def load_expert(path):
    df = pd.read_csv(path, header=0)          # first row: patient numbers 1..79
    df.columns = [int(c) for c in df.columns]
    return {pid: df[pid].to_numpy(dtype=np.int8) for pid in df.columns}


def events_from_binary(x, merge_gap=MERGE_GAP, min_dur=MIN_DUR):
    """Extract events (start_s, stop_s) from a 1 Hz binary series."""
    x = x.copy()
    # merge runs separated by small gaps (label noise at boundaries)
    if merge_gap > 0 and x.any():
        d = np.diff(x)
        gap_starts = np.where(d == -1)[0]
        gap_ends = np.where(d == 1)[0]
        for gs in gap_starts:
            ge = gap_ends[gap_ends > gs]
            if len(ge) and (ge[0] - gs) <= merge_gap:
                x[gs + 1:ge[0] + 1] = 1
    runs, start = [], None
    for i, v in enumerate(x):
        if v == 1 and start is None:
            start = i
        elif v == 0 and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(x)))
    return [(s, e) for s, e in runs if (e - s) >= min_dur]


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    inter = overlap(a, b)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def cohen_kappa(y1, y2):
    p_o = np.mean(y1 == y2)
    pe = np.mean(y1) * np.mean(y2) + (1 - np.mean(y1)) * (1 - np.mean(y2))
    return (p_o - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(mat):
    """mat: (n_seconds, 3 raters -> counts of 1s) as (n, 2) counts [n0, n1]."""
    n, k = mat.shape[0], mat.sum(axis=1)[0] if mat.shape[0] else 3
    counts = np.column_stack([3 - mat.sum(axis=1), mat.sum(axis=1)])
    p_j = counts.sum(axis=0) / (n * 3)
    P_i = ((counts ** 2).sum(axis=1) - 3) / (3 * 2)
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def main():
    experts = {e: load_expert(DATA / f"annotations_2017_{e}.csv") for e in "ABC"}
    pids = sorted(experts["A"].keys())
    # sanity: same length per patient across experts
    for pid in pids:
        assert len(experts["A"][pid]) == len(experts["B"][pid]) == len(experts["C"][pid])
    L = {pid: len(experts["A"][pid]) for pid in pids}
    total_h = sum(L.values()) / 3600
    print(f"patients: {len(pids)}; annotation rows total: {total_h:.1f} h (max rec {max(L.values())/3600:.1f} h)")

    # per-expert stats
    print("\n== Per-expert statistics ==")
    for e in "ABC":
        ev = {pid: events_from_binary(experts[e][pid]) for pid in pids}
        n_pat = sum(1 for pid in pids if ev[pid])
        n_ev = sum(len(v) for v in ev.values())
        dur = np.array([b - a for v in ev.values() for a, b in v])
        tot_h = dur.sum() / 3600
        print(f"Expert {e}: {n_pat} patients w/ seizures, {n_ev} events, "
              f"seizure time {tot_h:.1f} h, median event {np.median(dur):.0f} s (IQR {np.percentile(dur,25):.0f}-{np.percentile(dur,75):.0f})")

    # per-second pairwise kappa + event matching
    print("\n== Pairwise per-second agreement (pooled seconds) ==")
    for e1, e2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        y1 = np.concatenate([experts[e1][p] for p in pids])
        y2 = np.concatenate([experts[e2][p] for p in pids])
        k = cohen_kappa(y1, y2)
        jac = np.logical_and(y1, y2).sum() / np.logical_or(y1, y2).sum()
        print(f"{e1}-{e2}: kappa={k:.3f}, Jaccard={jac:.3f}, corr={np.corrcoef(y1, y2)[0,1]:.3f}")

    fleiss_mats = np.column_stack([np.concatenate([experts[e][p] for p in pids]) for e in "ABC"])
    print(f"Fleiss kappa (per-second, pooled): {fleiss_kappa(fleiss_mats):.3f}")

    print("\n== Event-level matching (any overlap) ==")
    for e1, e2 in [("A", "B"), ("A", "C"), ("B", "C")]:
        matched, total, ious = 0, 0, []
        for pid in pids:
            ev1 = events_from_binary(experts[e1][pid])
            ev2 = events_from_binary(experts[e2][pid])
            total += len(ev1)
            for a in ev1:
                hits = [iou(a, b) for b in ev2 if overlap(a, b) > 0]
                if hits:
                    matched += 1
                    ious.append(max(hits))
        print(f"{e1}-> {e2}: {matched}/{total} ({100*matched/total:.1f}%) events matched, "
              f"median IoU of matches {np.median(ious):.2f}")

    # consensus analysis: events by # experts agreeing
    print("\n== Consensus analysis (events seen by >=1/2/3 experts via any-overlap clustering) ==")
    n1 = n2 = n3 = 0
    for pid in pids:
        evs = {e: events_from_binary(experts[e][pid]) for e in "ABC"}
        marks = []
        for e in "ABC":
            for a, b in evs[e]:
                marks.append([a, b, e])
        used = set()
        clusters = []
        marks.sort(key=lambda m: m[0])
        for i, (a, b, e) in enumerate(marks):
            if i in used:
                continue
            cluster, stack = {e}, [i]
            used.add(i)
            while stack:
                j = stack.pop()
                ca, cb, ce = marks[j]
                for k2, (a2, b2, e2) in enumerate(marks):
                    if k2 not in used and overlap((ca, cb), (a2, b2)) > 0:
                        used.add(k2)
                        cluster.add(e2)
                        stack.append(k2)
            clusters.append(cluster)
        n3 += sum(1 for c in clusters if len(c) == 3)
        n2 += sum(1 for c in clusters if len(c) == 2)
        n1 += sum(1 for c in clusters if len(c) == 1)
    tot = n1 + n2 + n3
    print(f"total clustered events: {tot} | 3-expert: {n3} ({100*n3/tot:.0f}%), "
          f"2-expert: {n2} ({100*n2/tot:.0f}%), 1-expert: {n1} ({100*n1/tot:.0f}%)")
    print("(Stevenson et al. 2019 reported 1379 total: 889 / 295 / 195 = 65% / 21% / 14%)")

    # consensus (majority >=2) + union binary series
    consensus, union = {}, {}
    for pid in pids:
        m = np.stack([experts[e][pid] for e in "ABC"]).sum(axis=0)
        consensus[pid] = (m >= 2).astype(np.int8)
        union[pid] = (m >= 1).astype(np.int8)
    np.savez_compressed(
        DATA / "annotations.npz",
        pids=np.array(pids),
        lengths=np.array([L[p] for p in pids]),
        **{f"A_{p}": experts["A"][p] for p in pids},
        **{f"B_{p}": experts["B"][p] for p in pids},
        **{f"C_{p}": experts["C"][p] for p in pids},
        **{f"M_{p}": consensus[p] for p in pids},
        **{f"U_{p}": union[p] for p in pids},
    )
    print(f"\nSaved data/annotations.npz ({len(pids)} patients)")


if __name__ == "__main__":
    main()
