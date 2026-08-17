"""Preprocess Helsinki neonatal EEG into ML-ready per-patient arrays.

Pipeline (following Daly et al. 2024 / Hogan et al. 2024 conventions):
- 19 referential EEG electrodes -> 18-derivation bipolar double-banana montage
- Bandpass 0.5-12.8 Hz (FIR) on 256 Hz data
- Downsample to 32 Hz
- Trim per-second expert labels to recording duration, pad if short
- Save data/proc/eegN.npy (float32, 18 x T) + labels in shared arrays

Incremental: skips already-processed patients, so it can run while download continues.
"""
import sys
import time
from pathlib import Path

import mne
import numpy as np

mne.set_log_level("ERROR")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PROC = DATA / "proc"
PROC.mkdir(exist_ok=True)

SFREQ = 32.0
BIPOLARS = [
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    ("Fz", "Cz"), ("Cz", "Pz"),
]

ann = np.load(DATA / "annotations.npz", allow_pickle=True)
pids = list(ann["pids"])
lengths = dict(zip(pids, ann["lengths"]))


def process(pid):
    out = PROC / f"eeg{pid}.npz"
    if out.exists():
        return "cached"
    edf = DATA / f"eeg{pid}.edf"
    if not edf.exists():
        return "missing-edf"
    raw = mne.io.read_raw_edf(edf, preload=True, verbose="ERROR")
    # keep only the EEG electrodes used in the bipolar chain
    ch_map = {ch.split()[1].split("-")[0]: ch for ch in raw.ch_names}
    keep = [v for k, v in ch_map.items() if k in {a for a, _ in BIPOLARS} | {b for _, b in BIPOLARS}]
    raw.pick(keep)
    names = [ch.split()[1].split("-")[0] for ch in raw.ch_names]
    arr = raw.get_data()
    data = dict(zip(names, arr))
    bp = np.stack([data[a] - data[b] for a, b in BIPOLARS])  # (18, T) volts
    sf = raw.info["sfreq"]
    bp = mne.filter.filter_data(bp, sf, l_freq=0.5, h_freq=12.8, verbose="ERROR")
    if sf != SFREQ:
        n_out = int(bp.shape[1] * SFREQ / sf)
        bp = mne.filter.resample(bp, up=SFREQ / sf, npad="auto", axis=-1)[:, :n_out]
    dur_s = bp.shape[1] / SFREQ
    # scale to microvolts and to a sane numeric range (float32)
    bp = (bp * 1e6).astype(np.float32)
    # per-second labels trimmed/padded to true duration
    n_sec = int(round(dur_s))
    lab = {}
    for tag in list("ABC") + ["M", "U"]:
        a = ann[f"{tag}_{pid}"].astype(np.int8)
        lab[tag] = a[:n_sec] if len(a) >= n_sec else np.pad(a, (0, n_sec - len(a)))
    np.savez_compressed(out, x=bp, sfreq=SFREQ,
                        **{f"y{t}": v for t, v in lab.items()})
    return f"ok ({bp.shape[0]}x{bp.shape[1]}, {dur_s/60:.0f} min)"


def main():
    t0 = time.time()
    done = skip = miss = 0
    for pid in pids:
        r = process(pid)
        if r == "cached":
            skip += 1
        elif r == "missing-edf":
            miss += 1
        else:
            done += 1
            print(f"eeg{pid}: {r} [{time.time()-t0:.0f}s]", flush=True)
    print(f"processed={done} cached={skip} missing={miss} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
