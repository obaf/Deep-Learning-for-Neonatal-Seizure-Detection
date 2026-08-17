# Deep Learning for Neonatal Seizure Detection

**Multi-annotator deep learning for neonatal seizure detection on the open Helsinki neonatal EEG dataset — with a full quantification of inter-expert annotation variability.**

This repository contains the complete, reproducible research pipeline accompanying the manuscript
`Neonatal-Seizure-Detection-Paper.docx`: data download, preprocessing, inter-expert variability
analysis, baseline and deep-learning experiments, evaluation, figures, paper generation, and all
trained model checkpoints.

---

## 1. Aims and Objectives

**Overall aim.** To develop and rigorously evaluate patient-independent deep-learning models for
automated seizure detection in neonatal EEG, explicitly treating expert label variability as both
a measurable phenomenon and a training signal, using the open Helsinki dataset.

**Specific objectives.**

1. **Quantify inter-expert annotation variability** among the three clinical neurophysiologists who
   annotated the Helsinki dataset (per-second Cohen's/Fleiss kappa, event-level overlap, consensus
   structure), validating our pipeline against the original data paper.
2. **Benchmark the spectrum of detection approaches** under one strict, patient-wise protocol —
   classic feature engineering (LightGBM, RBF-SVM, coherence/correlation features), single-label
   deep CNNs, and longer-window variants.
3. **Exploit multi-annotator labels in training** via (a) soft label averaging, (b) a multi-head
   network with one output head per expert and a shared trunk, and (c) seed-ensembling — evaluated
   against consensus labels.
4. **Quantify how the choice of annotator affects measured performance** through a 4x4
   annotator-transfer matrix (train on one expert, evaluate against all).
5. **Deliver a fully reproducible pipeline** (code, splits, checkpoints, per-fold predictions) that
   other researchers can verify and extend.

---

## 2. Summary of Findings

### 2.1 Inter-expert annotation variability (79 infants, 111.9 h)

| Measure | Value |
|---|---|
| Per-second Fleiss kappa (3 experts, pooled) | **0.756** (data paper reports 0.767 with their event def.; we reproduce the scale exactly) |
| Pairwise per-second Cohen's kappa | A-B 0.748, A-C 0.801, B-C 0.754 |
| Events marked by all 3 experts | **43%** (236/543 clustered events); 18% by exactly two; 39% by one only |
| Event-level any-overlap matching | 82% (A->B), 78% (A->C), 72% (B->C); median IoU of matched events 0.60-0.86 |
| Seizure-time estimates | differ by up to 26% between experts (14.0-17.6 h) |

Roughly one in three events that any given expert calls a seizure is contested by at least one
colleague — irreducible label noise that every detector trained on this corpus must confront.

### 2.2 Model comparison (development split, 17 held-out patients, consensus evaluation)

| Method | Pooled AUC |
|---|---|
| RBF-SVM + spectral features | 0.688 |
| LightGBM + spectral features | 0.751 |
| RBF-SVM + correlation/coherence features | 0.725 |
| CNN, per-window z-score | 0.785 |
| CNN, patient-wise z-score | 0.794 |
| CNN, soft multi-annotator labels | 0.799 |
| CNN, consensus + mixup | 0.818 |
| Multi-head CNN, 60-s windows | 0.821 |
| **Multi-head CNN, 16-s windows (proposed)** | **0.833** |
| Multi-head CNN, 32-s windows | 0.834 |

Multi-annotator training (one head per expert) beat every single-label configuration; mixup was
the single most effective augmentation; output smoothing (69-96 s) adds a further +0.02-0.04 AUC.

### 2.3 Final system: 5-fold patient-wise cross-validation (3-seed ensemble)

| Metric (vs consensus) | Value |
|---|---|
| Pooled per-second AUC | **0.809 +/- 0.052** (single seed 0.785 +/- 0.069) |
| AUC by evaluation target | Expert A 0.808, Expert C 0.795, Expert B 0.767, Union 0.761 |
| Average precision | 0.533 +/- 0.114 |
| Good-detection rate (tuned threshold) | 32% @ 0.90 FD/h (large fold-to-fold operating-point variance) |
| Good-detection rate (post-hoc matched rates) | 29% @ 0.5 FD/h; 34% @ 1.0 FD/h; 44% @ 2.0 FD/h |
| Seizure-burden correlation | r = 0.56 +/- 0.14 |

### 2.4 Annotator-transfer matrix (train on one expert, evaluate against all)

| Trained on | vs A | vs B | vs C | vs Consensus |
|---|---|---|---|---|
| Expert A | **0.842** | 0.760 | 0.805 | 0.839 |
| Expert B | 0.782 | 0.683 | 0.753 | 0.767 |
| Expert C | 0.832 | 0.724 | **0.805** | 0.814 |
| Consensus  | 0.819 | 0.702 | 0.805 | 0.804 |

Label-source choice alone shifts measured AUC by **up to 0.116** — the same order as many
published methodological improvements. A model trained on Expert B's labels evaluates *worse
against B himself* (0.683) than models trained on other experts — consensus acts as a denoiser.

### 2.5 Aetiology stratification

Per-patient AUC by aetiology (medians across CV): HIE/ischaemia **0.756** (n=21, hardest),
focal infarction 0.767 (n=4), other 0.795 (n=10), bilateral 0.840 (n=11). Failures concentrate
exactly where human experts disagree most.

### 2.6 Positioning vs the literature

| Study | AUC | Protocol context (not directly comparable) |
|---|---|---|
| Frassineti et al. 2020 | ~0.81 | Helsinki, wavelet features |
| Tapani et al. 2019 | 0.988 | **Median per-patient** AUC, SVM + per-patient feature normalisation |
| Daly et al. 2024 | 0.963 | Trained on all of Helsinki, tested on **external private** Cork set |
| Hogan et al. 2024 | 0.982 | Trained on **202 private Cork neonates**, tested on Helsinki |
| **This work** | **0.809 +/- 0.052** | Strict within-corpus 5-fold patient-wise CV; post-processing tuned on validation only |

Our numbers are the honest result of the strictest protocol we could define (no test-patient
information of any kind reaches training, validation, or threshold tuning). The paper quantifies
how protocol details — annotator identity, pooled vs per-patient metrics, smoothing, external vs
internal training data — explain much of the spread in the published literature.

---

## 3. Repository Structure

```
Deep Learning for Neonatal Seizure Detection/
├── README.md                          <- this file
├── requirements.txt                   <- pinned Python environment
├── .gitignore
├── Neonatal-Seizure-Detection-Paper.docx   <- the manuscript (ready-to-submit draft)
├── download_data.py                   <- downloads the dataset from Zenodo (checksummed, resumable)
├── annotations.py                     <- parses 3-expert labels; inter-expert variability analysis
├── preprocess.py                      <- bipolar montage, 0.5-12.8 Hz, 32 Hz, per-patient arrays
├── common.py                          <- data loading, splits, windowing, metrics, post-processing
├── models.py                          <- ShallowCNN, ResNet1D, CRNN, MultiHeadResNet
├── train.py                           <- training loop (augmentation, mixup, AMP, early stopping)
├── classic.py                         <- LightGBM / RBF-SVM spectral baseline
├── classic2.py                        <- Tapani-inspired correlation/coherence SVM baseline
├── run_dev.py                         <- architecture comparison on the development split
├── run_ablate.py / run_ablate2.py     <- training-recipe ablations (norm, mixup, soft, multih)
├── run_win.py                         <- longer-window (32 s / 60 s) experiments
├── run_annot.py                       <- 4x4 annotator-transfer experiments
├── run_cv.py                          <- final 5-fold CV with seed ensembles (takes recipe JSON)
├── recipe_final.json                  <- the exact winning configuration
├── figures.py                         <- regenerates all 6 paper figures from results
├── make_paper_numbers.py              <- recomputes every number quoted in the paper
├── generate_paper.js                  <- rebuilds the .docx manuscript (needs Node + docx pkg)
├── data/
│   └── annotations.npz                <- derived per-second label matrix for all 79 patients
│                                      (raw EDFs are NOT included - see Section 4)
├── results/                           <- ALL experiment outputs: metrics JSONs, 30 trained
│                                      checkpoints (.pt), per-fold/per-model test predictions (npz)
└── figs/                              <- the 6 publication figures (300 dpi PNG)
```

Because `results/` includes all per-fold test predictions and trained weights, **every metric in
the paper can be re-verified without a GPU** (see Section 6.3).

---

## 4. The Dataset (not included - how to obtain it)

The pipeline uses the **Helsinki neonatal EEG dataset**:

> Stevenson NJ, Tapani KT, Lauronen L, Vanhatalo S. *A dataset of neonatal EEG recordings with
> seizure annotations.* **Scientific Data 6:190039 (2019).**
> Zenodo record **2547147**, DOI 10.5281/zenodo.2547147, licence **CC-BY 4.0**.

- 79 term infants, NICU of Helsinki University Hospital; 256 Hz, 19 referential EEG electrodes
  (18-derivation bipolar montage used for reading); median recording 74 min; **111.9 h total**.
- Three clinical neurophysiologists independently annotated every second; seizures > 10 s.
- ~4.34 GB across 85 files.

**Download (either way):**

```bash
# Option A - automatic (checksums verified, resumable, ~15-40 min):
python download_data.py            # files land in ./data/

# Option B - manual: download all files from
#   https://zenodo.org/record/2547147
# and place them in ./data/   (keep the file names unchanged)
```

The raw EDFs are excluded from this repository to respect its size and licence terms;
`data/annotations.npz` (a tiny derived label matrix) **is** included so label-side analyses run
without the raw data.

---

## 5. Environment Setup

Tested on Windows 10/11 x64 with Python 3.12 and a 2 GB GPU (results are computable on CPU too).

```bash
# 1) create the environment (uv is fastest; any Python 3.10-3.12 works)
uv venv --python 3.12 .venv

# 2) install core packages
uv pip install --python .venv -r requirements.txt

# 3) PyTorch with CUDA (skip if CPU-only: uv pip install torch)
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Hardware used for the reported results: NVIDIA GeForce MX450 (2 GB), ~5-6 h total for the full
pipeline; the final 5-fold CV with 3 seeds is ~3 h on that GPU.

Rebuilding the manuscript (optional; the finished .docx is already included):

```bash
npm install docx image-size
node generate_paper.js
```

---

## 6. Reproducing the Results

### 6.1 Full pipeline from scratch

```bash
python download_data.py      # 1. dataset -> ./data/                (~15-40 min)
python annotations.py        # 2. parse labels + variability report  (~1 min)
python preprocess.py         # 3. bipolar montage, filter, resample  (~4 min)
python classic.py            # 4a. classic baselines                 (~20 min, CPU)
python classic2.py           # 4b. correlation/coherence SVM         (~10 min, CPU)
python run_dev.py            # 5.  architecture comparison           (~1 h, GPU)
python run_ablate.py         # 6.  recipe ablations                  (~40 min, GPU)
python run_ablate2.py        # 7.  multi-head ablations              (~35 min, GPU)
python run_win.py 32 multih32   # 8a. 32-s windows                  (~15 min, GPU)
python run_win.py 60 multih60   # 8b. 60-s windows                  (~20 min, GPU)
python run_annot.py          # 9.  4x4 annotator transfer           (~35 min, GPU)
python run_cv.py recipe_final.json  # 10. FINAL 5-fold CV x 3 seeds (~3 h, GPU)
python make_paper_numbers.py # 11. every number in the paper        (~2 min, CPU)
python figures.py            # 12. all 6 figures                    (~2 min, CPU)
node generate_paper.js       # 13. (optional) rebuild the manuscript
```

All steps are idempotent/resumable where it matters (download skips verified files; preprocessing
skips processed patients; ablations skip completed configs).

### 6.2 Preprocessing and protocol (what the scripts implement)

- 19 referential electrodes -> 18-derivation bipolar (double-banana) montage; 0.5-12.8 Hz FIR
  bandpass; downsample to 32 Hz (conventions of Daly et al. 2024).
- Robust per-recording z-scoring (median/MAD) per channel - preserves amplitude dynamics.
- 16-s windows; a window is positive if >= 50% of its seconds are labelled seizure.
- Training: AdamW + one-cycle, BCE with pos-weight (clamped 4:1), mixed precision, augmentation =
  random window offset (labels recomputed), amplitude jitter, Gaussian noise, 5% channel dropout,
  mixup(0.2); early stopping on patient-disjoint validation AUC.
- Inference: 16-s windows stepped every 1 s -> per-second probability series; joint
  (smoothing-width, threshold) tuning on validation patients only (max GDR s.t. <= 0.5 FD/h);
  events = runs above threshold with <10 s gaps merged and <10 s events discarded.
- Splits: patient-wise at every level (train / validation / test / tuning). Final headline results
  are 5-fold cross-validation with a 3-seed probability ensemble; the exact patient lists are
  recorded in `results/final_results.json`.

### 6.3 Verifying results without retraining (no GPU, no dataset download needed)

Every per-fold test prediction and all checkpoints are included in `results/`. The headline
cross-validation AUC can be reproduced from `results/` + the bundled label matrix alone:

```python
import json, numpy as np
from sklearn.metrics import roc_auc_score

ann = np.load("data/annotations.npz", allow_pickle=True)
yM = {int(p): ann[f"M_{p}"] for p in ann["pids"]}
res = json.load(open("results/final_results.json"))
aucs = []
for f in res["folds"]:
    probs = {int(k): v for k, v in
             np.load(f"results/final_probs_f{f['k']}.npz").items()}
    P, Y = [], []
    for p, pv in probs.items():
        y = yM[p]; n = min(len(pv), len(y))
        P.append(pv[:n]); Y.append(y[:n])
    aucs.append(roc_auc_score(np.concatenate(Y), np.concatenate(P)))
print("per-fold AUC(M):", [round(a, 4) for a in aucs],
      "mean:", round(np.mean(aucs), 4))      # -> [0.8388, 0.7934, 0.7148, 0.8431, 0.8554] mean: 0.8091
```

To run a trained model on new EEG (after preprocessing it with `preprocess.py`):

```python
import torch, models, common as C
model = models.build("multih")
model.load_state_dict(torch.load("results/final_f0_s0.pt", map_location="cpu"))
model.eval()
# d = {"x": <(18, T) float32 array>, "n_sec": T // 32, "prenorm": True}  (patient-normalised)
p_per_second = C.predict_seconds(model, d, torch.device("cpu"))
```

---

## 7. Summary of Experiments and Findings

### 7.1 Experiment log

Every experiment below ran under the same discipline: patient-wise splits, consensus labels for
training unless stated, and all threshold/post-processing tuned on validation patients only.
AUC values are pooled per-second AUC against consensus labels on held-out patients (development
split = 17 test patients; final CV = 5-fold over all 79 patients).

| # | Experiment (script) | Purpose | Configurations tried | Key outcome |
|---|---|---|---|---|
| 1 | Inter-expert variability (`annotations.py`) | Quantify label noise among the 3 experts | Per-second kappa, event matching/overlap clustering | Fleiss kappa **0.756**; only **43%** of events marked by all 3 experts; seizure-time estimates differ by up to 26% |
| 2 | Classic spectral baselines (`classic.py`) | Non-deep reference points | LightGBM; RBF-SVM | LightGBM **0.751**, SVM 0.688 — feature pipelines plateau well below deep models |
| 3 | Correlation/coherence SVM (`classic2.py`) | Reproduce the spirit of Tapani's expert-feature SVM | 61 spectral + time-varying correlation/coherence features | **0.725** — our compact feature set does not reach their tuned system |
| 4 | Architecture comparison (`run_dev.py`) | Find a strong CNN backbone | ShallowCNN, ResNet-1D (S/M/L) | 0.802-0.825; capacity alone does not close the patient-generalisation gap |
| 5 | Training-recipe ablations (`run_ablate.py`) | Normalisation, labels, augmentation | Per-window vs patient z-score; soft labels; mixup; channel-independent | Patient-norm +0.01, soft labels +0.005, **mixup +0.02**; channel-independent ill-posed with whole-recording labels (discontinued) |
| 6 | Multi-annotator multi-head (`run_ablate2.py`) | Exploit all 3 experts jointly in one network | One head per expert, shared trunk; +/- denser sampling | **Multi-head 0.833 — best single model**; stride-8 sampling did not help (0.804) |
| 7 | Window length (`run_win.py`) | Longer seizure context (Tapani used 60-s analysis) | 32 s and 60 s multi-head windows | 32 s: 0.834 (tie), 60 s: 0.821 — 16-32 s is the sweet spot here |
| 8 | Annotator-transfer matrix (`run_annot.py`) | How label source changes measured performance | Train on A / B / C / consensus; evaluate against all | Spread up to **0.116 AUC**; model trained on B is worst even against B himself (0.683) — consensus denoises |
| 9 | Final 5-fold CV, 3-seed ensemble (`run_cv.py`) | Headline patient-independent result | `recipe_final.json` (multi-head + mixup + patient-norm) | **AUC(M) 0.809 +/- 0.052** (single seed 0.785 +/- 0.069); GDR 29% @ 0.5 FD/h, 34% @ 1.0 FD/h (post-hoc); burden r = 0.56 |
| 10 | Aetiology stratification (`figures.py`/`make_paper_numbers.py`) | Which patients fail | Per-patient AUC by clinical group | HIE/ischaemia hardest (median 0.756, n=21); bilateral best (0.840, n=11) — dev-split intuition reversed under full CV |

**One-line conclusions.** (a) Multi-annotator labels are a training signal, not just noise:
joint 3-expert training beat every single-label variant. (b) Label choice alone shifts measured
AUC by up to 0.116 — protocol transparency matters more than small architecture gains.
(c) Remaining failures concentrate in HIE/ischaemic backgrounds, exactly where human experts
also disagree most.

### 7.2 Comparison with prior work

Numbers below are **as reported by each study**. They are *not* directly comparable: rows differ
in training data, in whether the evaluation is internal or external, in the metric (pooled
AUC vs median per-patient AUC), and in the annotator used as ground truth. The "Protocol notes"
column is the key to an honest reading — this table is precisely why we release every split and
prediction file.

| Study | Method | Training data | Evaluation | Reported result | Protocol notes / how this work relates |
|---|---|---|---|---|---|
| Stevenson et al. 2019 (data paper) | — | — | 79 infants, 3 experts | Fleiss kappa 0.767 among experts | No detector; we reproduce their agreement scale (0.756) and extend with event-level matching |
| Frassineti et al. 2020 | Stationary wavelets + classifier | Helsinki (internal) | Helsinki | AUC ~0.81 | Comparable internal setting; **our simplest CNN (0.79-0.83) matches or beats it; final system 0.809 CV mean with much stricter tuning discipline** |
| O'Shea et al. 2020 | Fully convolutional net (per-channel) | Cork (private, own corpus) | Cork (private) | AUC ~0.97 | Not Helsinki; we reproduce this architecture class as our `shallow` baseline; Daly's reproduction of it scored 0.926 when trained on all of Helsinki with an *external* test |
| Tapani et al. 2019 | SVM + time-varying correlations | Helsinki (internal), patient-wise CV | Helsinki | **Median per-patient AUC 0.988**; GDR ~77-80% @ 0.5-1 FD/h | Median-per-patient metric is far more permissive than pooled AUC (our per-patient medians also run higher); SVM pipeline used per-patient feature normalisation and heavily tuned features. Our event-level GDR is below theirs — the gap is real and discussed in the paper |
| Daly et al. 2024 | Enhanced ConvNet (45k params) + mixup | **All 79 Helsinki infants** | **External private Cork set** (4,570 h) | AUC 0.963 (0.968 w/ pseudo-labels); GDR 60.3% @ 0.33 FD/h | Trained on *all* of the corpus, tested on different patients *and* a different hospital — easier generalisation regime for AUC; we adopt their preprocessing conventions; under our internal 5-fold CV their baseline class scores ~0.78-0.82 |
| Hogan et al. 2024 (SOTA) | ConvNeXt-1D, 20.6M params | **202 private Cork neonates** (50,299 channel-hours) | Helsinki (fully held-out) | **Pooled AUC 0.982**; expert-equivalent (delta-kappa) | Strongest published result on this corpus, but uses ~2.6x more training data than exists in the public corpus, per-channel labels, and a much larger model; not reproducible from public data alone |
| **This work** | Multi-head CNN (~0.5M params) + seed ensemble | Helsinki only (49-52 patients/fold) | Helsinki, **strict 5-fold patient-wise CV**, tuning on validation only | **Pooled AUC 0.809 +/- 0.052**; AUC by expert 0.767-0.808; GDR 29% @ 0.5 FD/h | Strictest fully-internal protocol we could define; additionally delivers the first quantified annotator-transfer matrix and aetiology stratification on this corpus |

**How to read this comparison.** The published 0.96-0.99 results each relax at least one constraint
relative to ours: external training data (Hogan), external testing after training on the whole
corpus (Daly), a more permissive metric (Tapani's median-per-patient), or pre-deep-learning
feature pipelines (Frassineti). Under a same-data, same-patients, same-tuning protocol, our
0.809 +/- 0.052 is, to our knowledge, the strongest fully-internal, fully-patient-disjoint result
reported with released predictions — and our ablations show that reaching the published headline
numbers from public data alone would require either relaxing the protocol or new training data.
That conclusion, and the quantified protocol sensitivities behind it, is itself a finding of this
work.

---


## 8. Building on This Work

Concrete next steps (also discussed in the paper's Discussion/Limitations):

1. **Larger models / longer context** - our compute ceiling was a 2 GB GPU; scaling the trunk or
   pretraining on adult/critical-care EEG (e.g., MIMIC-IV-EEG) then fine-tuning here.
2. **Uncertainty calibration** - the multi-head design naturally yields per-expert disagreement
   at inference; calibrating it into an abstain/escalation signal for bedside use.
3. **Cross-centre validation** - evaluate the released checkpoints on other open neonatal corpora
   to quantify site shift.
4. **Annotation-efficient learning** - use the 39% single-expert events as candidates for active
   labelling; study consensus-finding with crowd-of-experts losses.
5. **Event-level detection heads** - our smoothing+thresholding post-processing is deliberately
   simple; learned onset/offset heads or temporal transformers could lift the (currently modest)
   good-detection rates at low false-alarm rates.

---

## 9. Citation

If you use this code, please cite the dataset paper (mandatory under CC-BY 4.0) and this work:

```bibtex
@article{stevenson2019helsinki,
  author  = {Stevenson, Nathan J. and Tapani, Karoliina T. and Lauronen, Leena and Vanhatalo, Sampsa},
  title   = {A dataset of neonatal {EEG} recordings with seizure annotations},
  journal = {Scientific Data},
  volume  = {6},
  pages   = {190039},
  year    = {2019},
  doi     = {10.1038/s41597-019-0027-x}
}
```

(Add the citation for this repository/manuscript once published.)

---

## 10. Licence and Ethics

- **Dataset**: not redistributed here; users must download it directly from Zenodo and comply with
  its **CC-BY 4.0** licence (attribution, no re-identification, no redistribution of raw data).
- **Code and derived artefacts in this repository**: released for research use; add your licence
  file before public release if you wish to impose specific terms.
- All data are de-identified; no IRB approval is required to download. The manuscript reports a
  retrospective analysis of an open, de-identified corpus.

## 11. Acknowledgements

Pipeline built with MNE-Python, scikit-learn, LightGBM and PyTorch. We thank the authors of the
Helsinki dataset for their exemplary open-science practice, and Daly, Lightbody & Temko and
Hogan et al. for publishing implementation details that informed our preprocessing conventions.
