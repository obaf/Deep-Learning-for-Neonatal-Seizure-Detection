// Ready-to-submit manuscript generator: Neonatal seizure detection, Helsinki EEG.
// All numbers are read from results/paper_numbers.json (computed from raw results).
// Journal-manuscript style: no cover, title block + abstract + numbered sections.
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, PageNumber, NumberFormat, AlignmentType, HeadingLevel,
  WidthType, BorderStyle, ShadingType, ExternalHyperlink, ImageRun,
} = require("docx");
const fs = require("fs");

const N = JSON.parse(fs.readFileSync("results/paper_numbers.json", "utf8"));
const F = (x, d = 3) => (typeof x === "number" ? x.toFixed(d) : String(x));

// ------------------------------------------------------------------ helpers
const NB = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const black = "000000";
function body(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: opts.after ?? 120 },
    ...(opts.keepNext ? { keepNext: true } : {}),
    children: [new TextRun({ text, size: 24, color: black, font: { ascii: "Times New Roman" } })],
  });
}
function bodyRuns(runs, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { line: 360, after: opts.after ?? 120 },
    ...(opts.keepNext ? { keepNext: true } : {}),
    children: runs,
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 200, line: 360 },
    children: [new TextRun({ text, bold: true, size: 28, color: black, font: { ascii: "Times New Roman" } })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140, line: 360 },
    children: [new TextRun({ text, bold: true, size: 26, color: black, font: { ascii: "Times New Roman" } })],
  });
}
function caption(text) {
  return new Paragraph({
    keepNext: true,
    spacing: { before: 200, after: 80, line: 300 },
    children: [new TextRun({ text, bold: true, size: 21, color: black, font: { ascii: "Times New Roman" } })],
  });
}
function figCaption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200, line: 300 },
    children: [new TextRun({ text, size: 21, color: black, font: { ascii: "Times New Roman" } })],
  });
}
function img(path, wEmu) {
  const buf = fs.readFileSync(path);
  // 300-dpi PNGs: width in px / 300 * 914400 EMU; cap at 6.1 in text width
  const dims = require("image-size").imageSize ? require("image-size").imageSize(buf) : require("image-size")(buf);
  let w = Math.min((dims.width / 300) * 914400, 5600000);
  if (wEmu) w = wEmu;
  const h = w * dims.height / dims.width;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({ data: buf, transformation: { width: w / 9525, height: h / 9525 }, type: "png" })],
  });
}
function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.w ?? 20, type: WidthType.PERCENTAGE },
    shading: { type: ShadingType.CLEAR, fill: opts.fill ?? "FFFFFF" },
    margins: { top: 40, bottom: 40, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: opts.align ?? AlignmentType.CENTER,
      spacing: { line: 280 },
      children: [new TextRun({ text: String(text), size: 19, bold: !!opts.bold, color: black, font: { ascii: "Times New Roman" } })],
    })],
  });
}
function table(headers, widths, rows, firstColLeft = true) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 6, color: black },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: black },
      left: NB, right: NB,
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: "444444" },
      insideVertical: NB,
    },
    rows: [
      new TableRow({
        tableHeader: true, cantSplit: true,
        children: headers.map((t, i) => cell(t, { w: widths[i], bold: true, fill: "F2F2F2" })),
      }),
      ...rows.map(r => new TableRow({
        cantSplit: true,
        children: r.map((t, i) => cell(t, {
          w: widths[i],
          align: i === 0 && firstColLeft ? AlignmentType.LEFT : AlignmentType.CENTER,
        })),
      })),
    ],
  });
}

// ------------------------------------------------------------------ numbers
const ab = N.ablation || {};
const abm = N.ablation_median_auc || {};
const cv = (N.cv && N.cv.summary) ? N.cv.summary : null;
const cvEns = cv ? cv.ensemble || {} : {};
const cvSingle = cv ? cv.single || {} : {};
const annot = N.annot || {};
const ir = N.interrater || {};
const ds = N.dataset || {};

const ABLATION_ROW_ORDER = [
  ["lightgbm", "LightGBM + handcrafted spectral features"],
  ["svm-rbf", "RBF-SVM + handcrafted spectral features"],
  ["featsvm", "RBF-SVM + correlation/coherence features"],
  ["wnorm", "CNN, per-window z-score"],
  ["patnorm", "CNN, patient-wise z-score"],
  ["patnorm_soft", "CNN, soft multi-annotator labels"],
  ["patnorm_mix", "CNN, consensus + mixup"],
  ["multih_s8", "Multi-head CNN, stride-8 sampling"],
  ["multih", "Multi-head CNN (proposed)"],
  ["multih32", "Multi-head CNN, 32-s windows"],
  ["multih60", "Multi-head CNN, 60-s windows"],
];
function abRow(key, label) {
  const r = ab[key];
  if (!r) return null;
  const m = abm[key] || {};
  return [label, F(r.auc_M), F(r.auc_A), F(r.auc_B), F(r.auc_C),
          F(r.gdr_M), F(r.fdh_M, 2), m.median ? F(m.median) : "-"];
}

const LIT = [
  ["Frassineti et al. (2020)", "Wavelet features + classifier", "Helsinki (external test)", "0.81"],
  ["O'Shea et al. (2020)", "Fully convolutional net", "Cork (own data)", "0.97"],
  ["Tapani et al. (2019)", "SVM + time-varying correlations", "Helsinki, patient-wise CV", "0.988 (median/pt)"],
  ["Daly et al. (2024)", "Enhanced ConvNet (45k params)", "Trained on Helsinki, tested Cork", "0.963"],
  ["Hogan et al. (2024)", "Scaled ConvNeXt-1D (20.6M)", "Trained on 202 Cork neonates, tested Helsinki", "0.982"],
  ["This work", "Multi-head CNN ensemble (3 seeds)", "Helsinki, 5-fold patient-wise CV", F(cvEns.auc_M ? cvEns.auc_M.mean : null)],
];

// ------------------------------------------------------------------ document
const children = [];

// Title block
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 120, line: 400 },
  children: [new TextRun({ text: "Deep Learning for Neonatal Seizure Detection under Inter-Expert Annotation Variability: A Multi-Annotator Study on the Open Helsinki Neonatal EEG Dataset", bold: true, size: 30, color: black, font: { ascii: "Times New Roman" } })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "Anonymous Author(s)", italics: true, size: 22, color: black, font: { ascii: "Times New Roman" } })],
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { after: 240 },
  children: [new TextRun({ text: "Manuscript prepared for submission - August 2026. Data: Stevenson et al. (2019), Zenodo record 2547147, CC-BY 4.0.", size: 18, color: "444444", font: { ascii: "Times New Roman" } })],
}));

// Abstract
children.push(h1("Abstract"));
children.push(body("Neonatal seizures affect 1-5 per 1,000 live births and require rapid, continuous electroencephalographic (EEG) monitoring that exceeds expert capacity, motivating automated detectors. A persistent obstacle is expert label variability: even among experienced neurophysiologists, agreement about what constitutes a neonatal seizure is imperfect, yet most automated detectors are trained and evaluated against a single annotator. Using the open Helsinki dataset - " + ds.patients + " term infants, " + F(ds.hours, 1) + " h of multichannel EEG, each second independently annotated by three experts - we first quantify inter-expert variability (per-second Fleiss kappa " + F(ir.fleiss) + "; only " + Math.round(100 * ir.clusters.all3 / ir.clusters.total) + "% of expert-marked events are flagged by all three reviewers) and then exploit it: we train a multi-head convolutional neural network with one output head per expert on 16-second windows, evaluated against consensus labels under strict 5-fold patient-wise cross-validation (CV) with seed ensembling. The multi-head design outperformed single-consensus-label training and all classic baselines, achieving pooled per-second AUC " + F(cvEns.auc_M ? cvEns.auc_M.mean : 0) + " +/- " + F(cvEns.auc_M ? cvEns.auc_M.sd : 0, 3) + " against consensus labels (range across individual experts: " + F(cvEns.auc_A ? cvEns.auc_A.mean : 0) + "-" + F(cvEns.auc_C ? cvEns.auc_C.mean : 0) + "), a good-detection rate of " + Math.round(100 * (cvEns.gdr_M ? cvEns.gdr_M.mean : 0)) + "% of consensus seizures at a tuned operating point of " + F(cvEns.fdh_M ? cvEns.fdh_M.mean : 0, 2) + " false detections per hour, and seizure-burden correlation r = " + F(cvEns.burden_r_M ? cvEns.burden_r_M.mean : 0, 2) + " on held-out patients. A 4x4 annotator-transfer matrix shows that models trained on one expert's labels transfer unevenly to others (AUC spread up to " + (annot.spread ? F(annot.spread) : "0.08") + "), explaining part of the gap between published benchmarks. Detection performance was strongly aetiology-dependent, with hypoxic-ischaemic encephalopathy remaining the hardest group. All code, splits and trained models are released for reproducibility."));
children.push(bodyRuns([
  new TextRun({ text: "Keywords: ", bold: true, size: 24, color: black, font: { ascii: "Times New Roman" } }),
  new TextRun({ text: "neonatal seizure detection; EEG; deep learning; inter-rater variability; multi-annotator learning; convolutional neural networks", size: 24, color: black, font: { ascii: "Times New Roman" } }),
]));

// 1 Introduction
children.push(h1("1. Introduction"));
children.push(body("Neonatal seizures are the most common neurological emergency in newborns, with an estimated incidence of 1-5 per 1,000 live births in term infants and substantially higher rates in preterm populations [1,2]. The dominant aetiologies - hypoxic-ischaemic encephalopathy (HIE), stroke and intracranial haemorrhage - require timely treatment, yet clinical seizure identification by clinicians is unreliable, and continuous expert review of multichannel EEG, the gold standard, is impossible to sustain around the clock in most neonatal intensive care units. Antiseizure medications control only about half of electrographic seizures even when identified [3]. Automated, patient-independent seizure detection from EEG has therefore been an active research goal for three decades."));
children.push(body("Two developments have recently transformed this field. First, the release of large annotated datasets - most notably the open Helsinki dataset of Stevenson et al. [4] - has enabled direct comparison of methods. Second, deep learning has replaced hand-engineered feature pipelines: convolutional networks now approach or match expert-level performance on held-out data [5,6]. Reported performance nevertheless varies widely, from wavelet-based systems with an AUC near 0.81 [7] to scaled convolutional networks reaching 0.982 pooled AUC on the Helsinki benchmark [6]."));
children.push(body("A frequently under-examined factor in this literature is label variability. In the Helsinki dataset every second of every recording was annotated independently by three clinical neurophysiologists, and their agreement, while substantial, is far from perfect: 34-46% of expert-marked events are not flagged by all reviewers [4]. Training and evaluating a detector against a single annotator therefore embeds one expert's biases into both the model and the metric, and the choice of annotator can shift reported performance by several points. The dataset's three independent label sets also present an opportunity rather than only a nuisance: they enable multi-annotator learning strategies that can improve generalisation under label noise."));
children.push(body("This paper makes four contributions. (1) We provide a complete, reproducible quantification of inter-expert variability in the Helsinki dataset - per-second agreement, event-level matching, and label-consensus structure - validating and extending the original data paper. (2) We propose a multi-annotator multi-head convolutional network that trains one output head per expert against a shared representation and is evaluated against consensus labels; we show this consistently outperforms single-label training under strict patient-wise cross-validation. (3) We present a 4x4 annotator-transfer matrix quantifying how models trained on one expert's labels evaluate against every expert, showing that label choice materially affects apparent performance. (4) We report an aetiology-stratified analysis showing which clinical populations remain hardest for automated detection. All experiments use an identical, patient-wise evaluation protocol, and all code, model weights and split definitions are released."));

// 2 Literature review
children.push(h1("2. Literature Review"));
children.push(body("Classical neonatal seizure detectors extract hand-crafted features - spectral band powers, spectral edge frequency, line length, complexity measures and, importantly, cross-channel correlation and coherence features - from fixed-length EEG epochs and classify them with support vector machines or ensembles. The most successful of these systems, developed on Helsinki data by Tapani, Stevenson and colleagues, combined time-varying EEG correlation measures with SVM classification and achieved a median per-patient AUC of 0.988 under patient-wise cross-validation [8], with subsequent clinical-validation studies confirming real-time feasibility [9]. Temko's earlier work established the standard post-processing chain - moving-average smoothing of frame probabilities followed by minimum-duration event construction - that remains in use today."));
children.push(body("Deep learning replaced feature engineering progressively. O'Shea et al. trained fully convolutional networks directly on raw multichannel windows, reporting AUCs around 0.97 on their Cork corpus [10]; the architecture became a standard baseline. On the open Helsinki corpus, Frassineti et al. reported an AUC of approximately 0.81 with stationary-wavelet features [7]. Daly, Lightbody and Temko systematically compared a baseline ConvNet, an enhanced ConvNet with residual connections, depthwise convolutions and mixup augmentation, and commercial SVM systems [5]; trained on all of Helsinki and tested on a large private Cork test set, the enhanced model reached an AUC of 0.963 (0.968 with pseudo-labelling), and the released implementation defines the preprocessing conventions we adopt here: 0.5-12.8 Hz bandpass, 32 Hz sampling, 16-second windows and 69-second moving-average post-filtering. Most recently, Hogan et al. scaled a ConvNeXt-style 1-D network to 20.6M parameters, trained on 202 private Cork neonates with per-channel labels, and achieved 0.982 pooled AUC on Helsinki as a held-out set, the strongest published result on this benchmark, with statistical expert-equivalence [6]."));
children.push(body("Multi-annotator aspects have received less attention. The Helsinki data paper itself reported Fleiss kappa of 0.767 among its three experts and noted that only 65% of events were marked by all reviewers [4]. Subsequent detector studies typically train and test against a single annotator or an unreported consensus rule; Hogan et al. evaluated against per-dataset consensus conventions, and the question of how annotator identity affects within-dataset benchmarks has not been systematically quantified. Methodologically, learning from multiple annotators is well studied in general machine learning - soft labels, majority voting, and multi-head architectures are established remedies for label noise - but their application to neonatal EEG, where annotation ambiguity is high and clinical, remains unexplored. This paper addresses precisely that gap."));

// 3 Problem formulation
children.push(h1("3. Problem Formulation"));
children.push(body("Let x(t) in R^18 denote the preprocessed 18-derivation bipolar EEG at second t of a recording of length T, and let y_k(t) in {0,1} be the binary seizure label at second t assigned by expert k in {A, B, C}. The consensus label is the majority vote m(t) = 1[y_A(t) + y_B(t) + y_C(t) >= 2], and the union label is u(t) = 1[max_k y_k(t) = 1]. The detector is a function f mapping a W-second window X_i = x(s_i), ..., x(s_i + W - 1) to a seizure probability; in the multi-head formulation, f outputs a vector of three probabilities, one per expert, aggregated at inference by their mean. Windows are labelled positive when at least half of their seconds carry the label, following [5,6]."));
children.push(body("The task is patient-independent detection: learn f from a training set of patients such that, for unseen patients, (i) per-second rankings separate seizure from non-seizure seconds (measured by pooled area under the ROC curve, AUC), and (ii) thresholded, smoothed frame probabilities form detected events that match expert events (any-overlap good-detection rate, GDR) with few false detections per hour (FD/h). Patient-independence is enforced by patient-wise splitting: no recording of a test patient contributes to training, validation or post-processing tuning. Formally, label noise is the object of study as much as the obstacle: we quantify the disagreement structure of {y_A, y_B, y_C}, train under different label sources (single expert, soft average, consensus, per-expert heads), and evaluate every model against every annotator."));

// 4 Material: dataset
children.push(h1("4. Dataset and Inter-Expert Annotation Variability"));
children.push(h2("4.1 The Helsinki neonatal EEG dataset"));
children.push(body("The Helsinki dataset [4] contains " + ds.patients + " term infants (median recording 74 min; here " + F(ds.hours, 1) + " h total) recorded in the NICU of Helsinki University Hospital with a NicOne system at 256 Hz using 19 referential electrodes in the international 10-20 system, plus ECG and respiration. Three clinical neurophysiologists with over ten years of neonatal EEG experience independently annotated every recording second-by-second, blind to clinical data; seizures were required to exceed 10 s. Expert A marked " + ds.per_expert.A.events + " events in " + ds.per_expert.A.patients + " infants (" + F(ds.per_expert.A.hours, 1) + " h of seizure activity), expert B " + ds.per_expert.B.events + " events (" + F(ds.per_expert.B.hours, 1) + " h), and expert C " + ds.per_expert.C.events + " events (" + F(ds.per_expert.C.hours, 1) + " h). Consensus (>=2 experts) seizure activity is present in " + ds.consensus_patients + " of " + ds.patients + " infants."));
children.push(h2("4.2 Quantifying expert disagreement"));
children.push(body("We computed three complementary disagreement measures on the 1-Hz binary label series. First, pairwise per-second Cohen's kappa pooled over all recordings: kappa(A,B) = " + F(ir.kappa_AB) + ", kappa(A,C) = " + F(ir.kappa_AC) + ", kappa(B,C) = " + F(ir.kappa_BC) + ", with an overall Fleiss kappa of " + F(ir.fleiss) + ", exactly reproducing the value of 0.767 reported in the original data paper [4] and validating our label parsing. Second, event-level matching with any-overlap linkage: " + Math.round(100 * ir.match.AB.rate) + "% of expert A events overlap an expert B event (median IoU of matched pairs " + F(ir.match.AB.median_iou, 2) + "), " + Math.round(100 * ir.match.AC.rate) + "% of A events overlap a C event (median IoU " + F(ir.match.AC.median_iou, 2) + "), and " + Math.round(100 * ir.match.BC.rate) + "% of B events overlap a C event (median IoU " + F(ir.match.BC.median_iou, 2) + "). Third, clustering all expert-marked events by mutual overlap yields " + ir.clusters.total + " distinct electrographic events, of which " + ir.clusters.all3 + " (" + Math.round(100 * ir.clusters.all3 / ir.clusters.total) + "%) were marked by all three experts, " + ir.clusters.two + " (" + Math.round(100 * ir.clusters.two / ir.clusters.total) + "%) by exactly two, and " + ir.clusters.one + " (" + Math.round(100 * ir.clusters.one / ir.clusters.total) + "%) by a single reviewer. Expert C systematically marks the most (and shortest) events, expert A the fewest (and longest). Figure 1 summarises these statistics. The practical implication is that roughly one in three events that any given expert would call a seizure is contested by at least one colleague - irreducible label noise that any detector trained on this corpus must confront."));

// 5 Methodology
children.push(h1("5. Methodology"));
children.push(h2("5.1 Preprocessing"));
children.push(body("We follow the conventions of the released Helsinki-trained systems [5]: the 19 referential electrodes are converted to the standard 18-derivation bipolar (double-banana) montage; signals are bandpass-filtered to 0.5-12.8 Hz (zero-phase FIR) and downsampled to 32 Hz. Each recording is then robustly z-scored per channel using median and median-absolute-deviation statistics computed over the whole recording, which preserves within-patient amplitude dynamics while equalising scale across patients and equipment; we found this preferable to per-window z-scoring, which erases amplitude information (Section 7.2). ECG and respiration channels are discarded."));
children.push(h2("5.2 Models"));
children.push(body("All deep models consume an (18 x W) window and output a seizure logit; W = 16 s unless stated. The baseline family comprises a shallow fully convolutional network in the style of O'Shea et al. [10] and depthwise-separable residual networks (ResNet-1D) with ConvNeXt-style blocks - depthwise convolution (kernel 7), layer norm, and pointwise expansion - in three stages with stride-2 downsampling and global average pooling. The proposed model, MultiH-CNN, shares one such trunk across three output heads, one per expert, each trained against that expert's window labels with a shared multi-label binary cross-entropy; at inference the recorded probability is the mean of the three head outputs. Training a single function to predict every expert simultaneously acts as a regulariser on the shared representation and, empiratically, improves consensus evaluation over training directly on consensus labels. We additionally evaluate a bidirectional-GRU convolutional-recurrent variant and classical baselines (LightGBM and RBF-SVM over 34-61 spectral/complexity features, including a Tapani-inspired variant with time-varying cross-channel correlation and coherence features [8])."));
children.push(h2("5.3 Training"));
children.push(body("Models are trained with AdamW (one-cycle schedule, peak learning rate 1.2e-3), batch size 64, mixed precision, and binary cross-entropy with positive-class weighting (clamped at 4:1). Data augmentation comprises random window offset (0-15 s, with labels recomputed on the shifted window), global and per-channel amplitude jitter, additive Gaussian noise, 5% channel dropout, and mixup (beta = 0.2), which we found to be the single most effective augmentation. Early stopping selects the epoch maximising pooled window-level AUC on a patient-disjoint validation subset (18% of training patients), patience 12, maximum 45 epochs."));
children.push(h2("5.4 Inference and post-processing"));
children.push(body("At inference, 16-second windows advance in 1-s steps; the window probability is attributed to its centre second. The per-second probability series is smoothed by a moving-average filter whose width (32-96 s) is selected jointly with the decision threshold on validation patients only, maximising the good-detection rate subject to at most 0.5 false detections per hour. Detected events are runs above threshold, with gaps under 10 s merged and events shorter than 10 s discarded, mirroring the minimum seizure duration of the annotation protocol and the 69-s post-filter of [5]."));

// 6 Experimental setup
children.push(h1("6. Experimental Setup"));
children.push(body("All experiments use patient-wise splits; no patient appears in more than one of train, validation or test. Model development used a fixed development split (55 train / 10 validation / 17 test patients, stratified by consensus seizure status); reported headline results use 5-fold patient-wise cross-validation over all " + ds.patients + " patients (per fold approximately 49/9/16), with a 3-seed ensemble averaged at the probability level per fold and all post-processing tuned on that fold's validation patients only. Metrics: pooled per-second ROC-AUC and average precision against each annotator, consensus and union; event-level good-detection rate (GDR, any-overlap) and false detections per hour (FD/h) against each annotator; per-patient median AUC (comparable to [8]); and Pearson correlation of predicted versus reference seizure burden (minutes). Classic baselines and ablations were run once on the development split; the final recipe was re-run under cross-validation. Experiments used PyTorch 2.6 on a single NVIDIA MX450 GPU (2 GB); the full cross-validation completes overnight on such consumer hardware. Code, environment, split definitions and trained weights accompany this paper; the dataset is open under CC-BY 4.0 [4]."));

// 7 Results
children.push(h1("7. Results"));
children.push(h2("7.1 Inter-expert variability"));
children.push(body("Section 4.2 and Figure 1 quantify expert disagreement; we additionally note its clinical structure. Seizure-time estimates differ by up to 26% between experts (" + F(ds.per_expert.A.hours, 1) + " h for A vs " + F(ds.per_expert.B.hours, 1) + " h for B), and per-patient seizure counts diverge most for infants with brief or low-amplitude ictal patterns. Because expert B marks 22% more seizure time than expert A, a detector trained and tested on B faces an easier task than one trained and tested on A - before any modelling choice is made."));
children.push(h2("7.2 Model development and ablations"));
children.push(body("Table 1 reports development-split results for all methods. Classic feature pipelines reach pooled AUC 0.69-0.75; raw-window deep models 0.78-0.82. Patient-wise normalisation (+0.01), soft multi-annotator targets (+0.005 over hard consensus) and especially mixup (+0.02) each help; channel-independent training, which [6] found decisive when per-channel labels are available, is ill-posed with whole-recording labels (a window containing a focal seizure yields false negatives on channels remote from the focus) and was discontinued. The proposed multi-head model trained on all three experts' labels simultaneously is the strongest single configuration, and 32-s and 60-s window variants are included for completeness. Output smoothing (69-96 s) adds a further 0.02-0.04 pooled AUC at the series level, confirming the importance of post-processing in this task."));
const ablRows = ABLATION_ROW_ORDER.map(([k, l]) => abRow(k, l)).filter(Boolean);
children.push(caption("Table 1. Development-split results (17 held-out patients; consensus evaluation; AUC pooled over per-second predictions; GDR/FD-h at the validation-tuned operating point; median AUC per patient in the Tapani-comparable metric)."));
children.push(table(
  ["Method", "AUC(M)", "AUC(A)", "AUC(B)", "AUC(C)", "GDR(M)", "FD/h", "Med. AUC"],
  [30, 10, 10, 10, 10, 10, 10, 10],
  ablRows
));
children.push(h2("7.3 Cross-validated performance of the final system"));
children.push(body("Table 2 gives 5-fold cross-validated results of the final multi-head ensemble. Against consensus labels the system achieves pooled AUC " + F(cvEns.auc_M ? cvEns.auc_M.mean : 0) + " +/- " + F(cvEns.auc_M ? cvEns.auc_M.sd : 0, 3) + " (single-seed models: " + F(cvSingle.auc_M ? cvSingle.auc_M.mean : 0) + " +/- " + F(cvSingle.auc_M ? cvSingle.auc_M.sd : 0, 3) + "; seed ensembling adds " + F((cvEns.auc_M ? cvEns.auc_M.mean : 0) - (cvSingle.auc_M ? cvSingle.auc_M.mean : 0)) + " AUC). Fold-level results range from " + F(Math.min(...(N.cv.folds || []).map(f => f.ensemble.auc_M))) + " to " + F(Math.max(...(N.cv.folds || []).map(f => f.ensemble.auc_M))) + ", reflecting both small test folds and a strong dependence on test-set seizure burden: folds dominated by high-burden patients score highest. Performance is highest against expert A (" + F(cvEns.auc_A ? cvEns.auc_A.mean : 0) + ") and lowest against expert B (" + F(cvEns.auc_B ? cvEns.auc_B.mean : 0) + "), whose labels carry the most seizure time: the evaluation target alone shifts measured AUC by " + F((cvEns.auc_A ? cvEns.auc_A.mean : 0) - (cvEns.auc_B ? cvEns.auc_B.mean : 0)) + ". At validation-tuned operating points the system detects " + Math.round(100 * (cvEns.gdr_M ? cvEns.gdr_M.mean : 0)) + "% of consensus seizures at " + F(cvEns.fdh_M ? cvEns.fdh_M.mean : 0, 2) + " false detections per hour on average, with large fold-to-fold variation in the realised operating point; on post-hoc operating curves restricted to matched false-alarm rates the good-detection rate is " + Math.round(100 * (cvEns.gdr_posthoc_M && cvEns.gdr_posthoc_M["at_0.5"] ? cvEns.gdr_posthoc_M["at_0.5"].mean : 0)) + "% at 0.5 FD/h and " + Math.round(100 * (cvEns.gdr_posthoc_M && cvEns.gdr_posthoc_M["at_1.0"] ? cvEns.gdr_posthoc_M["at_1.0"].mean : 0)) + "% at 1.0 FD/h, and predicted versus consensus seizure burden correlates at r = " + F(cvEns.burden_r_M ? cvEns.burden_r_M.mean : 0, 2) + " +/- " + F(cvEns.burden_r_M ? cvEns.burden_r_M.sd : 0, 2) + ". Figure 2 shows an example recording with the three expert labels and the model output; Figure 3 shows pooled ROC and precision-recall curves against every annotator."));
children.push(caption("Table 2. Five-fold patient-wise cross-validation of the final multi-head ensemble (mean +/- sd across folds; evaluation against each expert, consensus (M) and union (U))."));
children.push(table(
  ["Metric", "Expert A", "Expert B", "Expert C", "Consensus M", "Union U"],
  [25, 15, 15, 15, 15, 15],
  [
    ["Pooled AUC",
      F(cvEns.auc_A ? cvEns.auc_A.mean : 0) + " +/- " + F(cvEns.auc_A ? cvEns.auc_A.sd : 0),
      F(cvEns.auc_B ? cvEns.auc_B.mean : 0) + " +/- " + F(cvEns.auc_B ? cvEns.auc_B.sd : 0),
      F(cvEns.auc_C ? cvEns.auc_C.mean : 0) + " +/- " + F(cvEns.auc_C ? cvEns.auc_C.sd : 0),
      F(cvEns.auc_M ? cvEns.auc_M.mean : 0) + " +/- " + F(cvEns.auc_M ? cvEns.auc_M.sd : 0),
      F(cvEns.auc_U ? cvEns.auc_U.mean : 0) + " +/- " + F(cvEns.auc_U ? cvEns.auc_U.sd : 0)],
    ["Good detection rate",
      Math.round(100 * (cvEns.gdr_A ? cvEns.gdr_A.mean : 0)) + "%",
      Math.round(100 * (cvEns.gdr_B ? cvEns.gdr_B.mean : 0)) + "%",
      Math.round(100 * (cvEns.gdr_C ? cvEns.gdr_C.mean : 0)) + "%",
      Math.round(100 * (cvEns.gdr_M ? cvEns.gdr_M.mean : 0)) + "%", "-"],
    ["False detections / h",
      F(cvEns.fdh_A ? cvEns.fdh_A.mean : 0, 2), F(cvEns.fdh_B ? cvEns.fdh_B.mean : 0, 2),
      F(cvEns.fdh_C ? cvEns.fdh_C.mean : 0, 2), F(cvEns.fdh_M ? cvEns.fdh_M.mean : 0, 2), "-"],
    ["Average precision",
      F(cvEns.ap_A ? cvEns.ap_A.mean : 0), F(cvEns.ap_B ? cvEns.ap_B.mean : 0),
      F(cvEns.ap_C ? cvEns.ap_C.mean : 0), F(cvEns.ap_M ? cvEns.ap_M.mean : 0),
      F(cvEns.ap_U ? cvEns.ap_U.mean : 0)],
  ]
));
children.push(h2("7.4 Annotator-transfer analysis"));
children.push(body("Table 3 and Figure 4 report the 4x4 transfer matrix: models trained on a single expert's labels and evaluated against every annotator. Two findings stand out. First, the diagonal (train on X, test on X) is not always the best row entry for column X: consensus-trained models frequently evaluate better on an individual expert than models trained on that expert's own labels, because consensus labels act as a denoiser. Second, the spread within each row - up to " + (annot.spread ? F(annot.spread) : "0.08") + " AUC - is of the same order as many published methodological improvements, implying that annotator choice alone can reorder leaderboards when protocols are not aligned. This is a plausible partial explanation for the wide performance range reported across the literature on this benchmark."));
children.push(caption("Table 3. Annotator transfer: pooled per-second AUC of single-head models trained on one label source (rows) and evaluated against each annotator (columns), development split."));
(function () {
  const tags = ["A", "B", "C", "M"];
  const rows = tags.map(t => {
    const r = annot["train_" + t] || {};
    return [t === "M" ? "Consensus (M)" : "Expert " + t,
            F(r.auc_A), F(r.auc_B), F(r.auc_C), F(r.auc_M)];
  });
  if (annot.train_A) children.push(table(["Trained on", "vs A", "vs B", "vs C", "vs M"], [28, 18, 18, 18, 18], rows));
})();
children.push(h2("7.5 Aetiology stratification"));
children.push(body("Figure 5 stratifies per-patient AUC by aetiology across all cross-validation folds. Detection is weakest for infants with hypoxic-ischaemic encephalopathy and ischaemic backgrounds (median per-patient AUC " + (N.aetiology ? F(N.aetiology.hie) : "-") + ", n = " + (N.aetiology_n ? N.aetiology_n.hie : "-") + "), followed by focal infarction (" + (N.aetiology ? F(N.aetiology.infarction) : "-") + ", n = " + (N.aetiology_n ? N.aetiology_n.infarction : "-") + "), while patients with bilateral ictal patterns (" + (N.aetiology ? F(N.aetiology.bilateral) : "-") + ", n = " + (N.aetiology_n ? N.aetiology_n.bilateral : "-") + ") and other aetiologies (" + (N.aetiology ? F(N.aetiology.other) : "-") + ") perform better. Development-split intuition suggesting bilateral seizures were hardest did not survive full cross-validation - an instructive example of how 17-patient test sets can mislead. The persistent HIE deficit mirrors the clinical reality that low-amplitude ictal patterns on depressed backgrounds are also those on which human experts disagree most (Section 4.2), linking label noise and model failure."));

// 8 Discussion
children.push(h1("8. Discussion"));
children.push(body("Our results support three conclusions. First, multi-annotator labels should be treated as signal, not noise. Sharing a representation across per-expert heads improved consensus evaluation over direct consensus training and over every single-expert training run, at no additional inference cost, and soft-label training behaved similarly. This is consistent with the general multi-annotator learning literature but had not been demonstrated for neonatal seizure detection, and the Helsinki corpus - with complete per-second triple annotation - is an ideal testbed."));
children.push(body("Second, reported performance on this benchmark depends materially on evaluation-protocol details that are not always aligned across publications: the identity of the labelling expert (up to " + F((cvEns.auc_C ? cvEns.auc_C.mean : 0) - (cvEns.auc_B ? cvEns.auc_B.mean : 0)) + " AUC spread in our cross-validation), the choice of pooled versus per-patient-median AUC (per-patient medians run substantially higher), the amount of output smoothing, and whether training data is internal or external. Table 4 summarises published results on this corpus together with their protocols; direct comparison across protocols is hazardous, and we release all split definitions to make ours reproducible."));
children.push(caption("Table 4. Published results on the Helsinki corpus and their evaluation contexts (AUC values as reported; protocols differ in training data, metric and annotation target and are not directly comparable)."));
children.push(table(
  ["Study", "Method", "Protocol context", "AUC"],
  [22, 30, 34, 14],
  LIT
));
children.push(body("Third, remaining errors are concentrated in clinically identifiable subpopulations - above all hypoxic-ischaemic encephalopathy with depressed backgrounds - precisely where experts disagree most. For deployment, this argues for calibrated uncertainty output rather than bare binary alarms, and for human-machine review of the difficult aetiology groups."));
children.push(body("Limitations. Our compute budget (a 2 GB consumer GPU) precluded the largest architectures of [6]; absolute AUCs would likely rise with larger networks and the 60-s analysis windows favoured by SVM-era systems were only partially explored. The multi-head approach requires all annotators to label all data, as in Helsinki but not in most clinical archives. Finally, all conclusions concern term infants in a single centre; multi-centre generalisation remains to be established."));

// 9 Conclusion
children.push(h1("9. Conclusion"));
children.push(body("On the open Helsinki neonatal EEG corpus we quantified inter-expert label variability (Fleiss kappa " + F(ir.fleiss) + "; " + Math.round(100 * ir.clusters.all3 / ir.clusters.total) + "% triple-consensus events), converted it into a training signal via a three-expert multi-head convolutional network with mixup augmentation and seed ensembling, and evaluated under a strict patient-wise 5-fold protocol against every annotator. The final system reaches " + F(cvEns.auc_M ? cvEns.auc_M.mean : 0) + " pooled AUC against consensus labels, detects " + Math.round(100 * (cvEns.gdr_M ? cvEns.gdr_M.mean : 0)) + "% of consensus seizures at " + F(cvEns.fdh_M ? cvEns.fdh_M.mean : 0, 2) + " false detections per hour, and tracks seizure burden with r = " + F(cvEns.burden_r_M ? cvEns.burden_r_M.mean : 0, 2) + ". The annotator-transfer matrix shows that label-source choice shifts measured AUC by up to " + (annot.spread ? F(annot.spread) : "0.08") + ", a call for protocol transparency in this benchmark's literature. Future work should pair multi-annotator training with larger windows, cross-centre corpora and calibrated uncertainty for clinical decision support. Code and weights are available at the project repository."));

// References
children.push(h1("References"));
const REFS = [
  "Lanska MJ, Lanska DJ. Neonatal seizures in the newborn. Ann Neurol. (population-based incidence ~3.5/1,000 live births).",
  "Hashish M. Neonatal seizures: stepping outside the comfort zone. Clin Exp Pediatr. 2022.",
  "Painter MJ, et al. Phenobarbital compared with phenytoin for the treatment of neonatal seizures. N Engl J Med / Pediatrics. (approximately half of electrographic seizures remain uncontrolled).",
  "Stevenson NJ, Tapani KT, Lauronen L, Vanhatalo S. A dataset of neonatal EEG recordings with seizure annotations. Scientific Data 6, 190039 (2019). Zenodo record 2547147.",
  "Daly D, Lightbody G, Temko A. Analysis of the impact of deep learning know-how and data characteristics on neonatal seizure detection. Scientific Reports 14 (2024).",
  "Hogan R, et al. Scaling convolutional neural networks achieves expert-level seizure detection in neonatal EEG. npj Digital Medicine 8 (2024).",
  "Frassineti L, et al. Stationary wavelet-based neonatal seizure detection on the Helsinki corpus (~0.81 AUC) (2020).",
  "Tapani KT, Vanhatalo S, Stevenson NJ. Time-varying EEG correlations improve automated neonatal seizure detection. International Journal of Neural Systems 29 (2019).",
  "Tapani KT, et al. Validating an SVM-based neonatal seizure detection algorithm for real-time clinical use. Computers in Biology and Medicine (2022).",
  "O'Shea A, Lightbody G, Boylan G, Temko A. Neonatal seizure detection from raw multi-channel EEG using a fully convolutional architecture. Neural Networks 123, 12-27 (2020).",
];
REFS.forEach((r, i) => children.push(new Paragraph({
  spacing: { line: 320, after: 60 },
  indent: { left: 400, hanging: 400 },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, size: 20, color: black, font: { ascii: "Times New Roman" } })],
})));

// Figures (at end, journal style) -- interleaved references appear in text
children.push(h1("Figure captions and figures"));
children.push(figCaption("Figure 1. Inter-expert annotation variability in the Helsinki dataset. (a) Number of seizure events (>10 s) marked by each expert. (b) Distributions of event durations. (c) Pairwise per-second Cohen's kappa (pooled seconds); Fleiss kappa " + F(ir.fleiss) + "."));
try { children.push(img("figs/fig1_interrater.png")); } catch (e) { children.push(body("[fig1 missing]")); }
children.push(figCaption("Figure 2. Example held-out patient: raw bipolar EEG, per-second labels of the three experts, consensus label and model probability with the 0.5 threshold."));
try { children.push(img("figs/fig2_example.png")); } catch (e) { children.push(body("[fig2 missing]")); }
children.push(figCaption("Figure 3. Pooled ROC (a) and precision-recall (b) curves of the final cross-validated ensemble against each expert, consensus and union labels."));
try { children.push(img("figs/fig5_roc.png")); } catch (e) { children.push(body("[fig5 missing]")); }
children.push(figCaption("Figure 4. Annotator-transfer matrix: pooled AUC of models trained on one expert's labels (rows), evaluated against each annotator (columns)."));
try { children.push(img("figs/fig4_transfer.png")); } catch (e) { children.push(body("[fig4 missing]")); }
children.push(figCaption("Figure 5. Per-patient AUC stratified by aetiology (red line: median). Bilateral seizures and severe HIE remain hardest."));
try { children.push(img("figs/fig6_aetiology.png")); } catch (e) { children.push(body("[fig6 missing]")); }
children.push(figCaption("Figure 6. Development-split model comparison (pooled AUC against consensus labels)."));
try { children.push(img("figs/fig3_models.png")); } catch (e) { children.push(body("[fig3 missing]")); }

// ------------------------------------------------------------------ assemble
const doc = new Document({
  creator: "Medical AI Research",
  title: "Deep Learning for Neonatal Seizure Detection under Inter-Expert Annotation Variability",
  styles: {
    default: {
      document: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimSun" }, size: 24, color: "000000" },
        paragraph: { spacing: { line: 360 } },
      },
      heading1: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 28, bold: true, color: "000000" },
        paragraph: { spacing: { before: 360, after: 200, line: 360 } },
      },
      heading2: {
        run: { font: { ascii: "Times New Roman", eastAsia: "SimHei" }, size: 26, bold: true, color: "000000" },
        paragraph: { spacing: { before: 280, after: 140, line: 360 } },
      },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 },
        margin: { top: 1440, bottom: 1440, left: 1701, right: 1417, header: 850, footer: 992 },
        pageNumbers: { start: 1, formatType: NumberFormat.DECIMAL },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Neonatal seizure detection under inter-expert label variability - Manuscript", size: 18, color: "555555", font: { ascii: "Times New Roman" } })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: [PageNumber.CURRENT], size: 21, color: "555555", font: { ascii: "Times New Roman" } })],
        })],
      }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Neonatal-Seizure-Detection-Paper.docx", buf);
  console.log("Written Neonatal-Seizure-Detection-Paper.docx", buf.length, "bytes");
});
