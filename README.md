# Calibrated sign recognition with a reject option — preliminary results

Preliminary figures for the senior design pitch: a sign classifier on real ASL
video, its confidence calibrated by temperature scaling, and a reject option
evaluated by a risk–coverage curve.

Everything here was measured, not estimated. Re-running `src/01` … `src/06`
reproduces every number and both figures.

---

## Headline numbers

Test split = **224 clips from 17 signers who appear in no other split**,
100 sign classes (chance = 1.0%).

| | value |
|---|---|
| **Top-1 accuracy (baseline)** | **0.232** (0.232 ± 0.024 over 5 seeds) |
| **ECE before calibration** | **0.247** (0.247 ± 0.031) |
| **ECE after temperature scaling** | **0.079** (0.079 ± 0.010) |
| ECE reduction | **68%** |
| Fitted temperature T* | 1.58 ± 0.10 |
| Mean confidence before → after | 0.476 → 0.280 |
| NLL before → after | 3.758 → 3.232 |

Accuracy is unchanged by temperature scaling (0.232 → 0.232),
which is expected: dividing the logits by a positive scalar cannot change the argmax.

**The overconfidence is the point.** Before calibration the model reports a mean
confidence of 0.476 while actually being right
0.232 of the time — a gap of
0.244. After scaling, the mean confidence is
0.280, essentially matching the true accuracy.

### Robustness

Two ECE binning schemes, because 15-bin equal-width ECE is unstable at n=224:

| estimator | before | after | reduction |
|---|---|---|---|
| ECE, 15 equal-width bins | 0.247 ± 0.031 | 0.079 ± 0.010 | 68% |
| ECE, 15 equal-mass bins | 0.251 ± 0.027 | 0.099 ± 0.010 | 61% |

95% bootstrap CIs over the test clips (seed 0, 2000 resamples):

| quantity | 95% CI |
|---|---|
| top-1 accuracy | [0.170, 0.281] |
| ECE before | [0.181, 0.280] |
| ECE after | [0.061, 0.139] |
| ECE reduction | [41%, 71%] |

The before/after ECE intervals do not overlap, so the calibration effect is the
one result here that is comfortably larger than the noise. The accuracy number
is weak and its interval is wide — say so rather than lean on it.

---

## Figures

### `figures/fig1_reliability.png` — reliability diagram, before vs after

Bars are observed accuracy per confidence bin; the hatched region is the gap to
the diagonal. Before calibration every well-populated bin sits below the
diagonal — textbook overconfidence. After scaling the bars track the diagonal.
Bins holding fewer than 5 clips are drawn faded, and the panels underneath give
the bin counts, so nobody reads a 2-clip bin as a result.

### `figures/fig2_risk_coverage.png` — the reject option

**Panel A, risk–coverage.** Accuracy rises from 0.223 at full
coverage to 0.328 at 30% coverage and 0.432 at
20% coverage. That is what abstention buys.

**Panel B is the one worth defending in the pitch.** Temperature scaling is a
strictly monotone transform, so it cannot reorder clips by confidence, so it
*cannot change Panel A at all* — at any fixed coverage the accepted set is
identical before and after. Anyone who plots a risk–coverage curve "before and
after calibration" and shows two different curves has made an error.

What calibration actually buys for a reject option is that the confidence
*means* something. Panel B plots, at each coverage, the accuracy the model
claims (mean confidence over accepted clips) against the accuracy it gets:

| coverage | actual accuracy | claimed, before | claimed, after | threshold τ before | τ after |
|---|---|---|---|---|---|
| 20% | 0.432 | 0.756 | 0.481 | 0.606 | 0.352 |
| 30% | 0.328 | 0.692 | 0.427 | 0.536 | 0.305 |
| 40% | 0.303 | 0.644 | 0.393 | 0.464 | 0.274 |
| 50% | 0.312 | 0.599 | 0.363 | 0.395 | 0.230 |
| 60% | 0.299 | 0.562 | 0.340 | 0.352 | 0.209 |
| 70% | 0.256 | 0.529 | 0.319 | 0.302 | 0.185 |
| 80% | 0.240 | 0.497 | 0.300 | 0.259 | 0.157 |
| 90% | 0.224 | 0.469 | 0.283 | 0.220 | 0.133 |
| 100% | 0.223 | 0.440 | 0.266 | 0.113 | 0.071 |

At 50% coverage the uncalibrated model believes it is 0.599
accurate while actually scoring 0.312. After scaling it claims
0.363. That is the difference between a threshold you can
set against a target and one you have to tune blindly.

---

## What the split is

**Signer-disjoint, four ways.** WLASL ships a `signer_id` per clip, so splits hold
out *signers*, not random clips. A random clip split would leak signer identity
and inflate both accuracy and apparent calibration.

| split | clips | glosses | signers |
|---|---|---|---|
| train | 616 | 100 | 20 |
| val | 112 | 67 | 15 |
| cal | 168 | 81 | 16 |
| test | 224 | 98 | 17 |

- **train** — network weights only.
- **val** — picks the training epoch. Different signers from train.
- **cal** — fits the temperature, and nothing else.
- **test** — every number above. Untouched until the end.

Keeping **cal and test on different signers** is the part that matters for this
project. If they shared signers, the temperature would be tuned on the very
condition it is being credited with handling, and the reported ECE reduction
would be optimistic in exactly the way the project claims to fix.

Assignment is greedy and deterministic: signers sorted by clip count, each given
to whichever split is furthest below its target share. No clip was dropped —
all 100 glosses appear in train, so every cal/test clip has a reachable label.

---

## Data

**Used: WLASL (tier b).** 1120 clips, the 100 most-represented glosses of
WLASL2000, pulled per-file from the ungated `Voxel51/WLASL` HuggingFace mirror
(481 MB, 0 download failures). Labels, `signer_id` and crop boxes come from
`WLASL_v0.3.json` in the original `dxli94/WLASL` repo.

**Dropped: SIGMA-ASL (tier a).** The repo distributes the dataset through a Baidu
Netdisk link only, with no sample subset and no partial-download path. That
needs an account and the Baidu client, which is not a few-hours proposition, so
it was abandoned within the first few minutes rather than burning the time box.
Worth revisiting for the real project — it is the only one of the three with
synchronized wrist IMU, which is the multi-modal half of your pitch. This
preliminary result is camera-only and does not test the fusion claim.

Tier (c), self-recorded webcam clips, was not needed.

---

## Method

1. **Landmarks** — MediaPipe Tasks `HandLandmarker` (2 hands) + `PoseLandmarker`
   (lite), 24 uniformly sampled frames per clip, cropped to the WLASL bounding
   box and resized to 512×512.
2. **Features**, 209 per frame, built to be roughly signer-invariant: 25
   upper-body pose points recentred on mid-shoulder and scaled by shoulder
   width; each hand's 21 points relative to its own wrist at the same scale;
   plus each wrist's position in signing space and a presence flag.
3. **Classifier** — single-layer bidirectional GRU (hidden 160), mean- and
   max-pooled over time, dropout 0.3, linear head. ~600 training clips over 100
   classes. AdamW, cosine schedule, 120 epochs, best epoch by val accuracy.
4. **Calibration** — one scalar temperature fit by LBFGS on the calibration
   split's NLL. Nothing else about the model changes.

Extraction quality over all 1120 clips: pose found in
**99.0%** of sampled frames, at least one hand in
**69.3%**, and **0** clips had zero hand
detections.

The model is deliberately weak — the project's claim is about confidence
behaviour, not accuracy, and a weak model makes the overconfidence legible.
0.232 top-1 on 100 classes is ~23× chance.

---

## Environment and how to re-run

Windows 11, Python 3.12 in a venv (MediaPipe has no 3.14 wheel, and the system
Python here is 3.14).

```
pip install mediapipe opencv-python numpy matplotlib scikit-learn torch
```

Installed versions are pinned in `requirements.txt`.

```
python src/01_select_and_download.py   # ~1 min, 481 MB to WLASL_VIDEO_DIR
python src/02_extract_landmarks.py     # ~6.5 min, 6 worker processes
python src/03_split.py
python src/04_train_calibrate.py       # ~2 min, 5 seeds
python src/05_figures.py
python src/06_make_readme.py
```

Videos are cached outside this folder (default `Z:\wlasl_cache`, override with
`WLASL_VIDEO_DIR`), so no video is ever written inside the repository.

**WLASL is under the Computational Use of Data Agreement (C-UDA-1.0) - academic
and non-commercial use only.** No dataset content is redistributed here: the
annotations, extracted landmarks and clip listings are gitignored and
regenerated by `src/01`-`src/02`. The committed logits, split summary and result
JSONs are enough to regenerate both figures and every number above via `src/05`
and `src/06` with no download. See [DATA.md](DATA.md) for full attribution and
licensing, including the MediaPipe model bundles.

Two Windows-specific things that cost time and are worth knowing:

- **MediaPipe 1.0 removed the legacy `mp.solutions` API.** This code uses the
  Tasks API. Most tutorials you will find are for the old one.
- **MediaPipe's C++ model loader cannot open non-ASCII paths**, and this project
  lives under `Masaüstü`. Model bundles are staged to a temp ASCII path at
  startup. Without that it fails with a bare `FileNotFoundError`.

**The GPU is not used.** The default PyPI torch wheel on Windows is CPU-only, and
the model is small enough that it trains in well under a minute on CPU; MediaPipe's
Python path is CPU-only on Windows regardless. Installing the CUDA wheel would
not change any number in this README.

---

## Honest limitations

- **224 test clips** is small for calibration work. The bootstrap CIs above are
  the honest version of every point estimate, and the accuracy CI in particular
  is wide.
- **Camera-only.** No IMU, so this says nothing yet about the multi-modal fusion
  claim. It establishes the calibration + reject-option half.
- **One temperature, global.** The ISAIA paper's contribution is making the
  temperature a *function of the varying condition* (SNR there, sensing quality
  here). This is the unconditional baseline that the conditional version has to
  beat. The signer-disjoint setup is the right harness for testing that next,
  but no conditional result is claimed here.
- **WLASL is noisy** — mixed sources, mixed framing, ~11 clips per class. Low
  accuracy is partly the dataset, not only the model.
- Calibration set and test set are different signers, but both are WLASL
  signers. Calibrating on one *dataset* and testing on another is a harder and
  more honest shift test, and is not done here.
