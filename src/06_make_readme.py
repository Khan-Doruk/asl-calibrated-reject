"""Generate README.md from the result files, so no number in the write-up is
typed by hand. Re-run after 04 and 05 and the README stays consistent."""
import json, pathlib, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
res = json.load(open(ROOT / "data" / "results.json"))
ci = json.load(open(ROOT / "data" / "bootstrap_ci.json"))
rc = json.load(open(ROOT / "data" / "risk_coverage_table.json"))
sp = json.load(open(ROOT / "data" / "splits.json"))
ex = json.load(open(ROOT / "data" / "extraction_stats.json"))

a = res["agg"]
s0 = res["per_seed"][0]
summ = sp["summary"]


def f(x, n=3):
    """Format a scalar, or the mean of a [mean, sd] pair."""
    if isinstance(x, (list, tuple)):
        x = x[0]
    return f"{x:.{n}f}"


def pm(key, n=3):
    m, s = a[key]
    return f"{m:.{n}f} ± {s:.{n}f}"


red = 100 * (a["before_ece"][0] - a["after_ece"][0]) / a["before_ece"][0]
red_em = 100 * (a["before_ece_em"][0] - a["after_ece_em"][0]) / a["before_ece_em"][0]

rows = [(k, summ["per_split"][k]["clips"], summ["per_split"][k]["glosses"],
         summ["per_split"][k]["signers"]) for k in ("train", "val", "cal", "test")]

rc_rows = "\n".join(
    f"| {r['coverage']:.0%} | {r['accuracy']:.3f} | {r['claimed_before']:.3f} | "
    f"{r['claimed_after']:.3f} | {r['tau_before']:.3f} | {r['tau_after']:.3f} |"
    for r in rc)

split_rows = "\n".join(
    f"| {k} | {n} | {g} | {s} |" for k, n, g, s in rows)

README = f"""# Calibrated sign recognition with a reject option — preliminary results

Preliminary figures for the senior design pitch: a sign classifier on real ASL
video, its confidence calibrated by temperature scaling, and a reject option
evaluated by a risk–coverage curve.

Everything here was measured, not estimated. Re-running `src/01` … `src/06`
reproduces every number and both figures.

---

## Headline numbers

Test split = **{res['n_test']} clips from {rows[3][3]} signers who appear in no other split**,
{res['n_classes']} sign classes (chance = 1.0%).

| | value |
|---|---|
| **Top-1 accuracy (baseline)** | **{f(a['before_acc'][0])}** ({pm('before_acc')} over {len(res['seeds'])} seeds) |
| **ECE before calibration** | **{f(a['before_ece'][0])}** ({pm('before_ece')}) |
| **ECE after temperature scaling** | **{f(a['after_ece'][0])}** ({pm('after_ece')}) |
| ECE reduction | **{red:.0f}%** |
| Fitted temperature T* | {pm('T', 2)} |
| Mean confidence before → after | {f(a['before_mean_conf'])} → {f(a['after_mean_conf'])} |
| NLL before → after | {f(a['before_nll'])} → {f(a['after_nll'])} |

Accuracy is unchanged by temperature scaling ({f(a['before_acc'][0])} → {f(a['after_acc'][0])}),
which is expected: dividing the logits by a positive scalar cannot change the argmax.

**The overconfidence is the point.** Before calibration the model reports a mean
confidence of {f(a['before_mean_conf'][0])} while actually being right
{f(a['before_acc'][0])} of the time — a gap of
{a['before_mean_conf'][0] - a['before_acc'][0]:.3f}. After scaling, the mean confidence is
{f(a['after_mean_conf'][0])}, essentially matching the true accuracy.

### Robustness

Two ECE binning schemes, because 15-bin equal-width ECE is unstable at n={res['n_test']}:

| estimator | before | after | reduction |
|---|---|---|---|
| ECE, 15 equal-width bins | {pm('before_ece')} | {pm('after_ece')} | {red:.0f}% |
| ECE, 15 equal-mass bins | {pm('before_ece_em')} | {pm('after_ece_em')} | {red_em:.0f}% |

95% bootstrap CIs over the test clips (seed 0, 2000 resamples):

| quantity | 95% CI |
|---|---|
| top-1 accuracy | [{f(ci['acc'][0])}, {f(ci['acc'][1])}] |
| ECE before | [{f(ci['ece_before'][0])}, {f(ci['ece_before'][1])}] |
| ECE after | [{f(ci['ece_after'][0])}, {f(ci['ece_after'][1])}] |
| ECE reduction | [{ci['ece_reduction_pct'][0]:.0f}%, {ci['ece_reduction_pct'][1]:.0f}%] |

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

**Panel A, risk–coverage.** Accuracy rises from {f(rc[-1]['accuracy'])} at full
coverage to {f(rc[1]['accuracy'])} at 30% coverage and {f(rc[0]['accuracy'])} at
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
{rc_rows}

At 50% coverage the uncalibrated model believes it is {f(rc[3]['claimed_before'])}
accurate while actually scoring {f(rc[3]['accuracy'])}. After scaling it claims
{f(rc[3]['claimed_after'])}. That is the difference between a threshold you can
set against a target and one you have to tune blindly.

---

## What the split is

**Signer-disjoint, four ways.** WLASL ships a `signer_id` per clip, so splits hold
out *signers*, not random clips. A random clip split would leak signer identity
and inflate both accuracy and apparent calibration.

| split | clips | glosses | signers |
|---|---|---|---|
{split_rows}

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

**Used: WLASL (tier b).** {summ['total_clips']} clips, the 100 most-represented glosses of
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
   (lite), {ex['frames_per_clip']} uniformly sampled frames per clip, cropped to the WLASL bounding
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

Extraction quality over all {ex['n_extracted']} clips: pose found in
**{ex['pose_rate']:.1%}** of sampled frames, at least one hand in
**{ex['hand_rate']:.1%}**, and **{ex['clips_zero_hands']}** clips had zero hand
detections.

The model is deliberately weak — the project's claim is about confidence
behaviour, not accuracy, and a weak model makes the overconfidence legible.
{f(a['before_acc'][0])} top-1 on 100 classes is ~{a['before_acc'][0] / 0.01:.0f}× chance.

---

## Getting set up (start here if you just cloned this)

**Python 3.12.** MediaPipe has no 3.13/3.14 wheel yet, so a newer Python fails at
`pip install`. Any OS is fine; this was developed on Windows 11.

```
python3.12 -m venv .venv
# Windows:   .venv/Scripts/Activate.ps1
# mac/linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Just want the figures and numbers? (no download, seconds)

Everything needed is already committed:

```
python src/05_figures.py      # regenerates both figures
python src/06_make_readme.py  # regenerates this README
```

### Want to reproduce from raw video? (~10 min, ~481 MB)

`src/01` downloads the clip subset; `src/02` fetches the MediaPipe model bundles
automatically on first run. Nothing is manual.

```
python src/01_select_and_download.py   # ~1 min, 481 MB to WLASL_VIDEO_DIR
python src/02_extract_landmarks.py     # ~6.5 min, 6 worker processes
python src/03_split.py
python src/04_train_calibrate.py       # ~2 min, 5 seeds
python src/05_figures.py
python src/06_make_readme.py
```

Clips are cached **outside the repository** at `~/.cache/wlasl_clips` by
default, so no video is ever committed and re-running costs nothing. Point it
elsewhere (external drive, shared scratch disk) with `WLASL_VIDEO_DIR`:

```
# Windows PowerShell
$env:WLASL_VIDEO_DIR = "D:/wlasl_clips"
# mac/linux
export WLASL_VIDEO_DIR=/mnt/scratch/wlasl_clips
```

**Teammates:** if one of you has already run `src/02`, sharing that
`data/landmarks.npz` directly lets everyone else skip steps 01-02 and start at
`src/03`. Pass it around privately - keep it off GitHub, it is WLASL-derived
(see [DATA.md](DATA.md)).

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

- **{res['n_test']} test clips** is small for calibration work. The bootstrap CIs above are
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
"""

(ROOT / "README.md").write_text(README, encoding="utf-8")
print(f"wrote README.md ({len(README)} chars)")
