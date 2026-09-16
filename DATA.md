# Data, licensing, and what this repository does not contain

The MIT license in `LICENSE` covers **the code in this repository only**. It does
not cover the WLASL dataset, any part of it, or the third-party model bundles.

## WLASL

Results here are computed on **WLASL** (Word-Level American Sign Language),
introduced in:

> D. Li, C. Rodriguez Opazo, X. Yu, H. Li. *Word-level Deep Sign Language
> Recognition from Video: A New Large-scale Dataset and Methods Comparison.*
> WACV 2020. https://github.com/dxli94/WLASL

**WLASL is released under the Computational Use of Data Agreement (C-UDA-1.0).
It is for academic and computational use only. No commercial use is permitted.**
Read `C-UDA-1.0.pdf` in the upstream repository and agree to its terms before
using the dataset. The upstream authors also ask to be contacted regarding any
copyright or privacy concern.

### Not redistributed here

To avoid redistributing dataset content, the following are excluded via
`.gitignore` and are regenerated locally by the pipeline:

| path | what it is | how to get it |
|---|---|---|
| `data/WLASL_v0.3.json` | upstream WLASL annotations | downloaded by `src/01` |
| `data/landmarks.npz` | landmarks extracted from WLASL video | produced by `src/02` |
| `data/matched.json`, `data/subset.json`, `data/hf_files.json` | derived clip listings | produced by `src/01` |
| video clips | WLASL mp4 files | downloaded by `src/01` to `WLASL_VIDEO_DIR`, **never** inside this repo |

No WLASL video, frame, annotation file, or extracted landmark array is committed.

### What *is* committed

Model outputs and experiment bookkeeping, which are results rather than dataset
content, and which are small:

- `data/logits_seed0.npz` — test/calibration logits and labels for seed 0
- `data/results.json`, `data/bootstrap_ci.json`, `data/risk_coverage_table.json`
- `data/splits.json` — split assignment plus a summary table
- `data/extraction_stats.json` — landmark detection rates
- `figures/*.png`

This is enough to re-derive every number in `README.md` and regenerate both
figures (`src/05`, `src/06`) **without** downloading anything. Reproducing from
raw video requires running `src/01`–`src/04`, which pulls ~481 MB of WLASL
clips and is subject to the C-UDA terms above.

## Video source mirror

Clips are fetched per-file from the ungated HuggingFace mirror
[`Voxel51/WLASL`](https://huggingface.co/datasets/Voxel51/WLASL), which hosts the
same clips as upstream WLASL. The C-UDA terms apply to that copy as well.

## MediaPipe model bundles

`src/02` downloads `hand_landmarker.task` and `pose_landmarker_lite.task` from
Google's MediaPipe model store. These are Google's models under the Apache
License 2.0 and are **not** committed here (`models/` is gitignored); the script
fetches them on first run.

- https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

## Related prior work by the author

The calibration method ported here originates in the author's paper on
SNR-adaptive temperature scaling for deep modulation classifiers (IEEE ISAIA
2026). This repository applies the unconditional (single global temperature)
baseline to sign recognition; it does not implement or claim the conditional
method from that paper.
