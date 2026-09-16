# Research plan

Planning notes for the senior design project this repository supports. The
README reports what was measured; this file records what we are trying to find
out, what we may and may not claim, and what runs next.

---

## The question

Published sign language recognition is evaluated on splits where the same
signer can appear in both training and test. On a signer-disjoint split the
same model collapses — and it stays confidently wrong while it does. On our
test split the model reports a mean confidence of 0.476 while scoring 0.232.

Global temperature scaling cuts calibration error by 68%, but it applies one
scalar to every signer. The project asks whether calibration should instead be
conditioned on signing quality, and whether a calibrated reject option makes an
otherwise unusable model usable.

**One-line version:** same method as our prior work on RF modulation
classifiers, new noise source — SNR becomes occlusion and signer variation.

---

## What we claim, and what we do not

**We do not improve accuracy.** Temperature scaling divides logits by a
positive scalar; that is monotone, so the argmax cannot move. Any claim of an
accuracy gain from calibration is an error.

What can improve accuracy is (a) a second sensing modality, (b) a better model,
and (c) the reject option, which raises accuracy on the answers that are
actually given. That last one is reported as a risk–coverage curve, not as a
headline accuracy number.

There is one indirect path worth testing: a fusion rule has to decide how much
to trust each modality per sample, so miscalibrated inputs make the weighting
wrong. Calibration inside a fusion rule is therefore not purely cosmetic.

**Framing:** we do not make the model more accurate. We make its accuracy
usable.

---

## Novelty position

An earlier, broader version of this claim does not survive a literature check.
The defensible wording is:

> To our knowledge, confidence calibration has not been evaluated for sign
> language recognition: no prior SLR work reports expected calibration error or
> reliability diagrams, and no SLR system provides a calibrated reject option
> characterised by a risk–coverage curve. While confidence-based rejection is
> established in myoelectric and Wi-Fi gesture control, and input-conditional
> temperature scaling has been proposed for generic action recognition (CARING,
> ICPR 2021), neither has been transferred to sign language recognition, nor has
> calibration been conditioned on a signing-quality operating variable.

Conformal prediction appears genuinely untouched in sign language and gesture
recognition, and is the strongest fully open claim available.

### Stage 1 support for this framing

Stage 1 tested the premise directly rather than asserting it. Holding model,
features, seeds and training volume fixed, moving from a signer-disjoint split
to WLASL's standard split moves top-1 from 0.232 to 0.342. The mechanism is
measurable: **98% of the standard test set (125 of 128 clips) comes from a
signer present in training.**

Two honest limits on that claim. The +0.110 is an **upper bound** — the two
protocols define different test sets, so it mixes signer overlap with test-set
difficulty, and this pool cannot separate them (both de-confounding designs are
infeasible; see RESULTS.md). And calibration itself did **not** degrade on the
standard split once bootstrap intervals are accounted for, so "standard splits
break calibration" is not a claim we can make. What we can say is that the
standard protocol overstates accuracy, and that the model stays overconfident
under both.

### Related work we must cite ourselves

A reviewer in this area will know at least one of these. Citing them first is
honesty; having them found for us is a weakness.

| Work | Why it matters to us |
|---|---|
| CARING, ICPR 2021 ([arXiv:2101.00468](https://arxiv.org/abs/2101.00468)) | Learns the temperature as a function of the video representation, i.e. input-conditional calibration for action recognition — the parent domain of isolated SLR. Closest prior work. |
| Scheme, Hudgins & Englehart, IEEE TBME 2013 | Confidence-based rejection for myoelectric gesture control. Reject options in gesture recognition are a decades-old idea. |
| Noisy-ASR calibration, ASRU 2025 ([arXiv:2509.07195](https://arxiv.org/abs/2509.07195)) | Temperature scaling applied selectively under low SNR. SNR-conditional calibration already exists in speech. |
| WiOpen ([arXiv:2402.00822](https://arxiv.org/pdf/2402.00822)) | Uncertainty quantification and rejection of unknown gestures, Wi-Fi sensing. |
| Multicalibration (Hébert-Johnson et al. 2018) | Group-conditional calibration as general theory; conditioning on a group variable is a known framework, not a new idea. |

---

## Where this sits in the literature

Published WLASL100 top-1, **standard split**, trained from scratch on pose:

| method | year | WLASL100 |
|---|---|---|
| Pose-GRU | 2020 | 46.51 |
| ST-GCN | — | 50.78 |
| Pose-TGCN | 2020 | 55.43 |
| SPOTER (MediaPipe keypoints) | 2022 | 78.29 |
| SignBart (0.76M params) | 2025 | 78.00 |
| Stack Transformer | 2026 | 79.45 – 82.95 |

With large-scale pretraining: VideoMAE 75.58 (Kinetics-400), SignBERT+ 79.84,
MASA 83.72, NLA-SLR 91.47, Uni-Sign 92.24 (pretrained on ~1985 h of video–text).

**Our numbers are not comparable to any of these, and stage 1 showed the reason
is worse than we thought.**

The original reason given here was the split protocol plus training volume
(~616 clips against "roughly 1400"). That 1400 is the *official* WLASL100 train
count. Our pool does not contain it. Measured in stage 1:

- Our 100 glosses were selected as the most-represented **on the HuggingFace
  mirror**, not from the official WLASL100 list. They overlap it by **58/100**.
- We hold **678 of the official benchmark's 2038 videos**, and only **1007 (49%)
  of them exist on the mirror at all**, so rebuilding the official list would
  still not close the gap.
- The standard split *within our pool* is 806/186/128 train/val/test, so after
  carving a calibration set it gives **638** training clips, not ~1400.

So neither protocol produces a number that can sit beside Pose-GRU (46.51) or
ST-GCN (50.78). Stage 1's standard-split result (0.383) is above our
signer-disjoint 0.232 but below the published band, and the shortfall is
explained by pool size and vocabulary, not only by the model.

Getting a genuinely comparable number would need a different data source than
the current mirror. Until then, published WLASL100 figures are context, not a
target to be measured against.

Two notes on using these numbers:

- **Use primary sources only.** Secondhand tables disagree. SignBart's table
  cites NLA-SLR at 93.08; the NLA-SLR paper itself reports 91.47 single-crop.
  At least one aggregator lists implausible WLASL2000 figures that appear to be
  subset mislabelling.
- **Do not conflate subsets.** WLASL100 and WLASL2000 numbers differ by roughly
  30 points for the same method.

---

## Sensing modality: an experiment, not an assumption

The camera fails under occlusion, poor lighting and motion blur — and in real
signing, hands cross and pass in front of the torso constantly. A second sensor
with different failure modes should help. Which one is an open question, and we
treat it as our first result rather than a design decision.

| option | wearable? | uncorrelated failures | notes |
|---|---|---|---|
| Second camera | no | weakly (both optical) | cheapest; not testable on any public multimodal set |
| Camera + depth | no | yes | available in SIGMA-ASL |
| Camera + mmWave radar | no | strongly (optical vs RF) | best complementarity; contactless; works in the dark; privacy-preserving; steepest processing chain |
| Camera + wrist IMU | **yes** | strongly | best studied; wireless sync is the hardest engineering; wearables invite a well-documented criticism (below) |

Radar keeps the problem in RF and in detection/estimation, which is where our
prior work sits. Camera-plus-radar fusion for isolated SLR already has
precedent (ICCV 2025 MSLR workshop), so the configuration is validated and our
contribution sits on top of it rather than in it.

**Blocker:** [SIGMA-ASL](https://arxiv.org/abs/2605.06351) is the only public
dataset with synchronized RGB-D, mmWave radar and wrist IMUs (20 participants,
160 ASL signs, 93,545 clips, millisecond alignment). Its repository distributes
the data through a Baidu Netdisk link only, with no sample subset and no
partial-download path, which we have not been able to use from the US. The
authors have been contacted for an alternative mirror. Until that resolves, the
fusion half of the project has no data and the camera-only half stands alone.

---

## Experiment roadmap

Run in order. Report after each stage rather than running everything and
summarising at the end — if something breaks we need to know which stage.

| # | Experiment | Compute | Purpose |
|---|---|---|---|
| 1 | Identical model on WLASL's **standard split** | CPU | **Done — see RESULTS.md.** Protocol gap +0.110 (0.232 → 0.342) with training volume held equal. 98% of the standard test set is signers the model trained on. |
| 2 | Input quality: hand-detection rate (currently 0.693 vs 0.76–0.80 typical), frames per clip 24 → 32/48, review normalization | CPU | SignBart reports +13.5% from normalization alone. Cheap and it gates everything downstream. |
| 3 | Fine-tune a pretrained model — VideoMAE (Kinetics-400) as the low-risk path, a SignBERT+/BEST/MASA skeleton encoder if checkpoints prove obtainable | GPU | Pretraining is worth roughly +25 points; architecture alone is +5–9. Target a *credible* baseline (~70–75), not SOTA. |
| 4 | Best model from stage 3, evaluated signer-disjoint | GPU | The actual research question. |
| 5 | **Per-signer ECE distribution** | CPU | Untested and potentially the real finding: global calibration may fix the average while individual unseen signers stay badly miscalibrated. If true, it directly motivates conditional calibration. If the variance is low, the story weakens and we need to know. |
| 6 | Conditional calibration + reject option vs global temperature | GPU | The contribution. |

Change one thing at a time. If stage 3 changes architecture, input quality and
split simultaneously, we cannot attribute the result.

**Graph models (ST-GCN and relatives)** are a reasonable upgrade — encoding the
skeleton's connectivity as a prior is more data-efficient than making a
transformer learn it, which matters at ~6 clips per class. Expect +5–9 points
(Pose-GRU 46.51 → Pose-TGCN 55.43, same authors, same protocol). Worth doing,
but after stages 1 and 2, and measured separately.

**Not worth doing:** a from-scratch transformer at this data scale. A controlled
study on a 37-class WLASL subset found recurrent models consistently beat a
transformer encoder under identical pipelines, attributed to the transformer's
higher data requirements. The current BiGRU is the right choice for ~6 clips
per class.

---

## Open risks

| Risk | Status |
|---|---|
| **No ASL-fluent consultant.** Nobody on the team signs. Needed to sanity-check vocabulary selection and to interpret error analysis. | Open. First action is NJIT accessibility services or a local ASL program. |
| **SIGMA-ASL access.** Fusion has no data without it. | Authors contacted; no reply yet. |
| **Accuracy credibility.** Calibration analysis on a 23% model is not convincing. | **Narrowed, not closed.** Stage 1 shows 0.232 is largely a protocol artefact: the same model scores 0.383 on the standard split. But our pool is not the official WLASL100 (58/100 glosses), so no number from it is publication-comparable. Stages 2–3 still needed. |
| **Domain gap.** SIGMA-ASL was recorded in a studio with an Azure Kinect; any live demo would use consumer hardware. | Measure it, do not hide it. |
| **Per-signer finding may not exist.** Stage 5 may show low variance. | Report it either way. |

### On wearables

Sign language gloves and similar wearables have been criticised at length by
Deaf researchers and linguists. The documented objections: such systems read
hands only while ASL grammar also lives in eyebrows, mouth shape and torso
position; they work in one direction, so the hearing person still cannot reply;
and they place the burden of accommodation on the Deaf person. Developers
routinely build them without consulting Deaf people.

Our reject option is **our engineering response** to the observation that these
systems present unreliable output as translation. It is not something the Deaf
community asked for, and we should not imply that it is.

This is a design argument for preferring a contactless second modality, and a
reason to raise the modality choice with an ASL consultant rather than settle it
among engineers.

---

## Decisions and rationale

| Decision | Why |
|---|---|
| Signer-disjoint split, four ways | A random clip split leaks signer identity and inflates both accuracy and apparent calibration. Calibration and test use different unseen signers so the temperature is never tuned on the condition it is credited with handling. |
| Camera-only for now | SIGMA-ASL is inaccessible; WLASL works. The calibration contribution does not require fusion. |
| BiGRU over a transformer | Correct for ~6 clips per class; see above. |
| Landmarks over raw RGB, for the fast loop | CPU-friendly, iterates in minutes. RGB re-enters at stage 3 via a pretrained model. |
| Accuracy is a precondition, not the goal | MASA already reports 83.72. Reproducing that is not a contribution. A credible model that reports trustworthy confidence is. |
