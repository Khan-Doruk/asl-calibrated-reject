"""Generate RESULTS.md from the stage JSONs, so no number is typed by hand.

Same discipline as src/06_make_readme.py. Reads whichever stage files exist and
writes the sections it can; missing stages are skipped rather than invented.
"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
S1 = ROOT / "data" / "stage1_split_comparison.json"
S2 = ROOT / "data" / "stage2_input_quality.json"
PROV = ROOT / "data" / "subset_provenance.json"
S1B = ROOT / "data" / "stage1b_split_variance.json"


def pm(pair, n=3):
    return f"{pair[0]:.{n}f} ± {pair[1]:.{n}f}"


def labelled(c):
    """train + val + cal: every labelled clip a condition consumes."""
    return sum(c["counts"][s]["clips"] for s in ("train", "val", "cal"))


def reduction_sentence(C, order):
    """Say whether temperature scaling reduced ECE, per condition, from whether
    the seed-0 bootstrap interval on the reduction excludes zero."""
    firm = [k for k in order if C[k]["bootstrap_ci"]["ece_reduction_pct"][0] > 0]
    soft = [k for k in order if k not in firm]
    fmt = lambda k: (f"{k} [{C[k]['bootstrap_ci']['ece_reduction_pct'][0]:.1f}%, "
                     f"{C[k]['bootstrap_ci']['ece_reduction_pct'][1]:.1f}%]")
    if not soft:
        return ("Temperature scaling's ECE reduction has a seed-0 bootstrap interval "
                "above zero in every condition: " + ", ".join(map(fmt, firm)) + ".")
    s = ("Whether scaling *reduces* ECE is established only where the seed-0 bootstrap "
         "interval on the reduction excludes zero. ")
    if firm:
        s += "It does for " + ", ".join(map(fmt, firm)) + ". "
    s += ("It does **not** for " + ", ".join(map(fmt, soft)) +
          f", so improvement there is not established on this test set.")
    return s


def stage1_section(d, prov):
    m, C = d["meta"], d["conditions"]
    order = [k for k in ("A", "B", "C") if k in C]
    dc = d.get("deconfound_feasibility", {})

    rows = "\n".join(
        f"| **{k}** | {C[k]['description']} | {C[k]['counts']['train']['clips']} | {labelled(C[k])} | "
        f"{C[k]['counts']['test']['clips']} | {C[k]['signer_overlap']['train_test']} | "
        f"{pm(C[k]['agg']['before_acc'])} | {pm(C[k]['agg']['before_ece'])} | "
        f"{pm(C[k]['agg']['after_ece'])} | {pm(C[k]['agg']['T'], 2)} |"
        for k in order)

    detail = []
    for k in order:
        c = C[k]; a = c["agg"]; ci = c["bootstrap_ci"]; n = c["counts"]
        counts = " · ".join(
            f"{s} {n[s]['clips']}/{n[s]['signers']}sig/{n[s]['glosses']}gl"
            for s in ("train", "val", "cal", "test"))
        detail.append(f"""
#### {k} — {c['description']}

{counts}

| metric | before | after |
|---|---|---|
| top-1 | {pm(a['before_acc'])} | {pm(a['after_acc'])} |
| ECE (equal-width) | {pm(a['before_ece'])} | {pm(a['after_ece'])} |
| ECE (equal-mass) | {pm(a['before_ece_em'])} | {pm(a['after_ece_em'])} |
| mean confidence | {pm(a['before_mean_conf'])} | {pm(a['after_mean_conf'])} |
| NLL | {pm(a['before_nll'])} | {pm(a['after_nll'])} |

Seed-0 bootstrap 95% CIs — top-1 [{ci['acc'][0]:.3f}, {ci['acc'][1]:.3f}] ·
ECE before [{ci['ece_before'][0]:.3f}, {ci['ece_before'][1]:.3f}] ·
ECE after [{ci['ece_after'][0]:.3f}, {ci['ece_after'][1]:.3f}] ·
ECE reduction [{ci['ece_reduction_pct'][0]:.1f}%, {ci['ece_reduction_pct'][1]:.1f}%]""")

    A, B, Cc = C["A"], C["B"], C["C"]
    accA, accB, accC = (x["agg"]["before_acc"][0] for x in (A, B, Cc))
    sdB, sdC = B["agg"]["before_acc"][1], Cc["agg"]["before_acc"][1]
    trA, trB, trC = (x["counts"]["train"]["clips"] for x in (A, B, Cc))
    ntest = Cc["counts"]["test"]["clips"]
    clip_diff = abs(accB - accC) * ntest
    so = Cc["signer_overlap"]

    p = prov["comparability"] if prov else None
    prov_text = ""
    if p:
        prov_text = f"""
**Our pool is not the official WLASL100.** It was built as the 100
most-represented glosses *on the HuggingFace mirror*, which hosts part of WLASL.
Against the official benchmark: **{p['gloss_overlap']}/100 glosses in common**, and we hold
{p['official_videos_we_hold']} of its {prov['official_wlasl100']['videos']} videos. The official split is
{prov['official_wlasl100']['split']['train']} train / {prov['official_wlasl100']['split']['val']} val / {prov['official_wlasl100']['split']['test']} test — that ~1400 is where the plan's
figure came from, but it is not what this pool contains.

So **no condition here is comparable to Pose-GRU (46.51) or ST-GCN (50.78)**,
on either protocol. Even rebuilding the official gloss list would not fix it:
only {p['official_videos_on_mirror']} of its {prov['official_wlasl100']['videos']} videos ({p['official_videos_on_mirror_frac']:.0%}) are on the mirror at all.
"""

    return f"""## Stage 1 — split protocol, everything else held fixed

Same features, same BiGRU, same {len(m['seeds'])} seeds, same temperature procedure, one shared
pool of {m['pool_clips']} clips over {m['n_classes']} glosses. Only the split assignment varies.

Condition A reproduced the committed baseline **bit-identically**, seed for
seed, which is what makes the three columns comparable.

| | split | train | labelled (train+val+cal) | test | signers shared train↔test | top-1 | ECE before | ECE after | T* |
|---|---|---|---|---|---|---|---|---|---|
{rows}

### What it says

**The protocol gap is {accC - accA:+.3f}** — {accA:.3f} (A) → {accC:.3f} (C) with the
*weight-fitting* set held equal at {trA} vs {trC} clips. That is not all the labelled
data each condition consumes: counting val and cal, A uses {labelled(A)} clips and C uses
{labelled(Cc)}, and C's larger validation set drives epoch selection. Stage 1b tests
whether that matters.

**B vs C is not a real difference.** They differ by {trB - trC} training clips and
{accB - accC:+.3f} accuracy, which is ~{clip_diff:.0f} clips on a {ntest}-clip test set, against seed
spreads of ±{sdB:.3f} and ±{sdC:.3f}. Treat B and C as replicates. The standard split
in this pool holds {m['standard_split_in_pool']['train']} train clips, so after carving a calibration set it
offers almost no volume advantage over A — the confound C was built to control
is nearly absent.

**Every condition is overconfident before scaling** (ECE {A['agg']['before_ece'][0]:.3f}, {B['agg']['before_ece'][0]:.3f}, {Cc['agg']['before_ece'][0]:.3f}).
{reduction_sentence(C, order)}

Residual ECE
looks worse on the standard split ({A['agg']['after_ece'][0]:.3f} vs {Cc['agg']['after_ece'][0]:.3f}), but the bootstrap
intervals overlap — A [{A['bootstrap_ci']['ece_after'][0]:.3f}, {A['bootstrap_ci']['ece_after'][1]:.3f}] against C [{Cc['bootstrap_ci']['ece_after'][0]:.3f}, {Cc['bootstrap_ci']['ece_after'][1]:.3f}] — and C's test
set is only {ntest} clips. **Do not claim calibration transfers worse on the standard
split.** It is not supported.

### The confound, and why it cannot be removed here

A and C use different test sets ({A['counts']['test']['clips']} vs {ntest} clips, {A['counts']['test']['glosses']} vs {Cc['counts']['test']['glosses']} glosses), because each
protocol defines its own. So {accC - accA:+.3f} mixes signer overlap with test-set
difficulty, and is an **upper bound** on the signer-overlap effect, not a
measurement of it.

A second overlap cannot be removed within the standard split: C's validation
set shares {Cc['signer_overlap']['val_test']} signers with its test set, and validation picks the epoch.
Stage 1b's condition D measures how much that selection signal is worth.

Two designs would isolate the signer-overlap effect, and neither is possible on
this pool:

- *Train on signers disjoint from the standard test set.* The standard test set
  draws on {dc.get('standard_test_signers', '?')} of the pool's signers, leaving only
  {dc.get('clips_from_signers_outside_standard_test', '?')} clips from other signers — against the
  {dc.get('clips_needed_to_match_condition_C', '?')} needed.
- *Split the standard test set by whether its signer was trained on.* Of its
  {ntest} clips, **{so.get('test_clips_from_seen_signer', '?')} come from a signer the model trained on and only
  {so.get('test_clips_from_unseen_signer', '?')} do not.**

That second number is the mechanism, measured directly:
**{so.get('test_frac_seen_signer', 0):.0%} of the standard test set is signers the model has already seen.**
That is the clearest statement of why the standard protocol flatters a model,
and it is worth more to the write-up than the {accC - accA:+.3f} itself.
{prov_text}
{"".join(detail)}
"""


def stage1b_section(d):
    m, sm, ref = d["meta"], d["summary"], d["reference_check"]
    dm = sm["draw_means"]
    s1 = sm["stage1_single_draw_reference"]

    def rng_(x):
        return f"{x['mean']:.3f} · sd {x['sd']:.3f} · [{x['min']:.3f}, {x['max']:.3f}]"

    def draw_rows(key):
        out = []
        for x in d[key]:
            n = x["counts"]; a = x["agg"]
            out.append(
                f"| {key}{x['draw']} | {n['train']['clips']} | {n['val']['clips']} | {n['cal']['clips']} | "
                f"{n['test']['clips']} | {x['labelled_clips_used']} | {x['signer_overlap']['val_test']} | "
                f"{pm(a['before_acc'])} | {pm(a['before_ece'])} | {pm(a['after_ece'])} |")
        return "\n".join(out)

    def inside(v, s):
        return "sits inside" if s["min"] <= v <= s["max"] else "falls outside"

    s1c = json.load(open(S1))["conditions"]
    s1_A_seed_sd = s1c["A"]["agg"]["before_acc"][1]
    seed_sd_A = sum(x["agg"]["before_acc"][1] for x in d["A"]) / len(d["A"])
    n_seeds = len(m["model_seeds"])
    seed_floor = seed_sd_A / n_seeds ** 0.5
    sA = dm["acc"]["A"]
    # Like-for-like: each draw mean averages n_seeds runs, so seed noise alone
    # would spread the draw means by about seed_sd / sqrt(n_seeds).
    a_var = (f"The per-draw means of A spread by sd {sA['sd']:.3f}. Seed noise alone would spread "
             f"them by about {seed_floor:.3f} (the {seed_sd_A:.3f} within-draw seed sd over "
             f"√{n_seeds} seeds). ")
    if sA["sd"] > 2 * seed_floor:
        a_var += (f"**The assignment itself is a real source of variation**, well beyond seed noise, "
                  f"so stage 1's seed-only ±{s1_A_seed_sd:.3f} on A understated its uncertainty. ")
    else:
        a_var += "The assignment adds little beyond seed noise. "
    a_var += (f"The committed single draw ({s1['A']:.3f}) {inside(s1['A'], sA)} the draw range "
              f"[{sA['min']:.3f}, {sA['max']:.3f}].")

    g = sm["gap_C_minus_A"]["pairwise_over_draws"]
    gap_txt = (f"Across all {g['n']} pairings of an A draw with a C draw, the gap is "
               f"**{g['mean']:+.3f}** (sd {g['sd']:.3f}, range [{g['min']:+.3f}, {g['max']:+.3f}]). ")
    gap_txt += ("Every pairing is positive: **the gap does not cross zero on any draw.**"
                if g["excludes_zero"] and g["min"] > 0 else
                "**The range crosses zero**, so on some pairings the protocol gap disappears or reverses.")

    e = sm["epoch_selection_D_minus_C"]
    eb, es = e["paired_by_draw"], e["paired_by_draw_and_seed"]
    gap_mean = sm["gap_C_minus_A"]["difference_of_draw_means"]
    frac = abs(eb["mean"] / gap_mean) if gap_mean else float("nan")
    SMALL = 0.25   # "small" = under a quarter of the C - A gap; stated in the text
    if not es["excludes_zero"] and frac < SMALL:
        sel_txt = (f"Paired seed by seed, D − C is {es['mean']:+.3f} (sd {es['sd']:.3f}, range "
                   f"[{es['min']:+.3f}, {es['max']:+.3f}]); the range includes zero. Averaged per draw "
                   f"it is {eb['mean']:+.3f}, {frac:.0%} of the C − A gap (under the {SMALL:.0%} this "
                   f"write-up treats as small). **Epoch selection on C's larger, test-overlapping "
                   f"validation set is not what drives the gap.**")
    elif not es["excludes_zero"]:
        sel_txt = (f"Paired seed by seed, D − C is {es['mean']:+.3f} (sd {es['sd']:.3f}, range "
                   f"[{es['min']:+.3f}, {es['max']:+.3f}]). The range includes zero, but averaged per "
                   f"draw D − C is {eb['mean']:+.3f}, {frac:.0%} of the C − A gap — too large to call "
                   f"negligible. **Inconclusive:** epoch selection may account for part of the gap.")
    elif es["mean"] < 0:
        sel_txt = (f"Paired seed by seed, D − C is {es['mean']:+.3f} (range [{es['min']:+.3f}, "
                   f"{es['max']:+.3f}]), below zero in every pairing. **Part of the gap was better "
                   f"model selection, not better generalisation:** per draw D − C is {eb['mean']:+.3f}, "
                   f"about {frac:.0%} of C − A.")
    else:
        sel_txt = (f"Paired seed by seed, D − C is {es['mean']:+.3f} (range [{es['min']:+.3f}, "
                   f"{es['max']:+.3f}]), above zero in every pairing: the smaller validation set picked "
                   f"*better* epochs, so C's selection signal does not inflate it.")

    def where(v, lo, hi):
        return "below" if v < lo else ("above" if v > hi else "inside")

    eb_A, ea_A = dm["ece_before"]["A"], dm["ece_after"]["A"]
    cb, ca = s1c["A"]["agg"]["before_ece"][0], s1c["A"]["agg"]["after_ece"][0]
    red = [100 * (x["agg"]["before_ece"][0] - x["agg"]["after_ece"][0]) / x["agg"]["before_ece"][0]
           for x in d["A"]]
    red_c = 100 * (cb - ca) / cb
    calib_txt = (
        f"Across the {len(d['A'])} A draws, ECE before scaling ranges "
        f"[{eb_A['min']:.3f}, {eb_A['max']:.3f}] and after scaling [{ea_A['min']:.3f}, {ea_A['max']:.3f}]. "
        f"The committed split's {cb:.3f} before is **{where(cb, eb_A['min'], eb_A['max'])}** that range and its "
        f"{ca:.3f} after is **{where(ca, ea_A['min'], ea_A['max'])}** it. Its {red_c:.0f}% reduction compares "
        f"with {min(red):.0f}–{max(red):.0f}% across draws (mean {sum(red)/len(red):.0f}%). ")
    if where(cb, eb_A['min'], eb_A['max']) == "below" or where(ca, ea_A['min'], ea_A['max']) == "below":
        calib_txt += ("**The committed signer-disjoint split is unusually well calibrated.** The README's "
                      "headline calibration figures come from it and sit at the favourable end of what a "
                      "signer-disjoint split produces; quote the draw range, not the single draw. ")
    all_draws = [x for k in ("A", "C", "D") for x in d[k]]
    n_red = sum(1 for x in all_draws if x["agg"]["after_ece"][0] < x["agg"]["before_ece"][0])
    if n_red == len(all_draws):
        calib_txt += (f"Mean ECE falls after scaling on all {len(all_draws)} draws across A, C and D, but "
                      "the size of the reduction depends on the split as much as on the method.")
    else:
        calib_txt += (f"Mean ECE falls after scaling on only {n_red} of {len(all_draws)} draws, so the "
                      "reduction itself is split-dependent.")

    gd = sm["gap_D_minus_A"]["pairwise_over_draws"]
    lab = s1["labelled_clips_used"]
    alab = dm["a_labelled_clips"]
    vt = [x["signer_overlap"]["val_test"] for x in d["D"]]

    return f"""## Stage 1b — how much of the gap is the split we happened to draw?

Stage 1's ± was model-seed spread only: A and C were each **one** split draw. Here
both are re-drawn {m['K']} times, with {len(m['model_seeds'])} model seeds per draw.

- **A draws** — src/03's greedy with the signer order shuffled. A draw is rejected
  and redrawn if train misses a gloss, or if any split lands more than
  {m['a_size_tolerance']} clips from its target. The size check matters: shuffling alone let one
  very large signer swing the train set by dozens of clips, mixing *who* is in
  train with *how much* data it holds. Keeping sizes exact by sorting big-first
  instead pinned that signer to train on almost every draw, which would never
  test the assignments this experiment is about. Rejection avoids both.
- **C draws** — WLASL's split is fixed upstream, so only the calibration
  carve-out and the train subsample are re-drawn.
- **D draws** — each C draw with validation subsampled to {m['d_val_target']} clips, A's size.

**These draws share one {m['pool_clips']}-clip pool and are not independent.** Every spread below
is descriptive. None of it is a confidence interval.

Torch CPU numerics depend on thread count, so this run pins {m['torch_threads']} threads. A
reference run under these settings **{'reproduces' if ref['reproduces'] else 'does NOT reproduce'}** the committed baseline
(T* {ref['T']:.6f} vs {ref['committed_T']:.6f}). A machine with a different thread count
will not match these numbers bit for bit.

### Per draw (mean ± sd over model seeds)

| draw | train | val | cal | test | labelled | val↔test signers | top-1 | ECE before | ECE after |
|---|---|---|---|---|---|---|---|---|---|
{draw_rows('A')}
{draw_rows('C')}
{draw_rows('D')}

### Across draws (mean · sd · [min, max] of per-draw means)

| | top-1 | ECE before | ECE after | stage 1 single draw |
|---|---|---|---|---|
| A | {rng_(dm['acc']['A'])} | {rng_(dm['ece_before']['A'])} | {rng_(dm['ece_after']['A'])} | {s1['A']:.3f} |
| C | {rng_(dm['acc']['C'])} | {rng_(dm['ece_before']['C'])} | {rng_(dm['ece_after']['C'])} | {s1['C']:.3f} |
| D | {rng_(dm['acc']['D'])} | {rng_(dm['ece_before']['D'])} | {rng_(dm['ece_after']['D'])} | — |

### What it says

**Split assignment.** {a_var}

**The protocol gap.** {gap_txt} Stage 1's single-draw figure was {s1['gap']:+.3f}.

**Calibration depends on the draw too.** {calib_txt}

**Epoch selection (D vs C).** C and D train on identical clips with identical
seeds, and evaluation consumes no RNG, so their weight trajectories are the same
and only the validation set that picks the epoch differs. {sel_txt}

With validation matched as well, the gap D − A is {gd['mean']:+.3f} (range
[{gd['min']:+.3f}, {gd['max']:+.3f}]).

**Labelled data, stated precisely.** Stage 1's A, B and C consume {lab['A']}, {lab['B']} and
{lab['C']} labelled clips (train + val + cal). Each D draw consumes {d['D'][0]['labelled_clips_used']}; A draws consume
{alab['min']:.0f}–{alab['max']:.0f}. D against A holds all three sets equal, not only the weights.

What D cannot remove: its validation set still shares signers with test
({min(vt)}–{max(vt)} per draw). That overlap is inherent to WLASL's standard split.
"""


def stage2_section(d, s1b):
    C, dg, sw, ex, dec, meta = (d["conditions"], d["diagnosis"], d["sweep"], d["extraction"],
                                d["decisions"], d["meta"])
    n_seeds = len(meta["seeds"])
    p_null = 2 * 0.5 ** n_seeds

    def delta(n, metric="before_acc"):
        return C[n]["delta_vs_ref"][metric]

    def est(n, metric="before_acc"):
        return delta(n, metric)["same_sign_all_seeds"]

    def verdict(n, metric="before_acc"):
        x = delta(n, metric)
        tail = "same sign on every seed" if x["same_sign_all_seeds"] else "crosses zero: **not established**"
        return f"{x['mean']:+.3f} [{x['min']:+.3f}, {x['max']:+.3f}] — {tail}"

    def row(n, label, extra="", as_ref=False):
        a = C[n]["agg"]
        v = verdict(n) if C[n].get("delta_vs_ref") and not as_ref else "reference"
        return (f"| {label} | {extra}{pm(a['before_acc'])} | {pm(a['before_ece'])} | {pm(a['after_ece'])} | {v} |")

    ref = C["base_T24"]["agg"]
    committed = meta["committed_reference"]

    # ---- 2a
    srcs = dg["by_source"]
    lo_src = min(srcs, key=lambda s: srcs[s]["hand_rate"])
    hi_src = max(srcs, key=lambda s: srcs[s]["hand_rate"])
    V = sw["variants"]; b = V["sub_base"]
    sweep_rows = "\n".join(
        f"| {t} | {r['description']} | {r['hand_rate']:.3f} ({r['hand_rate'] - b['hand_rate']:+.3f}) | "
        f"{r['pose_rate']:.3f} | {r['hand_rate_worst60']:.3f} | {r['hand_rate_random90']:.3f} | "
        f"{r['hand_rate_first_sixth']:.2f} / {r['hand_rate_middle']:.2f} / {r['hand_rate_last_sixth']:.2f} | "
        f"{r['wall_time_s']:.0f}s |" for t, r in V.items())
    pad_gain = max(V[t]["hand_rate"] for t in ("sub_pad10", "sub_pad20") if t in V) - b["hand_rate"]
    full_b = ex["base"]["per_T"]["24"]; full_d = ex["det_best"]["per_T"]["24"]
    bb = dg["bbox_beyond_frame"]

    if est("det_best_T24") and delta("det_best_T24")["mean"] > 0:
        a_conc = "The detection gain **did** buy accuracy."
    else:
        a_conc = ("**The detection gain did not reliably buy accuracy.** A higher hand-detection rate is "
                  "not, on its own, a better input here.")

    # ---- 2b
    fm = dg["frame_count_metadata"]
    bestT = dec["best_T"]
    tname = {24: "base_T24", 32: "T32", 48: "T48"}
    t_rows = "\n".join(
        f"| T={T} | {ex['base']['per_T'][str(T)]['hand_rate']:.3f} | {ex['base']['per_T'][str(T)]['clips_padded']} | "
        f"{ex['base']['per_T'][str(T)]['padded_frames_total']} | {ex['base']['per_T'][str(T)]['max_pad_fraction']:.0%} | "
        + row(tname[T], "", "").split("| ", 2)[2] for T in (24, 32, 48))
    if bestT == 24:
        b_conc = "No longer sequence beat T=24, so 2c runs at T=24."
    else:
        b_conc = (f"T={bestT} is best on mean accuracy ({verdict(tname[bestT])}), so 2c runs at T={bestT}. "
                  f"Longer T is not a clean single change under src/02's sampling: it also raises the share of "
                  f"end-padded frames, so the T={bestT} gain is measured *despite* more padding, not independently of it.")

    # ---- 2c
    names = [n for n in ("clip_scale", "velocity", "mirror") if n in C]
    c_ref = C[names[0]]["spec"]["ref"] if names else None
    fi = {n: C[n]["feature_info"] for n in names}
    c_rows = "\n".join(row(n, n) for n in names)
    notes = []
    if "clip_scale" in fi:
        notes.append(f"*clip_scale* — per-frame shoulder width varies by a median "
                     f"{fi['clip_scale']['per_frame_shoulder_width_cv_median']:.1%} within a clip "
                     f"(p90 {fi['clip_scale']['per_frame_shoulder_width_cv_p90']:.1%}), so there was little "
                     f"jitter to remove: {verdict('clip_scale')}.")
    if "velocity" in fi:
        hurts = est("velocity") and delta("velocity")["mean"] < 0
        notes.append(f"*velocity* — {fi['velocity']['feature_dim']} input dims instead of "
                     f"{C[c_ref]['feature_info']['feature_dim']}: {verdict('velocity')}."
                     + (" **It hurts.** Frames with no pose are all-zero, so their differences spike, and "
                        "the doubled input has to be learned from the same small training set; this variant "
                        "does not separate those causes." if hurts else ""))
    if "mirror" in fi:
        m = fi["mirror"]
        notes.append(
            f"*mirror* — decided per signer: {m['signers_mirrored']} of {m['signers_total']} signers "
            f"({m['clips_mirrored']} clips) mirrored. Per clip the dominance vote is unreliable: "
            f"{m['signers_3plus_with_mixed_clip_votes']} of {m['signers_with_3plus_clips']} signers with 3+ clips "
            f"get mixed per-clip votes (median minority share {m['mixed_signers_median_minority_share']:.0%}), which "
            f"is detection noise, not signers switching hands. {verdict('mirror')}.")

    # ---- context and roadmap
    all_named = ["det_best_T24", "T32", "T48"] + names
    established_pos = [n for n in all_named if est(n) and delta(n)["mean"] > 0]
    best_n = max(all_named, key=lambda n: delta(n)["mean"])
    best_gain = delta(best_n)["mean"]
    best_ref = C[best_n]["delta_vs_ref"]["ref"]
    # the best measured input configuration, paired seed by seed against base_T24 directly
    top = max(C, key=lambda n: C[n]["agg"]["before_acc"][0])
    cum = [a["before"]["acc"] - b["before"]["acc"]
           for a, b in zip(C[top]["per_seed"], C["base_T24"]["per_seed"])]
    cum_mean, cum_same = float(np.mean(cum)), all(x > 0 for x in cum) or all(x < 0 for x in cum)
    chain = {"mirror": "T=48 + per-signer mirroring", "clip_scale": "T=48 + clip scale",
             "velocity": "T=48 + velocity", "T48": "T=48", "T32": "T=32",
             "det_best_T24": "hand confidence 0.1", "base_T24": "the reference itself"}.get(top, top)
    cum_txt = (f"The best measured input configuration is **{top}** ({chain}): "
               f"{pm(C[top]['agg']['before_acc'])} against base_T24's {pm(C['base_T24']['agg']['before_acc'])}, "
               f"a direct paired difference of **{cum_mean:+.3f}** [{min(cum):+.3f}, {max(cum):+.3f}]"
               + (", same sign on every seed." if cum_same else ", which crosses zero."))
    draw_sd = s1b["summary"]["draw_means"]["acc"]["A"]["sd"] if s1b else None
    ctx = ""
    if draw_sd is not None:
        ctx = (f"Every delta here is on **one** split — the committed signer-disjoint draw. Stage 1b showed "
               f"that re-drawing that split moves accuracy by sd {draw_sd:.3f}; the stacked stage 2 gain "
               f"({cum_mean:+.3f}) is {'smaller than' if abs(cum_mean) < draw_sd else 'comparable to'} that, and "
               f"each single step ({best_gain:+.3f} at most) is smaller. The paired design (same split, same seeds) is what makes the "
               f"comparisons meaningful at all; whether the gains survive a different split is untested.")
    est_txt = (", ".join(f"{n} ({delta(n)['mean']:+.3f})" for n in established_pos)
               if established_pos else "none")
    MATERIAL = 0.05   # a gain this large would change the roadmap; stated in the text
    if not (cum_same and cum_mean >= MATERIAL):
        roadmap = (f"**For the roadmap.** Input quality is not where the accuracy gap lives. The best measured "
                   f"input configuration moves the reference by {cum_mean:+.3f}, short of this write-up's "
                   f"{MATERIAL:.2f} threshold for an established, roadmap-relevant gain. PLAN.md's stage 3 "
                   f"target has to come from the model, not from preprocessing, and stage 1 already showed it "
                   f"cannot be checked against a published benchmark on this pool. **The accuracy credibility "
                   f"risk is unchanged by stage 2.** What stage 2 settles is which inputs stage 3 should build "
                   f"on, and that a higher hand-detection rate is not by itself evidence of better inputs.")
    else:
        roadmap = (f"**For the roadmap.** Stacked, the stage 2 inputs ({chain}) move the reference by "
                   f"{cum_mean:+.3f} on every seed, past this write-up's {MATERIAL:.2f} threshold for a "
                   f"roadmap-relevant gain. That is real but modest: the model stays at "
                   f"{C[top]['agg']['before_acc'][0]:.3f}, far below PLAN.md's stage 3 target, which still has to come "
                   f"from the model rather than preprocessing. Stage 3 should build on these inputs. **The "
                   f"accuracy credibility risk is narrowed slightly by stage 2**, and only on one split: the gain "
                   f"is {'smaller than' if draw_sd is not None and cum_mean < draw_sd else 'comparable to'} stage 1b's "
                   f"split-to-split sd, so it needs checking on re-drawn splits before stage 3 relies on it. A "
                   f"higher hand-detection rate, on its own, bought nothing established.")

    return f"""## Stage 2 — input quality

All conditions use the committed signer-disjoint split, the committed model and
estimators, and {n_seeds} seeds. Each changes one input variable against a reference.

**Reference, trained in this stage.** Torch CPU numerics depend on thread count
(stage 1b), and stage 2 trains in parallel single-thread processes for speed, so
the reference is re-trained here rather than read from the committed run:
base_T24 scores {pm(ref['before_acc'])} (committed, 24 threads: {committed['before_acc'][0]:.3f}).
Every delta is against a reference trained under identical settings.

**What "established" means here.** A delta is paired seed by seed. It counts as
established only if it has the same sign on all {n_seeds} seeds; with no real effect
that happens by chance {p_null:.1%} of the time. Everything else is reported as not
established.

### 2a — hand detection

**Diagnosis first.** Across the pool, {dg['overall_hand_rate_T24']:.3f} of sampled frames have
at least one hand. The shortfall is at the ends of clips: **{dg['hand_rate_first_sixth']:.2f}** in
the first sixth of frames, **{dg['hand_rate_middle']:.2f}** in the middle, **{dg['hand_rate_last_sixth']:.2f}**
in the last sixth. That is the signature of hands at rest outside the frame
before and after the sign, not of a detector failing mid-sign. It also tracks the
source: {lo_src} clips {srcs[lo_src]['hand_rate']:.2f}, {hi_src} {srcs[hi_src]['hand_rate']:.2f}.

The bbox crop is not the cause. Padding it gained at most {pad_gain:+.3f} hand rate on
the subset. {bb['clips']} clip's annotation does not fit its video's resolution (hand rate
{', '.join(f'{r:.2f}' for r in bb['hand_rates'])}); it is a single bad annotation, recorded and left as is.

**Sweep** on a {sw['subset_clips']}-clip subset (the 60 worst clips plus 90 at random), one
setting changed at a time:

| tag | change | hand rate (Δ) | pose rate | worst-60 | random-90 | start / middle / end | wall |
|---|---|---|---|---|---|---|---|
{sweep_rows}

Selection rule: {sw['rule']}. Winner: **{sw['best']}**. On the full pool it raises the
hand rate from {full_b['hand_rate']:.3f} to {full_d['hand_rate']:.3f}, with pose rate
{full_b['pose_rate']:.3f} → {full_d['pose_rate']:.3f}.

**Does it move accuracy?** Detection rate is a proxy; this is the result.

| condition | top-1 | ECE before | ECE after | Δ top-1 vs base_T24 |
|---|---|---|---|---|
{row('base_T24', 'base_T24')}
{row('det_best_T24', 'det_best_T24')}

{a_conc} Δ ECE after scaling: {verdict('det_best_T24', 'after_ece')}.

### 2b — frames per clip

Container metadata overstates clip length: **{fm['clips_decoding_fewer_than_reported']}** clips decode fewer
frames than they report (median shortfall {fm['median_shortfall']:.0f}, max {fm['max_shortfall']}). src/02 samples
indices from the reported count, so the last indices do not exist and those clips
are end-padded with their final frame **even at T=24** — already true of the
committed baseline. Longer T adds genuine short clips on top: reported lengths run
{fm['reported_frames']['min']}–{fm['reported_frames']['max']} frames (median {fm['reported_frames']['median']:.0f}), and
{fm['clips_reporting_fewer_than']['48']} clips report fewer than 48. Padding is at the **end**, a frozen last
frame, not frames duplicated evenly through the sign.

| T | hand rate | clips end-padded | padded frames | worst clip padded | top-1 | ECE before | ECE after | Δ top-1 vs T=24 |
|---|---|---|---|---|---|---|---|---|
{t_rows}

{b_conc}

### 2c — normalization, at T={bestT}

Each variant changes one thing against {c_ref} and is computed from saved raw
coordinates (no re-extraction).

| variant | top-1 | ECE before | ECE after | Δ top-1 vs {c_ref} |
|---|---|---|---|---|
{row(c_ref, c_ref, as_ref=True)}
{c_rows}

{chr(10).join('- ' + x for x in notes)}

### What stage 2 means

Established single-step accuracy gains: **{est_txt}**. {cum_txt}

{ctx}

{roadmap}
"""


def main():
    parts = ["""# Results

Generated by `src/09_make_results.py` from the JSONs in `data/`.
Every number is read from a file; none is typed by hand.
"""]
    prov = json.load(open(PROV)) if PROV.exists() else None
    if S1.exists():
        parts.append(stage1_section(json.load(open(S1)), prov))
    else:
        print("no stage 1 JSON, skipping")
    if S1B.exists():
        parts.append(stage1b_section(json.load(open(S1B))))
    else:
        print("no stage 1b JSON, skipping")
    if S2.exists():
        parts.append(stage2_section(json.load(open(S2)),
                                    json.load(open(S1B)) if S1B.exists() else None))
    else:
        print("no stage 2 JSON, skipping")

    (ROOT / "RESULTS.md").write_text("\n---\n\n".join(parts), encoding="utf-8")
    print(f"wrote RESULTS.md ({sum(len(p) for p in parts)} chars)")


if __name__ == "__main__":
    main()
