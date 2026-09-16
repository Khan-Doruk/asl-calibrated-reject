"""Generate RESULTS.md from the stage JSONs, so no number is typed by hand.

Same discipline as src/06_make_readme.py. Reads whichever stage files exist and
writes the sections it can; missing stages are skipped rather than invented.
"""
import json, pathlib

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
        print("stage 2 JSON present but its section is not implemented yet")

    (ROOT / "RESULTS.md").write_text("\n---\n\n".join(parts), encoding="utf-8")
    print(f"wrote RESULTS.md ({sum(len(p) for p in parts)} chars)")


if __name__ == "__main__":
    main()
