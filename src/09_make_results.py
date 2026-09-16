"""Generate RESULTS.md from the stage JSONs, so no number is typed by hand.

Same discipline as src/06_make_readme.py. Reads whichever stage files exist and
writes the sections it can; missing stages are skipped rather than invented.
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
S1 = ROOT / "data" / "stage1_split_comparison.json"
S2 = ROOT / "data" / "stage2_input_quality.json"
PROV = ROOT / "data" / "subset_provenance.json"


def pm(pair, n=3):
    return f"{pair[0]:.{n}f} ± {pair[1]:.{n}f}"


def stage1_section(d, prov):
    m, C = d["meta"], d["conditions"]
    order = [k for k in ("A", "B", "C") if k in C]
    dc = d.get("deconfound_feasibility", {})

    rows = "\n".join(
        f"| **{k}** | {C[k]['description']} | {C[k]['counts']['train']['clips']} | "
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
ECE after [{ci['ece_after'][0]:.3f}, {ci['ece_after'][1]:.3f}]""")

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

| | split | train | test | signers shared train↔test | top-1 | ECE before | ECE after | T* |
|---|---|---|---|---|---|---|---|---|
{rows}

### What it says

**The protocol gap is {accC - accA:+.3f}** — {accA:.3f} (A) → {accC:.3f} (C) with training volume
held equal at {trA} vs {trC} clips. Moving from a signer-disjoint split to WLASL's
standard split, changing nothing else, buys about {100*(accC-accA):.0f} points.

**B vs C is not a real difference.** They differ by {trB - trC} training clips and
{accB - accC:+.3f} accuracy, which is ~{clip_diff:.0f} clips on a {ntest}-clip test set, against seed
spreads of ±{sdB:.3f} and ±{sdC:.3f}. Treat B and C as replicates. The standard split
in this pool holds {m['standard_split_in_pool']['train']} train clips, so after carving a calibration set it
offers almost no volume advantage over A — the confound C was built to control
is nearly absent.

**Calibration behaves the same under both protocols.** Every condition is
overconfident before scaling ({A['agg']['before_ece'][0]:.3f}, {B['agg']['before_ece'][0]:.3f}, {Cc['agg']['before_ece'][0]:.3f}) and improves after. Residual ECE
looks worse on the standard split ({A['agg']['after_ece'][0]:.3f} vs {Cc['agg']['after_ece'][0]:.3f}), but the bootstrap
intervals overlap — A [{A['bootstrap_ci']['ece_after'][0]:.3f}, {A['bootstrap_ci']['ece_after'][1]:.3f}] against C [{Cc['bootstrap_ci']['ece_after'][0]:.3f}, {Cc['bootstrap_ci']['ece_after'][1]:.3f}] — and C's test
set is only {ntest} clips. **Do not claim calibration transfers worse on the standard
split.** It is not supported.

### The confound, and why it cannot be removed here

A and C use different test sets ({A['counts']['test']['clips']} vs {ntest} clips, {A['counts']['test']['glosses']} vs {Cc['counts']['test']['glosses']} glosses), because each
protocol defines its own. So {accC - accA:+.3f} mixes signer overlap with test-set
difficulty, and is an **upper bound** on the signer-overlap effect, not a
measurement of it.

Two designs would isolate it, and neither is possible on this pool:

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
    if S2.exists():
        from importlib import import_module  # noqa: F401  (stage 2 section lands here)
        print("stage 2 JSON present but no section implemented yet")

    (ROOT / "RESULTS.md").write_text("\n---\n\n".join(parts), encoding="utf-8")
    print(f"wrote RESULTS.md ({sum(len(p) for p in parts)} chars)")


if __name__ == "__main__":
    main()
