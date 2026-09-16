"""Stage 1b - how much of the protocol gap is the particular split we drew?

Stage 1 reported A = 0.232 and C = 0.342 with a +/- that is *model-seed* spread
only. Both are single draws: src/03 is a deterministic greedy over signers, and
B/C's calibration carve-out and subsample were fixed at one RNG seed. This
script re-draws both sides.

  A draws  K signer-disjoint splits. Same greedy and target shares as src/03,
           but the signer order is shuffled per draw. A draw is rejected and
           redrawn - and the rejection recorded - if its train set misses a
           gloss, or if any split lands more than SIZE_TOL clips from target.

           The size check matters. A shuffled order lets a large signer land
           in a nearly full split (one signer holds 182 clips), which moved
           train between 582 and 657 in a dry run and would mix *who* is in
           train with *how much* data each split gets. The alternative that
           keeps sizes exact - big-first with a jittered sort - put that
           signer in train 99% of the time, so it would never test the
           assignments this experiment exists to probe.
  C draws  K standard-split variants. The split itself is WLASL's and cannot be
           redrawn; the calibration carve-out and the train subsample can.
  D draws  C with val subsampled to A's val size, everything else identical.

D isolates epoch selection exactly. For a given model seed, C and D train on the
same clips and evaluation runs in eval mode with no dropout, so it consumes no
RNG: the two follow the same weight trajectory and differ only in which epoch
the validation set picks. D - C is that effect, paired seed by seed.

All draws share one 1120-clip pool, so they are NOT independent samples. The
spread reported here is descriptive. It is not a confidence interval.

Sampling uses src/stratify.py (largest-remainder). src/07 and its JSON are left
as committed; its seed-0 conditions are reported alongside for reference.

Torch CPU results depend on the thread count, so it is pinned (TORCH_THREADS)
and a reference run checks that the committed baseline still reproduces.

Writes data/stage1b_split_variance.json.
"""
import json, pathlib, collections, importlib.util, sys, time
import numpy as np, torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from stratify import stratified_take, _self_check  # noqa: E402

OUT = ROOT / "data" / "stage1b_split_variance.json"
PARTIAL = ROOT / "data" / "stage1b_split_variance.partial.json"
K = 5
TORCH_THREADS = 24        # the committed runs used torch's default of 24 on the dev machine
TARGET = {"train": 0.55, "val": 0.10, "cal": 0.15, "test": 0.20}   # same as src/03
CAL_SHARE = 0.15
MAX_ATTEMPTS = 2000
SIZE_TOL = 10             # clips; max |split size - target| allowed per A draw
SPLITS = ("train", "val", "cal", "test")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tc = _load("train_calibrate", "04_train_calibrate.py")


# ------------------------------------------------------------------ splits
def signer_disjoint_draw(signer, gloss, n_classes, draw):
    """src/03's greedy with a shuffled signer order. Redraws until train covers
    every gloss and every split is within SIZE_TOL clips of target; returns the
    split and the record of every attempt."""
    counts = collections.Counter(signer.tolist())
    total = len(signer)
    attempts = []
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng([draw, attempt])
        signers = sorted(counts)
        order = [signers[i] for i in rng.permutation(len(signers))]
        assign, have = {}, {k: 0 for k in TARGET}
        for s in order:
            pick = max(TARGET, key=lambda k: TARGET[k] - have[k] / total)
            assign[s] = pick
            have[pick] += counts[s]
        split = np.array([assign[s] for s in signer.tolist()])
        train_gloss = len(set(gloss[split == "train"].tolist()))
        max_dev = max(abs(have[k] - TARGET[k] * total) for k in TARGET)
        reasons = []
        if train_gloss != n_classes:
            reasons.append(f"train covers {train_gloss}/{n_classes} glosses")
        if max_dev > SIZE_TOL:
            reasons.append(f"size deviation {max_dev:.0f} > {SIZE_TOL}")
        attempts.append(dict(attempt=attempt, train_glosses=train_gloss,
                             max_size_deviation=float(max_dev),
                             accepted=not reasons, rejected_for=reasons))
        if not reasons:
            idx = {k: np.where(split == k)[0] for k in SPLITS}
            return idx, attempts, {str(s): v for s, v in assign.items()}
    raise RuntimeError(f"draw {draw}: no acceptable split in {MAX_ATTEMPTS} attempts")


def standard_draw(std_split, y, r, n_train, n_val):
    """C and D for RNG draw r. Same test set every draw (WLASL's)."""
    std_train = np.where(std_split == "train")[0].tolist()
    cal, train_b = stratified_take(std_train, y, int(round(CAL_SHARE * len(y))),
                                   np.random.default_rng([r, 0]))
    dropped, train_c = stratified_take(train_b, y, len(train_b) - n_train,
                                       np.random.default_rng([r, 1]))
    val = np.where(std_split == "val")[0].tolist()
    val_dropped, val_d = stratified_take(val, y, len(val) - n_val,
                                         np.random.default_rng([r, 2]))
    idx_c = {"train": np.array(train_c), "val": np.array(val),
             "cal": np.array(cal), "test": np.where(std_split == "test")[0]}
    idx_d = dict(idx_c, val=np.array(val_d))
    prov = dict(cal=cal, train_dropped=dropped, val_dropped=val_dropped)
    return idx_c, idx_d, prov


# ------------------------------------------------------------------ runs
def describe(idx, gloss, signer):
    counts = {k: dict(clips=len(v), glosses=len(set(gloss[v].tolist())),
                      signers=len(set(signer[v].tolist()))) for k, v in idx.items()}
    sig = {k: set(signer[v].tolist()) for k, v in idx.items()}
    seen = np.array([s in sig["train"] for s in signer[idx["test"]].tolist()])
    overlap = dict(train_test=len(sig["train"] & sig["test"]),
                   val_test=len(sig["val"] & sig["test"]),
                   cal_test=len(sig["cal"] & sig["test"]),
                   test_frac_seen_signer=float(seen.mean()))
    labelled = sum(len(idx[k]) for k in ("train", "val", "cal"))
    return counts, overlap, labelled


def run_condition(label, idx, X, y, gloss, signer):
    rows = []
    for s in tc.SEEDS:
        t0 = time.time()
        lg, vacc = tc.run_seed(s, X, y, idx)
        T = tc.fit_temperature(lg["cal"], y[idx["cal"]])
        before, after = tc.summarize(lg["test"], y[idx["test"]], T)
        rows.append(dict(seed=s, T=T, val_acc=vacc, before=before, after=after))
        print(f"  {label} seed {s}: acc={before['acc']:.3f} ECE {before['ece']:.3f}"
              f"->{after['ece']:.3f} T*={T:.2f} ({time.time()-t0:.0f}s)", flush=True)
    agg = {}
    for phase in ("before", "after"):
        for m in ("acc", "ece", "ece_em", "mean_conf", "nll"):
            v = np.array([r[phase][m] for r in rows])
            agg[f"{phase}_{m}"] = [float(v.mean()), float(v.std())]
    agg["T"] = [float(np.mean([r["T"] for r in rows])), float(np.std([r["T"] for r in rows]))]
    counts, overlap, labelled = describe(idx, gloss, signer)
    return dict(counts=counts, signer_overlap=overlap, labelled_clips_used=labelled,
                per_seed=rows, agg=agg)


def spread(values):
    v = np.asarray(values, dtype=float)
    return dict(n=int(v.size), mean=float(v.mean()),
                sd=float(v.std(ddof=1)) if v.size > 1 else 0.0,
                min=float(v.min()), max=float(v.max()),
                excludes_zero=bool(v.min() > 0 or v.max() < 0))


def main():
    _self_check()
    torch.set_num_threads(TORCH_THREADS)
    t_start = time.time()

    d = np.load(ROOT / "data" / "landmarks.npz", allow_pickle=True)
    X, gloss, signer, vids = d["X"], d["gloss"], d["signer"], d["vids"].tolist()
    sel = json.load(open(ROOT / "data" / "subset.json"))
    assert [r[1] for r in sel] == vids, "subset.json and landmarks.npz are out of order"
    std_split = np.array([r[3] for r in sel])
    classes = sorted(set(gloss.tolist()))
    cmap = {g: i for i, g in enumerate(classes)}
    y = np.array([cmap[g] for g in gloss.tolist()])

    sp = json.load(open(ROOT / "data" / "splits.json"))
    committed_split = np.array(sp["split"])
    idx_committed = {k: np.where((committed_split == k) & np.array(sp["usable"]))[0]
                     for k in SPLITS}
    n_train = len(idx_committed["train"])
    n_val = len(idx_committed["val"])

    # ---- reference: does the committed baseline reproduce under these settings?
    ref = json.load(open(ROOT / "data" / "results.json"))["per_seed"][0]
    lg, _ = tc.run_seed(0, X, y, idx_committed)
    T = tc.fit_temperature(lg["cal"], y[idx_committed["cal"]])
    b, _ = tc.summarize(lg["test"], y[idx_committed["test"]], T)
    reference = dict(T=T, committed_T=ref["T"], acc=b["acc"], committed_acc=ref["before"]["acc"],
                     reproduces=abs(T - ref["T"]) < 1e-9 and abs(b["ece"] - ref["before"]["ece"]) < 1e-12)
    print(f"reference seed-0 run reproduces committed baseline: {reference['reproduces']}", flush=True)

    payload = dict(
        meta=dict(K=K, model_seeds=tc.SEEDS, torch_threads=TORCH_THREADS,
                  target_shares=TARGET, cal_share=CAL_SHARE, a_size_tolerance=SIZE_TOL,
                  c_train_target=n_train, d_val_target=n_val,
                  pool_clips=len(y), n_classes=len(classes),
                  independence_note=("All draws share one 1120-clip pool and are not independent "
                                     "samples. Spreads are descriptive, not confidence intervals."),
                  sampling="src/stratify.py largest-remainder"),
        reference_check=reference, A=[], C=[], D=[])

    def checkpoint():
        json.dump(payload, open(PARTIAL, "w"), indent=2)

    for draw in range(K):
        idx, attempts, assign = signer_disjoint_draw(signer, gloss, len(classes), draw)
        print(f"A draw {draw}: accepted on attempt {len(attempts)} " +
              " ".join(f"{k}={len(v)}" for k, v in idx.items()), flush=True)
        res = run_condition(f"A{draw}", idx, X, y, gloss, signer)
        res.update(draw=draw, signer_assignment=assign,
                   attempts_total=len(attempts),
                   rejected_gloss_coverage=sum(1 for a in attempts
                                               if any("glosses" in r for r in a["rejected_for"])),
                   rejected_size=sum(1 for a in attempts
                                     if any("size" in r for r in a["rejected_for"])),
                   accepted_max_size_deviation=attempts[-1]["max_size_deviation"])
        payload["A"].append(res)
        checkpoint()

    for r in range(K):
        idx_c, idx_d, prov = standard_draw(std_split, y, r, n_train, n_val)
        print(f"C/D draw {r}: " + " ".join(f"{k}={len(v)}" for k, v in idx_c.items()) +
              f" | D val={len(idx_d['val'])}", flush=True)
        c = run_condition(f"C{r}", idx_c, X, y, gloss, signer)
        c.update(draw=r, provenance=dict(
            cal_clip_ids=[vids[i] for i in prov["cal"]],
            train_dropped_clip_ids=[vids[i] for i in prov["train_dropped"]]))
        payload["C"].append(c)
        dd = run_condition(f"D{r}", idx_d, X, y, gloss, signer)
        dd.update(draw=r, provenance=dict(val_dropped_clip_ids=[vids[i] for i in prov["val_dropped"]]))
        payload["D"].append(dd)
        checkpoint()

    # ---- the gap, as distributions over draws
    acc = {k: [x["agg"]["before_acc"][0] for x in payload[k]] for k in ("A", "C", "D")}
    ece_b = {k: [x["agg"]["before_ece"][0] for x in payload[k]] for k in ("A", "C", "D")}
    ece_a = {k: [x["agg"]["after_ece"][0] for x in payload[k]] for k in ("A", "C", "D")}
    per_seed_dc = [dr["per_seed"][i]["before"]["acc"] - cr["per_seed"][i]["before"]["acc"]
                   for cr, dr in zip(payload["C"], payload["D"]) for i in range(len(tc.SEEDS))]
    s1 = json.load(open(ROOT / "data" / "stage1_split_comparison.json"))["conditions"]

    payload["summary"] = dict(
        draw_means=dict(
            acc={k: spread(v) for k, v in acc.items()},
            ece_before={k: spread(v) for k, v in ece_b.items()},
            ece_after={k: spread(v) for k, v in ece_a.items()},
            a_train_clips=spread([x["counts"]["train"]["clips"] for x in payload["A"]]),
            a_labelled_clips=spread([x["labelled_clips_used"] for x in payload["A"]]),
        ),
        gap_C_minus_A=dict(
            pairwise_over_draws=spread([c - a for c in acc["C"] for a in acc["A"]]),
            difference_of_draw_means=float(np.mean(acc["C"]) - np.mean(acc["A"])),
            note="All K x K pairings of A draws with C draws."),
        gap_D_minus_A=dict(
            pairwise_over_draws=spread([dv - a for dv in acc["D"] for a in acc["A"]]),
            difference_of_draw_means=float(np.mean(acc["D"]) - np.mean(acc["A"]))),
        epoch_selection_D_minus_C=dict(
            paired_by_draw=spread([dv - c for dv, c in zip(acc["D"], acc["C"])]),
            paired_by_draw_and_seed=spread(per_seed_dc),
            note=("Same training clips, same seed, identical weight trajectory; only the "
                  "validation set used to pick the epoch differs.")),
        stage1_single_draw_reference=dict(
            A=s1["A"]["agg"]["before_acc"][0], C=s1["C"]["agg"]["before_acc"][0],
            gap=s1["C"]["agg"]["before_acc"][0] - s1["A"]["agg"]["before_acc"][0],
            labelled_clips_used={k: sum(s1[k]["counts"][s]["clips"] for s in ("train", "val", "cal"))
                                 for k in ("A", "B", "C")}),
        wall_time_s=time.time() - t_start,
    )
    json.dump(payload, open(OUT, "w"), indent=2)
    PARTIAL.unlink(missing_ok=True)

    sm = payload["summary"]
    print("\n=== stage 1b (draw means; spreads are descriptive, draws share one pool) ===")
    for k in ("A", "C", "D"):
        a = sm["draw_means"]["acc"][k]
        print(f"{k}: acc mean {a['mean']:.3f} sd {a['sd']:.3f} range [{a['min']:.3f}, {a['max']:.3f}]")
    g = sm["gap_C_minus_A"]["pairwise_over_draws"]
    print(f"gap C-A: mean {g['mean']:+.3f} sd {g['sd']:.3f} range [{g['min']:+.3f}, {g['max']:+.3f}] "
          f"excludes zero: {g['excludes_zero']}")
    e = sm["epoch_selection_D_minus_C"]["paired_by_draw_and_seed"]
    print(f"D-C (epoch selection, paired): mean {e['mean']:+.3f} sd {e['sd']:.3f} "
          f"range [{e['min']:+.3f}, {e['max']:+.3f}]")
    print(f"wrote {OUT.relative_to(ROOT)} in {sm['wall_time_s']/60:.1f} min")


if __name__ == "__main__":
    main()
