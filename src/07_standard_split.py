"""Stage 1 - the same model and features under two split protocols.

Three conditions, sharing one clip pool, one feature set and one model so the
only thing that varies is how clips are assigned to splits:

  A  signer-disjoint (the committed baseline)      train ~616
  B  WLASL's standard split                        train ~638
  C  standard, train subsampled to match A         train  616

C is the one that answers the research question: B differs from A in both
protocol and training volume, C removes the volume difference.

Everything is read from the existing data/landmarks.npz and data/subset.json -
no re-download, no re-extraction. The standard split lives in the per-instance
`split` field of WLASL_v0.3.json, which src/01 already carries into
subset.json; src/02 drops it when writing landmarks.npz, so it is joined back
here by video_id (the two files are in the same order, which is asserted).

Estimators (ECE both binnings, temperature fitting) and the training loop are
imported from src/04 rather than reimplemented, so numbers stay comparable to
the baseline. Writes data/stage1_split_comparison.json. Touches nothing that
src/01-06 wrote.

Note on B and C: cal and test share signers there. That is the condition under
test, not an oversight.
"""
import json, pathlib, collections, importlib.util
import numpy as np, torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stage1_split_comparison.json"
CAL_SHARE = 0.15          # matches the baseline's cal share of the pool
RNG_SEED = 0              # for the cal carve-out and the condition-C subsample


def _load_trainer():
    """Import src/04 as a module (its name starts with a digit)."""
    path = ROOT / "src" / "04_train_calibrate.py"
    spec = importlib.util.spec_from_file_location("train_calibrate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tc = _load_trainer()


def stratified_take(pool, labels, n_take, rng, protect_min=1):
    """Take n_take indices from pool, spread across labels, leaving at least
    `protect_min` of every label behind. Returns (taken, remaining)."""
    by = collections.defaultdict(list)
    for i in pool:
        by[labels[i]].append(i)
    for v in by.values():
        rng.shuffle(v)

    remaining_counts = {k: len(v) for k, v in by.items()}
    quota = {k: min(len(v) - protect_min, int(round(len(v) / len(pool) * n_take)))
             for k, v in by.items()}
    taken = []
    # first pass: proportional quota
    for k, v in by.items():
        q = max(0, quota[k])
        taken.extend(v[:q])
        remaining_counts[k] -= q
    # top up / trim to land exactly on n_take, round-robin over labels that can spare
    keys = sorted(by)
    rng.shuffle(keys)
    i = 0
    while len(taken) < n_take:
        k = keys[i % len(keys)]
        used = len([t for t in taken if labels[t] == k])
        if used < len(by[k]) - protect_min:
            taken.append(by[k][used])
        i += 1
        if i > len(keys) * (max(len(v) for v in by.values()) + 2):
            raise RuntimeError(
                f"cannot take {n_take} from {len(pool)} while leaving "
                f"{protect_min} of each of {len(by)} labels behind")
    # Guard, not a fix. Rounding quotas independently can overshoot n_take, and
    # this function used to truncate silently here, zeroing whole labels. The
    # committed stage-1 runs never overshot (B's quotas summed to 163 of 168, C's
    # to 0 of 22), so the guard leaves their selections - and the committed
    # JSON - unchanged. New code uses src/stratify.py (largest-remainder), which
    # selects different clips, so it is deliberately not swapped in here.
    assert len(taken) == n_take, (
        f"quota overshoot: {len(taken)} taken for n_take={n_take}; "
        "use src/stratify.py")
    left = collections.Counter(labels[i] for i in set(pool) - set(taken))
    assert all(left[k] >= protect_min for k in by), "a label fell below protect_min"
    tset = set(taken)
    return sorted(tset), sorted(set(pool) - tset)


def summarize_condition(name, desc, idx, X, y, gloss, signer, seeds):
    ycal, ytest = y[idx["cal"]], y[idx["test"]]
    rows, logits0 = [], None
    for s in seeds:
        lg, vacc = tc.run_seed(s, X, y, idx)
        T = tc.fit_temperature(lg["cal"], ycal)
        before, after = tc.summarize(lg["test"], ytest, T)
        rows.append(dict(seed=s, T=T, val_acc=vacc, before=before, after=after))
        print(f"  {name} seed {s}: T*={T:.3f}  acc={before['acc']:.3f}  "
              f"ECE {before['ece']:.3f} -> {after['ece']:.3f}", flush=True)
        if logits0 is None:
            logits0 = lg

    agg = {}
    for phase in ("before", "after"):
        for m in ("acc", "ece", "ece_em", "mean_conf", "nll"):
            v = np.array([r[phase][m] for r in rows])
            agg[f"{phase}_{m}"] = [float(v.mean()), float(v.std())]
    agg["T"] = [float(np.mean([r["T"] for r in rows])),
                float(np.std([r["T"] for r in rows]))]

    # bootstrap CIs on seed 0, same procedure as src/05
    T0 = rows[0]["T"]
    p0 = torch.softmax(torch.tensor(logits0["test"]), 1).numpy()
    p1 = torch.softmax(torch.tensor(logits0["test"] / T0), 1).numpy()
    c0, ok0 = p0.max(1), (p0.argmax(1) == ytest).astype(float)
    c1, ok1 = p1.max(1), (p1.argmax(1) == ytest).astype(float)
    rng = np.random.default_rng(0)
    n = len(ytest)
    b = {"acc": [], "ece_before": [], "ece_after": [], "ece_reduction_pct": []}
    for _ in range(2000):
        k = rng.integers(0, n, n)
        e0 = tc.ece_equal_width(c0[k], ok0[k])
        e1 = tc.ece_equal_width(c1[k], ok1[k])
        b["acc"].append(ok0[k].mean())
        b["ece_before"].append(e0)
        b["ece_after"].append(e1)
        b["ece_reduction_pct"].append(100 * (e0 - e1) / e0 if e0 > 0 else np.nan)
    ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
          for k, v in b.items()}

    counts = {}
    for k, ix in idx.items():
        counts[k] = dict(clips=len(ix),
                         glosses=len(set(gloss[ix].tolist())),
                         signers=len(set(signer[ix].tolist())))
    st, sc, sv = (set(signer[idx[k]].tolist()) for k in ("train", "cal", "val"))
    ste = set(signer[idx["test"]].tolist())
    overlap = dict(train_test=len(st & ste), cal_test=len(sc & ste),
                   val_test=len(sv & ste))
    # the mechanism, measured directly: what fraction of the test set comes
    # from a signer the model actually trained on
    seen = np.array([s in st for s in signer[idx["test"]].tolist()])
    overlap["test_clips_from_seen_signer"] = int(seen.sum())
    overlap["test_clips_from_unseen_signer"] = int((~seen).sum())
    overlap["test_frac_seen_signer"] = float(seen.mean())

    return dict(name=name, description=desc, counts=counts,
                signer_overlap=overlap, per_seed=rows, agg=agg, bootstrap_ci=ci)


def main():
    d = np.load(ROOT / "data" / "landmarks.npz", allow_pickle=True)
    X, gloss, signer, vids = d["X"], d["gloss"], d["signer"], d["vids"].tolist()
    sel = json.load(open(ROOT / "data" / "subset.json"))
    assert [r[1] for r in sel] == vids, "subset.json and landmarks.npz are out of order"
    std_split = np.array([r[3] for r in sel])

    # one fixed class vocabulary for every condition, so label ids are identical
    classes = sorted(set(gloss.tolist()))
    cmap = {g: i for i, g in enumerate(classes)}
    y = np.array([cmap[g] for g in gloss.tolist()])
    print(f"pool: {len(y)} clips, {len(classes)} glosses, "
          f"{len(set(signer.tolist()))} signers, device={tc.DEV}")
    print("standard split in pool:", dict(collections.Counter(std_split.tolist())))

    # ---- A: the committed signer-disjoint split -------------------------
    sp = json.load(open(ROOT / "data" / "splits.json"))
    a_split = np.array(sp["split"]); usable = np.array(sp["usable"])
    idxA = {k: np.where((a_split == k) & usable)[0]
            for k in ("train", "val", "cal", "test")}

    # ---- B: WLASL standard split, with a cal set carved out of train ----
    rng = np.random.default_rng(RNG_SEED)
    std_train = np.where(std_split == "train")[0].tolist()
    n_cal = int(round(CAL_SHARE * len(y)))
    cal_idx, trainB = stratified_take(std_train, y, n_cal, rng)
    idxB = {"train": np.array(sorted(trainB)),
            "val": np.where(std_split == "val")[0],
            "cal": np.array(cal_idx),
            "test": np.where(std_split == "test")[0]}

    # ---- C: B with train subsampled down to A's train count -------------
    n_target = len(idxA["train"])
    # select the clips to DROP (not to keep): stratified_take protects every
    # label in the *remaining* set, which is exactly condition C's train set
    n_drop = len(trainB) - n_target
    droppedC, keepC = stratified_take(trainB, y, n_drop, np.random.default_rng(RNG_SEED))
    idxC = dict(idxB)
    idxC["train"] = np.array(sorted(keepC))

    conds = [
        ("A", "signer-disjoint (committed baseline)", idxA),
        ("B", "WLASL standard split, cal carved from train", idxB),
        ("C", f"standard split, train subsampled to {n_target} to match A", idxC),
    ]
    for nm, _, ix in conds:
        print(f"{nm}: " + "  ".join(f"{k}={len(v)}" for k, v in ix.items()))

    results = {}
    for nm, desc, ix in conds:
        assert len(set(gloss[ix["train"]].tolist())) == len(classes), \
            f"{nm}: train does not cover all {len(classes)} classes"
        results[nm] = summarize_condition(nm, desc, ix, X, y, gloss, signer, tc.SEEDS)

    # Could we isolate signer overlap with the test set held fixed? Two designs:
    #   (i)  train on signers disjoint from the standard test set
    #   (ii) split the standard test set by whether its signer was trained on
    # Record whether either is even possible on this pool.
    test_sig = set(signer[idxB["test"]].tolist())
    off_test = [i for i in range(len(y))
                if i not in set(idxB["test"].tolist()) and signer[i] not in test_sig]
    deconfound = dict(
        standard_test_signers=len(test_sig),
        clips_from_signers_outside_standard_test=len(off_test),
        glosses_covered_by_those_clips=len(set(gloss[off_test].tolist())),
        clips_needed_to_match_condition_C=len(idxC["train"]) + len(idxC["cal"]),
        design_i_feasible=len(off_test) >= len(idxC["train"]) + len(idxC["cal"]),
        # design (ii) is judged per condition via
        # conditions[*].signer_overlap.test_clips_from_unseen_signer
    )

    payload = dict(
        meta=dict(
            pool_clips=len(y), n_classes=len(classes),
            seeds=tc.SEEDS, cal_share=CAL_SHARE, rng_seed=RNG_SEED,
            standard_split_in_pool=dict(collections.Counter(std_split.tolist())),
            note=("B and C share signers between cal and test by construction; "
                  "that is the condition under test."),
        ),
        provenance=dict(
            cal_clip_ids=[vids[i] for i in idxB["cal"]],
            condition_C_dropped_clip_ids=[vids[i] for i in droppedC],
            condition_C_n_dropped=len(droppedC),
        ),
        deconfound_feasibility=deconfound,
        conditions=results,
    )
    json.dump(payload, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT.relative_to(ROOT)}")

    print("\n=== stage 1: mean +/- sd over %d seeds ===" % len(tc.SEEDS))
    hdr = f"{'cond':<5}{'train':>6}{'test':>6}{'top-1':>16}{'ECE before':>16}{'ECE after':>15}{'T*':>13}"
    print(hdr)
    for nm in ("A", "B", "C"):
        r = results[nm]; a = r["agg"]
        print(f"{nm:<5}{r['counts']['train']['clips']:>6}{r['counts']['test']['clips']:>6}"
              f"{a['before_acc'][0]:>9.3f} ±{a['before_acc'][1]:<5.3f}"
              f"{a['before_ece'][0]:>9.3f} ±{a['before_ece'][1]:<5.3f}"
              f"{a['after_ece'][0]:>8.3f} ±{a['after_ece'][1]:<5.3f}"
              f"{a['T'][0]:>7.2f} ±{a['T'][1]:<5.2f}")


if __name__ == "__main__":
    main()
