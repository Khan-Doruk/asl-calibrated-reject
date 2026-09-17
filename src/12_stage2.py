"""Stage 2 - input quality. Diagnosis, detection sweep summary, and training.

All training uses the committed signer-disjoint split (data/splits.json), the
committed model and estimators (src/04), and 5 model seeds. Each condition
changes one input variable against a reference condition trained under the
same settings in the same run.

Why the reference is re-trained rather than read from data/results.json: torch
CPU numerics depend on thread count (stage 1b). Training here runs many
single-thread processes in parallel - about 5x the throughput of one process -
so the committed 24-thread numbers do not reproduce bit for bit. Every delta is
therefore taken against base_T24 trained under identical settings; the
committed numbers are recorded beside it for reference only.

Usage
  python src/12_stage2.py --make-subset      # 150-clip diagnostic subset for the 2a sweep
  python src/12_stage2.py --diagnose         # detection diagnosis + sweep table (no training)
  python src/12_stage2.py --train base_T24,T32,T48
  python src/12_stage2.py --train clip_scale,velocity,mirror

Merges into data/stage2_input_quality.json; conditions already present are
skipped unless --force.
"""
import argparse, collections, importlib.util, json, os, pathlib, sys, tempfile, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import features  # noqa: E402

OUT = ROOT / "data" / "stage2_input_quality.json"
SUBSET = ROOT / "data" / "stage2_diag_subset.json"
VIDEO_DIR = pathlib.Path(os.environ.get(
    "WLASL_VIDEO_DIR", pathlib.Path.home() / ".cache" / "wlasl_clips"))
WORKERS = 16
THREADS_PER_WORKER = 1
SEEDS = [0, 1, 2, 3, 4]
SWEEP = [  # (tag, description) - one variable changed from sub_base
    ("sub_base", "src/02 settings"),
    ("sub_pad10", "bbox padded 10% per side"),
    ("sub_pad20", "bbox padded 20% per side"),
    ("sub_conf01", "hand confidence 0.3 -> 0.1"),
    ("sub_res640", "resize 512 -> 640"),
    ("sub_posefull", "pose lite -> full"),
]
POSE_TOLERANCE = 0.01     # a detection variant may not cost more pose rate than this
MIN_HAND_GAIN = 0.01      # nor count as better without at least this much hand rate
SMALL_SHARE = 0.25


# ------------------------------------------------------------------ io
def load_out():
    return json.load(open(OUT)) if OUT.exists() else dict(conditions={})


def save_out(d):
    json.dump(d, open(OUT, "w"), indent=2)


def committed_order():
    return np.load(ROOT / "data" / "landmarks.npz", allow_pickle=True)["vids"].tolist()


# ------------------------------------------------------------------ 2a subset + diagnosis
def make_subset():
    d = np.load(ROOT / "data" / "landmarks.npz", allow_pickle=True)
    X, vids = d["X"], d["vids"].tolist()
    rate = ((X[:, :, 75 + 66] > 0) | (X[:, :, 75 + 67 + 66] > 0)).mean(1)
    order = np.argsort(rate, kind="stable")
    worst = [vids[i] for i in order[:60]]
    rest = [v for v in vids if v not in set(worst)]
    rng = np.random.default_rng(0)
    rand = [rest[i] for i in sorted(rng.choice(len(rest), 90, replace=False))]
    json.dump(dict(worst60=worst, random90=rand, rule=(
        "60 clips with the lowest committed hand-detection rate, plus 90 drawn at random "
        "(seed 0) from the rest")), open(SUBSET, "w"), indent=2)
    print(f"wrote {SUBSET.relative_to(ROOT)}; worst-60 max rate {rate[order[59]]:.3f}")
    json.dump(worst + rand, open(ROOT / "data" / "stage2_diag_subset_ids.json", "w"))


def diagnose():
    import cv2
    out = load_out()
    base = json.load(open(ROOT / "data" / "extract_base.json"))["per_T"]["24"]
    ann = json.load(open(ROOT / "data" / "WLASL_v0.3.json"))
    meta = {i["video_id"]: i for g in ann for i in g["instances"]}
    per_clip = base["per_clip_hand_rate"]

    by_src = collections.defaultdict(list)
    for v, r in per_clip.items():
        by_src[meta[v]["source"]].append(r)

    bottom, height, beyond = [], [], []
    for v, r in per_clip.items():
        cap = cv2.VideoCapture(str(VIDEO_DIR / f"{v}.mp4"))
        W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if W <= 0 or H <= 0:
            raise SystemExit(f"cannot read {v} from {VIDEO_DIR}; set WLASL_VIDEO_DIR")
        x1, y1, x2, y2 = meta[v]["bbox"]
        bottom.append((y2 >= H - 2, r))
        height.append(((y2 - y1) / H, r))
        if max(x2 / W, y2 / H) > 1.05:      # annotation does not fit this video's resolution
            beyond.append((v, r))
    z = np.load(ROOT / "data" / "landmarks_base.npz", allow_pickle=True)
    rep, dec = z["n_reported"], z["n_decoded"]
    hq = np.quantile([h for h, _ in height], [0, .25, .5, .75, 1])
    pos = base["hand_rate_by_position"]
    edge = max(1, len(pos) // 6)

    out["diagnosis"] = dict(
        overall_hand_rate_T24=base["hand_rate"],
        hand_rate_by_position_T24=pos,
        hand_rate_first_sixth=float(np.mean(pos[:edge])),
        hand_rate_middle=float(np.mean(pos[edge:-edge])),
        hand_rate_last_sixth=float(np.mean(pos[-edge:])),
        by_source={s: dict(clips=len(v), hand_rate=float(np.mean(v)))
                   for s, v in sorted(by_src.items(), key=lambda kv: np.mean(kv[1]))},
        bbox_bottom_at_frame_edge=dict(
            clips=sum(1 for b, _ in bottom if b),
            hand_rate=float(np.mean([r for b, r in bottom if b])) if any(b for b, _ in bottom) else None,
            other_clips=sum(1 for b, _ in bottom if not b),
            other_hand_rate=float(np.mean([r for b, r in bottom if not b])) if any(not b for b, _ in bottom) else None),
        bbox_height_quartiles=[dict(lo=float(lo), hi=float(hi), clips=int(sum(1 for h, _ in height if lo <= h <= hi)),
                                    hand_rate=float(np.mean([r for h, r in height if lo <= h <= hi])))
                               for lo, hi in zip(hq[:-1], hq[1:])],
        bbox_beyond_frame=dict(clips=len(beyond), clip_ids=[v for v, _ in beyond],
                               hand_rates=[float(r) for _, r in beyond]),
        # container frame counts overstate what decodes, so src/02's last sampled
        # indices do not exist and those clips are end-padded even at T=24
        frame_count_metadata=dict(
            clips_decoding_fewer_than_reported=int((dec < rep).sum()),
            median_shortfall=float(np.median((rep - dec)[dec < rep])) if (dec < rep).any() else 0.0,
            max_shortfall=int((rep - dec).max()),
            clips_decoding_more=int((dec > rep).sum()),
            reported_frames=dict(min=int(rep.min()), median=float(np.median(rep)), max=int(rep.max())),
            clips_reporting_fewer_than=dict({str(T): int((rep < T).sum()) for T in (24, 32, 48)})),
    )

    keep = ("hand_rate", "pose_rate", "both_hands_rate", "clips_zero_hands", "clips_padded",
            "padded_frames_total", "max_pad_fraction", "mean_pad_fraction_padded_clips")
    ext = {}
    for tag in ("base", "det_best"):
        jp = ROOT / "data" / f"extract_{tag}.json"
        if jp.exists():
            j = json.load(open(jp))
            ext[tag] = dict(settings=j["settings"], wall_time_s=j["wall_time_s"], workers=j["workers"],
                            per_T={T: {k: v[k] for k in keep} for T, v in j["per_T"].items()})
    out["extraction"] = ext

    sub = json.load(open(SUBSET))
    groups = dict(worst60=sub["worst60"], random90=sub["random90"])
    n_sub = len(sub["worst60"]) + len(sub["random90"])
    rows = {}
    for tag, desc in SWEEP:
        jp = ROOT / "data" / f"extract_{tag}.json"
        if not jp.exists():
            continue
        j = json.load(open(jp))
        z = np.load(ROOT / "data" / f"landmarks_{tag}.npz", allow_pickle=True)
        vids = z["vids"].tolist()
        pr = dict(zip(vids, z["pose_ok_T24"].mean(1)))
        hr = j["per_T"]["24"]["per_clip_hand_rate"]
        rows[tag] = dict(
            description=desc, settings=j["settings"],
            hand_rate=j["per_T"]["24"]["hand_rate"], pose_rate=j["per_T"]["24"]["pose_rate"],
            both_hands_rate=j["per_T"]["24"]["both_hands_rate"],
            zero_hand_clips=j["per_T"]["24"]["clips_zero_hands"],
            hand_rate_first_sixth=float(np.mean(j["per_T"]["24"]["hand_rate_by_position"][:4])),
            hand_rate_middle=float(np.mean(j["per_T"]["24"]["hand_rate_by_position"][4:-4])),
            hand_rate_last_sixth=float(np.mean(j["per_T"]["24"]["hand_rate_by_position"][-4:])),
            **{f"hand_rate_{g}": float(np.mean([hr[v] for v in ids])) for g, ids in groups.items()},
            **{f"pose_rate_{g}": float(np.mean([pr[v] for v in ids])) for g, ids in groups.items()},
            wall_time_s=j["wall_time_s"], workers=j["workers"],
            projected_full_wall_time_s=j["wall_time_s"] * 1120 / n_sub)
    if "sub_base" in rows:
        b = rows["sub_base"]
        ok = {t: r for t, r in rows.items() if t != "sub_base"
              and r["pose_rate"] >= b["pose_rate"] - POSE_TOLERANCE
              and r["hand_rate"] >= b["hand_rate"] + MIN_HAND_GAIN}
        best = max(ok, key=lambda t: ok[t]["hand_rate"]) if ok else None
        out["sweep"] = dict(subset_clips=n_sub, variants=rows, best=best, rule=(
            f"highest subset hand rate among variants gaining >= {MIN_HAND_GAIN} hand rate over "
            f"sub_base without losing more than {POSE_TOLERANCE} pose rate"))
        print("best detection variant:", best)
    save_out(out)
    print("diagnosis written")


# ------------------------------------------------------------------ training
def _load_tc():
    spec = importlib.util.spec_from_file_location("train_calibrate", HERE / "04_train_calibrate.py")
    tc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tc)
    return tc


def _job(args):
    cond, seed, path = args
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)
    tc = _load_tc()
    z = np.load(path)
    X, y = z["X"], z["y"]
    idx = {k: z[f"idx_{k}"] for k in ("train", "val", "cal", "test")}
    t0 = time.time()
    lg, vacc = tc.run_seed(seed, X, y, idx)
    T = tc.fit_temperature(lg["cal"], y[idx["cal"]])
    before, after = tc.summarize(lg["test"], y[idx["test"]], T)
    return cond, seed, dict(seed=seed, T=T, val_acc=vacc, before=before, after=after,
                            seconds=time.time() - t0)


def condition_specs(out):
    """Conditions, resolved against earlier decisions (best detection variant, best T)."""
    specs = {
        "base_T24": dict(file="landmarks_base.npz", T=24, scheme="baseline", group="reference",
                         ref=None, description="src/02 settings, T=24 (stage 2 reference)"),
        "T32": dict(file="landmarks_base.npz", T=32, scheme="baseline", group="2b", ref="base_T24",
                    description="T=32, nothing else changed"),
        "T48": dict(file="landmarks_base.npz", T=48, scheme="baseline", group="2b", ref="base_T24",
                    description="T=48, nothing else changed"),
    }
    best = out.get("sweep", {}).get("best")
    if best:
        specs["det_best_T24"] = dict(file="landmarks_det_best.npz", T=24, scheme="baseline", group="2a",
                                     ref="base_T24", description=f"detection variant {best}, T=24")
    bestT = out.get("decisions", {}).get("best_T")
    if bestT:
        ref = {24: "base_T24", 32: "T32", 48: "T48"}[bestT]
        for scheme, desc in (("clip_scale", "per-clip median shoulder-width scale"),
                             ("velocity", "adds first-difference velocity features"),
                             ("mirror", "mirrors left-dominant clips")):
            specs[scheme] = dict(file="landmarks_base.npz", T=bestT, scheme=scheme, group="2c",
                                 ref=ref, description=f"{desc}, T={bestT}")
    return specs


def train(names, force):
    out = load_out()
    specs = condition_specs(out)
    unknown = [n for n in names if n not in specs]
    if unknown:
        raise SystemExit(f"unknown or not-yet-available conditions: {unknown} "
                         f"(available: {sorted(specs)})")
    todo = [n for n in names if force or n not in out["conditions"]]
    if not todo:
        print("nothing to do"); return

    sp = json.load(open(ROOT / "data" / "splits.json"))
    split, usable = np.array(sp["split"]), np.array(sp["usable"])
    order = committed_order()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="stage2_"))
    jobs, infos = [], {}
    for n in todo:
        s = specs[n]
        z = np.load(ROOT / "data" / s["file"], allow_pickle=True)
        assert z["vids"].tolist() == order, f"{s['file']} is not in committed clip order"
        fn = features.SCHEMES[s["scheme"]]
        X, info = fn(z, s["T"], z["signer"]) if s["scheme"] == "mirror" else fn(z, s["T"])
        if s["scheme"] == "baseline":
            assert np.array_equal(X, z[f"X_T{s['T']}"]), "baseline recompute drifted from extraction"
        gloss = z["gloss"]
        cmap = {g: i for i, g in enumerate(sorted(set(gloss.tolist())))}
        y = np.array([cmap[g] for g in gloss.tolist()])
        idx = {k: np.where((split == k) & usable)[0] for k in ("train", "val", "cal", "test")}
        path = tmp / f"{n}.npz"
        np.savez(path, X=X.astype(np.float32), y=y, **{f"idx_{k}": v for k, v in idx.items()})
        infos[n] = dict(info, feature_dim=int(X.shape[2]))
        jobs += [(n, seed, str(path)) for seed in SEEDS]

    print(f"training {len(todo)} conditions x {len(SEEDS)} seeds = {len(jobs)} runs "
          f"on {WORKERS} workers", flush=True)
    t0 = time.time()
    rows = collections.defaultdict(dict)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for k, (n, seed, row) in enumerate(ex.map(_job, jobs), 1):
            rows[n][seed] = row
            print(f"  [{k}/{len(jobs)} {time.time()-t0:.0f}s] {n} seed {seed}: "
                  f"acc {row['before']['acc']:.3f} ECE {row['before']['ece']:.3f}->{row['after']['ece']:.3f}",
                  flush=True)

    for n in todo:
        per_seed = [rows[n][s] for s in SEEDS]
        agg = {}
        for phase in ("before", "after"):
            for m in ("acc", "ece", "ece_em", "mean_conf", "nll"):
                v = np.array([r[phase][m] for r in per_seed])
                agg[f"{phase}_{m}"] = [float(v.mean()), float(v.std())]
        agg["T"] = [float(np.mean([r["T"] for r in per_seed])), float(np.std([r["T"] for r in per_seed]))]
        out["conditions"][n] = dict(spec=specs[n], feature_info=infos[n], per_seed=per_seed, agg=agg,
                                    batch_wall_time_s=time.time() - t0)
    out["meta"] = dict(split="committed signer-disjoint (data/splits.json)", seeds=SEEDS,
                       workers=WORKERS, threads_per_worker=THREADS_PER_WORKER,
                       committed_reference=json.load(open(ROOT / "data" / "results.json"))["agg"],
                       note=("Deltas are against a reference trained in this stage under the same "
                             "thread settings; committed numbers differ slightly by thread count."))
    add_deltas(out)
    save_out(out)
    print(f"batch done in {(time.time()-t0)/60:.1f} min")


def add_deltas(out):
    C = out["conditions"]
    for n, c in C.items():
        ref = c["spec"].get("ref")
        if not ref or ref not in C:
            continue
        r = C[ref]
        d = {}
        for m in ("acc", "ece", "ece_em"):
            for phase in ("before", "after"):
                key = f"{phase}_{m}"
                per = [a[phase][m] - b[phase][m] for a, b in zip(c["per_seed"], r["per_seed"])]
                d[key] = dict(mean=float(np.mean(per)), min=float(min(per)), max=float(max(per)),
                              same_sign_all_seeds=bool(all(x > 0 for x in per) or all(x < 0 for x in per)))
        c["delta_vs_ref"] = dict(ref=ref, **d)
    # decisions that later conditions depend on
    Tc = {24: "base_T24", 32: "T32", 48: "T48"}
    if all(v in C for v in Tc.values()):
        accs = {T: C[n]["agg"]["before_acc"][0] for T, n in Tc.items()}
        best = max(accs, key=lambda T: (round(accs[T], 6), -T))
        out.setdefault("decisions", {})["best_T"] = best
        out["decisions"]["best_T_rule"] = "highest mean top-1 over seeds; ties go to the smaller T"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-subset", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--train", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.make_subset:
        make_subset()
    if a.diagnose:
        diagnose()
    if a.train:
        train([x for x in a.train.split(",") if x], a.force)


if __name__ == "__main__":
    main()
