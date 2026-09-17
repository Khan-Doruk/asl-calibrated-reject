"""Stage 2 extraction: src/02 with its settings exposed, several T per pass, raw
coordinates kept.

src/02 and data/landmarks.npz are left untouched; they are the committed
baseline. This script reproduces src/02 exactly at its default settings
(checked against data/landmarks.npz by src/12_features.py), and additionally:

  * exposes the input-quality knobs stage 2a varies:
      --pad        grow the WLASL bbox by this fraction of its width/height on
                   every side, clamped to the frame (0 = src/02)
      --hand-conf  min_hand_detection_confidence and presence confidence (0.3)
      --res        square resize fed to MediaPipe (512)
      --pose       pose bundle, lite or full (lite)
  * samples several T in one pass. Frame indices are the union of every T's
    linspace; each unique frame is decoded and run through MediaPipe once, then
    each T's sequence is assembled exactly as src/02 would have built it -
    including its short-clip behaviour: indices are a set, so a clip with fewer
    frames than T yields fewer than T frames and is END-padded with its last
    frame. That behaviour is recorded per clip, not fixed, because stage 2b
    measures it.
  * saves raw MediaPipe coordinates, so normalization variants (stage 2c) are
    post-processing rather than re-extraction.

Stateless IMAGE mode is kept on purpose: see the comment at the top of src/02.

Writes data/landmarks_<tag>.npz (gitignored: WLASL-derived) and
data/extract_<tag>.json (settings and detection statistics only).
"""
import argparse, json, os, pathlib, shutil, sys, tempfile, time, urllib.request
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIDEO_DIR = pathlib.Path(os.environ.get(
    "WLASL_VIDEO_DIR", pathlib.Path.home() / ".cache" / "wlasl_clips"))
MODEL_DIR = pathlib.Path(tempfile.gettempdir()) / "aslcal_models"   # ASCII path; see src/02
POSE_KEEP = list(range(25))
FEAT_DIM = len(POSE_KEEP) * 3 + 2 * (21 * 3 + 3 + 1)
MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def ensure_models(names):
    (ROOT / "models").mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        src = ROOT / "models" / name
        if not src.exists() or src.stat().st_size < 100_000:
            print(f"downloading {name} ...", flush=True)
            urllib.request.urlretrieve(MODEL_URLS[name], src)
        if not (MODEL_DIR / name).exists():
            shutil.copy(src, MODEL_DIR / name)


_hl = _pl = _cfg = None


def _init(cfg):
    global _hl, _pl, _cfg
    _cfg = cfg
    from mediapipe.tasks.python import vision, BaseOptions
    _hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=cfg["hand_conf"],
        min_hand_presence_confidence=cfg["hand_conf"]))
    _pl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_DIR / f"pose_landmarker_{cfg['pose']}.task")),
        running_mode=vision.RunningMode.IMAGE, output_segmentation_masks=False))


def features_from_raw(pose, pose_ok, hand, hand_ok):
    """src/02's 209-dim features, from raw coordinates. Same float32 ops in the
    same order as src/02._frame_feats, so the result is bit-identical."""
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not pose_ok:
        return feat
    P = pose
    ls, rs = P[11], P[12]
    origin = (ls + rs) / 2.0
    scale = float(np.linalg.norm((ls - rs)[:2])) or 1e-3
    feat[:len(POSE_KEEP) * 3] = ((P[POSE_KEEP] - origin) / scale).ravel()
    off = len(POSE_KEEP) * 3
    for slot in (0, 1):
        if not hand_ok[slot]:
            continue
        H = hand[slot]
        wrist = H[0]
        base = off + slot * 67
        feat[base:base + 63] = ((H - wrist) / scale).ravel()
        feat[base + 63:base + 66] = (wrist - origin) / scale
        feat[base + 66] = 1.0
    return feat


def _detect(img):
    """Raw landmarks for one frame. Hand slots follow src/02: slot 0 = 'Left',
    slot 1 = 'Right', and a second hand with the same label overwrites the first."""
    pres = _pl.detect(img)
    hres = _hl.detect(img)
    pose = np.zeros((33, 3), np.float32)
    pose_ok = bool(pres.pose_landmarks)
    if pose_ok:
        pose = np.array([[p.x, p.y, p.z] for p in pres.pose_landmarks[0]], dtype=np.float32)
    hand = np.zeros((2, 21, 3), np.float32)
    hand_ok = np.zeros(2, bool)
    for h, hd in zip(hres.hand_landmarks or [], hres.handedness or []):
        slot = 0 if hd[0].category_name == "Left" else 1
        hand[slot] = np.array([[p.x, p.y, p.z] for p in h], dtype=np.float32)
        hand_ok[slot] = True
    return pose, pose_ok, hand, hand_ok, len(hres.hand_landmarks or [])


def process(rec):
    vid, bbox = rec
    Ts, pad, res = _cfg["Ts"], _cfg["pad"], _cfg["res"]
    import mediapipe as mp
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{vid}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, Hh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if n < 4:
        cap.release()
        return vid, None
    want = {T: set(np.linspace(0, n - 1, T).round().astype(int).tolist()) for T in Ts}
    union = set().union(*want.values())
    x1, y1, x2, y2 = bbox
    if pad:
        bw, bh = x2 - x1, y2 - y1
        x1, x2 = int(round(x1 - pad * bw)), int(round(x2 + pad * bw))
        y1, y2 = int(round(y1 - pad * bh)), int(round(y2 + pad * bh))
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(Hh, y2)
    frames, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in union:
            c = fr[max(0, y1):y2, max(0, x1):x2]
            f = cv2.resize(c if c.size else fr, (res, res))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            frames[i] = _detect(img)
        i += 1
    cap.release()

    out = dict(n_reported=n, n_decoded=i, frame_wh=(W, Hh))
    for T in Ts:
        kept = [j for j in sorted(want[T]) if j in frames]
        if len(kept) < 4:
            return vid, None
        n_real = len(kept)
        while len(kept) < T:
            kept.append(kept[-1])
        kept = kept[:T]
        pose = np.stack([frames[j][0] for j in kept])
        pose_ok = np.array([frames[j][1] for j in kept])
        hand = np.stack([frames[j][2] for j in kept])
        hand_ok = np.stack([frames[j][3] for j in kept])
        nh = np.array([frames[j][4] for j in kept])
        X = np.stack([features_from_raw(pose[k], pose_ok[k], hand[k], hand_ok[k]) for k in range(T)])
        out[T] = dict(X=X, pose=pose, pose_ok=pose_ok, hand=hand, hand_ok=hand_ok,
                      n_hands=nh, n_real=n_real)
    return vid, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pad", type=float, default=0.0)
    ap.add_argument("--hand-conf", type=float, default=0.3)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--pose", choices=("lite", "full"), default="lite")
    ap.add_argument("--T", default="24", help="comma-separated, e.g. 24,32,48")
    ap.add_argument("--clips", default="all", help="'all' or a JSON file listing video ids")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    if args.tag in ("", "baseline_committed") or "/" in args.tag:
        raise SystemExit("choose a plain tag; the committed data/landmarks.npz is never written here")
    out_npz = ROOT / "data" / f"landmarks_{args.tag}.npz"
    out_json = ROOT / "data" / f"extract_{args.tag}.json"
    assert out_npz.name != "landmarks.npz"

    Ts = sorted({int(t) for t in args.T.split(",")})
    ensure_models(["hand_landmarker.task", f"pose_landmarker_{args.pose}.task"])
    meta = {i["video_id"]: i for g in json.load(open(ROOT / "data" / "WLASL_v0.3.json"))
            for i in g["instances"]}
    sel = json.load(open(ROOT / "data" / "subset.json"))
    if args.clips != "all":
        keep = set(json.load(open(args.clips)))
        sel = [r for r in sel if r[1] in keep]
    missing = [r[1] for r in sel if not (VIDEO_DIR / f"{r[1]}.mp4").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} clips missing from {VIDEO_DIR} - run src/01 first")

    cfg = dict(Ts=Ts, pad=args.pad, res=args.res, hand_conf=args.hand_conf, pose=args.pose)
    recs = [(r[1], meta[r[1]]["bbox"]) for r in sel]
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(cfg,)) as ex:
        for k, (vid, o) in enumerate(ex.map(process, recs, chunksize=2), 1):
            if o is not None:
                results[vid] = o
            if k % 100 == 0:
                print(f"  {k}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    wall = time.time() - t0

    by = {r[1]: r for r in sel}
    vids = [r[1] for r in sel if r[1] in results]
    arrays = dict(vids=np.array(vids), gloss=np.array([by[v][0] for v in vids]),
                  signer=np.array([by[v][2] for v in vids]),
                  n_reported=np.array([results[v]["n_reported"] for v in vids]),
                  n_decoded=np.array([results[v]["n_decoded"] for v in vids]))
    stats = dict(settings=dict(pad=args.pad, hand_conf=args.hand_conf, res=args.res, pose=args.pose,
                               Ts=Ts, clips=args.clips, pad_definition=(
                                   "each side grown by pad x box width/height, clamped to frame")),
                 n_requested=len(sel), n_extracted=len(vids), wall_time_s=wall,
                 workers=args.workers, frames_reported=dict(
                     min=int(arrays["n_reported"].min()), median=float(np.median(arrays["n_reported"])),
                     max=int(arrays["n_reported"].max())),
                 per_T={})
    for T in Ts:
        for key in ("X", "pose", "pose_ok", "hand", "hand_ok", "n_hands", "n_real"):
            arrays[f"{key}_T{T}"] = np.stack([results[v][T][key] for v in vids])
        pose_ok = arrays[f"pose_ok_T{T}"]; nh = arrays[f"n_hands_T{T}"]
        n_real = arrays[f"n_real_T{T}"]
        any_h = nh > 0
        clip_hand = any_h.mean(1)
        stats["per_T"][str(T)] = dict(
            # src/02's definitions: fraction of the T sampled frames, pads included
            pose_rate=float(pose_ok.mean(1).mean()),
            hand_rate=float(clip_hand.mean()),
            both_hands_rate=float((nh >= 2).mean(1).mean()),
            clips_zero_hands=int((clip_hand == 0).sum()),
            hand_rate_by_position=[float(v) for v in any_h.mean(0)],
            # short clips: how many sequences are end-padded, and by how much
            clips_padded=int((n_real < T).sum()),
            padded_frames_total=int((T - n_real).clip(min=0).sum()),
            max_pad_fraction=float(((T - n_real).clip(min=0) / T).max()),
            mean_pad_fraction_padded_clips=float(
                ((T - n_real) / T)[n_real < T].mean()) if (n_real < T).any() else 0.0,
            per_clip_hand_rate={v: float(r) for v, r in zip(vids, clip_hand)},
        )
    np.savez_compressed(out_npz, **arrays)
    json.dump(stats, open(out_json, "w"), indent=2)
    print(f"wrote {out_npz.relative_to(ROOT)} and {out_json.relative_to(ROOT)} in {wall:.0f}s")
    for T in Ts:
        s = stats["per_T"][str(T)]
        print(f"  T={T}: hand {s['hand_rate']:.3f}  pose {s['pose_rate']:.3f}  zero-hand clips "
              f"{s['clips_zero_hands']}  padded clips {s['clips_padded']}")


if __name__ == "__main__":
    main()
