"""Extract MediaPipe hand + pose landmark sequences for the WLASL-100 subset.

MediaPipe 1.0 removed the legacy mp.solutions API, so this uses the Tasks API
(HandLandmarker + PoseLandmarker) in stateless IMAGE mode. VIDEO mode is not
usable here: Holistic's segmentation-smoothing buffer requires every frame in a
stream to share a resolution, and WLASL clips do not.

Features per frame (209 dims), built to be roughly signer-invariant:
  - 25 upper-body pose points (x,y,z), origin at mid-shoulder, scaled by
    shoulder width                                              -> 75
  - per hand: 21 points (x,y,z) relative to that hand's own wrist, same scale,
    + wrist position in signing space (3) + presence flag (1)   -> 67 x 2
"""
import json, os, sys, pathlib, time, tempfile, shutil
import numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIDEO_DIR = pathlib.Path(os.environ.get(
    "WLASL_VIDEO_DIR", pathlib.Path.home() / ".cache" / "wlasl_clips"))
# MediaPipe's C++ model loader cannot open paths with non-ASCII characters
# (this project lives under "Masaustu"), so models are staged to an ASCII path.
MODEL_DIR = pathlib.Path(tempfile.gettempdir()) / "aslcal_models"
T = 24               # frames sampled per clip
RES = 512            # square resize fed to MediaPipe
POSE_KEEP = list(range(25))   # upper body: face, shoulders, elbows, wrists, hips
FEAT_DIM = len(POSE_KEEP) * 3 + 2 * (21 * 3 + 3 + 1)

MODEL_URLS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def ensure_models():
    """Download the MediaPipe bundles if absent, then stage them to an ASCII path.

    The bundles are Google's (Apache-2.0) and are not redistributed in this repo,
    so a fresh clone fetches them here (~14 MB, once). Staging to an ASCII path is
    required because MediaPipe's C++ loader cannot open non-ASCII paths.
    """
    import urllib.request
    root_models = ROOT / "models"
    root_models.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        src = root_models / name
        if not src.exists() or src.stat().st_size < 100_000:
            print("downloading %s ..." % name, flush=True)
            urllib.request.urlretrieve(url, src)
        if not (MODEL_DIR / name).exists():
            shutil.copy(src, MODEL_DIR / name)
    print("models ready at", MODEL_DIR, flush=True)


_hl = _pl = None

def _init():
    global _hl, _pl
    import mediapipe as mp
    from mediapipe.tasks.python import vision, BaseOptions
    _hl = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "hand_landmarker.task")),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    _pl = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_DIR / "pose_landmarker_lite.task")),
        running_mode=vision.RunningMode.IMAGE, output_segmentation_masks=False))

def _frame_feats(img):
    """img: mp.Image. Returns (feat[209], pose_found, n_hands)."""
    pres = _pl.detect(img)
    hres = _hl.detect(img)
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not pres.pose_landmarks:
        return feat, False, len(hres.hand_landmarks or [])
    P = np.array([[p.x, p.y, p.z] for p in pres.pose_landmarks[0]], dtype=np.float32)
    ls, rs = P[11], P[12]
    origin = (ls + rs) / 2.0
    scale = float(np.linalg.norm((ls - rs)[:2])) or 1e-3
    feat[:len(POSE_KEEP) * 3] = ((P[POSE_KEEP] - origin) / scale).ravel()
    off = len(POSE_KEEP) * 3
    # slot 0 = handedness "Left", slot 1 = "Right" (consistent across clips)
    for hand, hd in zip(hres.hand_landmarks or [], hres.handedness or []):
        slot = 0 if hd[0].category_name == "Left" else 1
        H = np.array([[p.x, p.y, p.z] for p in hand], dtype=np.float32)
        wrist = H[0]
        base = off + slot * 67
        feat[base:base + 63] = ((H - wrist) / scale).ravel()
        feat[base + 63:base + 66] = (wrist - origin) / scale
        feat[base + 66] = 1.0
    return feat, True, len(hres.hand_landmarks or [])

def process(rec):
    """Stream the clip, keeping only the T sampled frames (already cropped and
    resized). Buffering every full-resolution frame blows up memory on 1080p
    clips and takes the worker pool down."""
    gloss, vid, signer, split, bbox = rec
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{vid}.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < 4:
        cap.release()
        return vid, None, 0.0, 0.0
    want = set(np.linspace(0, n - 1, T).round().astype(int).tolist())
    x1, y1, x2, y2 = bbox
    kept, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in want:
            c = fr[max(0, y1):y2, max(0, x1):x2]
            kept.append(cv2.resize(c if c.size else fr, (RES, RES)))
        i += 1
    cap.release()
    if len(kept) < 4:
        return vid, None, 0.0, 0.0
    while len(kept) < T:            # short/truncated clip: pad with last frame
        kept.append(kept[-1])
    kept = kept[:T]

    import mediapipe as mp
    seq = np.zeros((T, FEAT_DIM), dtype=np.float32)
    pose_hits = hand_hits = 0
    for k, f in enumerate(kept):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        feat, pok, nh = _frame_feats(img)
        seq[k] = feat
        pose_hits += int(pok); hand_hits += int(nh > 0)
    return vid, seq, pose_hits / T, hand_hits / T

def main():
    ensure_models()
    annot = ROOT / "data" / "WLASL_v0.3.json"
    if not annot.exists() or not (ROOT / "data" / "subset.json").exists():
        raise SystemExit("Run src/01_select_and_download.py first - it fetches the "
                         "WLASL annotations and the clip subset this step needs.")
    meta = {i["video_id"]: i for g in json.load(open(ROOT / "data" / "WLASL_v0.3.json"))
            for i in g["instances"]}
    sel = json.load(open(ROOT / "data" / "subset.json"))
    recs = [(r[0], r[1], r[2], r[3], meta[r[1]]["bbox"]) for r in sel]
    t0 = time.time()
    out, stats = {}, []
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        for n, (vid, seq, pr, hr) in enumerate(ex.map(process, recs, chunksize=4), 1):
            if seq is not None:
                out[vid] = seq
                stats.append((pr, hr))
            if n % 100 == 0:
                print(f"{n}/{len(recs)}  {time.time()-t0:.0f}s", flush=True)
    keys = list(out)
    X = np.stack([out[k] for k in keys])
    by = {r[1]: r for r in sel}
    np.savez_compressed(ROOT / "data" / "landmarks.npz", X=X,
                        vids=np.array(keys),
                        gloss=np.array([by[k][0] for k in keys]),
                        signer=np.array([by[k][2] for k in keys]))
    s = np.array(stats)
    json.dump(dict(n_extracted=len(keys), n_requested=len(recs),
                   pose_rate=float(s[:, 0].mean()),
                   hand_rate=float(s[:, 1].mean()),
                   clips_zero_hands=int((s[:, 1] == 0).sum()),
                   frames_per_clip=T),
              open(ROOT / "data" / "extraction_stats.json", "w"), indent=2)
    print(f"\nclips extracted: {len(keys)}/{len(recs)}")
    print(f"mean per-clip pose-detection rate: {s[:,0].mean():.3f}")
    print(f"mean per-clip >=1-hand rate:       {s[:,1].mean():.3f}")
    print(f"clips with zero hand frames:       {(s[:,1]==0).sum()}")
    print(f"elapsed {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
