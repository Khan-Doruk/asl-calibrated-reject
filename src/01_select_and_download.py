"""Select the WLASL-100 subset and download just those clips from the
ungated Voxel51/WLASL HuggingFace mirror. Videos land outside the project
folder (VIDEO_DIR) so OneDrive doesn't try to sync ~1k mp4s."""
import json, os, sys, collections, pathlib
from concurrent.futures import ThreadPoolExecutor
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
VIDEO_DIR = pathlib.Path(os.environ.get(
    "WLASL_VIDEO_DIR", pathlib.Path.home() / ".cache" / "wlasl_clips"))
BASE = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/"
N_CLASSES = 100

rows = json.load(open(ROOT / "data" / "matched.json"))  # [gloss, vid, signer, split, path]
cnt = collections.Counter(r[0] for r in rows)
top = [g for g, _ in cnt.most_common(N_CLASSES)]
sel = [r for r in rows if r[0] in top]
json.dump(sel, open(ROOT / "data" / "subset.json", "w"))
print(f"selected {len(sel)} clips over {len(top)} glosses", flush=True)

VIDEO_DIR.mkdir(parents=True, exist_ok=True)

def get(r):
    gloss, vid, signer, split, path = r
    dest = VIDEO_DIR / f"{vid}.mp4"
    if dest.exists() and dest.stat().st_size > 1000:
        return ("cached", vid)
    try:
        urllib.request.urlretrieve(BASE + path, dest)
        return ("ok", vid)
    except Exception as e:
        return (f"FAIL {type(e).__name__}", vid)

with ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(get, sel))

tally = collections.Counter(s for s, _ in res)
print(dict(tally), flush=True)
fails = [v for s, v in res if s.startswith("FAIL")]
json.dump(fails, open(ROOT / "data" / "download_failures.json", "w"))
print(f"failures: {len(fails)}", flush=True)
