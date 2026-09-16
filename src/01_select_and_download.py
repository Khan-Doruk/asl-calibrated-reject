"""Bootstrap the WLASL-100 subset from nothing, then download just those clips.

Runs end-to-end on a fresh clone:
  1. fetch the upstream WLASL annotations (WLASL_v0.3.json, ~12 MB)
  2. list what the ungated Voxel51/WLASL HuggingFace mirror actually hosts
  3. match annotations to available clips
  4. keep the N_CLASSES most-represented glosses
  5. download only those clips

Nothing in steps 1-3 is committed to the repo: the annotations are WLASL
content under C-UDA (see DATA.md), so they are re-fetched rather than
redistributed. Clips are written to VIDEO_DIR, which is outside the repo.
"""
import json, os, collections, pathlib
from concurrent.futures import ThreadPoolExecutor
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
VIDEO_DIR = pathlib.Path(os.environ.get(
    "WLASL_VIDEO_DIR", pathlib.Path.home() / ".cache" / "wlasl_clips"))

ANNOT_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
HF_API = "https://huggingface.co/api/datasets/Voxel51/WLASL"
BASE = "https://huggingface.co/datasets/Voxel51/WLASL/resolve/main/"
N_CLASSES = 100


def fetch_json(url, dest=None):
    """Fetch JSON, caching to dest if given."""
    if dest and dest.exists() and dest.stat().st_size > 1000:
        return json.load(open(dest, encoding="utf-8"))
    print(f"fetching {url.split('/')[-1] or url} ...", flush=True)
    with urllib.request.urlopen(url) as r:
        payload = json.loads(r.read().decode("utf-8"))
    if dest:
        json.dump(payload, open(dest, "w", encoding="utf-8"))
    return payload


def build_matched():
    """Cross-reference upstream annotations with clips the mirror actually has.

    Returns rows of [gloss, video_id, signer_id, upstream_split, mirror_path].
    """
    annot = fetch_json(ANNOT_URL, DATA / "WLASL_v0.3.json")
    meta = fetch_json(HF_API, DATA / "hf_api.json")
    avail = {}
    for s in meta["siblings"]:
        p = s["rfilename"]
        if p.endswith(".mp4"):
            avail[p.split("/")[-1][:-4]] = p
    rows = [[g["gloss"], i["video_id"], i["signer_id"], i["split"], avail[i["video_id"]]]
            for g in annot for i in g["instances"] if i["video_id"] in avail]
    print(f"mirror hosts {len(avail)} clips; matched {len(rows)} annotated instances",
          flush=True)
    json.dump(rows, open(DATA / "matched.json", "w"))
    return rows


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


def main():
    DATA.mkdir(exist_ok=True)
    cache = DATA / "matched.json"
    rows = json.load(open(cache)) if cache.exists() else build_matched()

    cnt = collections.Counter(r[0] for r in rows)
    top = {g for g, _ in cnt.most_common(N_CLASSES)}
    sel = [r for r in rows if r[0] in top]
    json.dump(sel, open(DATA / "subset.json", "w"))
    print(f"selected {len(sel)} clips over {len(top)} glosses", flush=True)

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading to {VIDEO_DIR} (set WLASL_VIDEO_DIR to change)", flush=True)
    with ThreadPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(get, sel))

    tally = collections.Counter(s for s, _ in res)
    print(dict(tally), flush=True)
    fails = [v for s, v in res if s.startswith("FAIL")]
    json.dump(fails, open(DATA / "download_failures.json", "w"))
    print(f"failures: {len(fails)}", flush=True)


if __name__ == "__main__":
    main()
