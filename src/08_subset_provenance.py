"""How our 100-gloss pool relates to the official WLASL100 benchmark.

This exists because it is easy to assume our subset *is* WLASL100 and to put our
numbers next to published ones. It is not, and they cannot be.

Our pool was built as the 100 most-represented glosses **on the HuggingFace
mirror**, which hosts only part of WLASL. The official WLASL100 is defined by
`code/I3D/preprocess/nslt_100.json` in the upstream repo. This script measures
the gap and writes data/subset_provenance.json, which is committed so the
comparability caveat can be regenerated without the gitignored annotations.

Run after src/01 (it needs data/WLASL_v0.3.json and data/matched.json).
"""
import json, pathlib, collections, urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NSLT_URL = ("https://raw.githubusercontent.com/dxli94/WLASL/master/"
            "code/I3D/preprocess/nslt_100.json")


def main():
    for req in ("WLASL_v0.3.json", "matched.json", "subset.json"):
        if not (DATA / req).exists():
            raise SystemExit(f"missing data/{req} - run src/01_select_and_download.py first")

    nslt_path = DATA / "nslt_100.json"
    if not nslt_path.exists():
        print("fetching official nslt_100.json ...", flush=True)
        urllib.request.urlretrieve(NSLT_URL, nslt_path)
    nslt = json.load(open(nslt_path))

    ann = json.load(open(DATA / "WLASL_v0.3.json"))
    v2g = {i["video_id"]: g["gloss"] for g in ann for i in g["instances"]}
    mirror = {r[1] for r in json.load(open(DATA / "matched.json"))}
    sel = json.load(open(DATA / "subset.json"))
    ours_gloss = {r[0] for r in sel}
    ours_vid = {r[1] for r in sel}

    off_vid = set(nslt)
    off_gloss = {v2g[v] for v in off_vid if v in v2g}
    recoverable = off_vid & mirror

    out = dict(
        our_pool=dict(clips=len(sel), glosses=len(ours_gloss),
                      selection_rule="100 most-represented glosses on the Voxel51/WLASL mirror"),
        official_wlasl100=dict(
            videos=len(off_vid), glosses=len(off_gloss),
            split=dict(collections.Counter(v["subset"] for v in nslt.values()))),
        comparability=dict(
            gloss_overlap=len(ours_gloss & off_gloss),
            glosses_we_have_official_lacks=sorted(ours_gloss - off_gloss),
            glosses_official_has_we_lack=sorted(off_gloss - ours_gloss),
            official_videos_we_hold=len(ours_vid & off_vid),
            official_videos_on_mirror=len(recoverable),
            official_videos_on_mirror_frac=len(recoverable) / len(off_vid),
            recoverable_official_split=dict(
                collections.Counter(nslt[v]["subset"] for v in recoverable)),
        ),
        conclusion=(
            "Our pool is not the official WLASL100. Numbers from it cannot be "
            "placed beside published WLASL100 results, on either split protocol."),
    )
    json.dump(out, open(DATA / "subset_provenance.json", "w"), indent=2)

    c = out["comparability"]
    print(f"our pool: {out['our_pool']['clips']} clips / {out['our_pool']['glosses']} glosses")
    print(f"official WLASL100: {out['official_wlasl100']['videos']} videos, "
          f"split {out['official_wlasl100']['split']}")
    print(f"gloss overlap: {c['gloss_overlap']}/100")
    print(f"official videos we hold: {c['official_videos_we_hold']}")
    print(f"official videos present on mirror: {c['official_videos_on_mirror']} "
          f"({c['official_videos_on_mirror_frac']:.0%}) -> split {c['recoverable_official_split']}")
    print("wrote data/subset_provenance.json")


if __name__ == "__main__":
    main()
