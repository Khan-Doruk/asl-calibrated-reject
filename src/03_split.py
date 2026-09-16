"""Signer-disjoint train / val / calibration / test split.

WLASL ships a signer_id per clip, so we hold out *signers*, not random clips.
All four splits get disjoint signers:
  train - fits the network weights
  val   - picks the training epoch (kept inside the "seen" budget conceptually,
          but still a different signer, so no clip-level leakage)
  cal   - fits the temperature, and ONLY that
  test  - reported numbers, never touched until the end

Keeping cal and test on different signers matters here: if they shared signers
the calibration would look better than it is, in exactly the way this project
is trying to measure (holding up under an unseen signer).

Greedy assignment: signers sorted by clip count, each handed to whichever split
is furthest below its target share. Deterministic, no RNG.
"""
import json, pathlib, collections
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = {"train": 0.55, "val": 0.10, "cal": 0.15, "test": 0.20}

d = np.load(ROOT / "data" / "landmarks.npz", allow_pickle=True)
gloss, signer = d["gloss"], d["signer"]

by_signer = collections.Counter(signer.tolist())
order = sorted(by_signer, key=lambda s: (-by_signer[s], s))

assign, have = {}, {k: 0 for k in TARGET}
total = len(signer)
for s in order:
    pick = max(TARGET, key=lambda k: TARGET[k] - have[k] / max(total, 1))
    assign[s] = pick
    have[pick] += by_signer[s]

split = np.array([assign[s] for s in signer.tolist()])

# a clip is only usable outside train if its gloss was actually seen in training
train_gloss = set(gloss[split == "train"].tolist())
usable = np.array([(sp == "train") or (g in train_gloss)
                   for g, sp in zip(gloss.tolist(), split.tolist())])

print(f"total clips {total}, signers {len(by_signer)}")
print("signers per split:", {k: sum(1 for v in assign.values() if v == k) for k in TARGET})
for k in TARGET:
    m = (split == k) & usable
    print(f"  {k:5s} clips={m.sum():4d}  glosses={len(set(gloss[m].tolist())):3d}  "
          f"signers={len(set(signer[m].tolist())):3d}")
print(f"dropped (gloss never seen in train): {(~usable).sum()}")

summary = {"total_clips": int(total), "n_signers": int(len(by_signer)),
           "per_split": {k: {"clips": int(((split == k) & usable).sum()),
                             "glosses": len(set(gloss[(split == k) & usable].tolist())),
                             "signers": len(set(signer[(split == k) & usable].tolist()))}
                         for k in TARGET}}

# summary is committed to the repo so the README can be regenerated without
# landmarks.npz, which is WLASL-derived and deliberately not redistributed
json.dump({"split": split.tolist(), "usable": usable.tolist(),
           "signer_assign": {str(k): v for k, v in assign.items()},
           "summary": summary},
          open(ROOT / "data" / "splits.json", "w"))
