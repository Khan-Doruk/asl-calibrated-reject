"""Stage 2c normalization schemes, computed from the raw coordinates saved by
src/11_extract_variants.py. Each scheme changes exactly one thing relative to
src/02's features:

  baseline         src/02: per-frame mid-shoulder origin and shoulder-width
                   scale, hands relative to their own wrist (209 dims)
  clip_scale       scale = median shoulder width over the clip's posed frames,
                   instead of each frame's own shoulder width. The origin stays
                   per-frame. Tests whether a noisy per-frame scale injects
                   jitter.
  velocity         baseline + first differences over time (418 dims). The first
                   frame's difference is zero. Frames with no pose are all-zero
                   in the baseline, so differences next to them spike; that is
                   left as-is rather than masked, so the variant is a single
                   change.
  mirror           baseline after mirroring every clip of a signer whose
                   majority-vote dominant hand is the minority slot, so the
                   dominant hand lands in the same feature slot. Decided per
                   signer, not per clip - see mirror() for why.
"""
import numpy as np

FEAT_DIM = 209
POSE_KEEP = list(range(25))
# MediaPipe pose left/right landmark pairs, swapped when mirroring
POSE_SWAP = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
             (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def _frame(pose, pose_ok, hand, hand_ok, scale=None):
    """src/02's per-frame features; `scale` overrides the shoulder width."""
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not pose_ok:
        return feat
    P = pose
    ls, rs = P[11], P[12]
    origin = (ls + rs) / 2.0
    if scale is None:
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


def _raw(z, T):
    return (z[f"pose_T{T}"], z[f"pose_ok_T{T}"], z[f"hand_T{T}"], z[f"hand_ok_T{T}"])


def baseline(z, T):
    pose, pose_ok, hand, hand_ok = _raw(z, T)
    N = pose.shape[0]
    X = np.stack([np.stack([_frame(pose[i, t], pose_ok[i, t], hand[i, t], hand_ok[i, t])
                            for t in range(T)]) for i in range(N)])
    return X, {}


def clip_scale(z, T):
    pose, pose_ok, hand, hand_ok = _raw(z, T)
    N = pose.shape[0]
    out, cv = [], []
    for i in range(N):
        w = np.linalg.norm((pose[i, :, 11] - pose[i, :, 12])[:, :2], axis=1)
        w = w[pose_ok[i] & (w > 0)]
        s = float(np.median(w)) if w.size else 1e-3
        if w.size > 1:
            cv.append(float(w.std() / w.mean()))
        out.append(np.stack([_frame(pose[i, t], pose_ok[i, t], hand[i, t], hand_ok[i, t], scale=s)
                             for t in range(T)]))
    return np.stack(out), dict(
        per_frame_shoulder_width_cv_median=float(np.median(cv)),
        per_frame_shoulder_width_cv_p90=float(np.percentile(cv, 90)))


def velocity(z, T):
    X, _ = baseline(z, T)
    V = np.zeros_like(X)
    V[:, 1:] = X[:, 1:] - X[:, :-1]
    return np.concatenate([X, V], axis=2), {}


def dominant_slot(pose_ok, hand, hand_ok):
    """The hand slot that does the signing: present in more frames (a resting
    hand usually is not detected); ties broken by total wrist travel. Returns
    None if neither hand is ever present."""
    counts = hand_ok.sum(0)
    if counts.max() == 0:
        return None
    if counts[0] != counts[1]:
        return int(np.argmax(counts))
    travel = []
    for s in (0, 1):
        wr = hand[:, s, 0, :2]
        both = hand_ok[1:, s] & hand_ok[:-1, s]
        travel.append(float(np.linalg.norm(wr[1:] - wr[:-1], axis=1)[both].sum()))
    return int(np.argmax(travel))


def mirror(z, T, signer):
    """Mirror every clip of a left-dominant *signer*, decided by majority vote
    over that signer's clips.

    Dominance is a property of the signer, and the per-clip vote is noisy: on
    this pool most signers with 3+ clips get mixed per-clip votes, which real
    signers do not do, so mirroring clip by clip would mostly mirror detection
    noise. The vote uses input features only, never labels, so it is valid for
    held-out signers too.
    """
    pose, pose_ok, hand, hand_ok = (a.copy() for a in _raw(z, T))
    N = pose.shape[0]
    dom = [dominant_slot(pose_ok[i], hand[i], hand_ok[i]) for i in range(N)]
    known = [x for x in dom if x is not None]
    majority = int(round(np.mean(known))) if known else 1
    votes = {}
    for i, dslot in enumerate(dom):
        if dslot is not None:
            votes.setdefault(str(signer[i]), []).append(dslot)
    # flip only on a strict minority vote; an even split is not evidence of
    # left-dominance (and Python's round(0.5) == 0 would otherwise decide it)
    share_minority = {s: (1 - np.mean(v)) if majority == 1 else np.mean(v) for s, v in votes.items()}
    flip_signers = {s for s, m in share_minority.items() if m > 0.5}
    flipped = []
    for i in range(N):
        if str(signer[i]) not in flip_signers:
            continue
        flipped.append(i)
        pose[i, :, :, 0] = np.where(pose_ok[i][:, None], 1.0 - pose[i, :, :, 0], 0.0)
        for a, b in POSE_SWAP:
            pose[i, :, [a, b]] = pose[i, :, [b, a]]
        hand[i, :, :, :, 0] = np.where(hand_ok[i][:, :, None], 1.0 - hand[i, :, :, :, 0], 0.0)
        # .copy(): the reversed slice is a view of the same memory being written
        hand[i] = hand[i][:, ::-1].copy()
        hand_ok[i] = hand_ok[i][:, ::-1].copy()
    X = np.stack([np.stack([_frame(pose[i, t], pose_ok[i, t], hand[i, t], hand_ok[i, t])
                            for t in range(T)]) for i in range(N)])
    mixed = {s: v for s, v in votes.items() if len(v) >= 3 and 0 < np.mean(v) < 1}
    info = dict(
        rule="per-signer majority vote of per-clip dominant slots; strict minority -> mirror",
        majority_dominant_slot=majority,
        signers_total=len(votes), signers_mirrored=len(flip_signers),
        clips_mirrored=len(flipped), clips_no_hand=sum(d is None for d in dom),
        # why the vote is per signer: per-clip votes disagree within most signers
        clip_votes_minority_slot=sum(1 for d in known if d != majority),
        signers_with_3plus_clips=sum(1 for v in votes.values() if len(v) >= 3),
        signers_3plus_with_mixed_clip_votes=len(mixed),
        mixed_signers_median_minority_share=float(np.median(
            [min(np.mean(v), 1 - np.mean(v)) for v in mixed.values()])) if mixed else 0.0)
    return X, info


SCHEMES = dict(baseline=baseline, clip_scale=clip_scale, velocity=velocity, mirror=mirror)
