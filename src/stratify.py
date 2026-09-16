"""Stratified take with largest-remainder apportionment.

Replaces the take logic that first lived in src/07. That version rounded each
label's proportional quota independently; when the rounded quotas summed above
n_take, its top-up loop never ran and a final `taken[:n_take]` truncated from
the tail of dict insertion order, so whole labels lost their entire quota while
the output count still looked right. This version apportions exactly and fails
loudly instead.

Run `python src/stratify.py` to execute the self-check.
"""
import collections
import math


def stratified_take(pool, labels, n_take, rng, protect_min=1):
    """Take exactly n_take indices from pool, proportional to label size.

    Every label keeps at least `protect_min` members in the remainder. Label
    shares use largest-remainder (Hamilton) apportionment, water-filled around
    each label's capacity (size - protect_min); remainder ties are broken by a
    random label order drawn from `rng`.

    Returns (taken, remaining), both sorted.
    """
    pool = list(pool)
    by = collections.defaultdict(list)
    for i in pool:
        by[labels[i]].append(i)
    keys = sorted(by)
    for k in keys:
        rng.shuffle(by[k])

    cap = {k: max(0, len(by[k]) - protect_min) for k in keys}
    if n_take > sum(cap.values()):
        raise ValueError(f"cannot take {n_take} of {len(pool)} while leaving "
                         f"{protect_min} of each of {len(keys)} labels behind")
    if n_take < 0:
        raise ValueError("n_take must be non-negative")

    ideal = {k: n_take * len(by[k]) / len(pool) for k in keys}
    alloc = {k: min(cap[k], math.floor(ideal[k])) for k in keys}
    order = list(keys)
    rng.shuffle(order)
    tiebreak = {k: r for r, k in enumerate(order)}

    short = n_take - sum(alloc.values())
    while short > 0:
        open_ = [k for k in keys if alloc[k] < cap[k]]
        open_.sort(key=lambda k: (-(ideal[k] - alloc[k]), tiebreak[k]))
        for k in open_[:short]:
            alloc[k] += 1
        short = n_take - sum(alloc.values())

    taken = [i for k in keys for i in by[k][:alloc[k]]]
    remaining = [i for k in keys for i in by[k][alloc[k]:]]

    assert len(taken) == n_take, (len(taken), n_take)
    left = collections.Counter(labels[i] for i in remaining)
    assert all(left[k] >= protect_min for k in keys), "a label fell below protect_min"
    return sorted(taken), sorted(remaining)


def _self_check():
    import numpy as np
    rng = np.random.default_rng(0)

    # Overshoot case that broke the old version: ten labels of 3, take 15.
    # Each ideal share is 1.5; independent rounding gives 2 x 10 = 20 > 15, and
    # the old truncation then zeroed the last labels in insertion order.
    labels = {i: i // 3 for i in range(30)}
    taken, rem = stratified_take(range(30), labels, 15, rng)
    per = collections.Counter(labels[i] for i in taken)
    assert len(taken) == 15
    assert all(per[k] >= 1 for k in range(10)), f"a label lost its whole share: {per}"
    assert max(per.values()) - min(per.values()) <= 1, per

    # Uneven labels, protect_min binding on the small ones.
    labels = {}
    i = 0
    for k, n in enumerate([1, 2, 2, 5, 20, 40]):
        for _ in range(n):
            labels[i] = k
            i += 1
    taken, rem = stratified_take(range(i), labels, 50, rng)
    left = collections.Counter(labels[j] for j in rem)
    assert len(taken) == 50 and all(left[k] >= 1 for k in range(6)), left

    # Impossible requests must raise, not silently under-deliver.
    try:
        stratified_take(range(i), labels, i - 5, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("over-capacity request did not raise")
    return True


if __name__ == "__main__":
    _self_check()
    print("stratify self-check passed")
