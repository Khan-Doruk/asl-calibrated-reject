"""Figure 1: reliability diagram (before / after temperature scaling).
   Figure 2: risk-coverage curve + what calibration actually buys.

Note on Figure 2. Temperature scaling is a strictly monotone transform of the
confidence, so it cannot reorder the test clips and therefore CANNOT move the
accuracy-vs-coverage curve: at any fixed coverage the accepted SET is identical
before and after. Panel A is that curve, and it is a property of the ranking
alone.

What calibration changes is whether the system knows how well it is doing.
Panel B plots, at each coverage, the accuracy the model *claims* (mean
confidence over the accepted clips) against the accuracy it actually gets.
Before scaling the claim is far too high; after scaling the two nearly
coincide. That is the property you need if a deployment threshold is going to
be chosen against a target accuracy.
"""
import json, pathlib
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
NB = 15
MIN_N = 5                      # bins thinner than this are drawn faded
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#4a5568"
RED, TEAL = "#e53e3e", "#2c7a7b"

d = np.load(ROOT / "data" / "logits_seed0.npz", allow_pickle=True)
res = json.load(open(ROOT / "data" / "results.json"))
lg, ytest, T = d["test"], d["ytest"], float(d["T"])


def probs(logits, t=1.0):
    p = torch.softmax(torch.tensor(logits / t), 1).numpy()
    return p.max(1), p.argmax(1)


def bins(conf, correct, nb=NB):
    edges = np.linspace(0, 1, nb + 1)
    xs, accs, confs, ns = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        xs.append((lo + hi) / 2)
        if m.sum():
            accs.append(correct[m].mean()); confs.append(conf[m].mean()); ns.append(int(m.sum()))
        else:
            accs.append(np.nan); confs.append(np.nan); ns.append(0)
    return [np.array(v) for v in (xs, accs, confs, ns)], edges


def ece_ew(conf, correct, nb=NB):
    (_, a, c, n), _ = bins(conf, correct, nb)
    k = n > 0
    return float((n[k] / n.sum() * np.abs(a[k] - c[k])).sum())


c0, p0 = probs(lg, 1.0);  ok0 = (p0 == ytest).astype(float)
c1, p1 = probs(lg, T);    ok1 = (p1 == ytest).astype(float)

# ---------------------------------------------------------------- Figure 1
fig, axes = plt.subplots(2, 2, figsize=(11, 8.2),
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07, "wspace": 0.2})
panels = [(c0, ok0, "Before calibration   (T = 1.00)", BLUE),
          (c1, ok1, "After temperature scaling   (T* = %.2f)" % T, ORANGE)]
for col, (conf, ok, ttl, col_c) in enumerate(panels):
    ax, axn = axes[0, col], axes[1, col]
    (x, a, cm, n), edges = bins(conf, ok)
    w = edges[1] - edges[0]
    k = n > 0
    solid = k & (n >= MIN_N)
    faint = k & (n < MIN_N)

    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.3, zorder=1, label="perfect calibration")
    ax.bar(x[solid], a[solid], width=w * 0.9, color=col_c, alpha=0.9,
           edgecolor="white", lw=0.8, zorder=2, label="observed accuracy")
    if faint.sum():
        ax.bar(x[faint], a[faint], width=w * 0.9, color=col_c, alpha=0.28,
               edgecolor="white", lw=0.8, zorder=2,
               label="observed accuracy (fewer than %d clips)" % MIN_N)

    # gap drawn between accuracy and confidence, in whichever direction it goes
    lo_ = np.minimum(a, cm)
    hgt = np.abs(a - cm)
    over = k & (cm > a)
    under = k & (a >= cm)
    ax.bar(x[over], hgt[over], bottom=lo_[over], width=w * 0.9, color=RED, alpha=0.32,
           edgecolor=RED, lw=0.8, hatch="//", zorder=3, label="gap: over-confident")
    if under.sum():
        ax.bar(x[under], hgt[under], bottom=lo_[under], width=w * 0.9, color=TEAL,
               alpha=0.32, edgecolor=TEAL, lw=0.8, hatch="xx", zorder=3,
               label="gap: under-confident")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(ttl, fontsize=11.5, pad=10)
    ax.set_ylabel("accuracy on clips in bin" if col == 0 else "")
    ax.text(0.035, 0.955,
            "ECE = %.3f\naccuracy = %.3f\nmean confidence = %.3f" % (
                ece_ew(conf, ok), ok.mean(), conf.mean()),
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="white", ec=GREY, alpha=0.93, boxstyle="round,pad=0.4"))
    if col == 0:
        handles, labels = ax.get_legend_handles_labels()
    ax.tick_params(labelbottom=False)
    ax.grid(alpha=0.18)

    axn.bar(x[k], n[k], width=w * 0.9, color=GREY, alpha=0.6)
    axn.axhline(MIN_N, ls=":", color=RED, lw=1)
    axn.set_xlim(0, 1); axn.set_xlabel("confidence")
    axn.set_ylabel("clips" if col == 0 else "")
    axn.grid(alpha=0.18)

fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.035))
fig.suptitle("Reliability - WLASL-100 sign classifier, signer-disjoint test split "
             "(n = %d clips, 100 classes)" % len(ytest), fontsize=13, y=0.965)
fig.savefig(FIG / "fig1_reliability.png", dpi=180, bbox_inches="tight")
print("wrote fig1_reliability.png")

# ---------------------------------------------------------------- Figure 2
order = np.argsort(-c0)                    # identical ranking after scaling
N = len(order)
cov = np.arange(1, N + 1) / N
sel_acc = np.cumsum(ok0[order]) / np.arange(1, N + 1)
claim_before = np.cumsum(c0[order]) / np.arange(1, N + 1)
claim_after = np.cumsum(np.sort(c1)[::-1]) / np.arange(1, N + 1)

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.8))

axA.plot(cov, sel_acc, color=BLUE, lw=2.2, zorder=3)
axA.axhline(ok0.mean(), ls="--", color=GREY, lw=1.2, zorder=1)
axA.text(0.985, ok0.mean() - 0.03, "answer everything: %.3f" % ok0.mean(),
         ha="right", va="top", fontsize=9, color=GREY)
for c in (0.25, 0.50, 0.75):
    i = int(c * N) - 1
    axA.plot([c], [sel_acc[i]], "o", color=ORANGE, ms=7, zorder=5)
    axA.annotate("%.2f" % sel_acc[i], xy=(c, sel_acc[i]), xytext=(0, 11),
                 textcoords="offset points", ha="center", fontsize=9.5,
                 color=ORANGE, fontweight="bold")
axA.set_xlabel("coverage - fraction of clips the system answers")
axA.set_ylabel("accuracy on answered clips")
axA.set_title("A. Risk-coverage: what the reject option buys", fontsize=11.5)
axA.set_xlim(0.03, 1.0); axA.set_ylim(0, 0.75); axA.grid(alpha=0.25)
axA.text(0.97, 0.05, "identical before/after scaling -\nT is monotone, so the ranking\n"
                     "and the accepted set do not change",
         transform=axA.transAxes, ha="right", va="bottom", fontsize=8.5,
         color=GREY, style="italic")

axB.plot(cov, sel_acc, color="black", lw=2.2, label="actual accuracy (same both ways)", zorder=4)
axB.plot(cov, claim_before, color=BLUE, lw=2, ls="--",
         label="accuracy the model claims - before", zorder=3)
axB.plot(cov, claim_after, color=ORANGE, lw=2, ls="--",
         label="accuracy the model claims - after", zorder=3)
axB.fill_between(cov, sel_acc, claim_before, color=BLUE, alpha=0.12, zorder=1)
axB.fill_between(cov, sel_acc, claim_after, color=ORANGE, alpha=0.20, zorder=2)
axB.set_xlabel("coverage - fraction of clips the system answers")
axB.set_ylabel("accuracy")
axB.set_title("B. Does the system know how well it is doing?", fontsize=11.5)
axB.set_xlim(0.03, 1.0); axB.set_ylim(0, 0.9); axB.grid(alpha=0.25)
axB.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
axB.text(0.03, 0.03, "shaded = self-assessment error\n(smaller is better)",
         transform=axB.transAxes, ha="left", va="bottom", fontsize=8.5,
         color=GREY, style="italic")

fig.suptitle("Selective prediction - signer-disjoint test split", fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(FIG / "fig2_risk_coverage.png", dpi=180, bbox_inches="tight")
print("wrote fig2_risk_coverage.png")

# ------------------------------------------------- table for the README
c1_sorted = np.sort(c1)[::-1]
tbl = []
for c in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    i = min(int(c * N) - 1, N - 1)
    tbl.append(dict(coverage=c, accuracy=float(sel_acc[i]),
                    claimed_before=float(claim_before[i]),
                    claimed_after=float(claim_after[i]),
                    tau_before=float(c0[order][i]),
                    tau_after=float(c1_sorted[i])))
json.dump(tbl, open(ROOT / "data" / "risk_coverage_table.json", "w"), indent=2)

# ------------------------------------------------- bootstrap CIs (seed 0)
# n = 224 is not large, so quote an interval rather than a bare point estimate.
rng = np.random.default_rng(0)
boot = {"acc": [], "ece_before": [], "ece_after": [], "ece_reduction_pct": []}
for _ in range(2000):
    s = rng.integers(0, N, N)
    e0, e1 = ece_ew(c0[s], ok0[s]), ece_ew(c1[s], ok1[s])
    boot["acc"].append(ok0[s].mean())
    boot["ece_before"].append(e0)
    boot["ece_after"].append(e1)
    boot["ece_reduction_pct"].append(100 * (e0 - e1) / e0 if e0 > 0 else np.nan)
ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
      for k, v in boot.items()}
json.dump(ci, open(ROOT / "data" / "bootstrap_ci.json", "w"), indent=2)
print("\n95%% bootstrap CIs over the %d test clips (seed 0, 2000 resamples):" % N)
for k, v in ci.items():
    print("  %-18s [%.3f, %.3f]" % (k, v[0], v[1]))
print("\ncov   acc    claimed_before  claimed_after  tau_before  tau_after")
for r in tbl:
    print(" %.2f  %.3f      %.3f           %.3f          %.3f       %.3f" % (
        r["coverage"], r["accuracy"], r["claimed_before"], r["claimed_after"],
        r["tau_before"], r["tau_after"]))
