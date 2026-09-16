"""Train a small BiGRU on the landmark sequences, then temperature-scale it.

Deliberately a weak model: ~600 training clips over 100 classes. The point of
the project is the confidence behaviour, not the accuracy.

Protocol:
  train  -> weights
  val    -> epoch selection (early stopping), different signers
  cal    -> temperature only, different signers again
  test   -> every reported number, never seen before this point
"""
import json, pathlib
import numpy as np, torch, torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEEDS = [0, 1, 2, 3, 4]
EPOCHS, BATCH = 120, 32
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SignGRU(nn.Module):
    def __init__(self, d_in, n_cls, hid=160, p=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.gru = nn.GRU(d_in, hid, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p)
        self.head = nn.Linear(hid * 4, n_cls)      # mean-pool ++ max-pool

    def forward(self, x):
        h, _ = self.gru(self.norm(x))
        z = torch.cat([h.mean(1), h.max(1).values], dim=1)
        return self.head(self.drop(z))


def ece_equal_width(conf, correct, n_bins=15):
    """Standard ECE: equal-width confidence bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return e


def ece_equal_mass(conf, correct, n_bins=15):
    """Adaptive ECE: equal-count bins. More stable when n is small, which it is."""
    order = np.argsort(conf)
    e, n = 0.0, len(conf)
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            e += len(chunk) / n * abs(correct[chunk].mean() - conf[chunk].mean())
    return e


def fit_temperature(logits, labels):
    """One scalar T minimising NLL on the calibration split."""
    T = torch.ones(1, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=200)
    lg, lb = torch.tensor(logits), torch.tensor(labels)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(lg / T.clamp(min=1e-2), lb)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().clamp(min=1e-2))


def run_seed(seed, X, y, idx):
    torch.manual_seed(seed); np.random.seed(seed)
    model = SignGRU(X.shape[2], int(y.max()) + 1).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    Xtr = torch.tensor(X[idx["train"]]).to(DEV); ytr = torch.tensor(y[idx["train"]]).to(DEV)
    Xva = torch.tensor(X[idx["val"]]).to(DEV);   yva = torch.tensor(y[idx["val"]]).to(DEV)

    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEV)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            nn.functional.cross_entropy(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    out = {}
    with torch.no_grad():
        for sp in ("cal", "test"):
            out[sp] = model(torch.tensor(X[idx[sp]]).to(DEV)).cpu().numpy().astype(np.float64)
    return out, best


def summarize(logits, labels, T):
    def stats(lg):
        p = torch.softmax(torch.tensor(lg), 1).numpy()
        conf, pred = p.max(1), p.argmax(1)
        correct = (pred == labels).astype(float)
        return dict(acc=float(correct.mean()),
                    ece=float(ece_equal_width(conf, correct)),
                    ece_em=float(ece_equal_mass(conf, correct)),
                    mean_conf=float(conf.mean()),
                    nll=float(nn.functional.cross_entropy(
                        torch.tensor(lg), torch.tensor(labels)).item()))
    return stats(logits), stats(logits / T)


def main():
    d = np.load(ROOT / "data" / "landmarks.npz", allow_pickle=True)
    X, gloss = d["X"], d["gloss"]
    sp = json.load(open(ROOT / "data" / "splits.json"))
    split = np.array(sp["split"]); usable = np.array(sp["usable"])

    classes = sorted(set(gloss[split == "train"].tolist()))
    cmap = {g: i for i, g in enumerate(classes)}
    y = np.array([cmap.get(g, -1) for g in gloss.tolist()])

    idx = {k: np.where((split == k) & usable & (y >= 0))[0]
           for k in ("train", "val", "cal", "test")}
    print({k: len(v) for k, v in idx.items()}, f"classes={len(classes)}  device={DEV}")

    ycal, ytest = y[idx["cal"]], y[idx["test"]]
    rows, per_seed = [], []
    for s in SEEDS:
        lg, vacc = run_seed(s, X, y, idx)
        T = fit_temperature(lg["cal"], ycal)
        before, after = summarize(lg["test"], ytest, T)
        print(f"seed {s}: val_acc={vacc:.3f}  T*={T:.3f}  "
              f"test_acc={before['acc']:.3f}  ECE {before['ece']:.3f} -> {after['ece']:.3f}")
        rows.append(dict(seed=s, T=T, val_acc=vacc, before=before, after=after))
        per_seed.append(lg)
        if s == SEEDS[0]:
            np.savez(ROOT / "data" / "logits_seed0.npz",
                     cal=lg["cal"], test=lg["test"], ycal=ycal, ytest=ytest, T=T,
                     classes=np.array(classes))

    agg = {}
    for phase in ("before", "after"):
        for m in ("acc", "ece", "ece_em", "mean_conf", "nll"):
            v = np.array([r[phase][m] for r in rows])
            agg[f"{phase}_{m}"] = [float(v.mean()), float(v.std())]
    agg["T"] = [float(np.mean([r["T"] for r in rows])), float(np.std([r["T"] for r in rows]))]
    json.dump(dict(per_seed=rows, agg=agg, n_test=len(ytest), n_cal=len(ycal),
                   n_classes=len(classes), seeds=SEEDS),
              open(ROOT / "data" / "results.json", "w"), indent=2)

    print("\n=== across seeds (mean +/- sd) ===")
    print(f"test top-1 accuracy : {agg['before_acc'][0]:.4f} +/- {agg['before_acc'][1]:.4f}")
    print(f"mean confidence     : {agg['before_mean_conf'][0]:.4f} (before) -> "
          f"{agg['after_mean_conf'][0]:.4f} (after)")
    print(f"ECE  (15 equal-width): {agg['before_ece'][0]:.4f} -> {agg['after_ece'][0]:.4f}")
    print(f"ECE  (15 equal-mass) : {agg['before_ece_em'][0]:.4f} -> {agg['after_ece_em'][0]:.4f}")
    print(f"NLL                  : {agg['before_nll'][0]:.4f} -> {agg['after_nll'][0]:.4f}")
    print(f"T*                   : {agg['T'][0]:.3f} +/- {agg['T'][1]:.3f}")


if __name__ == "__main__":
    main()
