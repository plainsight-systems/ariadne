"""Measurement 3: is the input tail a nonlinear echo of the core?

Same models, basis, core and tail as tail_identity.py (Pythia 1.4B, 2.8B,
6.9B, 12B input tables; unweighted principal directions of the centered
real rows; core = top k*(0.1); tail = the rest).

1. Nonlinear predictability. Predict the tail coordinates from the core
   coordinates for held-out tokens (fixed 80/20 split of the vocabulary,
   seed 0). Regressors: k-nearest neighbours in the core (Euclidean, k = 5
   and 20, prediction = mean of the neighbours' tails); a small MLP (core ->
   512 GELU -> tail, Adam 1e-3, 20 epochs, batch 512, seed 0); and ordinary
   least squares as a sanity check (principal coordinates are uncorrelated,
   so linear R^2 should be about 0). R^2 is pooled over all tail
   coordinates against the training mean. Baseline: the same regressor
   predicting a shuffled tail, with Measurement 1's five permutations of
   all tokens (seeds 1-5).

2. Geometry echo. Over the tokens in the phase-2 evaluation text, the
   Spearman correlation between pairwise cosines in the core alone and in
   the tail alone (rows centered on the all-token mean). Baseline: the same
   with the tail permuted within those tokens (seeds 1-5), so the set of
   tail vectors is unchanged.

Usage: .venv/bin/python nonlinear_echo.py
Writes results/nonlinear_echo.json and results/nonlinear_echo.md.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables
from project import eval_text, windows
from tail_identity import K_CORE, PERM_SEEDS

HERE = Path(__file__).resolve().parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
SPLIT_SEED, MLP_SEED = 0, 0
KS = (5, 20)


def r2(y, yhat, ybar):
    return float(1 - ((y - yhat) ** 2).sum() / ((y - ybar) ** 2).sum())


@torch.no_grad()
def knn_indices(train, test, kmax):
    tr = torch.as_tensor(train, dtype=torch.float32, device=DEVICE)
    trsq = (tr * tr).sum(1)
    out = []
    for b in range(0, len(test), 1000):
        te = torch.as_tensor(test[b:b + 1000], dtype=torch.float32, device=DEVICE)
        d2 = (te * te).sum(1, keepdim=True) - 2 * te @ tr.T + trsq[None, :]
        out.append(torch.topk(d2, kmax, largest=False).indices.cpu())
    return torch.cat(out).numpy()


def knn_r2(nbr, ytr, yte):
    ybar = ytr.mean(0)
    return {f"knn{k}": r2(yte, ytr[nbr[:, :k]].mean(1), ybar) for k in KS}


def ols_r2(xtr, xte, ytr, yte):
    A = np.hstack([xtr, np.ones((len(xtr), 1))])
    coef, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    return r2(yte, np.hstack([xte, np.ones((len(xte), 1))]) @ coef, ytr.mean(0))


def mlp_r2(xtr, xte, ytr, yte):
    torch.manual_seed(MLP_SEED)
    sx, sy = xtr.std(0) + 1e-12, ytr.std(0) + 1e-12
    mx, my = xtr.mean(0), ytr.mean(0)
    Xtr = torch.as_tensor((xtr - mx) / sx, dtype=torch.float32, device=DEVICE)
    Ytr = torch.as_tensor((ytr - my) / sy, dtype=torch.float32, device=DEVICE)
    net = torch.nn.Sequential(torch.nn.Linear(xtr.shape[1], 512), torch.nn.GELU(),
                              torch.nn.Linear(512, ytr.shape[1])).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(MLP_SEED)
    for _ in range(20):
        for idx in torch.randperm(len(Xtr), generator=g).split(512):
            idx = idx.to(DEVICE)
            loss = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(torch.as_tensor((xte - mx) / sx, dtype=torch.float32, device=DEVICE)).cpu().double().numpy()
    return r2(yte, pred * sy + my, ytr.mean(0))


def spearman_upper(a, b):
    """Spearman correlation of the upper-triangle entries of two square matrices."""
    iu = np.triu_indices(len(a), 1)
    x, y = a[iu], b[iu]
    rx = np.empty(len(x)); rx[np.argsort(x)] = np.arange(len(x))
    ry = np.empty(len(y)); ry[np.argsort(y)] = np.arange(len(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    spec = json.loads((HERE / "models.json").read_text())
    text = eval_text()
    rows = []
    for m in spec["models"]:
        name = m["repo"].split("/")[1]
        if name not in K_CORE:
            continue
        repo, rev, k = m["repo"], m["revision"], K_CORE[name]
        print(f"{repo}  core {k}", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        c = w - w.mean(0)
        lam, vec = np.linalg.eigh(c.T @ c / n)
        x = c @ vec[:, np.argsort(lam)[::-1]]
        core, tail = x[:, :k], x[:, k:]
        perms = [np.random.default_rng(sd).permutation(n) for sd in PERM_SEEDS]

        order = np.random.default_rng(SPLIT_SEED).permutation(n)
        te, tr = order[: n // 5], order[n // 5:]
        nbr = knn_indices(core[tr], core[te], max(KS))
        row = {"repo": repo, "revision": rev, "d": x.shape[1], "k_core": k, "train": len(tr), "test": len(te)}
        row["true"] = {**knn_r2(nbr, tail[tr], tail[te]), "ols": ols_r2(core[tr], core[te], tail[tr], tail[te]),
                       "mlp": mlp_r2(core[tr], core[te], tail[tr], tail[te])}
        row["shuffled"] = []
        for p in perms:
            ts = tail[p]
            row["shuffled"].append({**knn_r2(nbr, ts[tr], ts[te]), "ols": ols_r2(core[tr], core[te], ts[tr], ts[te]),
                                    "mlp": mlp_r2(core[tr], core[te], ts[tr], ts[te])})
        print(f"  R^2 true {row['true']}", flush=True)
        print(f"  R^2 shuffled knn5 {[round(s['knn5'], 4) for s in row['shuffled']]} "
              f"mlp {[round(s['mlp'], 4) for s in row['shuffled']]}", flush=True)

        batch, _ = windows(tok, text)
        seen = np.array(sorted(set(batch.flatten().tolist()) & set(range(n))))
        cc, ct = unit(core[seen]), unit(tail[seen])
        cos_core, cos_tail = cc @ cc.T, ct @ ct.T
        row["seen_tokens"] = int(len(seen))
        row["geometry_true"] = spearman_upper(cos_core, cos_tail)
        row["geometry_shuffled"] = []
        for sd in PERM_SEEDS:
            p = np.random.default_rng(sd).permutation(len(seen))
            cs = ct[p]
            row["geometry_shuffled"].append(spearman_upper(cos_core, cs @ cs.T))
        print(f"  geometry Spearman true {row['geometry_true']:.4f} shuffled "
              f"{[round(v, 4) for v in row['geometry_shuffled']]}", flush=True)
        rows.append(row)
        (RESULTS / "nonlinear_echo.json").write_text(json.dumps(rows, indent=2) + "\n")

    def band(vals):
        return f"{np.mean(vals):+.4f} [{min(vals):+.4f}, {max(vals):+.4f}]"

    lines = ["# Is the input tail a nonlinear echo of the core? (Pythia 1.4B-12B)", "",
             "Held-out R^2 of the tail predicted from the core (pooled over tail coordinates). Shuffled: mean",
             "[min, max] over 5 permutations. Geometry: Spearman correlation of pairwise cosines, core vs tail,",
             "over the evaluation-text tokens.", "",
             "| Model | core / tail dims | kNN-5 R^2 | shuffled | kNN-20 R^2 | shuffled | MLP R^2 | shuffled | OLS R^2 | shuffled |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, sh = r["true"], r["shuffled"]
        cells = []
        for key in ("knn5", "knn20", "mlp", "ols"):
            cells += [f"{t[key]:+.4f}", band([s[key] for s in sh])]
        lines.append(f"| {r['repo'].split('/')[1]} | {r['k_core']} / {r['d'] - r['k_core']} | " + " | ".join(cells) + " |")
    lines += ["", "| Model | tokens | geometry Spearman (true) | shuffled |", "|---|---|---|---|"]
    lines += [f"| {r['repo'].split('/')[1]} | {r['seen_tokens']} | {r['geometry_true']:+.4f} | "
              f"{band(r['geometry_shuffled'])} |" for r in rows]
    (RESULTS / "nonlinear_echo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
