"""Identity information of the Pythia embedding tables (definitions.md, section 1).

For each Pythia model in models.json (final checkpoint), each table (input,
output), each symbol distribution p (dictionary view: uniform over the
tokenizer's real tokens; text view: token frequencies in the phase-2
evaluation text, WikiText-103 test, first 16 x 1024 tokens), and each
resolution epsilon, estimate along k:

  I(k)    = I(V; Y_k)      bits of token identity
  I_B(k)  = I(G; Y_k)      between surface-variant classes
  I_W(k)  = I(k) - I_B(k)  within classes
  C(k)    capacity ceiling from the spectrum; efficiency I/C
  k_B50, k_W50  half-fill dimensions (definitions.md 1.3): smallest k at
          which I_B, or I_W, reaches half its own value at k = d

Estimator (definitions.md 1.5): draw v ~ p and z ~ N(0, sigma^2 I_d) once
(fixed seed); for each k use the first k coordinates of both, and the exact
posterior over all symbols. The same draws are used at every k.

Usage: .venv/bin/python identity_info.py [model-name ...]
Writes results/identity_info.json, results/identity_info.md and
results/identity_info.png.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from measure import OUT, RESULTS, TABLES, fetch, tokenizer_size
from measure_init import load_tables
from project import eval_text, windows

HERE = Path(__file__).resolve().parent
SEED = 0
SAMPLES = 4000
BATCH = 250
EPSILONS = [2.0, 4.0, 8.0]  # 0.25-1.0 saturate at once on pythia-70m (I(8) > H(V)/2)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LN2 = math.log(2)


def k_grid(d):
    ks = [8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048,
          2560, 3072, 3584, 4096, 4608, 5120]
    return [k for k in ks if k < d] + [d]


def surface_class(tokenizer, n):
    labels, ids = [], {}
    for v in range(n):
        key = tokenizer.decode([v]).strip().lower()
        key = key if key else f"<empty:{v}>"
        labels.append(ids.setdefault(key, len(ids)))
    return np.array(labels)


def entropy_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def measure_view(w, p, cls, ks, rng):
    """All estimates for one table and one symbol distribution p (over rows of w)."""
    keep = p > 0
    w, p, cls = w[keep], p[keep] / p[keep].sum(), cls[keep]
    # relabel classes to 0..G-1 and the class distribution
    _, cls = np.unique(cls, return_inverse=True)
    pg = np.bincount(cls, weights=p)
    mu = p @ w
    c = w - mu
    cov = (c * p[:, None]).T @ c
    lam, vec = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, vec = np.clip(lam[order], 0, None), vec[:, order]
    x = c @ vec  # coordinates in principal directions, columns in order
    d = w.shape[1]
    s = math.sqrt(lam.sum() / d)
    hv, hg = entropy_bits(p), entropy_bits(pg)

    xs = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    logp = torch.tensor(np.log(p), dtype=torch.float32, device=DEVICE)
    cls_t = torch.tensor(cls, device=DEVICE)
    draws = rng.choice(len(p), SAMPLES, p=p)
    z_unit = rng.standard_normal((SAMPLES, d))

    out = {"H_V": hv, "H_G": hg, "classes": int(len(pg)), "symbols": int(len(p)), "scale_s": s, "eps": {}}
    for eps in EPSILONS:
        sigma = eps * s
        rows = []
        for k in ks:
            hvy, hgy = [], []
            xk = xs[:, :k]
            xk_sq = (xk * xk).sum(1)
            for b in range(0, SAMPLES, BATCH):
                idx = draws[b:b + BATCH]
                y = xk[idx] + sigma * torch.tensor(z_unit[b:b + BATCH, :k], dtype=torch.float32, device=DEVICE)
                d2 = (y * y).sum(1, keepdim=True) - 2 * y @ xk.T + xk_sq[None, :]
                logits = logp[None, :] - d2 / (2 * sigma * sigma)
                logpost = logits - torch.logsumexp(logits, dim=1, keepdim=True)
                ti = torch.tensor(idx, device=DEVICE)
                hvy.append((-logpost[torch.arange(len(idx), device=DEVICE), ti] / LN2).cpu())
                # class posterior: sum of posterior mass within each class
                post = logpost.exp()
                pgy = torch.zeros(len(idx), len(pg), device=DEVICE).index_add_(1, cls_t, post)
                tg = cls_t[ti]
                hgy.append((-torch.log(pgy[torch.arange(len(idx), device=DEVICE), tg].clamp_min(1e-30)) / LN2).cpu())
            hvy, hgy = torch.cat(hvy).numpy(), torch.cat(hgy).numpy()
            i_all, i_b = hv - hvy.mean(), hg - hgy.mean()
            cap = float(0.5 * np.log2(1 + lam[:k] / sigma ** 2).sum())
            rows.append({"k": k, "I": float(i_all), "I_se": float(hvy.std() / math.sqrt(SAMPLES)),
                         "I_B": float(i_b), "I_B_se": float(hgy.std() / math.sqrt(SAMPLES)),
                         "I_W": float(i_all - i_b), "C": cap, "eff": float(i_all / cap) if cap > 0 else None})
        half = lambda key: next(r["k"] for r in rows if r[key] >= rows[-1][key] / 2)
        out["eps"][str(eps)] = {"sigma": sigma, "curve": rows,
                                "k_B50": half("I_B"), "k_W50": half("I_W")}
    return out


def main():
    spec = json.loads((HERE / "models.json").read_text())
    wanted = set(sys.argv[1:])
    text = eval_text()
    path = RESULTS / "identity_info.json"
    results = {r["repo"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    for m in spec["models"]:
        name = m["repo"].split("/")[1]
        if wanted and name not in wanted:
            continue
        repo, rev = m["repo"], m["revision"]
        print(repo, flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        tables = load_tables(repo, rev)
        cls = surface_class(tokenizer, n)
        batch, _ = windows(tokenizer, text)
        counts = np.bincount(batch.flatten().numpy(), minlength=n)[:n].astype(float)
        views = {"dictionary": np.full(n, 1.0 / n), "text": counts / counts.sum()}
        row = {"repo": repo, "revision": rev, "d": tables["input"].shape[1], "V": n}
        for side in TABLES:
            w = tables[side][:n]
            for view, p in views.items():
                rng = np.random.default_rng(SEED)
                row[f"{side}/{view}"] = measure_view(w, p, cls, k_grid(w.shape[1]), rng)
                e = row[f"{side}/{view}"]["eps"]["4.0"]["curve"][-1]
                print(f"  {side:6s} {view:10s} I(d) at eps 4 = {e['I']:.2f} of "
                      f"{row[f'{side}/{view}']['H_V']:.2f} bits (I_B {e['I_B']:.2f}, I_W {e['I_W']:.2f})", flush=True)
        results[repo] = row
        path.write_text(json.dumps(list(results.values()), indent=2) + "\n")
    write_report([results[m["repo"]] for m in spec["models"] if m["repo"] in results])


def write_report(rows):
    lines = ["# Identity information of the Pythia embedding tables", "",
             "definitions.md section 1. Classes: surface variants. I in bits; k_B50, k_W50 = half-fill dimensions.", ""]
    for view in ("dictionary", "text"):
        for eps in EPSILONS:
            lines += [f"## {view} view, epsilon = {eps}", "",
                      "| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                for side in TABLES:
                    v = r[f"{side}/{view}"]
                    e = v["eps"][str(eps)]
                    cur = e["curve"]
                    last = cur[-1]
                    half = next(c["k"] for c in cur if c["I"] >= last["I"] / 2)
                    lines.append(f"| {r['repo'].split('/')[1]} | {r['d']} | {side} | {v['H_V']:.2f} | "
                                 f"{last['I']:.2f} | {last['I_B']:.2f} | {last['I_W']:.2f} | {last['eff']:.3f} | "
                                 f"{half} | {e['k_B50']} | {e['k_W50']} |")
            lines.append("")
    (RESULTS / "identity_info.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for row_i, view in enumerate(("dictionary", "text")):
        ax_i, ax_b, ax_w = axes[row_i]
        for r in rows:
            cur = r[f"input/{view}"]["eps"]["4.0"]["curve"]
            ks = [c["k"] for c in cur]
            lbl = f"d={r['d']}"
            ax_i.plot(ks, [c["I"] for c in cur], marker=".", label=lbl)
            ax_b.plot(ks, [c["I_B"] / cur[-1]["I_B"] for c in cur], marker=".", label=lbl)
            ax_w.plot(ks, [c["I_W"] / cur[-1]["I_W"] for c in cur], marker=".", label=lbl)
        for ax, t in ((ax_i, "I(k): token identity (bits)"), (ax_b, "I_B(k) / I_B(d): between classes"),
                      (ax_w, "I_W(k) / I_W(d): within classes")):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("k (top principal directions)")
            ax.set_ylabel("bits" if ax is ax_i else "share of value at k = d")
            ax.set_title(f"input table, {view} view, eps = 4: {t}", fontsize=9)
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "identity_info.png", dpi=120)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
