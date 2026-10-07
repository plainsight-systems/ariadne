"""Experiment C, phase 1: how much of d do trained embedding tables use?

For each model in models.json (pinned revisions), reads the input table
(gpt_neox.embed_in) and output table (embed_out) from model.safetensors,
drops padding rows the tokenizer never produces, centers the rows, and
measures the spread across the d dimensions:

  erank   entropy effective rank (Roy and Vetterli 2007): exp of the entropy
          of the normalized singular values. Equals d for a perfectly even
          spread, 1 for a single direction.
  pr      participation ratio of the variances: (sum lam)^2 / sum lam^2.
  k90/k99 directions needed for 90% / 99% of the variance.

The same numbers for a random Gaussian table of the same shape (rows scaled
to the trained rows' lengths, fixed seed) show what "all of d used" looks
like at this shape.

Usage: .venv/bin/python measure.py   (downloads into out/models/)
Writes results/phase1.json, results/phase1.md, results/spectra.png.
"""

import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "models"
RESULTS = HERE / "results"
SEED = 0
TABLES = {"input": "gpt_neox.embed_in.weight", "output": "embed_out.weight"}


def fetch(repo, revision, filename):
    return hf_hub_download(repo, filename, revision=revision, cache_dir=OUT)


def tokenizer_size(tokenizer_json):
    tok = json.loads(Path(tokenizer_json).read_text())
    ids = set(tok["model"]["vocab"].values()) | {t["id"] for t in tok.get("added_tokens", [])}
    return max(ids) + 1


def spread(x):
    """Spread statistics of the rows of x (n x d), after centering."""
    x = x - x.mean(axis=0, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(x.T @ x)[::-1], 0.0, None)  # variances, descending
    sig = np.sqrt(lam)
    p = sig / sig.sum()
    p = p[p > 0]
    cum = np.cumsum(lam) / lam.sum()
    return {
        "erank": float(np.exp(-(p * np.log(p)).sum())),
        "pr": float(lam.sum() ** 2 / (lam ** 2).sum()),
        "k90": int(np.searchsorted(cum, 0.90) + 1),
        "k99": int(np.searchsorted(cum, 0.99) + 1),
        "spectrum": (lam / lam[0]).tolist(),
    }


def random_like(x, rng):
    """Gaussian table of x's shape, each row scaled to the length of x's centered row."""
    xc = x - x.mean(axis=0, keepdims=True)
    g = rng.standard_normal(x.shape)
    g *= (np.linalg.norm(xc, axis=1) / np.linalg.norm(g, axis=1))[:, None]
    return g


def measure(repo, revision, rng):
    weights = fetch(repo, revision, "model.safetensors")
    config = json.loads(Path(fetch(repo, revision, "config.json")).read_text())
    n_real = tokenizer_size(fetch(repo, revision, "tokenizer.json"))
    row = {
        "repo": repo,
        "revision": revision,
        "d": config["hidden_size"],
        "V_table": config["vocab_size"],
        "V_tokenizer": n_real,
        "tied": bool(config.get("tie_word_embeddings", False)),
    }
    with safe_open(weights, framework="numpy") as f:
        for side, name in TABLES.items():
            w = f.get_tensor(name).astype(np.float64)
            trained, padding = w[:n_real], w[n_real:]
            s = spread(trained)
            r = spread(random_like(trained, rng))
            row[side] = {
                **{k: s[k] for k in ("erank", "pr", "k90", "k99")},
                "erank_frac": s["erank"] / row["d"],
                "random": {k: r[k] for k in ("erank", "pr", "k90", "k99")},
                "padding_rows": int(padding.shape[0]),
                "padding_mean_norm": float(np.linalg.norm(padding, axis=1).mean()) if len(padding) else None,
                "trained_mean_norm": float(np.linalg.norm(trained, axis=1).mean()),
                "spectrum": s["spectrum"],
            }
    return row


def write_markdown(rows, path):
    lines = [
        "# Phase 1: spread of trained embedding tables (Pythia)",
        "",
        "Trained rows only (padding rows dropped), centered. `erank/d` is the share of d the",
        "spread covers; the random table of the same shape shows the ceiling at that shape.",
        "",
        "| Model | d | V | Table | erank | erank/d | random erank | PR | k90 | k99 | random k90 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for side in TABLES:
            t = r[side]
            lines.append(
                f"| {r['repo'].split('/')[1]} | {r['d']} | {r['V_tokenizer']} | {side} | "
                f"{t['erank']:.0f} | {t['erank_frac']:.2f} | {t['random']['erank']:.0f} | "
                f"{t['pr']:.1f} | {t['k90']} | {t['k99']} | {t['random']['k90']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for side, ax in zip(TABLES, axes[:2]):
        for r in rows:
            spec = np.array(r[side]["spectrum"])
            ax.semilogy(np.arange(1, len(spec) + 1) / r["d"], spec, label=f"d={r['d']}")
        ax.set_title(f"{side} table: variance per direction (normalized)")
        ax.set_xlabel("direction index / d")
        ax.set_ylabel("variance / largest")
        ax.legend(fontsize=8)
    ax = axes[2]
    ds = [r["d"] for r in rows]
    for side, marker in (("input", "o"), ("output", "s")):
        ax.plot(ds, [r[side]["erank"] for r in rows], marker=marker, label=f"{side} (trained)")
    ax.plot(ds, [r["input"]["random"]["erank"] for r in rows], "k--", label="random, same shape")
    ax.plot(ds, ds, ":", color="grey", label="erank = d")
    ax.set_xlabel("d (embedding dimension)")
    ax.set_ylabel("entropy effective rank")
    ax.set_title("Used room vs given room (fixed V)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    spec = json.loads((HERE / "models.json").read_text())
    rng = np.random.default_rng(SEED)
    rows = []
    for m in spec["models"]:
        print(f"measuring {m['repo']} @ {m['revision'][:8]}", flush=True)
        rows.append(measure(m["repo"], m["revision"], rng))
    RESULTS.mkdir(exist_ok=True)
    slim = [{**r, **{s: {k: v for k, v in r[s].items() if k != "spectrum"} for s in TABLES}} for r in rows]
    (RESULTS / "phase1.json").write_text(json.dumps(slim, indent=2) + "\n")
    write_markdown(rows, RESULTS / "phase1.md")
    plot(rows, RESULTS / "spectra.png")
    print((RESULTS / "phase1.md").read_text())


if __name__ == "__main__":
    main()
