"""Experiment C, phase 1b: which directions did training actually use?

A direction in the trained table can show variance only because it still
holds its random starting values. For each model, compare the trained table
with the same model at training step 0 (pinned in models.json):

  - Principal directions of the centered trained table (trained rows only).
  - Along each direction: trained variance, and step-0 variance of the same
    rows along the same direction. Their ratio says how much training grew
    that direction over its starting noise.
  - "Barely touched": directions where training less than doubled the
    variance (ratio < 2). Counted for ratio < 2 and ratio < 4.
  - erank of the change itself (trained minus step 0): the spread of what
    training wrote.

Usage: .venv/bin/python measure_init.py
Writes results/phase1b.json, results/phase1b.md, results/growth.png.
"""

import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from measure import OUT, RESULTS, TABLES, fetch, spread, tokenizer_size

HERE = Path(__file__).resolve().parent


def load_tables(repo, revision):
    """Both embedding tables at a revision, from a single or sharded safetensors file."""
    try:
        index = json.loads(Path(fetch(repo, revision, "model.safetensors.index.json")).read_text())
        files = {name: index["weight_map"][name] for name in TABLES.values()}
    except Exception:
        files = {name: "model.safetensors" for name in TABLES.values()}
    out = {}
    for side, name in TABLES.items():
        with safe_open(fetch(repo, revision, files[name]), framework="numpy") as f:
            out[side] = f.get_tensor(name).astype(np.float64)
    return out


def compare(trained, init):
    tc = trained - trained.mean(axis=0, keepdims=True)
    ic = init - init.mean(axis=0, keepdims=True)
    lam, vec = np.linalg.eigh(tc.T @ tc)
    order = np.argsort(lam)[::-1]
    lam, vec = np.clip(lam[order], 0, None), vec[:, order]
    init_var = np.einsum("ij,ij->j", vec, (ic.T @ ic) @ vec)
    ratio = lam / np.maximum(init_var, 1e-300)
    delta = spread(trained - init)
    return {
        "init_erank": spread(init)["erank"],
        "delta_erank": delta["erank"],
        "delta_k90": delta["k90"],
        "barely_touched_lt2": int((ratio < 2).sum()),
        "barely_touched_lt4": int((ratio < 4).sum()),
        "median_growth": float(np.median(ratio)),
        "growth": ratio.tolist(),
    }


def main():
    spec = json.loads((HERE / "models.json").read_text())
    rows = []
    for m in spec["models"]:
        repo = m["repo"]
        print(f"comparing {repo} trained vs step 0", flush=True)
        n_real = tokenizer_size(fetch(repo, m["revision"], "tokenizer.json"))
        d = json.loads(Path(fetch(repo, m["revision"], "config.json")).read_text())["hidden_size"]
        trained = load_tables(repo, m["revision"])
        init = load_tables(repo, m["step0_revision"])
        row = {"repo": repo, "d": d}
        for side in TABLES:
            row[side] = compare(trained[side][:n_real], init[side][:n_real])
        rows.append(row)

    RESULTS.mkdir(exist_ok=True)
    slim = [{**r, **{s: {k: v for k, v in r[s].items() if k != "growth"} for s in TABLES}} for r in rows]
    (RESULTS / "phase1b.json").write_text(json.dumps(slim, indent=2) + "\n")

    lines = [
        "# Phase 1b: trained table vs the same table at step 0 (Pythia)",
        "",
        "Growth = trained variance / step-0 variance along each principal direction of the",
        "trained table. Barely touched = directions where training less than doubled (or",
        "quadrupled) the starting variance.",
        "",
        "| Model | d | Table | median growth | barely touched (<2x) | (<4x) | share of d (<4x) | erank of change |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for side in TABLES:
            t = r[side]
            lines.append(
                f"| {r['repo'].split('/')[1]} | {r['d']} | {side} | {t['median_growth']:.1f} | "
                f"{t['barely_touched_lt2']} | {t['barely_touched_lt4']} | "
                f"{t['barely_touched_lt4'] / r['d']:.2f} | {t['delta_erank']:.0f} |"
            )
    (RESULTS / "phase1b.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for side, ax in zip(TABLES, axes):
        for r in rows:
            g = np.array(r[side]["growth"])
            ax.semilogy(np.arange(1, len(g) + 1) / r["d"], g, label=f"d={r['d']}")
        ax.axhline(1, color="grey", ls=":", lw=1)
        ax.set_title(f"{side} table: growth over step 0, per direction")
        ax.set_xlabel("direction index / d (trained table, largest first)")
        ax.set_ylabel("trained variance / step-0 variance")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "growth.png", dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
