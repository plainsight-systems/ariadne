"""Experiment C: how crowded are the token vectors? (d = 2048 models)

Observation, no model run: for each finished d = 2048 model in
cross_vocab.py, read the input and output tables (real tokens only), center
them, and measure overlap between token vectors:

  mean |cos|  over PAIRS random token pairs
  nn |cos|    for SAMPLE random tokens, the largest |cos| to any other token
              (how close each token's nearest neighbour is)

Each is compared with a random Gaussian table of the same shape (same V and
d, fixed seed): ratio > 1 means tokens overlap more than chance would give
in that room. Computed for all tokens, and for the tokens that occur in the
evaluation text (the first 70,000 characters of WikiText-103 test), a rough
stand-in for well-exposed tokens.

Usage: .venv/bin/python interference.py
Writes results/interference.json and results/interference.md.
"""

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cross_vocab import CHARS, MODELS
from measure import OUT, RESULTS
from project import eval_text

SEED = 0
PAIRS = 2_000_000
SAMPLE = 4_000


def unit_centered(w):
    w = w - w.mean(axis=0, keepdims=True)
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def overlap(u, rng, subset=None):
    """mean |cos| over random pairs and nearest-neighbour |cos|, rows of u unit length."""
    idx = np.arange(len(u)) if subset is None else np.asarray(subset)
    a, b = rng.choice(idx, PAIRS), rng.choice(idx, PAIRS)
    keep = a != b
    cos = np.abs(np.einsum("ij,ij->i", u[a[keep]], u[b[keep]]))
    probe = rng.choice(idx, min(SAMPLE, len(idx)), replace=False)
    sims = np.abs(u[probe] @ u[idx].T)
    sims[np.arange(len(probe)), np.searchsorted(idx, probe)] = 0  # drop self
    nn = sims.max(axis=1)
    return {"mean_abs_cos": float(cos.mean()), "nn_median": float(np.median(nn)),
            "nn_p90": float(np.percentile(nn, 90))}


def main():
    text = eval_text()[:CHARS]
    rows = []
    for repo, rev in MODELS:
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT, dtype=torch.float32)
        n = len(tok)
        seen = sorted(set(tok(text, add_special_tokens=False)["input_ids"]))
        rng = np.random.default_rng(SEED)
        rand = unit_centered(rng.standard_normal((n, model.config.hidden_size)))
        base_all, base_seen = overlap(rand, rng), overlap(rand, rng, seen)
        row = {"repo": repo, "revision": rev, "V": n, "d": model.config.hidden_size, "seen_tokens": len(seen),
               "random_all": base_all, "random_seen": base_seen}
        for side, w in (("input", model.get_input_embeddings().weight), ("output", model.get_output_embeddings().weight)):
            u = unit_centered(w.detach()[:n].double().numpy())
            row[side] = {"all": overlap(u, rng), "seen": overlap(u, rng, seen)}
        rows.append(row)
        del model

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "interference.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = [
        "# Crowding of token vectors (d = 2048, centered tables)",
        "",
        "x = ratio to a random table of the same V and d. mean |cos|: random token pairs.",
        "nn: median over sampled tokens of |cos| to the nearest other token.",
        "\"seen\": tokens that occur in the first 70,000 characters of WikiText-103 test.",
        "",
        "| Model | V | Table | mean abs cos (all) | x | nn median (all) | x | mean abs cos (seen) | x | nn median (seen) | x |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for side in ("input", "output"):
            a, s = r[side]["all"], r[side]["seen"]
            ra, rs = r["random_all"], r["random_seen"]
            lines.append(
                f"| {r['repo'].split('/')[1][:24]} | {r['V']:,} | {side} | "
                f"{a['mean_abs_cos']:.3f} | {a['mean_abs_cos'] / ra['mean_abs_cos']:.1f} | "
                f"{a['nn_median']:.3f} | {a['nn_median'] / ra['nn_median']:.1f} | "
                f"{s['mean_abs_cos']:.3f} | {s['mean_abs_cos'] / rs['mean_abs_cos']:.1f} | "
                f"{s['nn_median']:.3f} | {s['nn_median'] / rs['nn_median']:.1f} |")
    (RESULTS / "interference.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
