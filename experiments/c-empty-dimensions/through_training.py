"""Spread and crowding of pythia-2.8b's input table through training.

Observation, no model run. For step 0 and each checkpoint pinned in
models.json (weights picked by hash as in refine.py), read the input table
(real tokens only) and report:
  erank/d       share of d the centered rows spread over (measure.py)
  mean |cos|    over random token pairs, centered (interference.py)
  nn median     median |cos| to the nearest other token, centered
Writes results/through_training.json and .md.
"""

import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from interference import overlap, unit_centered
from measure import OUT, RESULTS, fetch, spread, tokenizer_size
from refine import checkpoint_dir

HERE = Path(__file__).resolve().parent
REPO = "EleutherAI/pythia-2.8b"
NAME = "gpt_neox.embed_in.weight"


def input_table(local, weights):
    files = list(weights)
    if files[0].endswith(".bin"):
        sd = torch.load(local / files[0], map_location="cpu", weights_only=True)
        return sd[NAME].float().numpy()
    for f in files:
        with safe_open(str(local / f), framework="numpy") as h:
            if NAME in h.keys():
                return h.get_tensor(NAME).astype(np.float64)
    raise KeyError(NAME)


spec = json.loads((HERE / "models.json").read_text())
main = next(m for m in spec["models"] if m["repo"] == REPO)
n = tokenizer_size(fetch(REPO, main["revision"], "tokenizer.json"))
labels = {"step0": main["step0_revision"], **spec["checkpoints"][REPO]}
rows = []
for label, rev in labels.items():
    local, weights = checkpoint_dir(REPO, label, rev, main["revision"])
    w = input_table(local, weights)[:n].astype(np.float64)
    s = spread(w)
    o = overlap(unit_centered(w), np.random.default_rng(0))
    rows.append({"label": label, "erank_frac": s["erank"] / w.shape[1], "k90_frac": s["k90"] / w.shape[1], **o})
    print(f"{label:11s} erank/d {rows[-1]['erank_frac']:.3f}  k90/d {rows[-1]['k90_frac']:.3f}  "
          f"mean|cos| {o['mean_abs_cos']:.4f}  nn median {o['nn_median']:.3f}", flush=True)
(RESULTS / "through_training.json").write_text(json.dumps(rows, indent=2) + "\n")
lines = ["# pythia-2.8b input table through training (centered)", "",
         "| Checkpoint | erank/d | k90/d | mean abs cos | nn median |", "|---|---|---|---|---|"]
lines += [f"| {r['label']} | {r['erank_frac']:.3f} | {r['k90_frac']:.3f} | {r['mean_abs_cos']:.4f} | {r['nn_median']:.3f} |" for r in rows]
(RESULTS / "through_training.md").write_text("\n".join(lines) + "\n")
