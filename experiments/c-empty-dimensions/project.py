"""Experiment C, phase 2: is the room needed, or just filled?

For each Pythia model, squash one embedding table onto its top k principal
directions (rows -> mean + projection of the centered row onto the top k
directions of the trained rows; padding rows untouched), run the model on
held-out text, and measure loss. Sweep k. Input table and output table are
squashed separately, the other one left as trained.

Evaluation text: WikiText-103 test set (raw), pinned revision, the first
WINDOWS x CONTEXT tokens in non-overlapping windows. Loss is reported in bits
per byte of the predicted text (comparable across tokenizations) and the
same text is used for every k, so differences between k are paired.

k* (reported, not the only output): the smallest k whose bits per byte is
within TOL of the unmodified model. The whole curve is saved.

Usage: .venv/bin/python project.py [model-name ...]   (default: all in models.json)
Writes results/phase2.json, results/phase2.md, results/loss_vs_k.png.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from measure import OUT, RESULTS, tokenizer_size, fetch

HERE = Path(__file__).resolve().parent
DATASET = ("Salesforce/wikitext", "b08601e04326c79dfdd32d625aee71d232d685c3",
           "wikitext-103-raw-v1/test-00000-of-00001.parquet")
WINDOWS, CONTEXT = 16, 1024
FRACTIONS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
TOL = 0.01  # bits per byte
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def eval_text():
    path = hf_hub_download(DATASET[0], DATASET[2], revision=DATASET[1], repo_type="dataset", cache_dir=OUT)
    return "".join(pq.read_table(path).column("text").to_pylist())


def windows(tokenizer, text):
    ids = tokenizer(text)["input_ids"][: WINDOWS * CONTEXT]
    assert len(ids) == WINDOWS * CONTEXT, "evaluation text too short"
    batch = torch.tensor(ids).view(WINDOWS, CONTEXT)
    # bytes of the predicted tokens (positions 1..CONTEXT-1 of each window)
    nbytes = sum(len(tokenizer.decode(row[1:].tolist()).encode("utf-8")) for row in batch)
    return batch, nbytes


@torch.no_grad()
def bits_per_byte(model, batch, nbytes):
    nats = 0.0
    for row in batch:
        x = row.unsqueeze(0).to(DEVICE)
        logits = model(x).logits[0, :-1].float()
        nats += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
    return nats / math.log(2) / nbytes


def principal(w):
    """Mean and principal directions (descending) of the rows of w, float64."""
    mu = w.mean(axis=0)
    c = w - mu
    lam, vec = np.linalg.eigh(c.T @ c)
    return mu, vec[:, np.argsort(lam)[::-1]]


def sweep(model, param, n_real, batch, nbytes, d, ks=None):
    """Loss with param's real rows squashed to each k in ks (default: FRACTIONS of d)."""
    ks = ks or [max(1, round(frac * d)) for frac in FRACTIONS]
    original = param.data.clone()
    w = original[:n_real].cpu().double().numpy()
    mu, vec = principal(w)
    curve = []
    for k in ks:
        vk = vec[:, :k]
        squashed = mu + (w - mu) @ vk @ vk.T
        param.data[:n_real] = torch.from_numpy(squashed).to(param.dtype).to(param.device)
        bpb = bits_per_byte(model, batch, nbytes)
        curve.append({"k": k, "frac": k / d, "bpb": bpb})
        print(f"    k={k:5d} ({k / d:.2f} d)  {bpb:.4f} bits/byte", flush=True)
    param.data.copy_(original)
    return curve


def main():
    spec = json.loads((HERE / "models.json").read_text())
    wanted = set(sys.argv[1:])
    models = [m for m in spec["models"] if not wanted or m["repo"].split("/")[1] in wanted]
    text = eval_text()
    rows = []
    for m in models:
        repo, rev = m["repo"], m["revision"]
        print(f"{repo} on {DEVICE}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT,
                                                   dtype=torch.float32).to(DEVICE).eval()
        n_real = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        d = model.config.hidden_size
        batch, nbytes = windows(tokenizer, text)
        base = bits_per_byte(model, batch, nbytes)
        print(f"  unmodified: {base:.4f} bits/byte", flush=True)
        row = {"repo": repo, "revision": rev, "d": d, "layers": model.config.num_hidden_layers,
               "baseline_bpb": base, "eval_tokens": WINDOWS * CONTEXT, "eval_bytes": nbytes}
        tables = (("input", model.get_input_embeddings().weight), ("output", model.get_output_embeddings().weight))
        assert tables[0][1].data_ptr() != tables[1][1].data_ptr(), "tables are tied; sweep would squash both"
        for side, param in tables:
            print(f"  {side} table", flush=True)
            curve = sweep(model, param, n_real, batch, nbytes, d)
            within = [c for c in curve if c["bpb"] - base <= TOL]
            row[side] = {"curve": curve, "k_star": min(c["k"] for c in within) if within else None}
        rows.append(row)
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "phase2.json"
    previous = {r["repo"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    previous.update({r["repo"]: r for r in rows})
    order = [m["repo"] for m in spec["models"]]
    rows = [previous[r] for r in order if r in previous]
    out.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        "# Phase 2: squash a table onto its top k directions, measure loss (Pythia)",
        "",
        f"WikiText-103 test, first {WINDOWS} x {CONTEXT} tokens. k* = smallest k within "
        f"{TOL} bits/byte of the unmodified model.",
        "",
        "| Model | d | layers | baseline bits/byte | input k* | input k*/d | output k* | output k*/d |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ki, ko = r["input"]["k_star"], r["output"]["k_star"]
        lines.append(f"| {r['repo'].split('/')[1]} | {r['d']} | {r['layers']} | {r['baseline_bpb']:.4f} | "
                     f"{ki} | {ki / r['d']:.2f} | {ko} | {ko / r['d']:.2f} |")
    (RESULTS / "phase2.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for side, ax in zip(("input", "output"), axes):
        for r in rows:
            c = r[side]["curve"]
            ax.plot([p["frac"] for p in c], [p["bpb"] - r["baseline_bpb"] for p in c], marker=".", label=f"d={r['d']}")
        ax.axhline(TOL, color="grey", ls=":", lw=1)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("k / d (directions kept)")
        ax.set_title(f"{side} table squashed to k directions")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("loss increase (bits per byte)")
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_vs_k.png", dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
