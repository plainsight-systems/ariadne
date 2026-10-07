"""Experiment C, step 2 start: same room (d = 2048), different vocabularies.

Runs the squash test (project.py) on finished models that all have
d = 2048 but different vocabularies and training data:

  TinyLlama 1.1B (3T tokens)   V = 32,000
  Pythia 1B and 1.4B (300B)    V = 50,277 (table padded to 50,304)
  OLMo-2 1B (about 4T)         V = 100,278 (table padded to 100,352)

Because tokenizers split text differently, every model reads the same span
of text: the first CHARS characters of the WikiText-103 test set (pinned in
project.py), tokenized by its own tokenizer and cut into windows of up to
CONTEXT tokens. Loss is in bits per byte of that span, so it is comparable
across vocabularies. Rows beyond the tokenizer's real vocabulary (padding)
are left untouched.

Usage: .venv/bin/python cross_vocab.py
Writes results/cross_vocab.json, .md, .png.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from measure import OUT, RESULTS
from project import DEVICE, TOL, eval_text, principal

MODELS = [
    ("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T", "59f6f375b26bde864a6ca194a9a3044570490064"),
    ("EleutherAI/pythia-1b", "f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2"),
    ("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"),
    ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf"),
]
CHARS = 70_000
CONTEXT = 1024
K_INPUT = list(range(256, 2048, 64)) + [2048]
K_OUTPUT = list(range(1024, 2048, 128)) + [2048]


def windows(tokenizer, text):
    """Token windows covering text; every token after a window's first is predicted."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [torch.tensor(ids[i:i + CONTEXT]) for i in range(0, len(ids), CONTEXT) if len(ids[i:i + CONTEXT]) > 1]


@torch.no_grad()
def bits_per_byte(model, wins, nbytes):
    nats = 0.0
    for w in wins:
        x = w.unsqueeze(0).to(DEVICE)
        logits = model(x).logits[0, :-1].float()
        nats += torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="sum").item()
    return nats / math.log(2) / nbytes


def sweep(model, param, n_real, wins, nbytes, ks):
    original = param.data.clone()
    w = original[:n_real].cpu().double().numpy()
    mu, vec = principal(w)
    curve = []
    for k in ks:
        vk = vec[:, :k]
        param.data[:n_real] = torch.from_numpy(mu + (w - mu) @ vk @ vk.T).to(param.dtype).to(param.device)
        bpb = bits_per_byte(model, wins, nbytes)
        curve.append({"k": k, "bpb": bpb})
        print(f"    k={k:5d}  {bpb:.4f} bits/byte", flush=True)
    param.data.copy_(original)
    return curve


def k_needed(curve, base, tol):
    """Smallest k within tol of base, interpolated between measured k."""
    pts = [(p["k"], p["bpb"] - base) for p in curve]
    if pts[0][1] <= tol:
        return float(pts[0][0])
    for (k0, e0), (k1, e1) in zip(pts, pts[1:]):
        if e0 > tol >= e1:
            return k0 + (k1 - k0) * (e0 - tol) / (e0 - e1)
    return None


def main():
    text = eval_text()[:CHARS]
    nbytes_text = len(text.encode("utf-8"))
    out = RESULTS / "cross_vocab.json"
    rows = {r["repo"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    for repo, rev in MODELS:
        print(f"{repo} on {DEVICE}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT,
                                                     dtype=torch.float32).to(DEVICE).eval()
        tables = {"input": model.get_input_embeddings().weight, "output": model.get_output_embeddings().weight}
        assert tables["input"].data_ptr() != tables["output"].data_ptr(), "tied tables"
        d = model.config.hidden_size
        assert d == 2048
        n_real = len(tokenizer)
        assert n_real <= tables["input"].shape[0]
        wins = windows(tokenizer, text)
        # bytes predicted: the whole span minus each window's unpredicted first token
        first_bytes = sum(len(tokenizer.decode([int(w[0])]).encode("utf-8")) for w in wins)
        nbytes = nbytes_text - first_bytes
        base = bits_per_byte(model, wins, nbytes)
        ntok = sum(len(w) for w in wins)
        print(f"  V={n_real} table rows={tables['input'].shape[0]} tokens={ntok} "
              f"bytes/token={nbytes_text / ntok:.2f} unmodified={base:.4f} bits/byte", flush=True)
        row = {"repo": repo, "revision": rev, "d": d, "V": n_real, "table_rows": int(tables["input"].shape[0]),
               "layers": model.config.num_hidden_layers, "eval_tokens": ntok, "eval_bytes": nbytes,
               "bytes_per_token": nbytes_text / ntok, "baseline_bpb": base}
        for side, ks in (("input", K_INPUT), ("output", K_OUTPUT)):
            print(f"  {side} table", flush=True)
            curve = sweep(model, tables[side], n_real, wins, nbytes, ks)
            row[side] = {"curve": curve, **{f"k_{t}": k_needed(curve, base, t) for t in (0.1, 0.03, 0.01, 0.003)}}
        rows[repo] = row
        out.write_text(json.dumps(list(rows.values()), indent=2) + "\n")
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    ordered = [rows[r] for r, _ in MODELS if r in rows]
    lines = [
        "# Same room, different vocabularies (d = 2048)",
        "",
        f"First {CHARS:,} characters of WikiText-103 test for every model. Input table: directions",
        "needed to stay within each tolerance (bits per byte) of the unmodified model, interpolated.",
        "",
        "| Model | V | trained on | bytes/token | bits/byte | k +0.1 | k +0.03 | k +0.01 | k +0.003 | output k +0.01 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    trained = {"TinyLlama": "3T", "pythia": "300B", "OLMo": "~4T"}
    for r in ordered:
        name = r["repo"].split("/")[1]
        t = next(v for k, v in trained.items() if k.lower() in name.lower())
        ki = r["input"]
        fmt = lambda v: f"{v:.0f}" if v is not None else "-"
        lines.append(f"| {name} | {r['V']:,} | {t} | {r['bytes_per_token']:.2f} | {r['baseline_bpb']:.3f} | "
                     f"{fmt(ki['k_0.1'])} | {fmt(ki['k_0.03'])} | {fmt(ki['k_0.01'])} | {fmt(ki['k_0.003'])} | "
                     f"{fmt(r['output']['k_0.01'])} |")
    (RESULTS / "cross_vocab.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for side, ax in zip(("input", "output"), axes):
        for r in ordered:
            c = r[side]["curve"]
            ax.plot([p["k"] for p in c], [p["bpb"] - r["baseline_bpb"] for p in c], marker=".",
                    label=f"{r['repo'].split('/')[1][:22]} (V={r['V']:,})")
        ax.axhline(TOL, color="grey", ls=":", lw=1)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("k (directions kept, d = 2048)")
        ax.set_ylabel("loss increase (bits per byte)")
        ax.set_title(f"{side} table")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "cross_vocab.png", dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
