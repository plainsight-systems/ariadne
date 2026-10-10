"""Two checks of existing claims in the empty-dimensions paper (rules set before running).

A. Loss at identity's 99% point. The paper says that squashing the input table
   to k_0.99 (text view, squash-test basis, sigma_res; results/identity_pinned_text.json,
   basis A) costs more than 0.1 bits per byte from 410M up, but at 410M, 6.9B and 12B
   that was inferred from the nearest larger measured k. Here the squash test of
   project.py is run at k_0.99 itself, for all eight Pythia input tables (and at both
   768 and 1,024 for 2.8B, whose k_0.99 is unresolved between them).
   Rule: the claim holds for a model if L(k_0.99) - L_0 > 0.1 bits per byte.

B. Repeated tail interventions. semantic_core.py used one shuffle and one random draw
   per model; at 1.4B and 2.8B the margins over zeroing were 0.01 to 0.02 bits per
   byte. Here, for Pythia 1.4B and 2.8B (core = top k*(0.1) directions, as there):
   tail zeroed once; tail shuffled across tokens with permutation seeds 1-10; tail
   replaced by matched Gaussian values with seeds 101-110.
   Rule: "a wrong tail costs more than no tail" holds for a model and a kind
   (shuffled, random) if all ten draws cost more than zeroing; it fails if any draw
   costs less. Mean, min and max are reported either way.

Same evaluation text, windows and bits-per-byte as project.py; weights in fp32 at
the pinned revisions in models.json.

Usage: .venv/bin/python claim_checks.py
Writes results/claim_checks.json and results/claim_checks.md.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from measure import OUT, RESULTS, fetch, tokenizer_size
from project import DEVICE, bits_per_byte, eval_text, principal, windows
from semantic_core import K_CORE

HERE = Path(__file__).resolve().parent
SHUFFLE_SEEDS = list(range(1, 11))
RANDOM_SEEDS = list(range(101, 111))
TAIL_MODELS = ("pythia-1.4b", "pythia-2.8b")


def set_rows(param, n, table):
    param.data[:n] = torch.from_numpy(table).to(param.dtype).to(param.device)


def main():
    spec = json.loads((HERE / "models.json").read_text())
    k99 = {r["repo"]: r["bases"]["A_unweighted"]["res"]["k99"]
           for r in json.loads((RESULTS / "identity_pinned_text.json").read_text())}
    text = eval_text()
    out = RESULTS / "claim_checks.json"
    rows = []
    for m in spec["models"]:
        repo, rev = m["repo"], m["revision"]
        name = repo.split("/")[1]
        if repo not in k99:
            continue
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT,
                                                   dtype=torch.float32).to(DEVICE).eval()
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        param = model.get_input_embeddings().weight
        original = param.data.clone()
        w = original[:n].cpu().double().numpy()
        mu, vec = principal(w)
        c = w - mu
        batch, nbytes = windows(tok, text)
        base = bits_per_byte(model, batch, nbytes)
        row = {"repo": repo, "revision": rev, "d": w.shape[1], "baseline_bpb": base, "A": [], "B": None}

        ks = [k99[repo]] + ([1024] if name == "pythia-2.8b" else [])
        for k in ks:
            vk = vec[:, :k]
            set_rows(param, n, mu + c @ vk @ vk.T)
            bpb = bits_per_byte(model, batch, nbytes)
            row["A"].append({"k": k, "bpb": bpb, "increase": bpb - base, "holds": bool(bpb - base > 0.1)})
            print(f"  A: k={k} +{bpb - base:.4f} bits/byte", flush=True)

        if name in TAIL_MODELS:
            k = K_CORE[name]
            core_b, tail_b = vec[:, :k], vec[:, k:]
            core, tail = c @ core_b, c @ tail_b
            zero = core @ core_b.T
            set_rows(param, n, mu + zero)
            z = bits_per_byte(model, batch, nbytes)
            shuf, rand = [], []
            for sd in SHUFFLE_SEEDS:
                perm = np.random.default_rng(sd).permutation(n)
                set_rows(param, n, mu + zero + tail[perm] @ tail_b.T)
                shuf.append(bits_per_byte(model, batch, nbytes))
            for sd in RANDOM_SEEDS:
                g = np.random.default_rng(sd).standard_normal(tail.shape) * tail.std(axis=0)
                set_rows(param, n, mu + zero + g @ tail_b.T)
                rand.append(bits_per_byte(model, batch, nbytes))
            row["B"] = {"k_core": k, "zeroed": z, "shuffled": shuf, "random": rand,
                        "shuffled_holds": bool(min(shuf) > z), "random_holds": bool(min(rand) > z)}
            print(f"  B: zeroed {z:.4f}; shuffled {min(shuf):.4f}-{max(shuf):.4f}; "
                  f"random {min(rand):.4f}-{max(rand):.4f}", flush=True)
        param.data.copy_(original)
        rows.append(row)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()
    write_report(rows)


def write_report(rows):
    lines = ["# Checks of two claims (Pythia input tables; rules in claim_checks.py)", "",
             "## A. Loss at identity's 99% point (text view, squash-test basis, sigma_res)", "",
             "| Model | d | baseline bits/byte | k | increase | > 0.1 |", "|---|---|---|---|---|---|"]
    for r in rows:
        for a in r["A"]:
            lines.append(f"| {r['repo'].split('/')[1]} | {r['d']} | {r['baseline_bpb']:.4f} | {a['k']} | "
                         f"{a['increase']:.4f} | {a['holds']} |")
    lines += ["", "## B. Tail interventions, ten draws each", "",
              "| Model | core | zeroed | shuffled mean [min, max] | all shuffled > zeroed | random mean [min, max] | all random > zeroed |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        b = r["B"]
        if not b:
            continue
        f = lambda v: f"{np.mean(v):.4f} [{min(v):.4f}, {max(v):.4f}]"
        lines.append(f"| {r['repo'].split('/')[1]} | {b['k_core']} | {b['zeroed']:.4f} | {f(b['shuffled'])} | "
                     f"{b['shuffled_holds']} | {f(b['random'])} | {b['random_holds']} |")
    (RESULTS / "claim_checks.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
