"""99% point of identity information at the pinned noise (definitions.md 1.10).

Pythia input tables, dictionary view, surface-variant classes, at each
model's eps_res and eps_prec (pinned_noise.py), with the same estimator and
seeds as identity_info.py. The 99% point is the smallest k on the grid with
I(k) >= 0.99 I(d). For comparison it is also read off the existing eps = 2,
4, 8 curves (results/identity_info.json).

Usage: .venv/bin/python identity_pinned.py
Writes results/identity_pinned.json and results/identity_pinned.md.
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import identity_info as ii
from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables

HERE = Path(__file__).resolve().parent


def k99(curve):
    top = curve[-1]["I"]
    return next(c["k"] for c in curve if c["I"] >= 0.99 * top)


def main():
    spec = json.loads((HERE / "models.json").read_text())
    pinned = {r["repo"]: r for r in json.loads((RESULTS / "pinned_noise.json").read_text())}
    earlier = {r["repo"]: r for r in json.loads((RESULTS / "identity_info.json").read_text())}
    rows = []
    for m in spec["models"]:
        repo, rev = m["repo"], m["revision"]
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        ii.EPSILONS = [pinned[repo]["eps_res"], pinned[repo]["eps_prec"]]
        r = ii.measure_view(w, np.full(n, 1.0 / n), ii.surface_class(tok, n), ii.k_grid(w.shape[1]),
                            np.random.default_rng(ii.SEED))
        res, prec = (r["eps"][str(e)] for e in ii.EPSILONS)
        old = earlier[repo]["input/dictionary"]["eps"]
        row = {"repo": repo, "d": w.shape[1], "eps_res": ii.EPSILONS[0], "eps_prec": ii.EPSILONS[1],
               "res": {"k99": k99(res["curve"]), "I_d": res["curve"][-1]["I"], "curve": res["curve"],
                       "k_B50": res["k_B50"], "k_W50": res["k_W50"]},
               "prec": {"k99": k99(prec["curve"]), "I_d": prec["curve"][-1]["I"]},
               "earlier_k99": {e: k99(old[e]["curve"]) for e in ("2.0", "4.0", "8.0")},
               "H_V": r["H_V"]}
        print(f"  eps_res {row['eps_res']:.2f}: k99 {row['res']['k99']}  I(d) {row['res']['I_d']:.2f}/{row['H_V']:.2f}"
              f" | eps_prec: k99 {row['prec']['k99']} | earlier k99 {row['earlier_k99']}", flush=True)
        rows.append(row)
    (RESULTS / "identity_pinned.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# 99% point of identity information at the pinned noise (Pythia input, dictionary view)", "",
             "k99 = smallest k on the grid with I(k) >= 0.99 I(d).", "",
             "| Model | d | eps_res | k99 at eps_res | I(d) at eps_res | k99 at eps 2 | 4 | 8 | k99 at eps_prec |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['repo'].split('/')[1]} | {r['d']} | {r['eps_res']:.2f} | {r['res']['k99']} | "
              f"{r['res']['I_d']:.2f} of {r['H_V']:.2f} | {r['earlier_k99']['2.0']} | {r['earlier_k99']['4.0']} | "
              f"{r['earlier_k99']['8.0']} | {r['prec']['k99']} |" for r in rows]
    (RESULTS / "identity_pinned.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
