"""Add-on to Measurement 1: Gaussian capacity of the core and of the tail.

Same models, basis, core and tail as tail_identity.py (Pythia 1.4B, 2.8B,
6.9B, 12B input tables; unweighted principal directions of the centered
real rows; core = top k*(0.1); tail = the rest). For eps 2, 4, 8 and the
pinned eps_res (pinned_noise.json), with sigma = eps * s and s from the full
table:

  C(block) = sum_i 0.5 log2(1 + lambda_i / sigma^2)

over the block's principal directions, lambda_i = variance of all real rows
along direction i (eigenvalues of C^T C / n, as in tail_identity.py). This is
the capacity of a Gaussian channel with the block's power spectrum, the
ceiling for any codebook with those second moments.

Reported beside the measured I(core), I(tail) and H(V) from
results/tail_identity.json, for both of its views. The surface view is
uniform over all real tokens, so its second moments are the lambda_i above.
The role view is uniform over the tokens with a lexicon role, a subset; for
it a second capacity is also reported, C_members, from the eigenvalues of
those members' own covariance within each block (centered on their mean),
so the measured I can be read against the moments of the codebook it used.
No new draws: the measured values are read from the saved results.

Usage: NLTK_DATA=out/nltk_data .venv/bin/python tail_capacity.py
Writes results/tail_capacity.json and results/tail_capacity.md.
"""

import json
from pathlib import Path

import numpy as np

from identity_roles import as_codes, build_lexicon, dictionary_classes, full_test_text, tag_spans
from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables
from tail_identity import K_CORE
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
EPS_KEYS = ("2", "4", "8", "res")


def capacity(lam, sigma):
    return float(0.5 * np.log2(1 + np.clip(lam, 0, None) / sigma ** 2).sum())


def main():
    spec = json.loads((HERE / "models.json").read_text())
    measured = {r["repo"]: r for r in json.loads((RESULTS / "tail_identity.json").read_text())}
    full = full_test_text()
    lexicon = build_lexicon(tag_spans(full), full)
    rows = []
    for mdl in spec["models"]:
        name = mdl["repo"].split("/")[1]
        if name not in K_CORE:
            continue
        repo, rev, k = mdl["repo"], mdl["revision"], K_CORE[name]
        prev = measured[repo]
        assert prev["revision"] == rev and prev["k_core"] == k
        print(f"{repo}  core {k}", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        c = w - w.mean(0)
        lam, vec = np.linalg.eigh(c.T @ c / n)
        order = np.argsort(lam)[::-1]
        lam, vec = lam[order], vec[:, order]
        d = len(lam)
        s = float(np.sqrt((c * c).sum() / n / d))
        assert abs(s - prev["s"]) < 1e-6 * max(1.0, s)
        assert abs(lam.sum() / d - s * s) < 1e-6 * s * s
        x = c @ vec
        rnames, _ = dictionary_classes(tok, n, lexicon)
        members = np.flatnonzero(as_codes(rnames) >= 0)
        xm = x[members] - x[members].mean(0)
        lam_m = {"core": np.linalg.eigvalsh(xm[:, :k].T @ xm[:, :k] / len(members)),
                 "tail": np.linalg.eigvalsh(xm[:, k:].T @ xm[:, k:] / len(members))}
        lam_b = {"core": lam[:k], "tail": lam[k:]}
        row = {"repo": repo, "revision": rev, "d": d, "k_core": k, "s": s, "role_members": int(len(members)),
               "power": {b: float(lam_b[b].sum()) for b in lam_b},
               "power_members": {b: float(lam_m[b].sum()) for b in lam_m}, "eps": {}}
        for key in EPS_KEYS:
            eps = prev["eps"][key]
            sigma = eps * s
            e = {"eps": eps, "sigma": sigma,
                 "C": {b: capacity(lam_b[b], sigma) for b in lam_b},
                 "C_members": {b: capacity(lam_m[b], sigma) for b in lam_m}}
            for view in ("surface", "role"):
                v = prev["results"][f"{view}/{key}"]
                e[view] = {"H_V": v["H_V"], **{f"I_{b}": v[b]["I"] for b in ("core", "tail")},
                           **{f"I_{b}_se": v[b]["I_se"] for b in ("core", "tail")}}
            row["eps"][key] = e
            print(f"  eps {key:3s} ({eps:.3g}) C core {e['C']['core']:.2f} tail {e['C']['tail']:.2f} | "
                  f"surface I core {e['surface']['I_core']:.2f} tail {e['surface']['I_tail']:.2f}", flush=True)
        rows.append(row)
    (RESULTS / "tail_capacity.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_report(rows)


def write_report(rows):
    f = lambda v, se: f"{v:.3f} ± {se:.3f}"
    lines = ["# Gaussian capacity of the input core and tail (Pythia 1.4B-12B)", "",
             "C = sum_i 0.5 log2(1 + lambda_i / sigma^2) over each block's unweighted principal directions,",
             "lambda_i over all real rows; sigma = eps * s, s from the full table. Measured I from",
             "results/tail_identity.json (4,000 draws, ± one standard error). Bits.", "",
             "## Surface view (uniform over all real tokens)", "",
             "| Model | eps | H(V) | C(core) | I(core) | C(tail) | I(tail) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        for key in EPS_KEYS:
            e, v = r["eps"][key], r["eps"][key]["surface"]
            lbl = key if key != "res" else f"res {e['eps']:.3g}"
            lines.append(f"| {r['repo'].split('/')[1]} | {lbl} | {v['H_V']:.3f} | {e['C']['core']:.2f} | "
                         f"{f(v['I_core'], v['I_core_se'])} | {e['C']['tail']:.2f} | {f(v['I_tail'], v['I_tail_se'])} |")
    lines += ["", "## Role view (uniform over tokens with a lexicon role)", "",
              "C_members: eigenvalues of the role tokens' own covariance within each block.", "",
              "| Model | eps | H(V) | C(core) | C_members(core) | I(core) | C(tail) | C_members(tail) | I(tail) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        for key in EPS_KEYS:
            e, v = r["eps"][key], r["eps"][key]["role"]
            lbl = key if key != "res" else f"res {e['eps']:.3g}"
            lines.append(f"| {r['repo'].split('/')[1]} | {lbl} | {v['H_V']:.3f} | {e['C']['core']:.2f} | "
                         f"{e['C_members']['core']:.2f} | {f(v['I_core'], v['I_core_se'])} | {e['C']['tail']:.2f} | "
                         f"{e['C_members']['tail']:.2f} | {f(v['I_tail'], v['I_tail_se'])} |")
    lines += ["", "## Block power (sum of lambda_i)", "",
              "| Model | core / tail dims | core power | tail power | tail share | role members |",
              "|---|---|---|---|---|---|"]
    for r in rows:
        pc, pt = r["power"]["core"], r["power"]["tail"]
        lines.append(f"| {r['repo'].split('/')[1]} | {r['k_core']} / {r['d'] - r['k_core']} | {pc:.4f} | {pt:.4f} | "
                     f"{pt / (pc + pt):.4f} | {r['role_members']} |")
    (RESULTS / "tail_capacity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
