"""Shannon's 1949 count, turned around: how many directions identity can need at least.

Shannon 1949 (Communication in the Presence of Noise, p. 17, eq. 21): at most
(sqrt((P+N)/N))^n signals are distinguishable in n dimensions, so telling M
apart needs n >= 2 log2 M / log2(1 + P/N). Here, for the eight Pythia input
tables at the receiver's noise sigma_res = eps_res * s_full (pinned_noise.json):

  Flat count (inputs: H(T) and eps only):
      n_flat = 2 H(T) / log2(1 + 1/eps^2)
  with eps = sigma_res / s, s the view's own scale (mean variance per direction).

  Capacity count (the table's own spectrum):
      C(k) = 0.5 log2 det(Id + Sigma_k / sigma^2),
  Sigma_k = covariance under p of the coordinates in the top k directions of the
  view's basis; k_C(q) = min{k : C(k) >= q I(d)}. Because I(k) <= C(k) (Gaussian
  maximum entropy, covariance under the channel's own p and basis), the exact
  k_q >= k_C(q): a lower bound on the directions identity needs, from the
  spectrum and the noise alone, with no Monte Carlo. C(k) is computed at every
  k from 1 to d (Cholesky of Id + Sigma / sigma^2: its leading minors).

Views, as measured before:
  dictionary: p uniform over the vocabulary, unweighted principal directions
      (Sigma diagonal); measured k99 and I(d) from results/identity_pinned.json.
  text A: p = token frequencies of the phase-2 text, squash-test basis
      (Sigma not diagonal under p, so the log-det is used); measured k99 and
      I(d) from results/identity_pinned_text.json, basis A.
  text B: same p, frequency-weighted principal directions (diagonal);
      results/identity_pinned_text.json, basis B.
No new draws.

Usage: .venv/bin/python capacity_count.py
Writes results/capacity_count.json and results/capacity_count.md.
"""

import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables
from project import eval_text, windows

HERE = Path(__file__).resolve().parent
QS = (0.95, 0.99, 0.999)


def logdet_curve(cov, sigma):
    """C(k) for k = 1..d: 0.5 log2 det of the leading k x k block of Id + cov / sigma^2."""
    m = np.eye(len(cov)) + cov / sigma ** 2
    L = np.linalg.cholesky(m)
    return np.cumsum(np.log2(np.diag(L)))  # 0.5 log2 det = sum log2 L_ii


def first_k(curve, target):
    hit = np.flatnonzero(curve >= target)
    return int(hit[0]) + 1 if len(hit) else None


def counts(curve, i_d, H, eps):
    return {"n_flat": 2 * H / math.log2(1 + 1 / eps ** 2), "C_d": float(curve[-1]),
            **{f"kC_{q}": first_k(curve, q * i_d) for q in QS}, "kC_H": first_k(curve, H)}


def main():
    spec = json.loads((HERE / "models.json").read_text())
    pinned = {r["repo"]: r for r in json.loads((RESULTS / "pinned_noise.json").read_text())}
    dic = {r["repo"]: r for r in json.loads((RESULTS / "identity_pinned.json").read_text())}
    txt = {r["repo"]: r for r in json.loads((RESULTS / "identity_pinned_text.json").read_text())}
    text = eval_text()
    rows = []
    for m in spec["models"]:
        repo, rev = m["repo"], m["revision"]
        if repo not in txt:
            continue
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        d = w.shape[1]
        s_full = pinned[repo]["s"]
        sigma = pinned[repo]["eps_res"] * s_full
        row = {"repo": repo, "revision": rev, "d": d, "sigma_res": sigma, "views": {}}

        # dictionary view
        c = w - w.mean(0)
        lam_u, vec_u = np.linalg.eigh(c.T @ c / n)
        order = np.argsort(lam_u)[::-1]
        lam_u, basis_a = np.clip(lam_u[order], 0, None), vec_u[:, order]
        assert abs(math.sqrt(lam_u.sum() / d) - s_full) < 1e-6
        curve = np.cumsum(0.5 * np.log2(1 + lam_u / sigma ** 2))
        D = dic[repo]
        row["views"]["dictionary"] = {"eps": sigma / s_full, "H": D["H_V"], "I_d": D["res"]["I_d"],
                                      "k99_measured": D["res"]["k99"],
                                      **counts(curve, D["res"]["I_d"], D["H_V"], sigma / s_full)}

        # text view, p from the phase-2 text
        batch, _ = windows(tok, text)
        cnt = np.bincount(batch.flatten().numpy(), minlength=n)[:n].astype(float)
        keep = cnt > 0
        p, wk = cnt[keep] / cnt.sum(), w[keep]
        ck = wk - p @ wk
        cov_p = (ck * p[:, None]).T @ ck
        s_text = math.sqrt(np.trace(cov_p) / d)
        T = txt[repo]
        assert abs(s_text - T["s_text"]) < 1e-6
        eps_t = sigma / s_text
        cov_a = basis_a.T @ cov_p @ basis_a
        lam_b = np.clip(np.sort(np.linalg.eigvalsh(cov_p))[::-1], 0, None)
        for label, crv in (("text_A", logdet_curve(cov_a, sigma)),
                           ("text_B", np.cumsum(0.5 * np.log2(1 + lam_b / sigma ** 2)))):
            b = T["bases"]["A_unweighted" if label == "text_A" else "B_weighted"]
            res = b["res"]
            row["views"][label] = {"eps": eps_t, "H": b["H_V"], "I_d": res["I_d"],
                                   "k95_measured": res["k95"], "k99_measured": res["k99"],
                                   "k999_measured": res["k999"], **counts(crv, res["I_d"], b["H_V"], eps_t)}
        for v, x in row["views"].items():
            print(f"  {v:10s} eps {x['eps']:.3f} n_flat {x['n_flat']:.0f} kC99 {x['kC_0.99']} "
                  f"k99 measured {x['k99_measured']} C(d) {x['C_d']:.1f}", flush=True)
        rows.append(row)
        (RESULTS / "capacity_count.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_report(rows)


def write_report(rows):
    lines = ["# Shannon's 1949 count against the measured directions identity needs (Pythia input tables)", "",
             "At sigma_res. n_flat = 2 H(T) / log2(1 + 1/eps^2) (inputs: H(T) and eps only).",
             "k_C(q) = min k with C(k) >= q I(d), C(k) = 0.5 log2 det(Id + Sigma_k / sigma^2) from the table's",
             "own covariance in the view's basis: a lower bound on the exact k_q. Measured k_q on the grid",
             "(results/identity_pinned.json, results/identity_pinned_text.json).", ""]
    for v, title in (("dictionary", "Dictionary view (uniform p, unweighted basis), H(T) = 15.62"),
                     ("text_A", "Text view, squash-test basis"),
                     ("text_B", "Text view, frequency-weighted basis")):
        lines += [f"## {title}", "",
                  "| Model | d | eps | n_flat | k_C(0.95) | k_C(0.99) | k_C(0.999) | k99 measured | C(d) bits |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            x = r["views"][v]
            lines.append(f"| {r['repo'].split('/')[1]} | {r['d']} | {x['eps']:.3f} | {x['n_flat']:.0f} | "
                         f"{x['kC_0.95']} | {x['kC_0.99']} | {x['kC_0.999']} | {x['k99_measured']} | {x['C_d']:.1f} |")
        lines.append("")
    (RESULTS / "capacity_count.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
