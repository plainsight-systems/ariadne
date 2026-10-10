"""Identity's 99% point on a fine grid, and Shannon's count against it (rule set before running).

Why: the comparison in capacity_count.py (observations 39-41) was made on the
measured grid, where seven of eight frequency-weighted intervals are the same
(128, 256], a factor of two, so it cannot tell Shannon's count from a
constant. This run measures the cut points finely and tests the count against
a constant, with the rule below committed before any value is computed.

Same eight Pythia input tables, surface-variant classes, estimator and seed
(identity_info.SEED) as identity_pinned.py and identity_pinned_text.py, at the
receiver's noise sigma_res = eps_res * s_full (pinned_noise.json). Two views:
  text B: p = token frequencies of the phase-2 text, frequency-weighted
      principal directions, eps = sigma_res / s_text (as identity_pinned_text.py).
  dictionary: p uniform over the vocabulary, its principal directions,
      eps = eps_res (as identity_pinned.py).
Grid: k = 16, 32, ..., 512 and d. Cut points k_q = min{k : I(k) >= q I(d)},
q = 0.95, 0.99, 0.999. Draws 4,000; if the 99% cut is not two standard errors
clear of 0.99 I(d) at k99 and at the grid point before it (shared draws), rerun
with 16,000, then 64,000, and say so. k95 and k99.9 get no separate check.

Shannon's count: n_flat = 2 H(T) / log2(1 + 1/eps^2), the k at which a table
whose every principal direction had its mean variance s^2 would have a Gaussian
ceiling C(k) equal to H(T).

Rule (stated before running), per view and per cut, never pooled:
  1. Per model: n_flat agrees with k_q if 0.8 <= n_flat / k_q <= 1.25.
  2. Against a constant: let L = mean over the eight models of
     |ln(n_flat / k_q)|, and L0 the same for the best single constant
     (the geometric mean of the eight k_q, which minimises it over constants
     to first order; L0 is computed exactly by a search over constants).
     The count carries information beyond a constant only if L < L0.
  The text-B view at q = 0.99 is the primary test. Both views and all three
  cuts are reported.

Usage: .venv/bin/python identity_fine.py
Writes results/identity_fine.json and results/identity_fine.md.
"""

import json
import math
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

import identity_info as ii
from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables
from project import eval_text, windows

HERE = Path(__file__).resolve().parent
DRAWS = (4000, 16000, 64000)
QS = (0.95, 0.99, 0.999)


def grid(d):
    return sorted({k for k in range(16, 513, 16) if k < d} | {d})


def cut(curve, q):
    top = curve[-1]["I"]
    return next(c["k"] for c in curve if c["I"] >= q * top)


def measure(w, p, cls, ks, eps):
    """Curve, cut points and the 99% separation check, raising draws until it separates."""
    for draws in DRAWS:
        ii.EPSILONS, ii.SAMPLES = [eps], draws
        r = ii.measure_view(w, p, cls, ks, np.random.default_rng(ii.SEED), keep_samples=True)
        curve = r["eps"][str(eps)]["curve"]
        hv = {k: r["_hv"][(eps, k)] for k in ks}
        d, H = ks[-1], r["H_V"]
        k99 = cut(curve, 0.99)

        def margin(k):  # I(k) - 0.99 I(d) on shared draws
            x = 0.01 * H - hv[k] + 0.99 * hv[d]
            return float(x.mean()), float(x.std() / math.sqrt(len(x)))

        i = ks.index(k99)
        m99, mprev = margin(k99), margin(ks[i - 1]) if i > 0 else None
        ok = m99[0] >= 2 * m99[1] and (mprev is None or mprev[0] <= -2 * mprev[1])
        if ok:
            break
    return {"H": r["H_V"], "draws": draws, "separates": ok, "margin_k99": m99, "margin_before": mprev,
            "I_d": curve[-1]["I"], **{f"k_{q}": cut(curve, q) for q in QS},
            "curve": [{kk: c[kk] for kk in ("k", "I", "I_se")} for c in curve]}


def n_flat(H, eps):
    return 2 * H / math.log2(1 + 1 / eps ** 2)


def verdict(nf, kq):
    nf, kq = np.array(nf), np.array(kq, dtype=float)
    L = float(np.abs(np.log(nf / kq)).mean())
    consts = np.exp(np.linspace(np.log(kq.min()), np.log(kq.max()), 20001))
    l0s = np.abs(np.log(consts[:, None] / kq[None, :])).mean(1)
    j = int(l0s.argmin())
    return {"agree": [bool(0.8 <= a / b <= 1.25) for a, b in zip(nf, kq)], "ratio": (nf / kq).tolist(),
            "L": L, "L0": float(l0s[j]), "best_constant": float(consts[j]), "beats_constant": L < float(l0s[j])}


def main():
    spec = json.loads((HERE / "models.json").read_text())
    pinned = {r["repo"]: r for r in json.loads((RESULTS / "pinned_noise.json").read_text())}
    text = eval_text()
    rows = []
    for m in spec["models"]:
        repo, rev = m["repo"], m["revision"]
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        d = w.shape[1]
        cls = ii.surface_class(tok, n)
        ks = grid(d)
        s_full, eps_res = pinned[repo]["s"], pinned[repo]["eps_res"]
        sigma = eps_res * s_full
        batch, _ = windows(tok, text)
        counts = np.bincount(batch.flatten().numpy(), minlength=n)[:n].astype(float)
        p = counts / counts.sum()
        mu = p @ w
        s_text = math.sqrt(float((p * ((w - mu) ** 2).sum(1)).sum()) / d)
        row = {"repo": repo, "revision": rev, "d": d, "sigma_res": sigma, "views": {}}
        for view, pv, eps in (("text_B", p, sigma / s_text), ("dictionary", np.full(n, 1.0 / n), eps_res)):
            res = measure(w, pv, cls, ks, eps)
            res["eps"], res["n_flat"] = eps, n_flat(res["H"], eps)
            row["views"][view] = res
            print(f"  {view:10s} eps {eps:.3f} draws {res['draws']} separates {res['separates']} "
                  f"k95/99/99.9 {res['k_0.95']}/{res['k_0.99']}/{res['k_0.999']} n_flat {res['n_flat']:.0f}",
                  flush=True)
        rows.append(row)
        (RESULTS / "identity_fine.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    tests = {}
    for view in ("text_B", "dictionary"):
        for q in QS:
            tests[f"{view}/{q}"] = verdict([r["views"][view]["n_flat"] for r in rows],
                                           [r["views"][view][f"k_{q}"] for r in rows])
    (RESULTS / "identity_fine.json").write_text(json.dumps({"rows": rows, "tests": tests}, indent=2) + "\n")
    write_report(rows, tests)


def write_report(rows, tests):
    names = [r["repo"].split("/")[1] for r in rows]
    lines = ["# Identity's cut points on a fine grid, and Shannon's count (Pythia input tables, sigma_res)", "",
             "Grid k = 16..512 in steps of 16, and d. n_flat = 2 H(T) / log2(1 + 1/eps^2). Rule in identity_fine.py,",
             "committed before running: agree if 0.8 <= n_flat / k_q <= 1.25; beats a constant if L < L0.", ""]
    for view, title in (("text_B", "Text view, frequency-weighted basis (primary)"), ("dictionary", "Dictionary view")):
        lines += [f"## {title}", "", "| Model | d | eps | n_flat | k95 | k99 | k99.9 | draws | separates |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for nm, r in zip(names, rows):
            x = r["views"][view]
            lines.append(f"| {nm} | {r['d']} | {x['eps']:.3f} | {x['n_flat']:.0f} | {x['k_0.95']} | {x['k_0.99']} | "
                         f"{x['k_0.999']} | {x['draws']} | {x['separates']} |")
        lines += ["", "| Cut | n_flat / k_q per model | agree (count) | L | L0 (best constant) | beats constant |",
                  "|---|---|---|---|---|---|"]
        for q in QS:
            t = tests[f"{view}/{q}"]
            lines.append(f"| {q} | {', '.join(f'{a:.2f}' for a in t['ratio'])} | {sum(t['agree'])} of 8 | "
                         f"{t['L']:.3f} | {t['L0']:.3f} ({t['best_constant']:.0f}) | {t['beats_constant']} |")
        lines.append("")
    lines.append("Models in order: " + ", ".join(names) + ".")
    (RESULTS / "identity_fine.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
