"""Measurement 4: the 99% point of identity in the text view, at the pinned noise.

Eight Pythia input tables, surface-variant classes, the estimator, seed
(identity_info.SEED) and k grid of identity_pinned.py, with:
  p = token frequencies in the phase-2 evaluation text (WikiText-103 test,
      first 16 x 1,024 tokens), built exactly as identity_info.py's text view.
  noise held fixed in absolute terms: sigma_res = eps_res * s_full and
      sigma_prec = eps_prec * s_full (pinned_noise.json); passed to the
      estimator as eps_text = sigma / s_text, where s_text = sqrt(tr Sigma_p / d)
      is the text view's own scale (trace, so the same in either basis).
  basis (A), primary: the unweighted principal directions (the squash
      test's), passed explicitly; (B), secondary: the p-weighted principal
      directions (measure_view's default for p).
The k grid is identity_info.k_grid(d) plus k*(0.01) itself, so I(k*) is
measured directly. k*(0.01) from the paper's table: 1b 1,984; 1.4b 1,664;
2.8b 1,664; 6.9b 2,816; 12b 3,072. For 70m, 160m and 410m the paper says
"more than 95% of d" (the phase-2 sweep, in steps of 5-10% of d, needed
every direction); k* = d is used for them.

Reported per model and basis, at sigma_res and at the fp16 floor: H(V),
I(d) (bits and share of H(V)), k95, k99, k99.9, I(k*)/I(d), the conditional
gains I(d) - I(k*) and I(d) - I(k99) with standard errors from shared
draws, and a precision check: at k99 and the grid point before it, the
margin I(k) - 0.99 I(d) must be at least two standard errors from zero on
the right side. A model and basis that fail it are rerun with 16,000
draws, then 64,000 (and say so).

Decision rule (stated before running): in basis (A), k99 at sigma_res at
least one grid step below k*(0.01) -> the gap between the directions
identity needs and the directions the network needs holds in the text
view for that model; k99 >= k* -> it does not, for that model.

Usage: .venv/bin/python identity_pinned_text.py
Writes results/identity_pinned_text.json and results/identity_pinned_text.md.
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
K_STAR = {"pythia-1b": 1984, "pythia-1.4b": 1664, "pythia-2.8b": 1664, "pythia-6.9b": 2816, "pythia-12b": 3072}
DRAWS, DRAWS_RETRY, DRAWS_FINAL = 4000, 16000, 64000


def cut(curve, frac):
    top = curve[-1]["I"]
    return next(c["k"] for c in curve if c["I"] >= frac * top)


def summarize(r, eps_text, grid, k_star):
    e = r["eps"][str(eps_text)]
    curve = e["curve"]
    hv = {k: r["_hv"][(eps_text, k)] for k in grid}
    d, H = grid[-1], r["H_V"]
    n = len(hv[d])
    k99 = cut(curve, 0.99)

    def gain(k):
        x = hv[k] - hv[d]
        return float(x.mean()), float(x.std() / math.sqrt(n))

    def margin(k):
        x = 0.01 * H - hv[k] + 0.99 * hv[d]
        return float(x.mean()), float(x.std() / math.sqrt(n))

    i_d = curve[-1]["I"]
    i_ks = next(c["I"] for c in curve if c["k"] == k_star)
    prev = grid[grid.index(k99) - 1] if grid.index(k99) > 0 else None
    m99, mprev = margin(k99), margin(prev) if prev else None
    ok = m99[0] >= 2 * m99[1] and (mprev is None or mprev[0] <= -2 * mprev[1])
    return {"sigma": e["sigma"], "eps_text": eps_text, "I_d": i_d, "I_d_se": curve[-1]["I_se"],
            "I_d_share": i_d / H, "k95": cut(curve, 0.95), "k99": k99, "k999": cut(curve, 0.999),
            "I_kstar_share": i_ks / i_d, "gain_kstar": gain(k_star), "gain_k99": gain(k99),
            "margin_k99": m99, "k_before_k99": prev, "margin_before": mprev, "separates": ok,
            "curve": [{kk: c[kk] for kk in ("k", "I", "I_se")} for c in curve]}


def run(w, p, cls, grid, eps_pair, basis, draws):
    ii.EPSILONS = list(eps_pair)
    ii.SAMPLES = draws
    return ii.measure_view(w, p, cls, grid, np.random.default_rng(ii.SEED), basis=basis, keep_samples=True)


def main():
    spec = json.loads((HERE / "models.json").read_text())
    pinned = {r["repo"]: r for r in json.loads((RESULTS / "pinned_noise.json").read_text())}
    text = eval_text()
    rows = []
    for m in spec["models"]:
        repo, rev = m["repo"], m["revision"]
        name = repo.split("/")[1]
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        d = w.shape[1]
        cls = ii.surface_class(tok, n)
        batch, _ = windows(tok, text)
        counts = np.bincount(batch.flatten().numpy(), minlength=n)[:n].astype(float)
        p = counts / counts.sum()
        cu = w - w.mean(0)
        lam_u, vec_u = np.linalg.eigh(cu.T @ cu / n)
        basis_a = vec_u[:, np.argsort(lam_u)[::-1]]
        mu = p @ w
        s_text = math.sqrt(float((p * ((w - mu) ** 2).sum(1)).sum()) / d)
        s_full = pinned[repo]["s"]
        sig_res, sig_prec = pinned[repo]["eps_res"] * s_full, pinned[repo]["eps_prec"] * s_full
        eps_res_t, eps_prec_t = sig_res / s_text, sig_prec / s_text
        k_star = K_STAR.get(name, d)
        grid = sorted(set(ii.k_grid(d)) | {k_star})
        row = {"repo": repo, "revision": rev, "d": d, "k_star": k_star, "k_star_source":
               "paper table" if name in K_STAR else "d (paper: more than 95% of d)",
               "s_full": s_full, "s_text": s_text, "sigma_res": sig_res, "sigma_prec": sig_prec,
               "eps_res_full": pinned[repo]["eps_res"], "eps_res_text": eps_res_t,
               "eps_prec_full": pinned[repo]["eps_prec"], "eps_prec_text": eps_prec_t, "bases": {}}
        for label, basis in (("A_unweighted", basis_a), ("B_weighted", None)):
            draws = DRAWS
            r = run(w, p, cls, grid, (eps_res_t, eps_prec_t), basis, draws)
            res = summarize(r, eps_res_t, grid, k_star)
            for more in (DRAWS_RETRY, DRAWS_FINAL):
                if res["separates"]:
                    break
                draws = more
                r = run(w, p, cls, grid, (eps_res_t, eps_prec_t), basis, draws)
                res = summarize(r, eps_res_t, grid, k_star)
            prec = summarize(r, eps_prec_t, grid, k_star)
            holds = res["k99"] < k_star
            row["bases"][label] = {"H_V": r["H_V"], "symbols": r["symbols"], "draws": draws, "res": res,
                                   "prec": prec, "gap_holds": holds if label == "A_unweighted" else None}
            print(f"  {label}: H(V) {r['H_V']:.2f}, draws {draws}, I(d) {res['I_d']:.3f} ({res['I_d_share']:.4f}), "
                  f"k95/99/99.9 {res['k95']}/{res['k99']}/{res['k999']} vs k* {k_star}, "
                  f"I(k*)/I(d) {res['I_kstar_share']:.5f}, separates {res['separates']}", flush=True)
        rows.append(row)
        (RESULTS / "identity_pinned_text.json").write_text(json.dumps(rows, indent=2) + "\n")
    write_report(rows)


def write_report(rows):
    pm = lambda v: f"{v[0]:.4f} ± {v[1]:.4f}"
    lines = ["# 99% point of identity in the text view, at the pinned noise (Pythia input tables)", "",
             "p = token frequencies in the phase-2 text; noise fixed in absolute terms (sigma = eps * s_full);",
             "basis A = unweighted principal directions (squash test's), B = frequency-weighted. Gains in bits,",
             "± one standard error from shared draws.", "",
             "## Noise conversion", "",
             "| Model | d | s_full | s_text | eps_res (full) | eps_res (text) | eps_prec (full) | eps_prec (text) |",
             "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['repo'].split('/')[1]} | {r['d']} | {r['s_full']:.4f} | {r['s_text']:.4f} | {r['eps_res_full']:.3f} | "
              f"{r['eps_res_text']:.3f} | {r['eps_prec_full']:.2e} | {r['eps_prec_text']:.2e} |" for r in rows]
    for label, title in (("A_unweighted", "Basis A (primary): unweighted principal directions"),
                         ("B_weighted", "Basis B (secondary): frequency-weighted principal directions")):
        lines += ["", f"## {title}, at sigma_res", "",
                  "| Model | H(V) | I(d) | share | k95 | k99 | k99.9 | k*(0.01) | I(k*)/I(d) | I(d)-I(k*) | I(d)-I(k99) | draws | separates | gap holds |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            b = r["bases"][label]
            x = b["res"]
            gh = "" if b["gap_holds"] is None else ("yes" if b["gap_holds"] else "NO")
            lines.append(f"| {r['repo'].split('/')[1]} | {b['H_V']:.3f} | {x['I_d']:.3f} | {x['I_d_share']:.4f} | "
                         f"{x['k95']} | {x['k99']} | {x['k999']} | {r['k_star']} | {x['I_kstar_share']:.5f} | "
                         f"{pm(x['gain_kstar'])} | {pm(x['gain_k99'])} | {b['draws']} | {x['separates']} | {gh} |")
        lines += ["", f"### {title}, at the fp16 floor", "",
                  "| Model | I(d) | share | k95 | k99 | k99.9 | I(k*)/I(d) |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            x = r["bases"][label]["prec"]
            lines.append(f"| {r['repo'].split('/')[1]} | {x['I_d']:.3f} | {x['I_d_share']:.4f} | {x['k95']} | "
                         f"{x['k99']} | {x['k999']} | {x['I_kstar_share']:.5f} |")
    lines += ["", "k*(0.01) for 70m, 160m, 410m is taken as d (the paper reports more than 95% of d)."]
    (RESULTS / "identity_pinned_text.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
