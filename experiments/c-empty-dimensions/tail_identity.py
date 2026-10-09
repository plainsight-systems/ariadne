"""What the input table's tail resolves on its own, against structure-free baselines.

Measurement 1 of the 2026-10-09 request. Pythia 1.4B, 2.8B, 6.9B, 12B input
tables. Basis: principal directions of the centered real rows with uniform
weights (the squash test's basis, which equals the dictionary view's).
Core = top k*(0.1) directions (1,408 / 1,344 / 2,048 / 2,048); tail = the
rest. Noise sigma = eps * s with s from the FULL table (identity_info.py).

For each model and eps (2, 4, 8, and the pinned eps_res and eps_prec from
pinned_noise.py), on the same draws (fixed seeds), channels:
  core           Y = x_core + Z_core
  tail           Y = x_tail + Z_tail
  full           Y = x + Z
  tail shuffled  tail rows permuted across the view's own tokens (5 fixed
                 permutations), so the set of tail vectors is unchanged and
                 I(T;Y_tail) differs only by sampling error
  tail random    independent Gaussian tail with each tail direction's variance
Views:
  surface: uniform over all real tokens, classes = surface variants
  role:    uniform over tokens with a lexicon role (identity_roles.py), classes
           = 12-tag universal POS. Same basis and same s as the surface view
           (the subset changes the symbol distribution, not the coordinates).
Per channel: I = I(T;Y), I_B, I_W with standard errors from per-sample
terms; for the true channels R = I_core + I_tail - I_full with its error
from the same per-sample draws.

Decision rule (stated in the request before running): for each partition,
the true tail's I_B clearly above the shuffled tail's I_B (more than two
combined standard errors, for every permutation) -> the tail carries class
structure redundantly (reading b); within two combined standard errors of
every permutation -> token-specific without that structure (reading a);
otherwise mixed.

Usage: NLTK_DATA=out/nltk_data .venv/bin/python tail_identity.py
Writes results/tail_identity.json and results/tail_identity.md.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from identity_info import surface_class
from identity_roles import as_codes, build_lexicon, dictionary_classes, full_test_text, tag_spans
from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables

HERE = Path(__file__).resolve().parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LN2 = math.log(2)
SAMPLES, BATCH = 4000, 250
DRAW_SEED, RANDOM_SEED, PERM_SEEDS = 0, 100, [1, 2, 3, 4, 5]
ONLY_VIEW = None  # set by --only VIEW to recompute one view and keep the other from the saved results
K_CORE = {"pythia-1.4b": 1408, "pythia-2.8b": 1344, "pythia-6.9b": 2048, "pythia-12b": 2048}


def per_sample(x, logp, draws, z, sigma, cls_codes):
    """-log2 p(true symbol | y) and -log2 p(true class | y) for each draw. x: (n, m) torch."""
    x_sq = (x * x).sum(1)
    n_cls = int(cls_codes.max()) + 1
    hv, hg = [], []
    for b in range(0, len(draws), BATCH):
        idx = torch.as_tensor(draws[b:b + BATCH], device=DEVICE)
        y = x[idx] + sigma * z[b:b + BATCH]
        logits = logp[None, :] - ((y * y).sum(1, keepdim=True) - 2 * y @ x.T + x_sq[None, :]) / (2 * sigma * sigma)
        logpost = logits - torch.logsumexp(logits, 1, keepdim=True)
        r = torch.arange(len(idx), device=DEVICE)
        hv.append((-logpost[r, idx] / LN2).cpu())
        pg = torch.zeros(len(idx), n_cls, device=DEVICE).index_add_(1, cls_codes, logpost.exp())
        hg.append((-torch.log(pg[r, cls_codes[idx]].clamp_min(1e-30)) / LN2).cpu())
    return torch.cat(hv).double().numpy(), torch.cat(hg).double().numpy()


def summarize(hv, hg, H_V, H_G):
    n = len(hv)
    return {"I": H_V - hv.mean(), "I_se": hv.std() / n ** 0.5,
            "I_B": H_G - hg.mean(), "I_B_se": hg.std() / n ** 0.5,
            "I_W": (H_V - H_G) - (hv - hg).mean(), "I_W_se": (hv - hg).std() / n ** 0.5}


def run_view(xfull, k, s, eps, members, cls, tail_np, tail_rand):
    """All channels for one view (members: symbol indices with uniform p; cls: class codes of members)."""
    n = len(members)
    logp = torch.full((n,), -math.log(n), device=DEVICE)
    _, cls = np.unique(cls, return_inverse=True)
    H_V = math.log2(n)
    pg = np.bincount(cls) / n
    H_G = float(-(pg * np.log2(pg)).sum())
    cls_t = torch.as_tensor(cls, device=DEVICE)
    rng = np.random.default_rng(DRAW_SEED)
    draws = rng.integers(0, n, SAMPLES)
    z = torch.as_tensor(rng.standard_normal((SAMPLES, xfull.shape[1])), dtype=torch.float32, device=DEVICE)
    sigma = eps * s
    m = torch.as_tensor(members, device=DEVICE)
    X = xfull[m]
    out = {"H_V": H_V, "H_G": H_G, "symbols": n, "classes": int(len(pg))}
    raw = {}
    for name, xs, zs in (("core", X[:, :k], z[:, :k]), ("tail", X[:, k:], z[:, k:]), ("full", X, z),
                         ("tail_random", tail_rand[m], z[:, k:])):
        raw[name] = per_sample(xs, logp, draws, zs, sigma, cls_t)
        out[name] = summarize(*raw[name], H_V, H_G)
    out["tail_shuffled"] = []
    own = tail_np[members]
    for sd in PERM_SEEDS:
        ts = torch.as_tensor(own[np.random.default_rng(sd).permutation(n)], dtype=torch.float32, device=DEVICE)
        out["tail_shuffled"].append(summarize(*per_sample(ts, logp, draws, z[:, k:], sigma, cls_t), H_V, H_G))
    r = H_V - raw["core"][0] - raw["tail"][0] + raw["full"][0]
    out["R"], out["R_se"] = float(r.mean()), float(r.std() / SAMPLES ** 0.5)
    t, sh = out["tail"], out["tail_shuffled"]
    gaps = [(t["I_B"] - x["I_B"]) / math.hypot(t["I_B_se"], x["I_B_se"]) for x in sh]
    out["reading"] = "b" if all(g > 2 for g in gaps) else "a" if all(abs(g) <= 2 for g in gaps) else "mixed"
    out["I_B_gap_in_se"] = gaps
    return out


def main():
    spec = json.loads((HERE / "models.json").read_text())
    pinned = {r["repo"]: r for r in json.loads((RESULTS / "pinned_noise.json").read_text())}
    previous = {r["repo"]: r for r in json.loads((RESULTS / "tail_identity.json").read_text())} \
        if ONLY_VIEW else {}
    full = full_test_text()
    lexicon = build_lexicon(tag_spans(full), full)
    rows = []
    for mdl in spec["models"]:
        name = mdl["repo"].split("/")[1]
        if name not in K_CORE:
            continue
        repo, rev, k = mdl["repo"], mdl["revision"], K_CORE[name]
        print(f"{repo}  core {k}", flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        w = load_tables(repo, rev)["input"][:n]
        c = w - w.mean(0)
        lam, vec = np.linalg.eigh(c.T @ c / n)
        vec = vec[:, np.argsort(lam)[::-1]]
        x = c @ vec
        d = x.shape[1]
        s = float(np.sqrt((c * c).sum() / n / d))
        assert abs(s - pinned[repo]["s"]) < 1e-6 * max(1.0, s)
        xfull = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
        tail = x[:, k:]
        tail_rand = torch.as_tensor(np.random.default_rng(RANDOM_SEED).standard_normal(tail.shape) * tail.std(0),
                                    dtype=torch.float32, device=DEVICE)
        surface = surface_class(tok, n)
        rnames, coverage = dictionary_classes(tok, n, lexicon)
        rcodes = as_codes(rnames)
        role_members = np.flatnonzero(rcodes >= 0)
        views = {"surface": (np.arange(n), surface), "role": (role_members, rcodes[role_members])}
        eps_list = {"2": 2.0, "4": 4.0, "8": 8.0, "res": pinned[repo]["eps_res"], "prec": pinned[repo]["eps_prec"]}
        row = {"repo": repo, "revision": rev, "d": d, "k_core": k, "s": s, "role_coverage": coverage,
               "eps": {key: float(v) for key, v in eps_list.items()}, "results": {}}
        for key, eps in eps_list.items():
            for view, (members, cls) in views.items():
                if ONLY_VIEW and view != ONLY_VIEW:
                    row["results"][f"{view}/{key}"] = previous[repo]["results"][f"{view}/{key}"]
                    continue
                r = run_view(xfull, k, s, eps, members, cls, tail, tail_rand)
                row["results"][f"{view}/{key}"] = r
                print(f"  eps {key:4s} ({eps:.3g}) {view:7s} tail I {r['tail']['I']:.2f} I_B {r['tail']['I_B']:.3f} | "
                      f"shuffled I_B {np.mean([x['I_B'] for x in r['tail_shuffled']]):.3f} | random I_B "
                      f"{r['tail_random']['I_B']:.3f} | R {r['R']:.2f} | reading {r['reading']}", flush=True)
        rows.append(row)
        (RESULTS / "tail_identity.json").write_text(json.dumps(rows, indent=2, default=float) + "\n")
    write_report(rows)


def write_report(rows):
    f = lambda v, se: f"{v:.3f} ± {se:.3f}"
    lines = ["# What the input tail resolves on its own (Pythia 1.4B-12B)", "",
             "Basis: unweighted principal directions; core = top k*(0.1); sigma = eps * s (s from the full table).",
             "eps res / prec = pinned noise (definitions.md 1.10). ± = one standard error (4,000 draws).",
             "Shuffled: mean over 5 permutations (range in brackets). Reading: decision rule in tail_identity.py.", ""]
    for view, cname in (("surface", "surface variant"), ("role", "grammatical role")):
        lines += [f"## Classes: {cname}", "",
                  "| Model | eps | I(core) | I(tail) | I(full) | R | tail I_B | shuffled I_B | random I_B | tail I_W | reading |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            for key in ("2", "4", "8", "res", "prec"):
                v = r["results"][f"{view}/{key}"]
                sh = [x["I_B"] for x in v["tail_shuffled"]]
                eps_lbl = key if key in ("2", "4", "8") else f"{key} {r['eps'][key]:.3g}"
                lines.append(
                    f"| {r['repo'].split('/')[1]} | {eps_lbl} | {f(v['core']['I'], v['core']['I_se'])} | "
                    f"{f(v['tail']['I'], v['tail']['I_se'])} | {f(v['full']['I'], v['full']['I_se'])} | "
                    f"{f(v['R'], v['R_se'])} | {f(v['tail']['I_B'], v['tail']['I_B_se'])} | "
                    f"{np.mean(sh):.3f} [{min(sh):.3f}, {max(sh):.3f}] | "
                    f"{f(v['tail_random']['I_B'], v['tail_random']['I_B_se'])} | "
                    f"{f(v['tail']['I_W'], v['tail']['I_W_se'])} | {v['reading']} |")
        lines.append("")
    lines += ["H(V), H(G) per view: " + "; ".join(
        f"{r['repo'].split('/')[1]} surface {r['results']['surface/4']['H_V']:.2f}/{r['results']['surface/4']['H_G']:.2f}, "
        f"role {r['results']['role/4']['H_V']:.2f}/{r['results']['role/4']['H_G']:.2f}" for r in rows)]
    (RESULTS / "tail_identity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    import sys
    if "--only" in sys.argv:
        ONLY_VIEW = sys.argv[sys.argv.index("--only") + 1]
    main()
