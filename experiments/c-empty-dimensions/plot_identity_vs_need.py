"""Figure for the empty-dimensions paper: identity's 99% point against the network's k*(0.01).

Per Pythia model, as intervals (grid point below, excluded, to the measured value):
  network:  k*(0.01) of the input table, squash test (phase2.json; phase3_fine.json
            for 1B-2.8B and phase3_large.json for 6.9B-12B, which refine it)
  identity, squash basis:   k_0.99, text view, sigma_res (identity_pinned_text.json, basis A)
  identity, weighted basis: k_0.99 on the grid of 16 (identity_fine.json, text_B)
and d itself. Log scale.

Usage: .venv/bin/python plot_identity_vs_need.py OUT.png
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from measure import RESULTS


def interval(curve, key, ok):
    """(previous measured k, first k satisfying ok) along a curve sorted by k."""
    pts = sorted(curve, key=lambda c: c["k"])
    for i, c in enumerate(pts):
        if ok(c):
            return (pts[i - 1]["k"] if i else 0), c["k"]
    raise ValueError("no point satisfies the condition")


def main(out):
    load = lambda f: {r["repo"]: r for r in json.loads((RESULTS / f).read_text())}
    p2, p3f, p3l = load("phase2.json"), load("phase3_fine.json"), load("phase3_large.json")
    txt = load("identity_pinned_text.json")
    fine = {r["repo"]: r for r in json.loads((RESULTS / "identity_fine.json").read_text())["rows"]}
    names, rows = [], []
    for repo, t in txt.items():
        d = t["d"]
        src = p3f.get(repo) or p3l.get(repo) or p2[repo]
        curve = src["input"]["curve"] + ([] if src is p2.get(repo) else
                                         [c for c in p2.get(repo, {"input": {"curve": []}})["input"]["curve"]
                                          if c["k"] < min(x["k"] for x in src["input"]["curve"])])
        base = src["baseline_bpb"]
        need = interval(curve, "bpb", lambda c: c["bpb"] - base <= 0.01)
        a = t["bases"]["A_unweighted"]["res"]
        ka = a["k99"] if repo != "EleutherAI/pythia-2.8b" else 1024  # 2.8B unresolved between 768 and 1,024
        ida = interval(a["curve"], "I", lambda c: c["k"] >= ka)
        if repo == "EleutherAI/pythia-2.8b":
            ida = (512, 1024)
        fb = fine[repo]["views"]["text_B"]
        idb = interval(fb["curve"], "I", lambda c: c["k"] >= fb["k_0.99"])
        names.append(repo.split("-")[-1].upper())
        rows.append((d, need, ida, idb))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = range(len(rows))
    for i, (d, need, ida, idb) in zip(x, rows):
        ax.plot([i], [d], marker="_", markersize=18, color="0.6")
        for off, (lo, hi), col, lab in ((-0.18, need, "#b2182b", "network, k*(0.01)"),
                                         (0.0, ida, "#2166ac", "identity, squash basis"),
                                         (0.18, idb, "#4dac26", "identity, weighted basis")):
            ax.plot([i + off, i + off], [max(lo, 8), hi], color=col, linewidth=5, solid_capstyle="butt",
                    label=lab if i == 0 else None)
    ax.plot([], [], marker="_", markersize=18, color="0.6", linestyle="none", label="d")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_xlabel("Pythia model")
    ax.set_ylabel("principal directions")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    for n, r in zip(names, rows):
        print(n, r)


if __name__ == "__main__":
    main(sys.argv[1])
