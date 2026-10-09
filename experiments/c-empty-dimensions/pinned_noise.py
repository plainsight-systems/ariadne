"""Pinned noise for the Pythia input tables (definitions.md 1.10).

For each Pythia model in models.json (final checkpoint):
  s          per-direction RMS spread of the full input table (real tokens,
             uniform weights): sqrt(tr Sigma / d), as in identity_info.py
  sigma_prec fp16 rounding floor: sqrt(mean over table entries of ulp(w)^2 / 12),
             ulp(w) = 2^(floor(log2|w|) - 10) for normal numbers
  sigma_res  residual-stream interference: per-coordinate variance of
             a_t = h_1(t) - e_{v_t} (what the first block adds) around its
             per-token mean, pooled over tokens seen at least twice, on the
             phase-2 evaluation text (16 x 1,024 tokens); also the total
             per-coordinate variance of a_t
and eps = sigma / s for each.

Usage: .venv/bin/python pinned_noise.py
Writes results/pinned_noise.json and results/pinned_noise.md.
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from measure import OUT, RESULTS, fetch, tokenizer_size
from project import DEVICE, eval_text, windows

HERE = Path(__file__).resolve().parent


def fp16_floor(w16):
    """sqrt(mean ulp^2 / 12) over the fp16 entries of w16 (subnormals use the subnormal ulp)."""
    a = np.abs(w16.astype(np.float64))
    e = np.floor(np.log2(np.where(a > 0, a, 2.0 ** -24)))
    ulp = np.where(a >= 2.0 ** -14, 2.0 ** (e - 10), 2.0 ** -24)
    return float(np.sqrt((ulp ** 2).mean() / 12))


@torch.no_grad()
def residual_interference(model, batch, n):
    """Pooled within-token and total per-coordinate variance of h_1 - e."""
    d = model.config.hidden_size
    sums = torch.zeros(n, d, dtype=torch.float64)
    sumsq = torch.zeros(n, dtype=torch.float64)
    counts = torch.zeros(n, dtype=torch.float64)
    total_sum = torch.zeros(d, dtype=torch.float64)
    total_sq, total_n = 0.0, 0
    for row in batch:
        out = model(row.unsqueeze(0).to(DEVICE), output_hidden_states=True)
        a = (out.hidden_states[1][0] - out.hidden_states[0][0]).cpu().double()
        ids = row.long()
        sums.index_add_(0, ids, a)
        sumsq.index_add_(0, ids, (a * a).sum(1))
        counts.index_add_(0, ids, torch.ones(len(ids), dtype=torch.float64))
        total_sum += a.sum(0)
        total_sq += float((a * a).sum())
        total_n += len(ids)
    keep = counts >= 2
    # within-token sum of squares: sum ||a||^2 - ||sum a||^2 / count, per token
    within_ss = (sumsq[keep] - (sums[keep] ** 2).sum(1) / counts[keep]).sum()
    within_df = (counts[keep] - 1).sum()
    var_within = float(within_ss / within_df / d)
    var_total = float((total_sq - float((total_sum ** 2).sum()) / total_n) / (total_n - 1) / d)
    return var_within, var_total, int(keep.sum()), int(counts[keep].sum())


def main():
    spec = json.loads((HERE / "models.json").read_text())
    text = eval_text()
    rows = []
    for m in spec["models"]:
        repo, rev = m["repo"], m["revision"]
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT,
                                                   dtype=torch.float32).to(DEVICE).eval()
        w = model.get_input_embeddings().weight.detach()[:n].cpu().double().numpy()
        d = w.shape[1]
        c = w - w.mean(0)
        s = float(np.sqrt((c * c).sum() / len(w) / d))
        sigma_prec = fp16_floor(w.astype(np.float16))
        batch, _ = windows(tok, text)
        var_within, var_total, ntok, nocc = residual_interference(model, batch, n)
        row = {"repo": repo, "revision": rev, "d": d, "s": s,
               "sigma_prec": sigma_prec, "eps_prec": sigma_prec / s,
               "sigma_res": var_within ** 0.5, "eps_res": var_within ** 0.5 / s,
               "sigma_res_total": var_total ** 0.5, "eps_res_total": var_total ** 0.5 / s,
               "res_tokens_pooled": ntok, "res_occurrences_pooled": nocc}
        print(f"  s {s:.4f}  eps_prec {row['eps_prec']:.2e}  eps_res {row['eps_res']:.2f}"
              f"  (total {row['eps_res_total']:.2f})", flush=True)
        rows.append(row)
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()
    (RESULTS / "pinned_noise.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Pinned noise for the Pythia input tables (definitions.md 1.10)", "",
             "| Model | d | s | eps_prec (fp16) | eps_res (within-token) | eps_res (total) |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['repo'].split('/')[1]} | {r['d']} | {r['s']:.4f} | {r['eps_prec']:.2e} | "
              f"{r['eps_res']:.2f} | {r['eps_res_total']:.2f} |" for r in rows]
    (RESULTS / "pinned_noise.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
