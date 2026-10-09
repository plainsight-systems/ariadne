"""Experiment C: semantic core vs pedantic refinement (Pythia input tables).

The input table's centered rows are split at k_core, the directions needed
at the loose tolerance (+0.1 bits per byte, from phase 3): the core is the
top k_core principal directions, the tail is the rest. Three observations:

1. Content or spread (model runs). Bits per byte on the phase-2 evaluation
   text with the table changed as follows, everything else untouched:
     original
     tail zeroed                  (= the squash test at k_core)
     tail shuffled across tokens  (each token gets another token's whole tail:
                                   same values and variance, wrong owner)
     tail random, matched         (Gaussian, each tail direction's variance kept)
     tail zeroed, lengths kept    (core rescaled to each row's original length)
     core shuffled across tokens  (control: the core's values, wrong owners)
   If shuffling the tail costs about as much as zeroing it, the tail carries
   token-specific information. If it costs much less, the tail mainly holds
   spread.

2. What the tail separates (table only). For tokens that occur in the
   evaluation text, find each one's nearest neighbour by cosine in the core
   directions alone, then measure how much the tail pushes the pair apart
   (core cosine minus full cosine). Pairs are classed by string rules on the
   decoded tokens:
     surface: same text after stripping whitespace and lowercasing
     stem:    otherwise share a prefix of at least 4 characters covering at
              least 60% of the shorter one
     other:   everything else

3. More used, emptier (table only). Median nearest-neighbour cosine among
   the same tokens using only the top k directions, for k from 64 to d.

Usage: .venv/bin/python semantic_core.py [model-name ...]
Writes results/semantic_core.json and results/semantic_core.md.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from measure import OUT, RESULTS, fetch, tokenizer_size
from project import DEVICE, bits_per_byte, eval_text, principal, windows

HERE = Path(__file__).resolve().parent
SEED = 0
# k at +0.1 bits per byte (phase 3 fine for 1.4b and 2.8b, steps of 64; large for 6.9b and 12b, steps of 256)
K_CORE = {"pythia-1.4b": 1408, "pythia-2.8b": 1344, "pythia-6.9b": 2048, "pythia-12b": 2048}
EMPTINESS_K = [64, 128, 256, 512, 768, 1024, 1536, 2048, 2560, 3072, 4096, 5120]


def classify(a, b):
    na, nb = a.strip().lower(), b.strip().lower()
    if na and na == nb:
        return "surface"
    p = 0
    for x, y in zip(na, nb):
        if x != y:
            break
        p += 1
    if p >= 4 and p >= 0.6 * min(len(na), len(nb)):
        return "stem"
    return "other"


def unit(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1, n)


def nearest(u):
    """Index and cosine of each row's nearest other row (rows unit length)."""
    s = u @ u.T
    np.fill_diagonal(s, -np.inf)
    j = s.argmax(axis=1)
    return j, s[np.arange(len(u)), j]


def table_observations(c, vec, k_core, seen, tokenizer):
    """Observations 2 and 3 on the centered table c (rows: all real tokens)."""
    cs = c[seen]
    full = unit(cs)
    core = unit(cs @ vec[:, :k_core])
    j, cos_core = nearest(core)
    cos_full = np.einsum("ij,ij->i", full, full[j])
    text = [tokenizer.decode([int(t)]) for t in seen]
    classes = [classify(text[i], text[j[i]]) for i in range(len(seen))]
    by_class = {}
    for name in ("surface", "stem", "other"):
        m = np.array([cl == name for cl in classes])
        if m.any():
            by_class[name] = {"pairs": int(m.sum()), "share": float(m.mean()),
                              "core_cos": float(cos_core[m].mean()), "full_cos": float(cos_full[m].mean()),
                              "tail_push": float((cos_core[m] - cos_full[m]).mean())}
    order = np.argsort(-(cos_core - cos_full))
    examples = [{"a": text[i], "b": text[j[i]], "class": classes[i],
                 "core_cos": float(cos_core[i]), "full_cos": float(cos_full[i])} for i in order[:25]]
    emptiness = []
    d = c.shape[1]
    for k in [k for k in EMPTINESS_K if k <= d]:
        _, nn = nearest(unit(cs @ vec[:, :k]))
        emptiness.append({"k": k, "nn_median": float(np.median(nn))})
    return {"by_class": by_class, "most_separated_by_tail": examples, "emptiness": emptiness}


def variants(w, mu, vec, k, rng):
    """Input tables for observation 1, rows = real tokens (float64)."""
    c = w - mu
    core_b, tail_b = vec[:, :k], vec[:, k:]
    core, tail = c @ core_b, c @ tail_b
    perm = rng.permutation(len(w))
    zero = core @ core_b.T
    lengths = np.linalg.norm(c, axis=1, keepdims=True)
    zl = np.linalg.norm(zero, axis=1, keepdims=True)
    return {
        "original": w,
        "tail zeroed": mu + zero,
        "tail shuffled": mu + zero + tail[perm] @ tail_b.T,
        "tail random, matched": mu + zero + (rng.standard_normal(tail.shape) * tail.std(axis=0)) @ tail_b.T,
        "tail zeroed, lengths kept": mu + zero * (lengths / np.where(zl == 0, 1, zl)),
        "core shuffled": mu + core[perm] @ core_b.T + tail @ tail_b.T,
    }


def main():
    spec = json.loads((HERE / "models.json").read_text())
    wanted = sys.argv[1:] or list(K_CORE)
    text = eval_text()
    out = RESULTS / "semantic_core.json"
    rows = {r["repo"]: r for r in json.loads(out.read_text())} if out.exists() else {}
    for m in spec["models"]:
        name = m["repo"].split("/")[1]
        if name not in wanted:
            continue
        repo, rev, k_core = m["repo"], m["revision"], K_CORE[name]
        print(f"{repo}  k_core = {k_core}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        model = GPTNeoXForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT,
                                                   dtype=torch.float32).to(DEVICE).eval()
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        param = model.get_input_embeddings().weight
        original = param.data.clone()
        w = original[:n].cpu().double().numpy()
        mu, vec = principal(w)
        batch, nbytes = windows(tokenizer, text)
        seen = np.array(sorted(set(batch.flatten().tolist()) & set(range(n))))

        print("  table observations", flush=True)
        row = {"repo": repo, "revision": rev, "d": w.shape[1], "k_core": k_core, "seen_tokens": int(len(seen)),
               **table_observations(w - mu, vec, k_core, seen, tokenizer)}

        print("  content or spread", flush=True)
        rng = np.random.default_rng(SEED)
        row["bpb"] = {}
        for label, table in variants(w, mu, vec, k_core, rng).items():
            param.data[:n] = torch.from_numpy(table).to(param.dtype).to(param.device)
            row["bpb"][label] = bits_per_byte(model, batch, nbytes)
            print(f"    {label:28s} {row['bpb'][label]:.4f}", flush=True)
        param.data.copy_(original)
        rows[repo] = row
        out.write_text(json.dumps(list(rows.values()), indent=2) + "\n")
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()

    ordered = [rows[m["repo"]] for m in spec["models"] if m["repo"] in rows]
    lines = ["# Semantic core vs pedantic refinement (Pythia input tables)", "",
             "k_core = directions needed at +0.1 bits per byte. Loss in bits per byte on the phase-2 text.", "",
             "## 1. Content or spread", "",
             "| Model | d | k_core | original | tail zeroed | tail shuffled | tail random | zeroed, lengths kept | core shuffled |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in ordered:
        b = r["bpb"]
        lines.append(f"| {r['repo'].split('/')[1]} | {r['d']} | {r['k_core']} | {b['original']:.3f} | "
                     f"{b['tail zeroed']:.3f} | {b['tail shuffled']:.3f} | {b['tail random, matched']:.3f} | "
                     f"{b['tail zeroed, lengths kept']:.3f} | {b['core shuffled']:.3f} |")
    lines += ["", "## 2. What the tail separates", "",
              "Nearest neighbour in the core directions; tail push = core cosine minus full cosine.", "",
              "| Model | class | share of pairs | core cos | full cos | tail push |", "|---|---|---|---|---|---|"]
    for r in ordered:
        for cl, v in r["by_class"].items():
            lines.append(f"| {r['repo'].split('/')[1]} | {cl} | {v['share']:.2f} | {v['core_cos']:.3f} | "
                         f"{v['full_cos']:.3f} | {v['tail_push']:.3f} |")
    lines += ["", "## 3. More used, emptier", "", "Median nearest-neighbour cosine using only the top k directions.", "",
              "| Model | " + " | ".join(f"k={k}" for k in EMPTINESS_K) + " |", "|---|" + "---|" * len(EMPTINESS_K)]
    for r in ordered:
        e = {x["k"]: x["nn_median"] for x in r["emptiness"]}
        lines.append(f"| {r['repo'].split('/')[1]} | " + " | ".join(f"{e[k]:.3f}" if k in e else "" for k in EMPTINESS_K) + " |")
    lines += ["", "## Pairs the tail separates most (2.8b)", ""]
    for r in ordered:
        if r["repo"].endswith("2.8b"):
            for x in r["most_separated_by_tail"][:15]:
                lines.append(f"- {x['a']!r} / {x['b']!r} ({x['class']}): core {x['core_cos']:.2f}, full {x['full_cos']:.2f}")
    (RESULTS / "semantic_core.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
