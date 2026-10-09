"""Identity information with grammatical-role classes, all ten models.

Same measure as identity_info.py (definitions.md section 1), with the class
function g changed from surface variants to grammatical role: the 12-tag
universal part-of-speech set of Petrov, Das and McDonald (2012) as mapped
by NLTK (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, NUM, CONJ, PRT, "." for
punctuation, X), tagged with NLTK's averaged-perceptron tagger, plus SPACE
for tokens that decode to whitespace only.

  Text view: the first 70,000 characters of the WikiText-103 test set
  (cross_vocab.py), tagged in context. Each token occurrence takes the tag
  of the word its first non-space character falls in; a token type's class
  is its most frequent tag. p = token frequencies in that text.

  Dictionary view: uniform over the tokens with a known role. A lexicon
  (lowercased word -> most frequent tag) is built by tagging the whole
  WikiText-103 test set; a token's decoded text, stripped and lowercased,
  is looked up there. Rules first: whitespace-only -> SPACE; every
  character punctuation or symbol -> "."; every character a digit -> NUM.
  Tokens not covered (mostly word fragments) are left out of this view;
  the coverage is reported.

Words are split for tagging with the regex [A-Za-z]+(?:'[A-Za-z]+)?|\\d+|[^\\w\\s].
Reports I, I_B, I_W, both comparisons (crossover, with a noise check, and
half-fill), for epsilon 2, 4, 8, both tables.

Usage: NLTK_DATA=out/nltk_data .venv/bin/python identity_roles.py [model-name ...]
Writes results/identity_roles.json and results/identity_roles.md.
"""

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from nltk.tag import pos_tag
from transformers import AutoModelForCausalLM, AutoTokenizer

import identity_info as ii
from cross_vocab import CHARS, MODELS as XV_MODELS
from measure import OUT, RESULTS, fetch, tokenizer_size
from measure_init import load_tables
from project import DATASET

HERE = Path(__file__).resolve().parent
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]")


def full_test_text():
    path = hf_hub_download(DATASET[0], DATASET[2], revision=DATASET[1], repo_type="dataset", cache_dir=OUT)
    return "".join(pq.read_table(path).column("text").to_pylist())


def tag_spans(text):
    """[(start, end, tag)] for the words of text, tagged in context."""
    spans = [(m.start(), m.end()) for m in WORD.finditer(text)]
    tags = pos_tag([text[a:b] for a, b in spans], tagset="universal")
    return [(a, b, t) for (a, b), (_, t) in zip(spans, tags)]


def rule_class(s):
    if not s:
        return "SPACE"
    if all(unicodedata.category(ch)[0] in "PS" for ch in s):
        return "."
    if s.isdigit():
        return "NUM"
    return None


def build_lexicon(spans, text):
    votes = defaultdict(Counter)
    for a, b, t in spans:
        votes[text[a:b].lower()][t] += 1
    return {w: c.most_common(1)[0][0] for w, c in votes.items()}


def dictionary_classes(tokenizer, n, lexicon):
    names, covered = [], 0
    for v in range(n):
        s = tokenizer.decode([v]).strip()
        c = rule_class(s) or lexicon.get(s.lower())
        names.append(c)
        covered += c is not None
    return names, covered / n


def text_classes(tokenizer, n, text, spans):
    """Per token type, the most frequent tag of the word its occurrences start in; and its counts."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    starts = np.array([a for a, _, _ in spans])
    votes, counts = defaultdict(Counter), np.zeros(n)
    for tid, (a, b) in zip(enc["input_ids"], enc["offset_mapping"]):
        if tid >= n:
            continue
        counts[tid] += 1
        piece = text[a:b]
        off = a + (len(piece) - len(piece.lstrip()))
        if not piece.strip():
            votes[tid]["SPACE"] += 1
            continue
        i = np.searchsorted(starts, off, side="right") - 1
        if i >= 0 and spans[i][0] <= off < spans[i][1]:
            votes[tid][spans[i][2]] += 1
        else:
            votes[tid][rule_class(text[off:b].strip()) or "X"] += 1
    names = [votes[v].most_common(1)[0][0] if votes[v] else None for v in range(n)]
    return names, counts


def as_codes(names):
    table = {}
    return np.array([table.setdefault(c, len(table)) if c is not None else -1 for c in names])


def model_tables(repo, rev):
    spec = json.loads((HERE / "models.json").read_text())
    if any(m["repo"] == repo for m in spec["models"]):
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        n = tokenizer_size(fetch(repo, rev, "tokenizer.json"))
        t = load_tables(repo, rev)
        return tok, n, {s: t[s][:n] for s in t}
    tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
    model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT, dtype=torch.float32)
    n = len(tok)
    out = {"input": model.get_input_embeddings().weight.detach()[:n].double().numpy(),
           "output": model.get_output_embeddings().weight.detach()[:n].double().numpy()}
    return tok, n, out


def all_models():
    spec = json.loads((HERE / "models.json").read_text())
    pythia = [(m["repo"], m["revision"]) for m in spec["models"]]
    return pythia + [m for m in XV_MODELS if "pythia" not in m[0]]


def noise_level(e):
    if not e["k_cross"]:
        return None
    c = {x["k"]: x for x in e["curve"]}
    a, b = c[e["k_cross"][0]], c[e["k_cross"][1]]
    se = max(a["I_se"], b["I_se"], a["I_B_se"], b["I_B_se"])
    return max(abs(b["I_B"] - a["I_B"]), abs(b["I_W"] - a["I_W"])) < 2 * se


def main():
    wanted = set(sys.argv[1:])
    full = full_test_text()
    text = full[:CHARS]
    lexicon = build_lexicon(tag_spans(full), full)
    spans = tag_spans(text)
    path = RESULTS / "identity_roles.json"
    results = {r["repo"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    for repo, rev in all_models():
        name = repo.split("/")[1]
        if wanted and name not in wanted:
            continue
        print(repo, flush=True)
        tok, n, tables = model_tables(repo, rev)
        dnames, coverage = dictionary_classes(tok, n, lexicon)
        tnames, counts = text_classes(tok, n, text, spans)
        views = {
            "dictionary": (np.array([1.0 if c is not None else 0.0 for c in dnames]), as_codes(dnames)),
            "text": (counts, as_codes(tnames)),
        }
        row = {"repo": repo, "revision": rev, "d": tables["input"].shape[1], "V": n,
               "dictionary_coverage": coverage,
               "class_counts": {"dictionary": Counter(c for c in dnames if c),
                                "text": Counter(c for c, k in zip(tnames, counts) if c and k > 0)}}
        for side, w in tables.items():
            for view, (p, codes) in views.items():
                p = np.where(codes >= 0, p, 0.0)
                p = p / p.sum()
                r = ii.measure_view(w, p, np.where(codes >= 0, codes, 0), ii.k_grid(w.shape[1]),
                                    np.random.default_rng(ii.SEED))
                for e in r["eps"].values():
                    e["k_cross_noise"] = noise_level(e)
                row[f"{side}/{view}"] = r
                e = r["eps"]["4.0"]
                print(f"  {side:6s} {view:10s} H(G) {r['H_G']:.2f} of H(V) {r['H_V']:.2f} | eps 4: "
                      f"k_B50 {e['k_B50']} k_W50 {e['k_W50']} k_x {e['k_cross']}"
                      f"{' (noise)' if e['k_cross_noise'] else ''}", flush=True)
        results[repo] = row
        path.write_text(json.dumps(list(results.values()), indent=2, default=dict) + "\n")
    write_report([results[r] for r, _ in all_models() if r in results])


def write_report(rows):
    lines = ["# Identity information with grammatical-role classes", "",
             "definitions.md section 1. Classes: 12-tag universal part of speech (NLTK), plus SPACE.",
             "k_B50, k_W50: half-fill dimensions. k_x: crossover; (n) marks a crossover within sampling noise.", ""]
    for view in ("dictionary", "text"):
        for eps in ii.EPSILONS:
            lines += [f"## {view} view, epsilon = {eps}", "",
                      "| Model | V | d | table | H(V) | H(G) | I(d) | I_B(d) | I_W(d) | k_B50 | k_W50 | k_x |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in rows:
                for side in ("input", "output"):
                    v = r[f"{side}/{view}"]
                    e = v["eps"][str(eps)]
                    last = e["curve"][-1]
                    kx = "none" if not e["k_cross"] else "-".join(map(str, e["k_cross"])) + (" (n)" if e["k_cross_noise"] else "")
                    lines.append(f"| {r['repo'].split('/')[1][:24]} | {r['V']:,} | {r['d']} | {side} | {v['H_V']:.2f} | "
                                 f"{v['H_G']:.2f} | {last['I']:.2f} | {last['I_B']:.2f} | {last['I_W']:.2f} | "
                                 f"{e['k_B50']} | {e['k_W50']} | {kx} |")
            lines.append("")
    lines += ["## Dictionary-view coverage", "", "| Model | V | tokens with a known role |", "|---|---|---|"]
    lines += [f"| {r['repo'].split('/')[1][:24]} | {r['V']:,} | {r['dictionary_coverage']:.2f} |" for r in rows]
    (RESULTS / "identity_roles.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
