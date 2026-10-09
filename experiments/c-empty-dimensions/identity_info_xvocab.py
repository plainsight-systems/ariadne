"""Identity information (definitions.md section 1) on TinyLlama and OLMo-2 1B.

Same measure, settings and report as identity_info.py, applied to the two
non-Pythia d = 2048 models from cross_vocab.py: TinyLlama 1.1B (V = 32,000)
and OLMo-2 1B (V = 100,278). Classes: surface variants. Text view: token
frequencies in the first 70,000 characters of the WikiText-103 test set,
tokenized by each model's own tokenizer (cross_vocab.py). Reports both the
crossover and the half-fill dimensions.

Usage: .venv/bin/python identity_info_xvocab.py
Writes results/identity_info_xvocab.json, .md and .png.
"""

import json
import shutil

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import identity_info as ii
from cross_vocab import CHARS, MODELS, windows
from measure import OUT, RESULTS
from project import eval_text

PICK = [m for m in MODELS if "pythia" not in m[0]]


def main():
    text = eval_text()[:CHARS]
    rows = []
    for repo, rev in PICK:
        print(repo, flush=True)
        tok = AutoTokenizer.from_pretrained(repo, revision=rev, cache_dir=OUT)
        model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, cache_dir=OUT, dtype=torch.float32)
        n = len(tok)
        tables = {"input": model.get_input_embeddings().weight.detach()[:n].double().numpy(),
                  "output": model.get_output_embeddings().weight.detach()[:n].double().numpy()}
        del model
        cls = ii.surface_class(tok, n)
        ids = torch.cat(windows(tok, text)).numpy()
        counts = np.bincount(ids, minlength=n)[:n].astype(float)
        views = {"dictionary": np.full(n, 1.0 / n), "text": counts / counts.sum()}
        d = tables["input"].shape[1]
        row = {"repo": repo, "revision": rev, "d": d, "V": n}
        for side, w in tables.items():
            for view, p in views.items():
                row[f"{side}/{view}"] = ii.measure_view(w, p, cls, ii.k_grid(d), np.random.default_rng(ii.SEED))
                e = row[f"{side}/{view}"]["eps"]["4.0"]
                print(f"  {side:6s} {view:10s} I(d) {e['curve'][-1]['I']:.2f} of {row[f'{side}/{view}']['H_V']:.2f}"
                      f"  k_B50 {e['k_B50']} k_W50 {e['k_W50']} k_x {e['k_cross']}", flush=True)
        rows.append(row)
    (RESULTS / "identity_info_xvocab.json").write_text(json.dumps(rows, indent=2) + "\n")
    # reuse the Pythia report writer, then move its outputs to this run's names
    pythia_md, pythia_png = RESULTS / "identity_info.md", RESULTS / "identity_info.png"
    keep_md, keep_png = pythia_md.read_bytes(), pythia_png.read_bytes()
    ii.write_report(rows)
    shutil.move(pythia_md, RESULTS / "identity_info_xvocab.md")
    shutil.move(pythia_png, RESULTS / "identity_info_xvocab.png")
    pythia_md.write_bytes(keep_md)
    pythia_png.write_bytes(keep_png)


if __name__ == "__main__":
    main()
