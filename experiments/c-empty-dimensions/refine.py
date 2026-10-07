"""Experiment C, phase 3: the knee in finer steps, larger models, and checkpoints.

Same squash-and-measure test as project.py (same text, same tolerance), with
an explicit list of k values and a choice of table, for any model in
models.json and, for pythia-2.8b, any training checkpoint pinned there.

Usage:
  .venv/bin/python refine.py NAME --models pythia-1b pythia-1.4b --k 1024:2048:64
  .venv/bin/python refine.py NAME --models pythia-2.8b --checkpoints step1000 step8000 --k 512:2560:128
  --tables input output   (default: input)
  --k a:b:step (inclusive of b) or a comma list

Writes results/NAME.json (merged across runs, keyed by repo and revision),
results/NAME.md and results/NAME.png.
"""

import argparse
import json
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from measure import OUT, RESULTS, fetch, tokenizer_size
from project import DEVICE, TOL, bits_per_byte, eval_text, sweep, windows

HERE = Path(__file__).resolve().parent


def parse_ks(spec, d):
    if ":" in spec:
        a, b, step = (int(x) for x in spec.split(":"))
        ks = list(range(a, b + 1, step))
    else:
        ks = [int(x) for x in spec.split(",")]
    ks = sorted({min(k, d) for k in ks} | {d})
    return ks


def checkpoint_dir(repo, label, rev, main_rev):
    """Local folder holding a checkpoint's own weights.

    On the Pythia hub repos, some checkpoint branches carry a copy of the
    final model under the standard filename (same LFS hash as main), with the
    checkpoint's real weights in other files. Pick the weight files whose
    hashes differ from main's; fail if there are none.
    """
    api = HfApi()
    def tree(r):
        return {f.path: f.lfs.sha256 for f in api.list_repo_tree(repo, revision=r) if getattr(f, "lfs", None)}
    files, main_hashes = tree(rev), set(tree(main_rev).values())
    shards = sorted(f for f in files if f.startswith("model-0") and f.endswith(".safetensors"))
    if shards and not any(files[f] in main_hashes for f in shards):
        chosen, extra = shards, ["model.safetensors.index.json"]
    elif "model.safetensors" in files and files["model.safetensors"] not in main_hashes:
        chosen, extra = ["model.safetensors"], []
    elif "pytorch_model.bin" in files and files["pytorch_model.bin"] not in main_hashes:
        chosen, extra = ["pytorch_model.bin"], []
    else:
        raise RuntimeError(f"{repo} {label}: no weight file that differs from the final model")
    local = OUT / "checkpoints" / repo.replace("/", "--") / label
    local.mkdir(parents=True, exist_ok=True)
    for name in chosen + extra + ["config.json"]:
        target = local / name
        if not target.exists():
            target.symlink_to(hf_hub_download(repo, name, revision=rev, cache_dir=OUT))
    return local, {name: files[name] for name in chosen}


def runs(spec, names, checkpoints):
    by_name = {m["repo"].split("/")[1]: m for m in spec["models"]}
    for name in names:
        m = by_name[name]
        if checkpoints:
            for label in checkpoints:
                        yield m["repo"], label, spec["checkpoints"][m["repo"]][label], m["revision"]
        else:
            yield m["repo"], "final", m["revision"], m["revision"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--k", required=True)
    ap.add_argument("--tables", nargs="+", default=["input"])
    args = ap.parse_args()

    spec = json.loads((HERE / "models.json").read_text())
    text = eval_text()
    out = RESULTS / f"{args.name}.json"
    rows = {(r["repo"], r["label"]): r for r in json.loads(out.read_text())} if out.exists() else {}

    for repo, label, rev, main_rev in runs(spec, args.models, args.checkpoints):
        print(f"{repo} {label} on {DEVICE}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(repo, revision=main_rev, cache_dir=OUT)
        weights = None
        if rev == main_rev:
            source = dict(pretrained_model_name_or_path=repo, revision=rev, cache_dir=OUT)
        else:
            local, weights = checkpoint_dir(repo, label, rev, main_rev)
            print(f"  weights: {weights}", flush=True)
            source = dict(pretrained_model_name_or_path=str(local),
                          use_safetensors=not any(w.endswith(".bin") for w in weights))
        model = GPTNeoXForCausalLM.from_pretrained(**source, dtype=torch.float32).to(DEVICE).eval()
        n_real = tokenizer_size(fetch(repo, main_rev, "tokenizer.json"))
        d = model.config.hidden_size
        batch, nbytes = windows(tokenizer, text)
        base = bits_per_byte(model, batch, nbytes)
        print(f"  unmodified: {base:.4f} bits/byte", flush=True)
        row = {"repo": repo, "label": label, "revision": rev, "weights": weights, "d": d,
               "layers": model.config.num_hidden_layers, "baseline_bpb": base}
        params = {"input": model.get_input_embeddings().weight, "output": model.get_output_embeddings().weight}
        assert params["input"].data_ptr() != params["output"].data_ptr()
        for side in args.tables:
            print(f"  {side} table", flush=True)
            curve = sweep(model, params[side], n_real, batch, nbytes, d, parse_ks(args.k, d))
            within = [c for c in curve if c["bpb"] - base <= TOL]
            row[side] = {"curve": curve, "k_star": min(c["k"] for c in within)}
        rows[(repo, label)] = row
        del model
        if DEVICE == "mps":
            torch.mps.empty_cache()
        out.write_text(json.dumps(list(rows.values()), indent=2) + "\n")  # save after each model

    ordered = list(rows.values())
    sides = [s for s in ("input", "output") if any(s in r for r in ordered)]
    head = "| Model | checkpoint | d | baseline bits/byte | " + " | ".join(f"{s} k* | {s} k*/d" for s in sides) + " |"
    lines = [f"# {args.name}: squash test (k* within {TOL} bits/byte of unmodified)", "", head,
             "|" + "---|" * (4 + 2 * len(sides))]
    for r in ordered:
        cells = []
        for s in sides:
            k = r.get(s, {}).get("k_star")
            cells += [str(k) if k else "", f"{k / r['d']:.2f}" if k else ""]
        lines.append(f"| {r['repo'].split('/')[1]} | {r['label']} | {r['d']} | {r['baseline_bpb']:.4f} | " + " | ".join(cells) + " |")
    (RESULTS / f"{args.name}.md").write_text("\n".join(lines) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(sides), figsize=(7 * len(sides), 4.5), squeeze=False)
    for side, ax in zip(sides, axes[0]):
        for r in ordered:
            if side not in r:
                continue
            c = r[side]["curve"]
            ax.plot([p["k"] for p in c], [p["bpb"] - r["baseline_bpb"] for p in c], marker=".",
                    label=f"{r['repo'].split('/')[1]} {r['label']}")
        ax.axhline(TOL, color="grey", ls=":", lw=1)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_xlabel("k (directions kept)")
        ax.set_ylabel("loss increase (bits per byte)")
        ax.set_title(f"{side} table")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{args.name}.png", dpi=130)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
