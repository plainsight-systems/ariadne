# Experiment C: do models have empty dimensions?

Exploratory. Started 2026-10-07 from the ideas entry "Vocabulary, order,
room and shape".

## Question

V and d are chosen, and training fills in the V x d embedding table. Does
the trained table use all of d?

Andrew's larger theory: V x d together set the expressiveness of the learned
language, and d should be derivable from the vocabulary and the grammar.
**Prediction:** if V and d are chosen arbitrarily, then at a fixed V the
embedding dimension is usually too large, and the trained table leaves room
unused.

If that holds, the next question is whether the vocabulary and the grammar
drive the dimension actually used.

## Setup

Pythia holds V fixed (50,277 tokens) while d grows (512 to 2560 here), on
the same data in the same order. Small models first because the runs are
cheap; a larger open-weight model to confirm. Weights in full precision,
pinned revisions in `models.json`.

## Runs

| Script | What | Results |
|---|---|---|
| `measure.py` | Spread of the trained input and output tables across d, vs a random table of the same shape | `results/phase1.md`, `results/spectra.png` |
| `measure_init.py` | Trained table vs the same table at training step 0 | `results/phase1b.md` (read its correction), `results/growth.png` |

Not yet run: the causal test (squash the table onto its top k directions,
run the model, see where loss stops changing). Spread can include variance
that does not matter for the model's output; this test is what decides
"empty".

## Reproduce

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python measure.py
.venv/bin/python measure_init.py
```

Python version in `python-version.txt`. Downloads go to `out/` (git-ignored),
about 25 GB for both scripts.
