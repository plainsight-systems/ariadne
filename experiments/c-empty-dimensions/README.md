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

Pythia only (decided 2026-10-07): the same vocabulary (V = 50,277), the same
data in the same order and the same number of training tokens at every
size, with d varying (512 to 2560 so far). Depth and head count also change
between sizes; pythia-1b and pythia-1.4b share d = 2048 and differ mainly in
depth, a built-in control. Small sizes first because the runs are cheap;
the larger Pythias (6.9B, 12B) to confirm. Weights in full precision,
pinned revisions in `models.json`.

## Runs

| Script | What | Results |
|---|---|---|
| `measure.py` | Spread of the trained input and output tables across d, vs a random table of the same shape | `results/phase1.md`, `results/spectra.png` |
| `measure_init.py` | Trained table vs the same table at training step 0 | `results/phase1b.md` (read its correction), `results/growth.png` |

| `project.py` | Causal test: squash a table onto its top k directions, run the model on WikiText-103 test, find where loss stops changing | `results/phase2.md`, `results/loss_vs_k.png`, `results/phase2.log` |

## What we have so far (2026-10-07)

- Spread (phase 1): both tables spread over 91-98% of d at every size. The
  vocabulary gets stretched to fill whatever room it is given.
- Need (phase 2), input table: every direction is needed up to d = 1024.
  From d = 2048 on, the needed number levels off around 1800-1950 (k* =
  1946 at 1b, 1843 at 1.4b, 1792 at 2.8b), so the share of d needed falls:
  1.00, 1.00, 1.00, 0.95, 0.90, 0.70. At 2.8b the knee sits between 1280 and
  1792 directions.
- Need (phase 2), output table: all of d is needed at every size. Dropping
  the weakest 5% costs 0.06 to 0.62 bits per byte.
- Rough reading: for this vocabulary, the input side seems to want about
  1.8-1.9K dimensions and leaves the rest empty; the output side uses
  everything it is given. Resolution is coarse (k in steps of 0.05 to 0.1 of
  d), on 16K tokens of text, with a 0.01 bits-per-byte tolerance.

Next: finer k steps around the knee for 1b, 1.4b and 2.8b; then pythia-6.9b
(d = 4096) and pythia-12b (d = 5120). If the input side really wants about
1.8-1.9K dimensions for this vocabulary, k* should stay near there while d
doubles.

## Reproduce

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python measure.py
.venv/bin/python measure_init.py
```

Python version in `python-version.txt`. Downloads go to `out/` (git-ignored),
about 25 GB for both scripts.
