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

| `refine.py` | Finer k steps, larger models, checkpoints | `results/phase3_*` |
| `cross_vocab.py` | Same d, different vocabularies (TinyLlama, Pythia, OLMo-2) | `results/cross_vocab.*` |
| `interference.py` | Crowding of token vectors in the d = 2048 models | `results/interference.*` |
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

## Observed trends (as of 2026-10-07)

Observations only, from the runs below. No interpretation here.

1. The vocabulary stretches to fill the room it is given: trained tables
   spread over 91-98% of d at every size, needed or not.
2. The input side has slack once the room is big enough: small rooms are
   fully used; past a point part of the room can be removed without loss,
   and the share used falls as d grows at fixed vocabulary (1.0 down to
   about 0.6-0.7).
3. The output side never has slack: it uses all of d in every model, size
   and vocabulary tested.
4. Used input room grows with exposure: through training it keeps climbing
   and has not stopped at 300B tokens.
5. Used input room grows with model size at fixed vocabulary, slower than
   d (about 1.7K, 2.8K, 3.1K at d = 2.5K, 4K, 5K).
6. A bigger vocabulary uses more room at the same d: core about 1.5K for
   32K and 50K vocabularies; all 2048 for 100K.
7. Token vectors are, on average, about as separate as random vectors in
   the same room (input: mean pairwise overlap 1.1-1.5x random), but almost
   every token has a close neighbour (nearest-neighbour overlap 3-6x
   random). The crowding is local, not general. (`interference.py`,
   `results/interference.md`; d = 2048 models.)
8. The output side is more crowded than the input side, most of all among
   tokens that occur in ordinary text (mean overlap 2.6-2.7x random for
   TinyLlama and Pythia, 7.7x for OLMo-2).
9. Among tokens that occur in ordinary text, input nearest-neighbour
   crowding rises with vocabulary size: 3.2x (32K), 4.1-4.2x (50K), 4.8x
   (100K).

## Phase 3 (2026-10-07): finer steps, larger models, checkpoints

`refine.py`; results in `results/phase3_*.md`, `.json`, `.png`, `.log`.

Input table, directions needed (k) at several loss tolerances (bits per byte
above the unmodified model):

| Model | d | +0.1 | +0.03 | +0.01 | +0.003 | k/d at +0.01 |
|---|---|---|---|---|---|---|
| 1b | 2048 | 1664 | 1856 | 1984 | 1984 | 0.97 |
| 1.4b | 2048 | 1408 | 1600 | 1664 | 1856 | 0.81 |
| 2.8b | 2560 | 1344 | 1472 | 1664 | 1856 | 0.65 |
| 6.9b | 4096 | 2048 | 2560 | 2816 | 3328 | 0.69 |
| 12b | 5120 | 2048 | 2560 | 3072 | 3840 | 0.60 |

(1b to 2.8b in steps of 64; 6.9b and 12b in steps of 256.)

Output table, 6.9b and 12b: all of d needed, as at every smaller size.

pythia-2.8b through training (steps of 128):

| Step | bits/byte | +0.1 | +0.03 | +0.01 | +0.003 |
|---|---|---|---|---|---|
| 1000 | 1.602 | 768 | 1408 | 1920 | 2304 |
| 8000 | 0.996 | 640 | 1024 | 1408 | 1920 |
| 16000 | 0.934 | 768 | 1024 | 1408 | 1792 |
| 33000 | 0.888 | 1024 | 1280 | 1408 | 1792 |
| 66000 | 0.847 | 1280 | 1408 | 1536 | 1792 |
| 100000 | 0.820 | 1408 | 1536 | 1664 | 1920 |
| 143000 | 0.806 | 1408 | 1536 | 1664 | 1920 |

What this says:

- **Empty room: yes, on the input side.** From 1.4b up, 20 to 40% of the
  input table's room can be removed for 0.01 bits per byte. The output table
  uses all of d at every size.
- **The plateau prediction failed.** The needed count did not stay near
  1.8-1.9K: it keeps growing with d (1664 at 2.8b, 2816 at 6.9b, 3072 at 12b
  at +0.01). The share used settles around 60-70% of d instead. At looser
  tolerances (+0.1, +0.03) 6.9b and 12b need the same (2048, 2560): the core
  may level off while finer detail keeps using more room.
- **Exposure matters.** Within 2.8b, after an early phase (step 1000, still
  close to random), the needed count grows with training: 640 to 1408 at
  +0.1, 1408 to 1664 at +0.01. The room used is not fixed by the vocabulary
  alone; it grows as the model learns more about each token.

Data problem found and fixed: on the Pythia hub repos, several checkpoint
branches hold a copy of the final model under the standard filename (same
LFS hash as main); the checkpoint's own weights are in other files. The
first checkpoint run measured the final model seven times and was
discarded. `refine.py` now picks the weight files whose hashes differ from
main's, records them, and fails if there are none. The step-0 comparison in
phase 1b used genuine step-0 files (checked by hash).

## Reproduce

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python measure.py
.venv/bin/python measure_init.py
```

Python version in `python-version.txt`. Downloads go to `out/` (git-ignored),
about 25 GB for both scripts.

## Rate of new directions through training (2026-10-07), parked

Scripts: `plot_rate_lr.py`, `plot_lr_ceiling.py`. Plots: `results/rate_vs_step.png`,
`results/rate_vs_lr.png`, `results/lr_ceiling.png`. Learning-rate schedule
from Pythia's own 2.8B config (lr 1.6e-4, floor 1.6e-5, cosine over 143,000
steps, 1% warmup; commit pinned in `plot_rate_lr.py`).

pythia-2.8b, input table, new directions (within 0.01 bits per byte) per
1,000 steps, and the learning rate scaled into the tightest ceiling over
those points:

| Step (midpoint) | Rate | Ceiling | Rate / ceiling |
|---|---|---|---|
| 12,000 | 1.35 | 9.65 | 0.14 |
| 24,500 | 2.72 | 9.19 | 0.30 |
| 49,500 | 3.05 | 7.45 | 0.41 |
| 83,000 | 1.88 | 4.36 | 0.43 |
| 121,500 | 1.62 | 1.62 | 1.00 |

- The rate rises, peaks mid-training, then falls, and is still positive at
  the end of training (300B tokens). No bound in sight.
- Early on the learning rate is far above the rate: it is not what limits
  new directions. The rate climbs toward the ceiling, and only at the end
  does the learning rate press down on it. The late slowdown looks like the
  schedule, not the language running out.
- Per unit of learning rate spent, new directions keep rising (8.5, 18.1,
  25.0, 26.4, 61.1), but the low-rate end of training also works
  differently (annealing), so this ratio is not a clean measure.
- Weak points: one model, seven checkpoints, k steps of 128 before
  interpolation; the last interval sets the ceiling's height.

Parked (Andrew): the training-rate curve mostly shows the learning-rate
schedule, so it is not the thread to chase for the natural-dimension
question.

## Same room, different vocabularies (2026-10-07)

`cross_vocab.py`; results in `results/cross_vocab.*`. Four finished models
with d = 2048, all read on the same text (first 70,000 characters of the
WikiText-103 test set), loss in bits per byte. Input table: directions
needed at each tolerance.

| Model | V | trained on | k +0.1 | k +0.03 | k +0.01 | output k +0.01 |
|---|---|---|---|---|---|---|
| TinyLlama 1.1B | 32,000 | 3T | 1479 | 1828 | 2019 | 2045 |
| Pythia 1B | 50,277 | 300B | 1601 | 1821 | 1921 | 2043 |
| Pythia 1.4B | 50,277 | 300B | 1370 | 1553 | 1652 | 2044 |
| OLMo-2 1B | 100,278 | ~4T | 2030 | 2042 | 2046 | 2045 |

- OLMo-2, the largest vocabulary, needs all 2048 input directions even at
  the loosest tolerance: the room is too small for it.
- The core (loosest tolerance) roughly follows vocabulary size: about
  1.4-1.6K for 32K and 50K, all of the room for 100K.
- At strict tolerance the two long-trained models (TinyLlama, OLMo-2) fill
  the room; Pythia, trained 10x less, leaves the most empty. Training amount
  and vocabulary are confounded across these families (and so are data,
  architecture and recipe), and the checkpoint run showed used room grows
  with training.
- Output tables use all of d in every model.
