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
| `through_training.py` | Spread and crowding of 2.8b's input table through training | `results/through_training.*` |
| `semantic_core.py` | Semantic core vs pedantic refinement: tail content, what the tail separates, emptiness | `results/semantic_core.*` |
| `identity_info.py` | Identity information (definitions.md section 1) on all Pythia tables | `results/identity_info.*` |
| `identity_info_xvocab.py` | The same on TinyLlama and OLMo-2 (d = 2048), with both the crossover and the half-fill comparison | `results/identity_info_xvocab.*` |
| `identity_roles.py` | Identity information with grammatical-role classes, all ten models | `results/identity_roles.*` |
| `pinned_noise.py` | Noise derived from the receiver (definitions.md 1.10) for all Pythia input tables | `results/pinned_noise.*` |
| `identity_pinned.py` | 99% point of identity information at the pinned noise | `results/identity_pinned.*` |
| `nonlinear_echo.py` | Is the tail a nonlinear echo of the core: kNN, MLP and OLS prediction of the tail from the core; core-tail geometry correlation | `results/nonlinear_echo.*` |
| `tail_identity.py` | What the tail resolves on its own, against shuffled and random tails; redundancy with the core | `results/tail_identity.*` (the first run's role view permuted across all tokens; rerun permuting within the view's tokens, `results/tail_identity_role_rerun.log`) |
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
  the weakest 5% costs 0.08 to 0.42 bits per byte (70m to 2.8b; corrected
  2026-10-09 from "0.06 to 0.62", which was wrong).
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
10. Through training (pythia-2.8b input table), spread does not grow
    with the directions needed. From the random start, training first
    concentrates the table (share of d covered: 0.99 at step 0 to 0.89 at
    step 33,000; directions for 90% of variance: 0.85 d to 0.65 d) while
    nearest neighbours pull close (0.09 to 0.49). After that, spread edges
    back up (0.89 to 0.91) and crowding levels off (about 0.51), while the
    directions needed keep rising (observation 4). (`through_training.py`,
    `results/through_training.md`.)
11. The tail (input directions past the loose-tolerance core) carries
    token-specific content, not only spread. Giving each token another
    token's tail, or matched random values, costs at least as much as
    zeroing the tail in every model, and much more in the largest (12b:
    zeroed +0.10, shuffled +0.29, random +0.30 bits per byte). Wrong tail
    values are worse than none. Restoring row lengths after zeroing helps
    in three models and hurts in one (2.8b). (`semantic_core.py`,
    `results/semantic_core.md`.)
12. For each token's nearest neighbour in the core, the tail pushes the
    pair apart most when they are surface variants of one word (case,
    leading space), less for shared stems, least for different words, in
    every model (12b: 0.110, 0.094, 0.075 in cosine). The push grows with
    model size. The pairs it separates most are punctuation and function
    words that the core nearly merges ('.'/',', ' was'/' were',
    ' the'/' to', ' 0'/' 1').
13. Nearest neighbours get steadily further apart as directions are added
    (median cosine about 0.70 at 64 directions to 0.21-0.35 at full d),
    with no break at the core boundary. At the same number of directions,
    the larger models are emptier.
14. Token identity needs few directions. Identity information
    (definitions.md section 1; `identity_info.py`,
    `results/identity_info.md`) at epsilon 4, dictionary view: the input
    tables carry 6-14 of 15.6 bits in their top 64 directions and nearly
    all of it by 256 to 1024, far below the thousands of directions the
    loss-based test needs (observations 2, 5).
15. Surface-variant information fills earlier than between-word
    information. By the half-fill dimensions (definitions.md 1.3), the
    within-class part reaches half its full-room value first in 78 of 96
    cases (8 models x 2 tables x 2 views x 3 resolutions), at the same k in
    the other 18, and never later (epsilon 4, dictionary: within at 8-64
    directions, between at 32-128; output tables at 8-16).
16. Efficiency (identity carried over the spectrum's capacity ceiling) falls
    as d grows: at epsilon 4, dictionary view, input table 0.67 at d = 512
    down to 0.07 at d = 5120; the output tables match within 0.01-0.04.
17. At the same number of directions, the input table carries more identity
    than the output table (epsilon 4, dictionary, 64 directions: 9.5 vs 7.2
    bits at 1b, 14.3 vs 9.2 at 12b).
18. At the same number of directions, larger models carry more identity
    (input, epsilon 4, dictionary, 64 directions: 6.2 bits at 70m to 14.3
    at 12b). Part of this comes from the noise scale being set per model
    (definitions.md 1.2), so cross-size comparisons at fixed epsilon are
    not like for like.
19. No real crossover. With surface-variant classes, the crossover
    (definitions.md 1.3) appears in 23 of 144 cases (Pythia, TinyLlama,
    OLMo-2; 2 tables, 2 views, 3 resolutions), and in all 23 it falls after
    both parts have saturated, where both per-direction gains are within
    two standard errors of zero. Wherever either part is still growing,
    each direction adds more between-word bits than within-word bits.
    (`identity_info_xvocab.py`, `results/identity_info_xvocab.md`.)
20. The half-fill order holds across vocabularies. In TinyLlama (32K) and
    OLMo-2 (100K) as in Pythia (50K), the within-class part fills first or
    at the same k, never later. The output tables put surface-variant
    information in their top 8 directions in all three families (epsilon 4,
    dictionary); the input tables reach half of it at 32 (TinyLlama,
    Pythia) to 64 (OLMo-2) directions.
21. With grammatical-role classes (12-tag universal part of speech, tagged
    in context; `identity_roles.py`, `results/identity_roles.md`), in the
    text view grammatical role fills before the word within the role: role
    first in 53 of 60 cases (10 models x 2 tables x 3 resolutions), tied in
    7, never later. At epsilon 4, role reaches half its value by 8-16
    directions (OLMo-2 input: 32); the word within the role by 32-64.
22. In the dictionary view (every token with a known role counted equally;
    roles from a lexicon, covering 34-48% of each vocabulary) there is no
    consistent order: role first in 11 of 60 cases, tied in 27, word first
    in 22.
23. With role classes the crossover appears in all 120 cases, at 8-16 (or
    16-32) directions, never within sampling noise. Role carries about 2-3
    bits and the word within a role about 7-12, so absolute within-class
    gains overtake at once. With surface-variant classes (observation 19)
    the reverse holds and no real crossover appears.
24. Pinned noise (definitions.md 1.10, derived before computing;
    `pinned_noise.py`, `results/pinned_noise.md`). Residual-stream
    interference, the context-driven part of what the first block adds to
    a token's vector, gives eps_res = 2.2-2.8 for 70m-410m and 3.5-4.2 from
    1b up. Counting the token-determined part too gives 8.1-20.9. The fp16
    storage floor gives 2.1e-4.
25. At eps_res, identity information reaches 99% of its full-room value by
    128-512 directions in all eight Pythia input tables (dictionary view;
    `identity_pinned.py`, `results/identity_pinned.md`). At eps = 8 the same
    point is 1,280-1,792 directions (512-1,024 for the three smallest).
26. On its own, the input tail (directions past the k*(0.1) core) resolves
    most of the token's identity: at eps_res 10.4 of 15.6 bits (1.4b), 14.0
    (2.8b), essentially all (6.9b, 12b). Nearly all of it is shared with
    the core: the redundancy R is within 0.01 bits of I(tail) at eps_res;
    at eps = 8 the tail adds 0.46 (1.4b), 0.35 (2.8b), 0.06 (6.9b) and 0.03
    (12b) bits beyond the core. (`tail_identity.py`,
    `results/tail_identity.md`.)
27. The tail's class structure matches a shuffled tail. Wherever the tail
    does not already resolve the token (1.4b and 2.8b at eps 4, 8, res;
    6.9b and 12b at eps 8), the true tail's between-class information is
    within two standard errors of all five shuffled tails for
    surface-variant classes in every case, and for role classes in all but
    one: in 12b at eps 8 the true tail carries less role information than
    shuffled (-0.5 to -2.6 standard errors). Gaps above two standard errors
    appear only for role classes at eps = 2 in 1.4b and 2.8b, where the
    tail already resolves 99.6-99.9% of identity, and amount to about 0.01
    bits. Where the tail resolves all of it (6.9b and 12b at eps 2, 4 and
    res; every model at the fp16 floor) the comparison has no power.
28. A Gaussian tail with the true tail's per-direction variances resolves
    more identity than the true tail in the surface view (1.4b, eps 4:
    10.62 vs 10.26 bits; 12b, eps 8: 14.09 vs 13.95). The true tail is not
    isotropic, and its structure lowers distinguishability; by
    observation 27 that structure is not surface-variant or role structure.
29. Nearest neighbours in the core have tails more alike than chance.
    Predicting held-out tokens' tail coordinates from their nearest
    neighbours in the core gives R^2 well above the shuffled-tail baseline
    in every model (k = 5: +0.036, +0.032, -0.056, -0.054 against -0.200 to
    -0.201 for 1.4b, 2.8b, 6.9b, 12b; k = 20: +0.013, +0.011, -0.017, -0.013
    against -0.050; spread across permutations under 0.005). In absolute
    terms the core predicts almost none of the tail: every R^2 is within
    0.06 of zero. (`nonlinear_echo.py`, `results/nonlinear_echo.md`.)
30. A small MLP from core to tail does no better than on a shuffled tail
    (held-out R^2 -0.072 vs -0.068, -0.044 vs -0.044, -0.053 vs -0.044,
    -0.044 vs -0.035). Ordinary least squares is below its shuffled
    baseline (-0.067 vs -0.036 at 1.4b), as expected: principal coordinates
    are uncorrelated over the whole vocabulary, so any correlation fitted on
    the training tokens is reversed on the held-out ones.
31. Over the evaluation-text tokens (3,536), pairwise cosines in the core
    and in the tail are weakly anti-correlated (Spearman -0.024, -0.030,
    -0.032, -0.033), against shuffled-tail baselines within 0.001 of zero.
    Pairs that are close in the core tend, slightly, to be farther apart
    in the tail.

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
