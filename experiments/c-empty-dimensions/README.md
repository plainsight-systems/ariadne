# Experiment C: do models have empty dimensions?

Exploratory. Started 2026-10-07 from the ideas entry "Vocabulary, order,
room and shape". Written up as working paper 1, `pieces/empty-dimensions/`.

## Question and prediction

V and d are chosen, and training fills in the V x d embedding table. Does
the trained table use all of d?

Andrew's larger theory (brief.md, H5): V x d together set the expressiveness
of the learned language, and d should be derivable from the vocabulary and
the grammar. **Prediction, made before the first run:** if V and d are
chosen arbitrarily, then at a fixed V the embedding dimension is usually too
large, and the trained table leaves room unused. A second prediction, that
the number of directions needed levels off at a value set by the vocabulary,
failed (observation 5).

## Setup

Pythia (V = 50,277; d from 512 to 5,120; the same data in the same order at
every size). Depth and head count change with d; pythia-1b and pythia-1.4b
share d = 2048, a built-in control. TinyLlama 1.1B and OLMo-2 1B (d = 2048)
for the cross-vocabulary runs. Weights in full precision, pinned revisions in
`models.json`. Evaluation text: WikiText-103 test, first 16 x 1,024 tokens for
Pythia, first 70,000 characters for the cross-vocabulary and role runs.

## Runs

Superseded and parked runs are kept in `results/superseded/` and `results/parked/`, each with a README saying why.


| Script | What | Results |
|---|---|---|
| `measure.py` | Spread of the trained input and output tables across d, vs a random table of the same shape | `results/phase1.md`, `results/spectra.png` |
| `measure_init.py` | Trained table vs the same table at training step 0 (superseded) | `results/superseded/phase1b.md` (read its correction), `results/superseded/growth.png` |
| `refine.py` | Finer k steps, larger models, checkpoints | `results/phase3_*` |
| `cross_vocab.py` | Same d, different vocabularies (TinyLlama, Pythia, OLMo-2) | `results/cross_vocab.*` |
| `interference.py` | Crowding of token vectors in the d = 2048 models | `results/interference.*` |
| `through_training.py` | (Parked) Spread and crowding of 2.8b's input table through training | `results/parked/through_training.*` |
| `semantic_core.py` | Semantic core vs pedantic refinement: tail content, what the tail separates, emptiness | `results/semantic_core.*` |
| `identity_info.py` | Identity information (definitions.md section 1) on all Pythia tables | `results/identity_info.*` |
| `identity_info_xvocab.py` | The same on TinyLlama and OLMo-2 (d = 2048), with both the crossover and the half-fill comparison | `results/identity_info_xvocab.*` |
| `identity_roles.py` | Identity information with grammatical-role classes, all ten models | `results/identity_roles.*` |
| `pinned_noise.py` | Noise derived from the receiver (definitions.md 1.10) for all Pythia input tables | `results/pinned_noise.*` |
| `identity_pinned.py` | 99% point of identity information at the pinned noise | `results/identity_pinned.*` |
| `identity_pinned_text.py` | The same in the text view, noise fixed in absolute terms, in the squash-test basis and the frequency-weighted basis | `results/identity_pinned_text.*` |
| `nonlinear_echo.py` | (Superseded) Is the tail a nonlinear echo of the core: kNN, MLP and OLS prediction of the tail from the core; core-tail geometry correlation | `results/superseded/nonlinear_echo.*` |
| `tail_identity.py` | What the tail resolves on its own, against shuffled and random tails; redundancy with the core | `results/tail_identity.*` (the first run's role view permuted across all tokens; rerun permuting within the view's tokens, `results/tail_identity_role_rerun.log`) |
| `tail_capacity.py` | Gaussian capacity of the core and the tail beside their measured identity (no new draws) | `results/tail_capacity.*` |
| `capacity_count.py` | Shannon's 1949 count (token entropy and noise only) and the capacity lower bound on the directions identity needs, beside the measured k99 (no new draws) | `results/capacity_count.*` |
| `identity_fine.py` | Identity's cut points on a grid of 16, and Shannon's count against them with a rule committed before running | `results/identity_fine.*` |
| `claim_checks.py` | Loss measured at identity's k99; ten shuffles and ten random tails at 1.4b and 2.8b | `results/claim_checks.*` |
| `plot_identity_vs_need.py` | The paper's figure of identity's k99 against the network's k*(0.01) | `pieces/empty-dimensions/figures/identity-vs-need.png` |
| `project.py` | Causal test: squash a table onto its top k directions, run the model on WikiText-103 test, find where loss stops changing | `results/phase2.md`, `results/loss_vs_k.png`, `results/phase2.log` |

## Where it stands

- Numbered observations, kept apart from interpretation: `OBSERVATIONS.md`
  (1 to 47).
- Dated narrative of the phases, with the readings made at the time:
  `LOG.md`.
- The write-up and its interpretation: `pieces/empty-dimensions/paper.md`.
- Next: experiment D, the constructed toddler language (brief.md).

## Reproduce

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python measure.py
.venv/bin/python measure_init.py
```

Python version in `python-version.txt`. Downloads go to `out/` (git-ignored),
about 25 GB for both scripts.
