# Experiment C: do models have empty dimensions?

*Design sketch, 2026-10-07. Nothing has run. Not yet in the brief's
experiment list (A and B); it comes from the ideas entry "Vocabulary, order,
room and shape".*

## Question

When a model is trained, V (vocabulary size) and d (embedding dimension) are
chosen, and training fills in the V x d embedding table. Does the trained
table actually use all d dimensions, or does it leave some empty?

This is step 1 of two. Step 2, only if step 1 finds empty dimensions: do the
vocabulary and the grammar drive how many dimensions get used? (Andrew's
hypothesis: d should be derivable from the vocabulary and the grammar, not
set as a hyperparameter.)

## Scope

- **Measured:** the trained embedding tables. The input table (token to
  vector) and the output table (vector to token scores) are both V x d, so
  both get measured. Models that tie the two have one table.
- **Not measured yet:** the activations (what the model computes while
  reading text). The later layers reuse the same d dimensions, so a direction
  the vocabulary leaves empty could still carry grammar later. "Empty in the
  table" means empty for the vocabulary, not necessarily for the model. The
  activations are the follow-up.
- **Order:** small models first because the runs are cheap; then a larger
  open-weight model to confirm. The question is not about small models.

## Models

To confirm in phase 0 (from memory, unchecked):

- **Pythia** (EleutherAI): one family trained on the same data in the same
  order at many sizes, d from 512 up, with checkpoints saved through
  training, including step 0. Separate input and output tables. The table is
  believed to be padded past the tokenizer's real vocabulary; if so, the
  padding rows were never trained and give a built-in "unused" reference.
- **Confirmation:** a larger Pythia, plus one model from another family
  (candidates: Llama 3.1 8B, OLMo 2 7B), so the result is not one family's
  habit.

Weights are read in full precision from the original release, not from
quantized files, so quantization noise does not fill in the small
directions. Every model is pinned to an exact release revision.

## Measurements

### 1. Spread (no model run needed)

1. Take the table, drop rows for tokens that never occur in training (count
   them; report separately).
2. Center it (subtract the mean row). Trained tables share a large common
   offset that would otherwise count as a dimension.
3. Singular value decomposition: d numbers saying how much the rows spread
   along each direction.
4. Report the spectrum, plus three summaries:
   - entropy effective rank (primary, chosen now so it is not picked after
     seeing results): e to the entropy of the normalized singular values
   - participation ratio
   - directions needed for 90% and 99% of the variance
5. Twice: every row counted equally (the room the dictionary uses), and rows
   weighted by token frequency in the evaluation text (the room the text
   uses).

### 2. Empty or quiet (needs model runs)

Low spread does not prove a direction is unused. The causal test:

1. Replace every row of the table with its projection onto the top k
   directions (keeping the mean).
2. Run the model on held-out text and measure loss in bits per byte.
3. Sweep k from small to d. Plot loss against k.
4. The used dimension, k*, is the smallest k whose loss is within a tolerance
   of the unmodified model (tolerance fixed before any run; see open
   decisions). Report the whole curve, not just k*.
5. Input table and output table separately.

### 3. Baselines

- **Random table** of the same shape and row lengths: what "all d used"
  looks like.
- **Step 0** (initialization): what the table looked like before training.
- **Never-trained rows** (padding, unseen tokens): what "unused" looks like
  inside the same model.

### 4. Through training (Pythia only)

Repeat measurement 1, and measurement 2 at a few points, on checkpoints
through training for one or two sizes. Shows whether the used dimension
grows, shrinks or settles as training goes on.

## Phases

| Phase | What | Cost |
|---|---|---|
| 0 | Pin models, revisions, evaluation text; confirm each model's V, d, tied or not, padding | none |
| 1 | Spread on small Pythia models | seconds per model, no model runs |
| 2 | Empty-or-quiet test on the same models | small model runs |
| 3 | Through-training checkpoints, one or two sizes | small |
| 4 | Confirm on the larger models | the expensive phase |

**Stop rule:** if phases 1 and 2 find no empty dimensions, record that as the
result and stop before phase 4.

## What would count as "yes"

Proposed, for Andrew to confirm or replace **before any run**: k* is at most
80% of d for the input table in at least two Pythia sizes, and the larger
models confirm it.

## Prediction made in advance

Not yet recorded. Andrew states it here before phase 1 runs. (Kept separate
from any guess Claude has made in conversation.)

## Things that could fool us

- **The common offset:** handled by centering.
- **A few huge "outlier" dimensions:** known in trained transformers (lead,
  unchecked: Timkey and van Schijndel 2021, rogue dimensions). They can make
  the spread look narrower than it is.
- **Weight decay** pushes weights toward low rank, so empty room might come
  from the optimizer, not the language. The step-0 baseline, the
  through-training checkpoints and a second model family help separate the
  two.
- **Never-trained rows** can distort the spread either way: excluded, and
  reported separately.
- **The tolerance and the 90%/99% cut-offs are knobs.** Fixed in advance,
  with full curves reported.
- **Evaluation text frequencies are a stand-in** for training exposure.

## Open decisions

- **Evaluation text:** must be openly licensed, fixed and pinned. Candidate:
  the WikiText-103 test set. It overlaps the kind of text Pythia trained on,
  which is fine for comparing a model against itself.
- **Tolerance for k*:** for example 0.01 bits per byte.
- **Hardware:** the small models run on a laptop CPU. The larger ones need
  more memory than a laptop has in full precision; decide where phase 4 runs.
- **Environment:** Python, with every dependency pinned. Charlotte is not
  used here: it runs quantized files, and it does not train.

## Ariadne's framing rule

Every *claim* keeps a non-neural receiver in it. Step 1 is a measurement of
neural models, not a claim. If step 2 makes a claim, it needs a non-neural
counterpart (for example a vector-quantization codebook, or experiment A's
bounded-context predictor).

## Layout (once it runs)

```
c-empty-dimensions/
  README.md        this design
  ...              code, pinned environment
  results/         small committed summaries: tables, spectra, loss-vs-k curves
  out/             raw output, git-ignored
```
