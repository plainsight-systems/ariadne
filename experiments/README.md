# Experiments

One folder per experiment, named after the brief: `a-bounded-context/`,
`b-neural/`.

Each experiment folder holds:

- a README stating the hypothesis it tests, the prediction made in advance, and
  how to reproduce it
- code, with every dependency version pinned
- `results/`: small, committed summaries (tables, plots) of a run, with the
  commit and parameters that produced them
- `out/`: raw regenerated output, git-ignored

Experiments that drive another repository (experiment B uses Charlotte) record
the exact commit of that repository they ran against.

A result that contradicts its prediction is still recorded and kept.
