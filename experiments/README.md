# Experiments

One folder per experiment, named after the brief: `a-bounded-context/`,
`b-neural/`.

Each experiment folder holds:

- `README.md`: the hypothesis it tests (linked to its row in `../brief.md`), the
  prediction made in advance, setup, a runs table, and how to reproduce it
- `OBSERVATIONS.md`: numbered observations only, in the order made, kept apart
  from interpretation; superseded ones are marked in place and keep their numbers
- `LOG.md` (optional): dated narrative, with the readings made at the time
- code, with every dependency version pinned; a rule for a test is committed
  before the run it governs
- `results/`: small, committed summaries (tables, plots, logs) of each run;
  `results/superseded/` and `results/parked/` hold runs that were replaced or
  set aside, each with a README saying why. Nothing is deleted.
- `out/`: raw regenerated output, git-ignored

Experiments that drive another repository (experiment B uses Charlotte) record
the exact commit of that repository they ran against.

A result that contradicts its prediction is still recorded and kept.
