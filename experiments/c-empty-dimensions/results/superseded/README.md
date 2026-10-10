# Superseded results

Kept because results are kept, including ones that turned out wrong or were
replaced. Nothing here is cited by the paper. Rerunning a script writes to
`results/`, not here.

| Files | Script | Why superseded |
|---|---|---|
| `phase1b.*`, `growth.png` | `measure_init.py` | The growth metric was misleading; see the correction in `phase1b.md` |
| `nonlinear_echo.*` | `nonlinear_echo.py` | Tested the tail beyond k*(0.1) against unmatched or unpaired baselines; observations 29-31 |
| `tail_identity_first_run.log` | `tail_identity.py` | The first run's role-view baseline permuted across all tokens; rerun in `../tail_identity_role_rerun.log` |
