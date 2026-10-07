# Phase 1b: trained table vs the same table at step 0 (Pythia)

Growth = trained variance / step-0 variance along each principal direction of the
trained table. Barely touched = directions where training less than doubled (or
quadrupled) the starting variance.

| Model | d | Table | median growth | barely touched (<2x) | (<4x) | share of d (<4x) | erank of change |
|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | input | 1.0 | 459 | 503 | 0.98 | 506 |
| pythia-70m | 512 | output | 0.9 | 435 | 489 | 0.96 | 496 |
| pythia-160m | 768 | input | 1.3 | 657 | 750 | 0.98 | 760 |
| pythia-160m | 768 | output | 1.6 | 502 | 705 | 0.92 | 733 |
| pythia-410m | 1024 | input | 1.3 | 850 | 990 | 0.97 | 1008 |
| pythia-410m | 1024 | output | 1.3 | 931 | 1018 | 0.99 | 1013 |
| pythia-1b | 2048 | input | 2.2 | 894 | 1720 | 0.84 | 1985 |
| pythia-1b | 2048 | output | 2.2 | 849 | 1933 | 0.94 | 2015 |
| pythia-1.4b | 2048 | input | 1.8 | 1157 | 1802 | 0.88 | 1968 |
| pythia-1.4b | 2048 | output | 2.0 | 1079 | 1979 | 0.97 | 2016 |
| pythia-2.8b | 2560 | input | 1.6 | 1515 | 2179 | 0.85 | 2391 |
| pythia-2.8b | 2560 | output | 2.1 | 1158 | 2425 | 0.95 | 2511 |

## Correction: the growth measure does not show "untouched" directions

Checked after the run (2026-10-07). Trained rows point in almost unrelated
directions from their step-0 rows: mean cosine(trained row, step-0 row) is
0.00 to 0.05 for pythia-70m, 0.02 to 0.18 for pythia-410m, 0.09 to 0.35 for
pythia-2.8b (by fifths of the token-id range, lower ids higher). Row lengths
are similar (step 0: 0.63; trained: 0.70 to 0.99 input). So training
rewrote the rows with new vectors of about the same size, spread across all
directions. Variance per direction then barely changes, which is what the
growth ratio measures. A growth near 1 here does not mean a direction was
left alone. The "barely touched" columns above should not be read as empty
room.

One thing the check does show: the leftover share of the random start rises
with model size (cosine about 0.03 at d = 512, about 0.2 at d = 2560).
Larger tables are rewritten less completely.
