# Is the input tail a nonlinear echo of the core? (Pythia 1.4B-12B)

Held-out R^2 of the tail predicted from the core (pooled over tail coordinates). Shuffled: mean
[min, max] over 5 permutations. Geometry: Spearman correlation of pairwise cosines, core vs tail,
over the evaluation-text tokens.

| Model | core / tail dims | kNN-5 R^2 | shuffled | kNN-20 R^2 | shuffled | MLP R^2 | shuffled | OLS R^2 | shuffled |
|---|---|---|---|---|---|---|---|---|---|
| pythia-1.4b | 1408 / 640 | +0.0363 | -0.2002 [-0.2039, -0.1964] | +0.0130 | -0.0500 [-0.0509, -0.0489] | -0.0716 | -0.0675 [-0.0677, -0.0672] | -0.0666 | -0.0362 [-0.0363, -0.0360] |
| pythia-2.8b | 1344 / 1216 | +0.0320 | -0.2005 [-0.2022, -0.1975] | +0.0113 | -0.0502 [-0.0507, -0.0495] | -0.0439 | -0.0437 [-0.0439, -0.0436] | -0.0632 | -0.0346 [-0.0346, -0.0345] |
| pythia-6.9b | 2048 / 2048 | -0.0563 | -0.2012 [-0.2040, -0.1981] | -0.0169 | -0.0503 [-0.0511, -0.0496] | -0.0530 | -0.0437 [-0.0439, -0.0436] | -0.0988 | -0.0536 [-0.0538, -0.0534] |
| pythia-12b | 2048 / 3072 | -0.0538 | -0.2012 [-0.2025, -0.1994] | -0.0133 | -0.0503 [-0.0507, -0.0500] | -0.0438 | -0.0345 [-0.0347, -0.0344] | -0.0988 | -0.0536 [-0.0537, -0.0535] |

| Model | tokens | geometry Spearman (true) | shuffled |
|---|---|---|---|
| pythia-1.4b | 3536 | -0.0235 | -0.0001 [-0.0006, +0.0005] |
| pythia-2.8b | 3536 | -0.0301 | +0.0002 [-0.0002, +0.0007] |
| pythia-6.9b | 3536 | -0.0315 | -0.0001 [-0.0009, +0.0006] |
| pythia-12b | 3536 | -0.0325 | -0.0003 [-0.0005, -0.0000] |
