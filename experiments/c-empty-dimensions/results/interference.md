# Crowding of token vectors (d = 2048, centered tables)

x = ratio to a random table of the same V and d. mean |cos|: random token pairs.
nn: median over sampled tokens of |cos| to the nearest other token.
"seen": tokens that occur in the first 70,000 characters of WikiText-103 test.

| Model | V | Table | mean abs cos (all) | x | nn median (all) | x | mean abs cos (seen) | x | nn median (seen) | x |
|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermedi | 32,000 | input | 0.020 | 1.1 | 0.340 | 3.6 | 0.023 | 1.3 | 0.261 | 3.2 |
| TinyLlama-1.1B-intermedi | 32,000 | output | 0.025 | 1.4 | 0.497 | 5.3 | 0.048 | 2.7 | 0.448 | 5.5 |
| pythia-1b | 50,277 | input | 0.022 | 1.2 | 0.505 | 5.3 | 0.026 | 1.5 | 0.336 | 4.1 |
| pythia-1b | 50,277 | output | 0.022 | 1.3 | 0.525 | 5.5 | 0.045 | 2.6 | 0.433 | 5.3 |
| pythia-1.4b | 50,277 | input | 0.023 | 1.3 | 0.518 | 5.4 | 0.027 | 1.5 | 0.344 | 4.2 |
| pythia-1.4b | 50,277 | output | 0.022 | 1.3 | 0.556 | 5.8 | 0.047 | 2.7 | 0.448 | 5.5 |
| OLMo-2-0425-1B | 100,278 | input | 0.021 | 1.2 | 0.461 | 4.6 | 0.026 | 1.5 | 0.394 | 4.8 |
| OLMo-2-0425-1B | 100,278 | output | 0.055 | 3.1 | 0.475 | 4.8 | 0.135 | 7.7 | 0.517 | 6.3 |
