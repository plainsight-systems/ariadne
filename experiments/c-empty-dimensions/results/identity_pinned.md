# 99% point of identity information at the pinned noise (Pythia input, dictionary view)

k99 = smallest k on the grid with I(k) >= 0.99 I(d).

| Model | d | eps_res | k99 at eps_res | I(d) at eps_res | k99 at eps 2 | 4 | 8 | k99 at eps_prec |
|---|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | 2.75 | 384 | 15.58 of 15.62 | 128 | 512 | 512 | 8 |
| pythia-160m | 768 | 2.64 | 256 | 15.58 of 15.62 | 128 | 768 | 768 | 8 |
| pythia-410m | 1024 | 2.20 | 128 | 15.59 of 15.62 | 128 | 512 | 1024 | 8 |
| pythia-1b | 2048 | 4.20 | 512 | 15.59 of 15.62 | 128 | 384 | 1792 | 8 |
| pythia-1.4b | 2048 | 3.96 | 384 | 15.59 of 15.62 | 64 | 384 | 1792 | 8 |
| pythia-2.8b | 2560 | 4.10 | 256 | 15.60 of 15.62 | 64 | 256 | 1792 | 8 |
| pythia-6.9b | 4096 | 3.53 | 256 | 15.62 of 15.62 | 64 | 256 | 1792 | 8 |
| pythia-12b | 5120 | 3.69 | 128 | 15.62 of 15.62 | 32 | 256 | 1280 | 8 |
