# Semantic core vs pedantic refinement (Pythia input tables)

k_core = directions needed at +0.1 bits per byte. Loss in bits per byte on the phase-2 text.

## 1. Content or spread

| Model | d | k_core | original | tail zeroed | tail shuffled | tail random | zeroed, lengths kept | core shuffled |
|---|---|---|---|---|---|---|---|---|
| pythia-1.4b | 2048 | 1408 | 0.863 | 0.940 | 0.961 | 0.960 | 0.930 | 3.629 |
| pythia-2.8b | 2560 | 1344 | 0.806 | 0.878 | 0.888 | 0.894 | 0.907 | 3.711 |
| pythia-6.9b | 4096 | 2048 | 0.777 | 0.838 | 0.899 | 0.888 | 0.824 | 3.826 |
| pythia-12b | 5120 | 2048 | 0.744 | 0.844 | 1.032 | 1.047 | 0.821 | 3.570 |

## 2. What the tail separates

Nearest neighbour in the core directions; tail push = core cosine minus full cosine.

| Model | class | share of pairs | core cos | full cos | tail push |
|---|---|---|---|---|---|
| pythia-1.4b | surface | 0.19 | 0.549 | 0.503 | 0.046 |
| pythia-1.4b | stem | 0.20 | 0.521 | 0.481 | 0.040 |
| pythia-1.4b | other | 0.60 | 0.322 | 0.291 | 0.031 |
| pythia-2.8b | surface | 0.19 | 0.563 | 0.493 | 0.070 |
| pythia-2.8b | stem | 0.21 | 0.525 | 0.465 | 0.060 |
| pythia-2.8b | other | 0.60 | 0.330 | 0.282 | 0.048 |
| pythia-6.9b | surface | 0.18 | 0.443 | 0.357 | 0.087 |
| pythia-6.9b | stem | 0.20 | 0.408 | 0.332 | 0.077 |
| pythia-6.9b | other | 0.63 | 0.255 | 0.196 | 0.058 |
| pythia-12b | surface | 0.18 | 0.454 | 0.344 | 0.110 |
| pythia-12b | stem | 0.19 | 0.399 | 0.304 | 0.094 |
| pythia-12b | other | 0.63 | 0.261 | 0.186 | 0.075 |

## 3. More used, emptier

Median nearest-neighbour cosine using only the top k directions.

| Model | k=64 | k=128 | k=256 | k=512 | k=768 | k=1024 | k=1536 | k=2048 | k=2560 | k=3072 | k=4096 | k=5120 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pythia-1.4b | 0.698 | 0.627 | 0.563 | 0.494 | 0.451 | 0.418 | 0.372 | 0.345 |  |  |  |  |
| pythia-2.8b | 0.691 | 0.629 | 0.562 | 0.495 | 0.454 | 0.423 | 0.381 | 0.352 | 0.335 |  |  |  |
| pythia-6.9b | 0.716 | 0.636 | 0.554 | 0.468 | 0.416 | 0.376 | 0.326 | 0.291 | 0.265 | 0.249 | 0.226 |  |
| pythia-12b | 0.717 | 0.648 | 0.569 | 0.477 | 0.422 | 0.385 | 0.330 | 0.294 | 0.270 | 0.251 | 0.225 | 0.210 |

## Pairs the tail separates most (2.8b)

- '.' / ',' (other): core 0.69, full 0.49
- ' -' / '-' (surface): core 0.50, full 0.32
- ' "' / " '" (other): core 0.60, full 0.42
- " '" / ' "' (other): core 0.60, full 0.42
- ' +' / ' -' (other): core 0.42, full 0.26
- ' ,' / ',' (surface): core 0.38, full 0.22
- ' and' / ',' (other): core 0.63, full 0.47
- ' .' / ' ,' (other): core 0.36, full 0.22
- '-' / ' to' (other): core 0.64, full 0.51
- ' was' / ' were' (other): core 0.62, full 0.49
- ' were' / ' was' (other): core 0.62, full 0.49
- ',' / ' to' (other): core 0.69, full 0.56
- ' 0' / ' 1' (other): core 0.48, full 0.35
- ' the' / ' to' (other): core 0.67, full 0.55
- ' by' / ' to' (other): core 0.58, full 0.46
