# Shannon's 1949 count against the measured directions identity needs (Pythia input tables)

At sigma_res. n_flat = 2 H(T) / log2(1 + 1/eps^2) (inputs: H(T) and eps only).
k_C(q) = min k with C(k) >= q I(d), C(k) = 0.5 log2 det(Id + Sigma_k / sigma^2) from the table's
own covariance in the view's basis: a lower bound on the exact k_q. Measured k_q on the grid
(results/identity_pinned.json, results/identity_pinned_text.json).

## Dictionary view (uniform p, unweighted basis), H(T) = 15.62

| Model | d | eps | n_flat | k_C(0.95) | k_C(0.99) | k_C(0.999) | k99 measured | C(d) bits |
|---|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | 2.746 | 174 | 85 | 91 | 92 | 384 | 45.0 |
| pythia-160m | 768 | 2.637 | 161 | 73 | 77 | 79 | 256 | 73.2 |
| pythia-410m | 1024 | 2.202 | 115 | 37 | 39 | 40 | 128 | 134.3 |
| pythia-1b | 2048 | 4.202 | 393 | 121 | 129 | 130 | 512 | 80.1 |
| pythia-1.4b | 2048 | 3.960 | 350 | 88 | 93 | 95 | 384 | 89.3 |
| pythia-2.8b | 2560 | 4.095 | 374 | 66 | 70 | 71 | 256 | 103.4 |
| pythia-6.9b | 4096 | 3.532 | 281 | 38 | 41 | 41 | 256 | 220.1 |
| pythia-12b | 5120 | 3.692 | 306 | 34 | 36 | 36 | 128 | 251.7 |

## Text view, squash-test basis

| Model | d | eps | n_flat | k_C(0.95) | k_C(0.99) | k_C(0.999) | k99 measured | C(d) bits |
|---|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | 2.975 | 122 | 76 | 81 | 82 | 512 | 35.0 |
| pythia-160m | 768 | 2.912 | 117 | 63 | 67 | 68 | 768 | 53.4 |
| pythia-410m | 1024 | 2.389 | 81 | 31 | 33 | 34 | 384 | 96.9 |
| pythia-1b | 2048 | 4.669 | 291 | 106 | 113 | 114 | 1536 | 56.9 |
| pythia-1.4b | 2048 | 4.294 | 248 | 78 | 83 | 84 | 1024 | 65.3 |
| pythia-2.8b | 2560 | 4.462 | 267 | 63 | 67 | 68 | 768 | 74.9 |
| pythia-6.9b | 4096 | 3.866 | 202 | 33 | 36 | 36 | 512 | 147.1 |
| pythia-12b | 5120 | 4.040 | 220 | 28 | 30 | 30 | 384 | 165.5 |

## Text view, frequency-weighted basis

| Model | d | eps | n_flat | k_C(0.95) | k_C(0.99) | k_C(0.999) | k99 measured | C(d) bits |
|---|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | 2.975 | 122 | 20 | 21 | 22 | 256 | 35.0 |
| pythia-160m | 768 | 2.912 | 117 | 13 | 14 | 15 | 256 | 53.4 |
| pythia-410m | 1024 | 2.389 | 81 | 8 | 9 | 9 | 128 | 96.9 |
| pythia-1b | 2048 | 4.669 | 291 | 14 | 15 | 15 | 256 | 56.9 |
| pythia-1.4b | 2048 | 4.294 | 248 | 12 | 12 | 12 | 256 | 65.3 |
| pythia-2.8b | 2560 | 4.462 | 267 | 10 | 11 | 11 | 256 | 74.9 |
| pythia-6.9b | 4096 | 3.866 | 202 | 6 | 7 | 7 | 256 | 147.1 |
| pythia-12b | 5120 | 4.040 | 220 | 6 | 6 | 6 | 256 | 165.5 |
