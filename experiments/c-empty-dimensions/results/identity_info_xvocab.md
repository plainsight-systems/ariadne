# Identity information of the Pythia embedding tables

definitions.md section 1. Classes: surface variants. I in bits; k_B50, k_W50 = half-fill dimensions;
k_x = crossover interval.

## dictionary view, epsilon = 2.0

| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 | k_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | input | 14.97 | 14.95 | 14.17 | 0.78 | 0.047 | 16 | 16 | 8 | 1024-1280 |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | output | 14.97 | 14.95 | 14.17 | 0.78 | 0.047 | 32 | 32 | 8 | none |
| OLMo-2-0425-1B | 2048 | input | 16.61 | 16.61 | 15.61 | 1.00 | 0.052 | 16 | 16 | 8 | none |
| OLMo-2-0425-1B | 2048 | output | 16.61 | 16.61 | 15.61 | 1.00 | 0.055 | 16 | 16 | 8 | none |

## dictionary view, epsilon = 4.0

| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 | k_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | input | 14.97 | 14.94 | 14.16 | 0.78 | 0.168 | 64 | 64 | 32 | none |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | output | 14.97 | 14.95 | 14.17 | 0.78 | 0.170 | 128 | 128 | 8 | none |
| OLMo-2-0425-1B | 2048 | input | 16.61 | 16.55 | 15.56 | 0.99 | 0.187 | 64 | 64 | 64 | none |
| OLMo-2-0425-1B | 2048 | output | 16.61 | 16.59 | 15.59 | 1.00 | 0.198 | 128 | 128 | 8 | none |

## dictionary view, epsilon = 8.0

| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 | k_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | input | 14.97 | 14.14 | 13.38 | 0.76 | 0.619 | 384 | 384 | 256 | none |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | output | 14.97 | 14.25 | 13.50 | 0.74 | 0.627 | 384 | 512 | 256 | none |
| OLMo-2-0425-1B | 2048 | input | 16.61 | 14.73 | 13.80 | 0.93 | 0.645 | 384 | 384 | 256 | none |
| OLMo-2-0425-1B | 2048 | output | 16.61 | 15.21 | 14.24 | 0.97 | 0.691 | 512 | 512 | 128 | none |

## text view, epsilon = 2.0

| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 | k_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | input | 9.09 | 9.09 | 8.79 | 0.30 | 0.036 | 8 | 8 | 8 | none |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | output | 9.09 | 9.09 | 8.79 | 0.30 | 0.040 | 8 | 8 | 8 | none |
| OLMo-2-0425-1B | 2048 | input | 9.24 | 9.23 | 8.99 | 0.24 | 0.035 | 16 | 16 | 8 | 1792-2048 |
| OLMo-2-0425-1B | 2048 | output | 9.24 | 9.24 | 9.00 | 0.24 | 0.044 | 8 | 8 | 8 | none |

## text view, epsilon = 4.0

| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 | k_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | input | 9.09 | 9.09 | 8.79 | 0.30 | 0.117 | 32 | 32 | 16 | none |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | output | 9.09 | 9.09 | 8.79 | 0.30 | 0.128 | 32 | 32 | 8 | 768-1024 |
| OLMo-2-0425-1B | 2048 | input | 9.24 | 9.15 | 8.91 | 0.24 | 0.115 | 32 | 32 | 32 | none |
| OLMo-2-0425-1B | 2048 | output | 9.24 | 9.24 | 9.00 | 0.24 | 0.138 | 16 | 32 | 8 | 1536-1792 |

## text view, epsilon = 8.0

| Model | d | table | H(V) | I(d) | I_B(d) | I_W(d) | I/C at d | k for half of I(d) | k_B50 | k_W50 | k_x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | input | 9.09 | 8.90 | 8.60 | 0.30 | 0.410 | 64 | 64 | 64 | none |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 2048 | output | 9.09 | 8.90 | 8.60 | 0.30 | 0.437 | 64 | 64 | 16 | 1792-2048 |
| OLMo-2-0425-1B | 2048 | input | 9.24 | 8.17 | 7.95 | 0.21 | 0.372 | 64 | 64 | 64 | none |
| OLMo-2-0425-1B | 2048 | output | 9.24 | 9.06 | 8.83 | 0.23 | 0.465 | 64 | 64 | 8 | none |

