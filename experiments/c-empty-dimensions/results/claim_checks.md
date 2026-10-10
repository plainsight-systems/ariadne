# Checks of two claims (Pythia input tables; rules in claim_checks.py)

## A. Loss at identity's 99% point (text view, squash-test basis, sigma_res)

| Model | d | baseline bits/byte | k | increase | > 0.1 |
|---|---|---|---|---|---|
| pythia-70m | 512 | 1.3449 | 512 | 0.0000 | False |
| pythia-160m | 768 | 1.1519 | 768 | 0.0000 | False |
| pythia-410m | 1024 | 0.9731 | 384 | 0.7767 | True |
| pythia-1b | 2048 | 0.9089 | 1536 | 0.1387 | True |
| pythia-1.4b | 2048 | 0.8635 | 1024 | 0.4701 | True |
| pythia-2.8b | 2560 | 0.8060 | 768 | 1.3333 | True |
| pythia-2.8b | 2560 | 0.8060 | 1024 | 0.8683 | True |
| pythia-6.9b | 4096 | 0.7770 | 512 | 1.7332 | True |
| pythia-12b | 5120 | 0.7444 | 384 | 1.6094 | True |

## B. Tail interventions, ten draws each

| Model | core | zeroed | shuffled mean [min, max] | all shuffled > zeroed | random mean [min, max] | all random > zeroed |
|---|---|---|---|---|---|---|
| pythia-1.4b | 1408 | 0.9400 | 0.9620 [0.9514, 0.9766] | True | 0.9551 [0.9451, 0.9746] | True |
| pythia-2.8b | 1344 | 0.8776 | 0.8984 [0.8915, 0.9145] | True | 0.8921 [0.8840, 0.9017] | True |
