# phase3_checkpoints: squash test (k* within 0.01 bits/byte of unmodified)

| Model | checkpoint | d | baseline bits/byte | input k* | input k*/d |
|---|---|---|---|---|---|
| pythia-2.8b | step1000 | 2560 | 1.6021 | 1920 | 0.75 |
| pythia-2.8b | step8000 | 2560 | 0.9959 | 1408 | 0.55 |
| pythia-2.8b | step16000 | 2560 | 0.9343 | 1408 | 0.55 |
| pythia-2.8b | step33000 | 2560 | 0.8884 | 1408 | 0.55 |
| pythia-2.8b | step66000 | 2560 | 0.8473 | 1536 | 0.60 |
| pythia-2.8b | step100000 | 2560 | 0.8204 | 1664 | 0.65 |
| pythia-2.8b | step143000 | 2560 | 0.8060 | 1664 | 0.65 |
