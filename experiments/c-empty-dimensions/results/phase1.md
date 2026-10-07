# Phase 1: spread of trained embedding tables (Pythia)

Trained rows only (padding rows dropped), centered. `erank/d` is the share of d the
spread covers; the random table of the same shape shows the ceiling at that shape.

| Model | d | V | Table | erank | erank/d | random erank | PR | k90 | k99 | random k90 |
|---|---|---|---|---|---|---|---|---|---|---|
| pythia-70m | 512 | 50277 | input | 496 | 0.97 | 511 | 352.2 | 420 | 499 | 452 |
| pythia-70m | 512 | 50277 | output | 468 | 0.91 | 511 | 153.6 | 354 | 485 | 451 |
| pythia-160m | 768 | 50277 | input | 751 | 0.98 | 767 | 566.9 | 641 | 750 | 674 |
| pythia-160m | 768 | 50277 | output | 711 | 0.93 | 766 | 63.9 | 572 | 741 | 671 |
| pythia-410m | 1024 | 50277 | input | 991 | 0.97 | 1021 | 675.8 | 827 | 999 | 894 |
| pythia-410m | 1024 | 50277 | output | 1001 | 0.98 | 1021 | 501.3 | 838 | 993 | 893 |
| pythia-1b | 2048 | 50277 | input | 1952 | 0.95 | 2037 | 1213.5 | 1554 | 1978 | 1760 |
| pythia-1b | 2048 | 50277 | output | 2000 | 0.98 | 2037 | 1027.7 | 1667 | 1991 | 1758 |
| pythia-1.4b | 2048 | 50277 | input | 1926 | 0.94 | 2037 | 1067.9 | 1495 | 1966 | 1760 |
| pythia-1.4b | 2048 | 50277 | output | 2001 | 0.98 | 2037 | 1001.5 | 1665 | 1988 | 1758 |
| pythia-2.8b | 2560 | 50277 | input | 2325 | 0.91 | 2543 | 1039.6 | 1681 | 2415 | 2184 |
| pythia-2.8b | 2560 | 50277 | output | 2496 | 0.97 | 2543 | 1283.7 | 2066 | 2482 | 2181 |
