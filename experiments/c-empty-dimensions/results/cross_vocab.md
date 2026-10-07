# Same room, different vocabularies (d = 2048)

First 70,000 characters of WikiText-103 test for every model. Input table: directions
needed to stay within each tolerance (bits per byte) of the unmodified model, interpolated.

| Model | V | trained on | bytes/token | bits/byte | k +0.1 | k +0.03 | k +0.01 | k +0.003 | output k +0.01 |
|---|---|---|---|---|---|---|---|---|---|
| TinyLlama-1.1B-intermediate-step-1431k-3T | 32,000 | 3T | 3.63 | 0.853 | 1479 | 1828 | 2019 | 2039 | 2045 |
| pythia-1b | 50,277 | 300B | 4.31 | 0.910 | 1601 | 1821 | 1921 | 1977 | 2043 |
| pythia-1.4b | 50,277 | 300B | 4.31 | 0.864 | 1370 | 1553 | 1652 | 1804 | 2044 |
| OLMo-2-0425-1B | 100,278 | ~4T | 4.21 | 0.776 | 2030 | 2042 | 2046 | 2047 | 2045 |
