# Prior art for the empty-dimensions paper (2026-10-09)

Three searches run by fresh-context agents during the paper's revision, one per
claim cluster. Status of each source: "opened" means the search agent read the
page (abstract or full text, as marked); nothing here is in `refs.bib` until it
is checked against the paper itself.

## 1. A dimension count from vocabulary and noise (Shannon's count)

Verdict: the measured comparison (identity information of a trained LM table
under Gaussian noise against 2 H(T) / log2(1 + SNR)) was not found. The counting
form is not new. Confidence: moderate-high that the measurement is new; high that
the form is not.

Must cite and distinguish:
- Weller, Boratko, Naim, Lee, "On the Theoretical Limitations of Embedding-Based
  Retrieval", ICLR 2026, arXiv 2508.21038. Thm 1 / Eq. 2: d >= log C(n,k) /
  log(1 + 1/gamma) by sphere-packing volume. Margin, not noise; retrieval, not
  identity; log n, not entropy; a bound, not a measurement. Opened.
- Yin and Shen, "On the Dimensionality of Word Embedding", NeurIPS 2018,
  arXiv 1812.04224. Thm 3: optimal dimension set by signal spectrum against
  noise (estimation noise on PMI). Conceptual precedent for "dimension from
  signal and noise". Opened.
- Kataiwa, Hakaze/Cho, Ohki, arXiv 2503.02142 (already cited as kataiwa2025):
  Pythia input-embedding ID 25-122.
- Quemy, "A Hub of Short Rows Inflates Intrinsic Dimension Estimation of Token
  Embeddings", arXiv 2608.29702 (Aug 2026, NeurReps submission). Trimmed ID
  collapses to a narrow range; Pythia's rise with size disappears. Abstract only.
  The paper must say why identity's few hundred differs from ID of 10-120.
- Lan et al., ALBERT, ICLR 2020, arXiv 1909.11942: E << H chosen empirically.
- Kuratov et al., arXiv 2502.13063, Eq. 1: L <= d b / log2|V| (precision form).
- Shi, Li, Han, Hernandez-Lobato, arXiv 2609.06862 (Sep 2026), Table 2:
  log2 N <= m log2(1 + 2R/delta); "width alone constrains nothing without a
  precision or noise model". Partly read.
- Elhage et al. 2022, Toy Models of Superposition (near-orthogonal packing).
- Guha, arXiv 2606.02765 (already in the library). Abstract only.
- Feyisetan et al., WSDM 2020, arXiv 1910.08902: identity survival under noise
  added to word embeddings, empirical.
- Frady, Kleyko, Sommer, arXiv 1707.01429: Shannon capacity of decoding
  superposed random codes (a non-neural-LM receiver). Abstract only.
- Patel and Bhattacharyya, IJCNLP 2017, ACL I17-2006: geometric lower bound.

## 2. The gap: what the network uses beyond identity

- Claim A (input squashable, output not): partly anticipated. ALBERT Table 3;
  Wies et al., ICML 2021, arXiv 2105.03928 (embedding rank bottleneck); a blog
  by Giles Thomas, 9 Oct 2026 (GPT-2 small from scratch: rank bottleneck on the
  input barely hurts, on the output hurts). Cho et al., COLING 2025, arXiv
  2406.01468: 30-40% of axis-aligned OUTPUT dimensions of Pythia-2.8B zeroed
  with MAUVE > 0.8. In tension with "output needs all d" at a much looser test;
  reconcile. SliceGPT already cited.
  Not to be used: the searcher offered the softmax bottleneck and Borenstein's
  output-rank bound as explanations of the output result. Andrew rejected that
  explanation (2026-10-08); the output result stays an observation.
- Claim B (identity few hundred vs network thousands): appears open. Referee
  will say "embeddings are known to be low-dimensional" (Kataiwa, Quemy); our
  result cuts against reading ID as unused dimensions.
- Claim C (grammar hypothesis): open as a hypothesis. The referee's alternative
  is bigram statistics: Elhage et al. 2021 (zero-layer transformers model
  bigrams); Chang and Bergen, NeurIPS 2025, arXiv 2504.15471 (bigram
  subnetworks in Pythia-1B's first MLP); Papadimitriou and Prince, arXiv
  2510.07613. The grammar test must separate state from bigram statistics.
  Theory: Borenstein 2024 (cited, empirical result only), Svete and Cotterell
  arXiv 2310.05161, Hewitt et al. arXiv 2010.07515.
- Claim D (frequency-weighted basis): method anticipated. Yokoi, Bao, Kurita,
  Shimodaira, "Zipfian Whitening", NeurIPS 2024, arXiv 2411.00680 (must cite);
  Gao et al., ICLR 2019, arXiv 1907.12009 (degeneration from rare words);
  Raunak et al. 2020 (cited).

## 3. The receiver's noise and identity information

- Noise from block 1: nothing found that does this. Must distinguish:
  Ethayarajh, EMNLP 2019, arXiv 1909.00512 (self-similarity, context variance);
  Abrahao, "Neural Collapse Is Forbidden: Information Floors in Language
  Models", arXiv 2607.09487 (July 2026; within-token context variance 79-91%
  across 14 models incl. Pythia, read as stored information, not noise; the
  closest risk to calling context "noise"); Feucht et al., arXiv 2406.20086
  (token erasure); Lad et al., arXiv 2406.19384 (detokenization stage);
  Kamoda et al., NAACL Findings 2025, arXiv 2501.15754.
- Identity information swept over ordered PCs with a chain-rule split: the
  combination not found. Must cite: Pimentel et al., ACL 2020, arXiv
  2004.03061 (conditional-MI probing); Hewitt et al., EMNLP 2021, arXiv
  2109.09234 (conditional probing); Torroba Hennigen et al., EMNLP 2020, arXiv
  2010.02812 (information against number of dimensions, greedy selection);
  Greenewald et al., ISIT 2023, arXiv 2305.04712 (PCA plus smoothed entropy);
  Goldfeld et al., k-sliced and max-sliced MI, arXiv 2206.08526, 2309.16200;
  Lozano, Tulino, Verdu, IEEE Trans. IT 52(7):3033-3051, 2006, DOI
  10.1109/TIT.2006.876220 (mercury/waterfilling over parallel channels with
  arbitrary inputs; Crossref only). Not opened, worth checking: Chechik et al.
  2005 Gaussian IB (JMLR 6).
- Case and spacing before word, POS before word: nothing found.
- Credits' metadata check (Crossref): Ungerboeck 1982, Wachsmann 1999,
  Perez-Cruz 2010, Goldfeld 2019, Brunner 2020 match. MCR2 venue (NeurIPS 2020)
  not confirmed from the arXiv page.
