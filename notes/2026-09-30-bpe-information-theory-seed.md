---
status: seed
kind: ic-piece
posted:
scheduled:
---

# BPE has an information theory, and it's not the one we talk about

> **Superseded 2026-10-03 by Ariadne** (`brief.md` at the repository root). The project is not about BPE; BPE is
> how the question arose. This note stays as the detailed research record the brief cites.

*Seed captured 2026-09-30. Not drafted. IC voice (see memory ic-showcase-voice). Natural home: Charlotte, which will
train toy models with their own tokenizers, so this can be demonstrated rather than asserted.*

## Andrew's question (the origin)
Tokenizer discussions are always practical: vocabulary length versus compute. Is there an information-theory base?
"We are combining signals to roots that contain a combination of expressiveness but reuse at the same time."

## The answer in one line
BPE was born as a compression algorithm, it greedily builds a compressed grammar of the corpus, and the
expressiveness-versus-reuse trade Andrew described has a precise name: Minimum Description Length (MDL). BPE measures
reuse but never prices expressiveness.

## Lineage
- **1994, Philip Gage, "A New Algorithm for Data Compression"** (C Users Journal): replace the most frequent byte pair
  with an unused byte; pure compression.
- **1999–2000, Re-Pair** (Larsson & Moffat): the same algorithm as grammar-based compression; each merge is a rule
  X → ab, and the merge table is a small grammar that regenerates the text.
- **Smallest grammar problem** (Charikar et al., IEEE Trans. IT, 2005): NP-hard; BPE is a greedy approximation.
- **2016, Sennrich, Haddow & Birch:** BPE as subword units for NMT; today's tokenizers inherit this.

## Three lenses
1. **Source coding (Shannon):** a merge of a pair seen c times shortens the sequence by c. Greedy on frequency, not
   probability; never asks how many bits a token is worth. **Unigram LM tokenizer** (Kudo 2018, SentencePiece)
   chooses the vocabulary by corpus likelihood: the explicitly information-theoretic sibling.
2. **MDL (Rissanen), the core of the piece:** total description = bits for the model (vocabulary: embedding rows,
   undertrained rare tokens) + bits for the data given the model (sequence length: compute, context). A token earns
   its place when reuse (count × length saved) pays for its description. The vocab-size/compute debate IS the MDL
   trade-off; BPE leaves vocab size as a hand-set hyperparameter where MDL says it should fall out of the data.
3. **Roots = boundary entropy:** Zellig Harris (1955), "From Phoneme to Morpheme": morpheme boundaries sit where
   next-character uncertainty spikes ("walk|ing"). **Morfessor** (Creutz & Lagus) found morphemes with an explicit
   MDL objective. BPE merges by frequency, so its tokens often cut across real morphemes.

## Recent research (read before quoting specifics)
- **Zouhar et al. 2023, "Tokenization and the Noiseless Channel"** (ACL): tokenization as channel coding; Rényi
  efficiency predicts downstream MT quality better than plain compression.
- **Zouhar et al. 2023, "A Formal Perspective on Byte-Pair Encoding"** (Findings of ACL): BPE as greedy optimization
  of a compression objective with an approximation guarantee. VERIFY the exact bound before citing.
- **Counterpoint: Schmidt et al. 2024, "Tokenization Is More Than Compression"** (EMNLP): a minimal-token tokenizer
  did not produce better models.
- **Delétang et al., "Language Modeling Is Compression":** tokenizer = cheap first compression stage, model does the
  rest; why bits per byte, not perplexity per token, is the fair cross-tokenizer comparison.

## Andrew's angle (draft claims)
- Right: tokens are reusable codewords; the trade is MDL's model cost vs data cost.
- Gap: BPE prices reuse and ignores expressiveness; frequency stands in for information; vocab size chosen outside
  the algorithm; "roots" are frequent strings, not morphemes.
- Open question to end on: tokenization quality isn't just compression (Schmidt et al.); something about which units
  a model can learn from matters, and nobody has nailed it.

## Demonstration idea (Charlotte)
Train tiny models on the same corpus with BPE vs Unigram vs an MDL-chosen vocab size; report bits per byte; visualize
merges on a word cloud of morphemes vs BPE splits. Visual hook per the IC formula (a picture of where BPE cuts words
vs where entropy says the boundaries are).

## Before publishing
- Read both Zouhar papers and Schmidt et al. in full; confirm findings and any bounds quoted.
- Credit every lineage item above; claim only the synthesis (MDL as the missing frame for the practical debate) and
  any Charlotte results.
- Check whether someone has already written "BPE through MDL" for a general audience; credit if so.

## Research pass (2026-10-02)

**Verified:**
- Zouhar, Meister, Gastaldi, Du, Vieira, Sachan, Cotterell, "A Formal Perspective on Byte-Pair Encoding" (ACL 2023,
  arXiv 2306.16837): BPE as a combinatorial optimization over merge sequences; greedy achieves a
  (1/σ)(1−e^{−σ}) approximation via submodularity (σ = total backward curvature), empirically ≈ 0.37; faster BPE
  O(N log M). (The approximation is to optimal *compression*, not downstream quality.)
- Zouhar et al., "Tokenization and the Noiseless Channel" (ACL 2023, arXiv 2306.16842): Rényi efficiency (α = 2.5)
  correlates with BLEU at 0.78, versus −0.32 for compressed length. Penalizes very skewed token-frequency distributions.
- Schmidt, Reddy, Zhang, Alameddine, Uzan, Pinter, Tanner, "Tokenization Is More Than Compression" (EMNLP 2024, arXiv
  2402.18376): PathPiece (minimum tokens for a fixed vocab) does not improve downstream performance; pre-tokenization
  and BPE-based vocabulary initialization matter.

**PRIOR ART on the MDL framing (so MDL is NOT Andrew's novel claim):**
- Chung, Garrette, et al., "Improving Multilingual Models with Language-Clustered Vocabularies" (EMNLP Findings 2020,
  arXiv 2010.12777): explicitly uses MDL (Rissanen 1989), following Goldwater (2007), defining description length for a
  subword vocabulary as vocabulary size plus the number of encoded integers.
- Nouri, "MDL-Calibrated Significance-Gain Pair Encoding" (arXiv 2609.31705, 2026-09-20): MDL criterion (sequence cost +
  vocabulary penalty + merge-rule cost) stops merging automatically, no target vocab size. On a small benchmark: 3.16%
  better test bits-per-character than frequency BPE, while frequency BPE compresses more (tokens per character).
- Unigram LM (Kudo 2018) is widely described as MDL-flavored likelihood optimization.

**Revised angle (what's left for Andrew):** a practitioner synthesis, credited, not a theory claim. Three independent
results converge on one practical message the vocab-size/compute debate misses: fewer tokens is not the objective.
Zouhar (compressed length anti-correlates with BLEU; balanced usage predicts it), Schmidt (minimum tokens doesn't help),
Nouri (less compression, better bits per character under an MDL stop rule). The lineage (Gage compression → Re-Pair
grammar → MDL) explains why: BPE optimizes the data half of the description and ignores the codebook half.
Original contribution candidates: the practitioner bridge + Charlotte results (BPE vs Unigram vs MDL-stopped vocab on
toy models, reported in bits per byte, with a visual of where each cuts words).

## Andrew's push (2026-10-02): vocab size should fall out, and the learning link

**Andrew's position:**
1. A hyperparameter is a sign the theory is incomplete (like the Standard Model's free parameters). Vocabulary size
   should fall out of the grammar of what you compress, or at least an optimization with a cliff point.
2. There's a known-but-unexplained coupling between vocab and model learning; expressiveness vs compute, and possibly
   lossy behavior past a drop-off point in learning.

**Framing we landed on:** V is a free knob because the theory only has half the objective.
- Savings side = the data's grammar (MDL: merging saves encoded length).
- Cost side = the learner, not the data: each entry costs parameters (embedding + output rows, V × d), softmax compute
  per token (∝ V), and learnability (rare tokens get too few gradient updates).
- Lossless on paper, lossy in practice: undertrained tokens (0.1–1% of real vocabs severely undertrained, Land &
  Bartolo, "Fishing for Magikarp", EMNLP 2024 Outstanding Paper) and sub-token blindness (spelling inside a token).

**Evidence the cost side moves with the learner:**
- Tao et al., "Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies" (NeurIPS 2024, arXiv
  2407.13623): optimal V grows with compute; Llama 2 70B should have used ≥216K vs 32K; 32K→43K at fixed compute took
  ARC-Challenge 29.1→32.0.
- Chung & Kim, "Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training" (NeurIPS 2025, arXiv
  2508.15390): scaling V from 24K to 196K at fixed data/compute lowers loss almost entirely on the ~2,500 most frequent
  words while rare-word loss rises; reframe as "lowering complexity of tokenized text helps." No formula for choosing V
  from training exposure.

**The learnability-floor hypothesis (PRIOR ART EXISTS):**
- Andrew's/our hypothesis: add a token only if training will show it enough times to learn it; the optimal V is where
  the compression-savings curve meets a learnability floor set by training budget and model size.
- **Prior art:** Gowda & May, "Finding the Optimal Vocabulary Size for Neural Machine Translation" (Findings of EMNLP
  2020): empirical heuristic for Transformer NMT, "use the largest BPE vocabulary such that at least 95% of classes have
  100 or more examples in training", motivated by classification class-imbalance. A heuristic, not derived; NMT-scale.
- **What appears open (moderate confidence, no paper found deriving it):** whether an exposure floor, scaled by
  training tokens and model size, *predicts* Tao et al.'s compute-optimal V for LLMs, i.e., one joint objective that
  removes V as a free parameter.

**Charlotte experiment (original-contribution candidate):**
1. Fix corpus; sweep V (e.g., 512 → 32K) at fixed compute on toy models.
2. Measure bits per byte (fair across tokenizers) + count undertrained tokens (Magikarp-style weight indicators) + the
   fraction of tokens with ≥k training occurrences (Gowda-style).
3. Find the knee; then vary training budget and model size and test whether the knee moves the way an exposure floor
   predicts. If yes: evidence for a derived V. If no: the coupling is something else (worth saying either way).
4. Visual hook: the knee plot, plus where BPE vs MDL-stopped vocabularies cut the same words.

**Credit list for any public piece:** Gage 1994; Larsson & Moffat (Re-Pair); Charikar et al. 2005; Sennrich et al. 2016;
Rissanen (MDL); Goldwater 2007; Chung, Garrette et al. 2020; Kudo 2018; Zouhar et al. 2023 (x2); Schmidt et al. 2024;
Nouri 2026; Tao et al. 2024; Chung & Kim 2025; Land & Bartolo 2024; Gowda & May 2020; Harris 1955; Creutz & Lagus.

## Shannon: the root (2026-10-02)

**What Shannon says:**
1. **Information belongs to the source, not the alphabet** (Shannon, "A Mathematical Theory of Communication", 1948).
   Re-chunking text into tokens redistributes the information, it doesn't remove it: bits/token = bits/char ×
   chars/token. With an ideal model, bits per byte is invariant to the tokenizer. A tokenizer can only move work
   between a free deterministic lookup and the model.
2. **Block coding approaches entropy.** The source coding theorem codes longer blocks to bring average code length
   toward H. Merging is a crude version of that, but probability-blind: Shannon's optimal code length is −log₂ p; BPE
   counts frequency and never assigns bits (Unigram does use probabilities).
3. **Language modeling is his guessing game** (Shannon, "Prediction and Entropy of Printed English", 1951): humans
   predicting the next letter; English estimated at ~0.6–1.3 bits per character. An LLM is Shannon's predictor;
   bits per byte is his measure.

**Lineage table:**
| Year | Step | Adds |
|---|---|---|
| 1948 | Shannon, source coding | entropy is the floor; block coding approaches it |
| 1951 | Shannon, prediction game | language as next-symbol prediction, in bits per character |
| 1952 | Huffman | optimal fixed-to-variable codes |
| 1967 | Tunstall (Georgia Tech thesis) | variable-to-fixed codes: dictionary of variable-length strings → fixed-size codewords = what a tokenizer is |
| 1977–78 | Lempel–Ziv | dictionaries built adaptively from data |
| 1994 | Gage, BPE | greedy pair merging as compression |
| 1999–2000 | Larsson & Moffat, Re-Pair | BPE as grammar compression |
| 2016 | Sennrich, Haddow & Birch | BPE as subword units for neural models |

Tunstall note: dictionary size follows from the chosen codeword length (2^k); for a simple (memoryless) source,
efficiency keeps improving as the dictionary grows. No natural cliff in the coding theory; the trade-off appears only
when the dictionary's cost is counted.

**Synthesis (ours; needs a literature check before claiming):** if an ideal model makes bits per byte invariant to
tokenization, vocabulary size matters only because the learner isn't ideal (finite capacity, data, exposure per token).
The hyperparameter is an unfinished theory of the learner, not of the language; Shannon finished the language part.
Consistent with Tao et al. (V moves with compute), Chung & Kim (frequent-word gains, rare-word losses), Magikarp
(undertrained tokens), Gowda & May (exposure floor). Any cliff lives in the learner's limits.

**Candidate opening line:** "Shannon proved the tokenizer can't matter. It matters anyway, which tells you something
about the model."

**Before publishing, also check:** whether the "invariance with an ideal model → V is a property of the learner" framing
already exists (likely places: Delétang et al. "Language Modeling Is Compression"; tokenization-theory papers by
Gastaldi, Cotterell et al.; "Tokenization Is More Than Compression" discussion).

## Prior-art check: Shannon + the imperfect learner (2026-10-02)

Andrew's question: someone must have extended Shannon to a non-perfect learner, some form of signal loss. **Yes. This
is established, and recent.**

- **V-information** (Xu, Zhao, Song, Stewart, Ermon, "A Theory of Usable Information Under Computational
  Constraints", ICLR 2020): a variational extension of Shannon that accounts for the observer's modeling power and
  compute. Unlike Shannon mutual information, and violating the data-processing inequality, V-information *can be
  created by computation*. Lens: a tokenizer is a deterministic computation that can increase the information a bounded
  learner can use, while Shannon information is unchanged.
- **Epiplexity** (Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson, "From Entropy to Epiplexity: Rethinking Information for
  Computationally Bounded Intelligence", arXiv 2601.03220, Jan 2026): Shannon and Kolmogorov "assume observers with
  unlimited computational capacity." Splits data into epiplexity (structure a bounded learner can extract) and
  time-bounded entropy (what looks random to that learner = Andrew's "signal loss"). "Paradox 2": classical information
  is invariant to factorization, bounded learners are not. Uses time-bounded MDL and prequential coding. Reports that
  VQ tokenization significantly raises epiplexity for image data. No discussion of text vocabulary size or BPE found
  (grepped the full paper).
- **Rajaraman, Jiao, Ramchandran, "Toward a Theory of Tokenization in LLMs"** (NeurIPS 2024, arXiv 2404.08335):
  transformers trained on k-th-order Markov data without tokenization fall back to a unigram model over characters;
  with tokenization, even unigram models over tokens reach near-optimal cross-entropy. Tokenization compensates for
  learner limits. Direct support for "the tokenizer matters because the learner is imperfect."
- **Erdogan, Gorle, Chandak, Pilanci, Weissman, "An Information-Theoretic Perspective on LLM Tokenizers"** (arXiv
  2601.09039, Jan 2026): tokenizer scale redistributes entropy (higher unigram entropy, lower higher-order conditional
  entropy); channel capacity/utilization metrics; compression-aware BPE.
- **ByteSpan** (Goriely, Salhan, Lesci, Cheng, Buttery, arXiv 2506.18639, 2025): uses a byte-level LM's prediction
  error spikes to place boundaries (Harris's 1955 idea, operationalized); better morphological alignment than BPE;
  similar compression and Rényi efficiency. No automatic vocab size.
- **Gastaldi, Terilla, Malagutti, DuSell, Vieira, Cotterell, "The Foundations of Tokenization"** (arXiv 2407.11606,
  2024): stochastic-map formalism; conditions for tokenizers to preserve estimator consistency.
- Zouhar et al. "Noiseless Channel" (2023): Rényi efficiency as a learner-aware efficiency measure.

**So the framing "Shannon's invariance → V is about the learner" is NOT novel** (epiplexity's Paradox 2, V-information,
Rajaraman et al.). Credit them.

**What still appears open (moderate confidence; no paper found):** using these bounded-learner measures to *derive the
vocabulary size*: V* = the representation that maximizes usable information (epiplexity / V-information) for a given
learner and budget, with the exposure floor (Gowda & May) as the mechanism and Tao et al.'s compute scaling as the
prediction to match.

**Experiment upgrade (Charlotte):** for each vocab size in the sweep, estimate epiplexity the way Finzi et al. do
(prequential coding: area under the training loss curve above final loss), alongside bits per byte and undertrained-
token counts. Hypothesis: the knee in bits per byte coincides with peak epiplexity per unit compute, and both move with
training budget as the exposure floor predicts.

## Andrew's push #2: "compute of what?" and flatland information theory (2026-10-02)

**Andrew's question:** Shannon assumed a perfect receiver. The bounded-learner papers constrain compute, but compute of
what? Tokenization is lossless, so what is lost is *expression*, which we implicitly fix by the dimension the tokens
live in. Was Shannon's receiver unlimited because its representational space was unlimited? Has anyone done a
"flatlander" information theory?

**What Shannon's perfect receiver is (fact):** the decoder may be any function of the received block; block length goes
to infinity (codebooks of size 2^{nR}, exponential memory); it knows the source distribution exactly; compute is free.
So it is unbounded in representation, memory, data, and compute at once. Its model *is* the source. Andrew's "limitless
space → infinite expression" reading is right.

**Correction to the premise:** V-information does partly ask "of what": V is a predictive *family* (e.g. linear
predictors), a representational constraint, though stated as a function class rather than a dimension. Epiplexity is
the time-bounded one.

**Clean decomposition (prior art: Bottou & Bousquet, "The Tradeoffs of Large Scale Learning", NeurIPS 2008):**
model cross-entropy = H(source) + approximation gap (best representable model, the "expression" term) + estimation gap
(finite data) + optimization gap (finite compute). Shannon sets the last three to zero. Epiplexity works the
optimization term; the exposure floor is the estimation term; **the flatland question is the approximation term.**
The tokenizer moves all three at once.

**Flatland prior art (verified unless noted):**
- Shannon 1949, "Communication in the Presence of Noise": signals as points in 2TW-dimensional space; capacity from
  sphere packing in that dimension. Shannon did count dimensions, of the *channel*, never of the receiver's model.
  (Well known; not re-fetched.)
- Kolmogorov 1936 n-widths: best error approximating a function class by n-dimensional subspaces. (Not re-fetched.)
- Hellman & Cover, "Learning with Finite Memory", Ann. Math. Stat. 1970: optimal S-state hypothesis testers; a receiver
  with a finite state count.
- Lin & Tegmark, "Critical Behavior in Physics and Probabilistic Formal Languages", Entropy 2017: mutual information
  decays exponentially under any probabilistic regular grammar (finite state), can decay as a power law under
  context-free grammars; natural language shows power law.
- Yang, Dai, Salakhutdinov, Cohen, "Breaking the Softmax Bottleneck", ICLR 2018: language modeling as matrix
  factorization; log-probability matrix rank ≤ d+1, so a d-dim model cannot express arbitrary distributions over V
  tokens. **This is literally flatland: a V-dimensional world seen by a d-dimensional receiver.**
- Wies, Levine, Jannai, Shashua, "Which Transformer Architecture Fits My Data? A Vocabulary Bottleneck in
  Self-Attention", ICML 2021: embedding rank bottleneck; "a small vocabulary size or rank dictates an added advantage
  of depth over width"; 25–50% size redundancy in ALBERT/T5.
- Elhage et al., "Toy Models of Superposition", Anthropic 2022: more features than dimensions via near-orthogonal
  directions; interference is the cost. (Not re-fetched.)
- Godey, de la Clergerie, Sagot, "Why do small language models underperform?", COLM 2024 (arXiv 2404.07647):
  saturation from hidden dimension vs high-rank target distribution; models under ~1000 hidden dims degenerate late in
  pretraining.
- Huang et al., "Over-Tokenized Transformer: Vocabulary is Generally Worth Scaling", arXiv 2501.16975 (2025):
  decouples input and output vocab; input vocab scaling log-linearly improves loss at every size; larger output vocab
  can hurt small models. **Input side and output side want different V.**
- **Godey & Artzi, "Lost in Backpropagation: The LM Head is a Gradient Bottleneck", arXiv 2603.10145 (Mar 2026):** the
  softmax bottleneck is also an optimization bottleneck; 95–99% of gradient norm is suppressed by the rank-D output
  layer; makes trivial patterns unlearnable. **This is the hinge for "compute of what": compute is spent through a
  flat projection, so dimension and compute are not independent constraints.**

**So "flatland information theory" exists in pieces** (channel geometry, approximation widths, finite-memory learners,
softmax bottleneck, superposition, gradient bottleneck), but nobody found unifies it with tokenization.

**What looks open (inference; moderate-low confidence it is unclaimed):** the tokenizer sits exactly where the
V-dimensional symbol world meets the d-dimensional receiver. Sharpened hypothesis:
- "The vocab size" is two numbers. V_out is bounded by the receiver's dimension (expressivity rank d+1 and the
  gradient bottleneck). V_in is bounded by data (exposure floor), not by d (Over-Tokenized evidence).
- Then V_out* falls out of d and V_in* falls out of the data budget: the hyperparameter is replaced by two
  architecture/data quantities.

**Charlotte experiment addition:** sweep V_out at several d (e.g. 256/512/1024); measure effective rank of the logit
matrix, fraction of gradient norm surviving the LM head (Godey & Artzi method), bits per byte. Prediction: the
bits-per-byte knee in V_out shifts with d; the V_in sweep does not.
