# Ariadne: project brief

*Captured 2026-10-03. Named 2026-10-03. Status: research program defined, nothing written or built. Supersedes the
"BPE" seed, now at `notes/2026-09-30-bpe-information-theory-seed.md`, whose research is carried over below.*

## The name

Lineage, per the naming convention (real machine, then the pop-culture hop):

- **Theseus** (Claude Shannon, Bell Labs, 1950): a wheeled mouse that solved a 5 x 5, 25-square maze by trial and error
  and remembered the solution in a bank of telephone relays under the floor. A learner with a fixed, finite memory,
  built by the man whose theory assumes a receiver with no limits at all.
- **Ariadne** gave Theseus the thread that got him out of the labyrinth: a path through a space too large to see whole.
- **Inception** (2010): Ariadne is the architect who designs the mazes the dreamers move through.

The project is about finite receivers moving through spaces they cannot hold whole. The thread is also where the
reasoning-tokens work joins later (a receiver steering a path through its own representation space).

Visual identity: original art only (a thread through a maze, a relay bank, a Square looking at a sphere). No film
stills or likenesses, same rules as Seymour and Serenity.

## The question

Shannon's theory assumes a receiver that can represent anything. Real receivers are finite: a neural network has a
fixed width, a codec a fixed table, a sensor a fixed precision, a person a fixed working memory.

1. How much of a message can a receiver of fixed capacity actually use?
2. How does the alphabet a message arrives in change that?
3. Does the right alphabet fall out of the receiver's capacity, so it is derived rather than tuned?

**This is not a machine-learning project.** Byte-pair encoding is how Andrew arrived at the question (Charlotte's
tokenizer, and the vocabulary-size knob that nothing derives). Vocabulary size is one instance of question 3. The same
question appears wherever a finite receiver meets a code.

## Andrew's positions (decisions)

- **A hyperparameter is a sign the theory is incomplete**, like the Standard Model's free parameters. A number that
  has to be tuned is a question nobody has answered. Vocabulary size should fall out of the structure of what is being
  received, or at least out of an optimization with a cliff.
- **Recoding is lossless; what is lost is expression.** Re-chunking a message does not remove information, but a
  finite receiver cannot represent everything, so the alphabet decides what it can use. Expression lives in the
  dimension the receiver works in.
- **"Flatlander" information theory:** Shannon's receiver was effectively unlimited in the space it could occupy, so
  anything could be learned. Ask what happens when it is not.
- **Start at the start and work forward:** Shannon 1948 first, looking for what was assumed, what was accepted, and why.
- **Not scoped to ML idioms.** Frame and test across receiver kinds.

## What Shannon assumed (facts, from the seed research)

- **Information belongs to the source, not the alphabet** (Shannon, "A Mathematical Theory of Communication", 1948).
  Lossless recoding redistributes information without removing it: bits per symbol scale with symbol length. With an
  ideal receiver, the alphabet cannot matter.
- **The perfect receiver:** the decoder may be any function of the received block; block length goes to infinity
  (codebooks of size 2^{nR}, exponential memory); it knows the source distribution exactly; computation is free. It is
  unbounded in representation, memory, data and compute at once. Its model *is* the source.
- **Shannon did count dimensions, of the channel** ("Communication in the Presence of Noise", 1949: signals as points
  in a 2TW-dimensional space, capacity from sphere packing in that dimension). Never of the receiver's model.
- **Prediction as the measure** ("Prediction and Entropy of Printed English", 1951): next-letter guessing, English at
  roughly 0.6 to 1.3 bits per character.

## Framework: where the loss lives

Prior art (Bottou and Bousquet, "The Tradeoffs of Large Scale Learning", NIPS 2007): a receiver's loss is

**source entropy + approximation gap + estimation gap + optimization gap**

| Term | Limited by | Ariadne's reading |
|---|---|---|
| Approximation | what the receiver can represent at all | **expression; the flatland term** |
| Estimation | finite data / exposure | the exposure floor (how often each symbol is seen) |
| Optimization | finite compute | what epiplexity and time-bounded entropy measure |

Shannon sets the last three to zero. The alphabet moves all three at once, which is why it looks like a free knob.

## Working hypotheses (hypotheses, not claims)

| # | Hypothesis | Status |
|---|---|---|
| H1 | The alphabet matters only because the receiver is bounded. | **Prior art; credit, do not claim.** Epiplexity's "Paradox 2", V-information, Rajaraman et al. |
| H2 | The right alphabet size can be derived from the receiver's capacity and budget, not tuned. | **Appears open** (moderate confidence; no paper found deriving it). The core of the program. |
| H3 | It is two numbers: what a receiver can take in is bounded by exposure; what it can express out is bounded by its dimension. | **Appears open** (moderate-low confidence). Evidence: the softmax bottleneck (output rank at most d + 1); Over-Tokenized Transformer (input vocabulary scales freely, output vocabulary hurts small models). |
| H4 | Capacity and compute are not independent: compute is spent through the receiver's limited dimension. | Strong evidence in one receiver kind: Godey and Artzi 2026 (95 to 99% of gradient norm suppressed at a rank-limited output). Needs a general statement. |

## Where the question shows up (instances to test across)

| Receiver | Its limit | The alphabet question | Status |
|---|---|---|---|
| Neural language model | width d, training budget | vocabulary size (input and output) | seed research done |
| Variable-to-fixed coder (Tunstall, 1967) | fixed codeword length | dictionary size; no natural cliff for a memoryless source until the dictionary's cost is counted | seed note |
| Bounded-context compressor | context order or window | which alphabet makes a short context sufficient | to research |
| Finite-state learner (Hellman and Cover, 1970) | number of states S | how much an S-state receiver can learn | prior art, verified |
| Grammar-limited processes (Lin and Tegmark, 2017) | finite state vs context-free | mutual information decays exponentially under regular grammars, can decay as a power law under context-free ones | prior art, verified |
| Human reader | working memory, chunking | the unit people chunk in (Miller, 1956, "The Magical Number Seven, Plus or Minus Two") | to research |
| Representation geometry | features vs dimensions | superposition: more features than dimensions, interference as the cost (Elhage et al. 2022) | prior art, not re-fetched |

## Pieces (a program, not one paper)

Each piece stands alone and goes where it fits: a paper on the site, an article on andrewphunter.com, or a
standalone LinkedIn post (see the write-ups rule). Order is a proposal.

1. **The receiver Shannon left out** (essay). The infinite-receiver assumption, what it hides, and why hyperparameters
   show up exactly where it breaks. Candidate line: "Shannon proved the alphabet can't matter. It matters anyway,
   which tells you something about the receiver."
2. **Bounded receivers, from Theseus to now** (prior-art map). Everyone who has bounded Shannon's receiver, credited:
   finite memory, function classes, compute, dimension. Ends on what nobody has joined up.
3. **A derived alphabet** (formal). A measure of what a receiver of given capacity can use, and the conditions under
   which the alphabet size falls out of it (H2, H3).
4. **Tests across receiver kinds** (paper). Predictions checked on at least two kinds so the result does not depend on
   machine learning (experiments below).

Venue options for the formal and test pieces depend on how they land: information-theory venues if the derivation is
clean, ML venues if the result is a measured curve (survey at
`~/Documents/plainsight-publication-venues-2026-10.md`).

## Experiments

**A. Classical receiver (cheap, deterministic, no ML).** Synthetic sources with known entropy (k-th order Markov),
received by bounded-context predictors of order m over alphabets of increasing block size. Prediction: the best
alphabet size is set by the gap between the source's order and the receiver's context, and the knee is computable
in advance. Tie-in: Rajaraman et al. showed the neural version (no tokenization falls back to a unigram model).

**B. Neural receiver (Charlotte).** Carried over from the seed:
1. Fix the corpus; sweep vocabulary size (for example 512 to 32K) at fixed compute on toy models.
2. Measure bits per byte (fair across alphabets), undertrained symbols (Magikarp-style indicators), and the fraction
   of symbols seen at least k times (Gowda and May style).
3. Estimate epiplexity per alphabet the way Finzi et al. do (prequential coding: area under the loss curve above the
   final loss).
4. Sweep output vocabulary at several widths (for example d = 256, 512, 1024); measure the effective rank of the logit
   matrix and the fraction of gradient norm surviving the output layer (Godey and Artzi method).
5. Predictions: the bits-per-byte knee coincides with peak epiplexity per unit compute and moves with budget (H2);
   the output-vocabulary knee shifts with d while the input-vocabulary sweep does not (H3). If not, say so: the
   coupling is something else, and that is a result too.

**Visual hook (IC formula):** the knee plot across receiver widths; a thread-through-maze diagram of the
decomposition; where different alphabets cut the same words.

## Prior art carried over (credit list)

**Coding lineage:** Shannon 1948, 1949, 1951; Huffman 1952; Tunstall 1967; Lempel and Ziv 1977 to 1978; Gage 1994
(BPE); Larsson and Moffat 1999 to 2000 (Re-Pair); Charikar et al. 2005 (smallest grammar problem); Sennrich, Haddow
and Birch 2016; Rissanen (MDL); Harris 1955 (boundary entropy); Creutz and Lagus (Morfessor).

**Tokenization theory:** Kudo 2018 (Unigram LM); Zouhar et al. 2023, "A Formal Perspective on Byte-Pair Encoding"
(greedy approximation about 0.37 to optimal compression); Zouhar et al. 2023, "Tokenization and the Noiseless Channel"
(Rényi efficiency correlates 0.78 with BLEU, compressed length -0.32); Schmidt et al. 2024, "Tokenization Is More Than
Compression"; Chung, Garrette et al. 2020 (MDL for vocabularies); Goldwater 2007; Nouri 2026 (MDL-stopped pair
encoding); Gastaldi et al. 2024, "The Foundations of Tokenization"; Erdogan et al. 2026; ByteSpan 2025; Delétang et
al., "Language Modeling Is Compression".

**Bounded learners:** Xu et al. 2020 (V-information); Finzi et al. 2026 (epiplexity, time-bounded entropy); Rajaraman,
Jiao, Ramchandran 2024; Hellman and Cover 1970; Bottou and Bousquet 2007.

**Vocabulary and the learner:** Tao et al. 2024 (vocabulary scaling laws); Chung and Kim 2025 (frequency imbalance);
Land and Bartolo 2024 (undertrained tokens); Gowda and May 2020 (exposure heuristic).

**Dimension:** Kolmogorov 1936 (n-widths); Yang et al. 2018 (softmax bottleneck); Wies et al. 2021 (vocabulary
bottleneck); Elhage et al. 2022 (superposition); Godey et al. 2024 (small-model saturation); Huang et al. 2025
(Over-Tokenized Transformer); Godey and Artzi 2026 (gradient bottleneck); Lin and Tegmark 2017.

Full notes, figures and verification status for each: `notes/2026-09-30-bpe-information-theory-seed.md`.

## Risks

- **Novelty.** H1 is taken. The program stands on H2 and H3 and on joining the pieces across receiver kinds. Run a
  fresh prior-art check before each piece publishes; claim only what remains.
- **Overreach.** A general "receiver capacity" measure is easy to state and hard to make precise. The formal piece
  should start from one tractable receiver (experiment A) and generalize only as far as the math holds.
- **Scope creep into ML.** Keep at least one non-neural receiver in every claim.

## Related threads

- **Reasoning tokens and the hyperplane:** "Reasoning Tokens Steer the Trajectory" (andrewphunter.com, 2026-09-22).
  A finite receiver steering a path through its own representation space; likely a later Ariadne piece.
- **Charlotte:** the neural receiver for experiment B.
- **Serenity:** separate project; shares the habit of finding one field's solved problem in another.

## Next steps

1. Andrew reviews this brief; settle the order of pieces.
2. Fresh prior-art pass on H2 and H3 specifically (derived alphabet size; separate input and output bounds), outside
   ML as well as inside it.
3. Design experiment A in detail (sources, receiver orders, alphabet construction, the predicted knee).
4. Outline piece 1.
