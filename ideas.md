# Ideas

Half-formed thoughts and questions to explore. Not hypotheses: nothing here
is claimed, checked or ranked. An entry can be one line.

When an idea firms up, it moves on: a hypothesis goes into `brief.md`, a
research pass into a dated note in `notes/`. Leave the entry here with a
pointer to where it went, so the trail stays.

Newest first. Format:

```
## YYYY-MM-DD: short title

The thought or question, as rough as it is.

From: what prompted it (optional)
Moved to: brief.md H5 / notes/... (once it has moved)
```

## 2026-10-08: Attention as relaxed selection; reasoning blocks as recoding

Softmax is the solution of an optimization: for scores s, softmax(s) =
argmax over distributions p of <p, s> + H(p), linear selection plus an
entropy term, with the 1/sqrt(d) scaling acting as temperature. So attention
is a barrier-regularized relaxation of choosing which earlier tokens to
read, held at a fixed temperature. An interior-point method drives its
barrier to zero to reach the exact selection; attention never does. The
entropy term is literally information-theoretic: what sets the temperature,
and is sqrt(d) derived or convention?

From: Markus Hartikainen's LinkedIn post (2026-10-08) on test-time compute
as mathematical programming over reusable token blocks, solved by GPU
interior-point methods; his follow-up: "the search space and solver
architecture matter just as much as the base model."
Rough notes from talking it through:
- His proposal in Ariadne's terms: recoding. Same reasoning, coarser units
  (blocks instead of tokens), for a searcher that cannot handle the fine
  alphabet. Variable-to-fixed, like Tunstall: variable-length strings, one
  fixed decision each. Recoding loses no information; what changes is what
  a bounded searcher can use (questions 2 and 3).
- Block library = vocabulary; the solver's constraints = grammar made
  explicit (the 2026-10-05 vocabulary/order split, one level up). In an LLM
  grammar lives implicitly in order and weights; here it moves into
  constraints solved exactly.
- A new receiver kind for the table: a reasoning searcher (model plus
  solver), limited by search depth and compute; the alphabet question
  becomes block granularity. Spectrum: tokens (fine, discrete), blocks
  (coarse, discrete), latent world-model states (continuous, no alphabet).
- Block granularity is a new hyperparameter: too fine is token branching,
  too coarse and the library balloons and blocks stop composing. Predict a
  knee; can it be derived from the searcher's budget (H2)?
- A MIP solver reports a bound on how far it may be from optimal, so moving
  reasoning onto solvers would make the optimization term (Bottou and
  Bousquet) measurable. LLM sampling never reports it.
- The block library is learned, so it inherits the exposure floor:
  rarely seen blocks are poorly formed.
- Where blocks would live, and why it matters: as text (reusable anywhere,
  a fresh prefill per assembly), as KV segments (cheap, but past layer 1
  each encodes the context it was computed in, so exact only in its
  original position and order; CacheBlend, arXiv:2405.16444, and EPIC,
  arXiv:2410.15332, patch reuse by recomputing part of each chunk), as
  vocabulary macro-tokens (native, one step each, grows V), or as latent
  vectors (toward the continuous side). His post does not say which.
- Attention couples blocks: block B's value depends on what came before it,
  so a solver's objective needs pairwise terms, a block-to-block
  compatibility matrix, which is what qK^T computes between tokens.
- Confidence: the softmax identity is standard. "Frozen partway along the
  central path" is an analogy, not a theorem: the entropic barrier is a
  valid barrier for the simplex, but attention does not run an
  interior-point method.

## 2026-10-08: Is the strict limit a limit, or a hold on spread?

The input's directions-needed count grows with d at strict tolerances,
which may be trivial: training spreads every vector across all of d, and
the larger models keep more of their random start. Squashing any spread-out
variance costs a little loss, and the later layers were trained on
whatever is there. So the strict count may measure dependence, not need.
The loose count is the part that isn't trivial: it stops growing.

From: drafting the LinkedIn post on experiment C (2026-10-08)
Rough notes from talking it through:
- Evidence the strict count is inflated by spread: at step 1000 (table
  still near random) pythia-2.8b needs 2304 of 2560 directions at +0.003
  but 768 at +0.1. A flat random spectrum makes every direction cost
  something at a strict tolerance.
- Larger tables are rewritten less completely: mean cosine of trained row
  to step-0 row about 0.03 at d = 512, about 0.2 at d = 2560 (phase 1b
  correction).
- The loose count holds: 6.9b and 12b need the same 2048 at +0.1 and 2560
  at +0.03, although 12b has 1024 more dimensions. Only the strict counts
  (+0.01, +0.003) keep scaling with d.
- Question: is the strict limit a real limit, or a pedantic hold on
  spread-out, partly random structure? That is the semantic/pedantic cliff
  question itself.
- Test 1: cost of removing a trained direction against removing one from a
  random table of the same shape (does the trained table's tail cost more
  than random variance would?).
- Test 2: squash, then retrain briefly; if the model recovers from losing
  the strict directions, they held dependence, not need.
- Prior art checked the same day: input-embedding redundancy is known
  (ALBERT 2019; Kataiwa et al. 2025, arXiv:2503.02142, Pythia token
  embeddings ID about 25; Quemy 2026, arXiv:2608.29702, ID 10-17 at every
  size once a hub of short rows is removed). The gap between geometric ID
  (tens) and the loss-based count (thousands) is its own thread. Output
  full use is consistent with the softmax bottleneck (Yang et al. 2018;
  Godey et al. 2024, arXiv:2404.07647) but not shown by it.

## 2026-10-07: Semantic core vs pedantic refinement

Of course training expands into the dimensions: through exposure the model
keeps learning the differences between tokens. At some point the
differences become pedantic rather than semantic. Spatially that means
that while more dimensions are used, the empty space grows.

From: experiment C, directions needed through training
Rough notes from talking it through:
- Hint already in the data: at the loose tolerance (0.1 bits per byte) the
  directions needed through training climb then flatten; at the strict
  tolerance (0.01) they keep climbing. Loose = semantic core (saturates);
  strict adds pedantic refinements (each real, each worth little).
  Observation 10: late directions don't change spread or crowding, so they
  are thin, low-variance refinements, not new major axes.
- "More used, emptier": volume grows exponentially with dimension, so the
  same tokens separated along more directions occupy a shrinking fraction
  of the room. Measurable: nearest-neighbour distance within the top k
  directions as k grows past the core.
- Might rescue the natural dimension: the plateau failed for the strict
  count but may hold for the semantic core; past it, pedantic refinement
  keeps growing as long as training continues.
- Links to "the loss from source to signal is the meaning": which
  differences matter is the semantic/pedantic line.
- Caution: small contribution to average loss is a proxy for pedantic, not
  a definition (rare but genuinely different words also contribute little).
  A better line between semantic and pedantic is itself a theory question.
- Next (2026-10-08): frame it up.

## 2026-10-06: The knob held fixed in a paper about the knob

Pulipaka 2026 (arXiv:2608.07727, in the library as pulipaka2026) asks how vocabulary and capacity should be
allocated across four Dravidian languages, and concludes that "how vocabulary and model capacity are allocated can
matter just as much as model size." Every model uses GPT-2 small's shape (12 layers, d = 768, 12 heads); the four
monolingual tokenizers are 32K each and the joint one 64K. Neither size is derived, and none is varied. Asked on
LinkedIn (2026-10-06) how 32K was chosen, the author said it was kept fixed to keep the comparison consistent across
languages, with a sweep left for future work.

Candidate citation for piece 1: a clean, current example of a vocabulary size used as an experimental control rather
than as the question, inside a paper whose subject is vocabulary allocation. Also the GPT-2 small shape (2019) still
serving as the default body in 2026: convention, inherited.

Cite the paper's own text (setup in section 3.2 and 3.3), not the LinkedIn exchange, and keep it neutral: it is the
field's habit, not one author's lapse. Supporting numbers from the paper: dedicated fertility 1.59 (Kannada) to 1.84
(Malayalam) tokens per word; bits per byte favors the monolingual models on all four languages.

From: comment sweep for Ariadne-aligned posts, 2026-10-06

## 2026-10-05: Vocabulary, order, room and shape

We shifted to "alphabet" for tokenization, but that was incorrect. It is
vocabulary: not just phonemes, but the vocabulary of the corpus being
compressed. That is Shannon's "meaning doesn't matter", and why compression
plus codec is lossless. The vocabulary isn't lost, it's translated.

If vocabulary is that, then the order of the tokens is where implied grammar
lives. Hence why ordering is important.

The embedding dimension is where the expressiveness lives, where vocabulary
and grammar can become meaning. It's both: the number of dimensions is the
size of the room, and the values in those dimensions are the shape carved in.

Rough notes from talking it through:
- BPE's vocabulary is one of reuse, not meaning ("ing", " the").
- Order could be measured: entropy from token frequencies alone minus the
  true entropy rate is what order carries (Shannon's 1948 approximation
  series is this, step by step). Attention without position signal cannot
  see order at all; the rotary parameters are what let it see grammar.
- The tokenizer draws the line between vocabulary and grammar: merging
  "New York" moves structure out of order into vocabulary. Maybe the
  vocabulary-size knob is where that line goes (cleaner H2?).
- Room = approximation term; shape = estimation and optimization terms
  (Bottou and Bousquet, from the inside).
- Which binds? Compare the embedding's effective rank with d. Near d: the
  room is the limit. Far below: the carving is (exposure, compute).
  Superposition can over-fill the room too. Could check in Charlotte on the
  open-weight models it loads. (Claude's side guess, not Andrew's: small
  models room-limited, large ones carving-limited.)
- Andrew's hypothesis (2026-10-07): vocabulary and embedding dimension
  together specify the expressiveness of the learned language, and the
  dimension should be derivable from knowing the vocabulary and the grammar.
  It should not be a hyperparameter.
- Plan (Andrew, 2026-10-07): first question, do models have empty
  dimensions? Not limited to small models; run small ones first because the
  runs are cheap, then confirm on a larger open-weight model. Only if that
  holds, the larger question: do vocabulary and grammar drive the effective
  dimension?
- Experiment C has run (experiments/c-empty-dimensions/). Its observed
  trends are listed there under "Observed trends", kept apart from
  interpretation. Next: check the trends point one way, then theory, then
  hypotheses derived from it.
- Non-neural version: a receiver that sees the last m tokens sees as much
  grammar as fits in m; longer tokens pull grammar into range (experiment A).

## 2026-10-05: The hyperparameters are a map of where the theory stops

Working through my WebGPU transformer, it is hyperparameters everywhere, with
vague answers of "well, this kinda tells us". Did ML just walk away from
information theory? It flows all the way through: tokenization, the attention
block itself (Q/K/V, their values and dimensionality), the number of heads,
the rotation parameters, and so on.

From: Charlotte (~/repositories/tech-demos/charlotte), the WebGPU inference
harness
Candidate: the opening of piece 1, built from Charlotte's own code. Walk the
forward pass and mark each number as derived, fitted or convention.
Knobs to walk (status from memory, unchecked):
- Vocabulary size: fitted (Tao et al. 2024). H2.
- Width d: fitted ratios; nothing derives it from the vocabulary or the data.
- Q/K/V width per head, usually d / heads: convention.
- Number of heads: convention.
- The 1/sqrt(d_k) in attention: half-derived, a variance argument (Vaswani
  et al. 2017), not an information argument.
- Rotary position base (10000 in RoFormer, Su et al. 2021; raised later for
  long context): chosen, then tuned. Check whether anyone has derived it from
  context length.
- MLP width (often 4d, or about 8/3 d for gated units): convention.
- Layers, normalization epsilon, warmup, Adam beta2, weight decay: empirical.
- Learning rate and initialization across width: derived in part by muP
  (Yang et al., Tensor Programs V, 2022), the clearest counterexample.
- Counter-trend to credit: V-information, epiplexity, Nixon 2026, Deletang
  et al., all in the library.

Second strand: even "what does this mean" gets circular answers. "The query is
what a token looks for, the key what it offers, the value what it passes" is
a database-lookup metaphor attached to three matrix multiplies, not a
definition.
- The mechanical account is not circular. Checked against Elhage et al.
  2021, "A Mathematical Framework for Transformer Circuits" (in the library):
  Q, K and V are intermediate results of two low-rank matrices, W_Q^T W_K and
  W_O W_V, each of rank at most d_head, and the paper says transformers can
  usefully be described without reference to Q, K and V. The QK circuit is a
  bilinear form deciding which source position a destination position reads
  from; the OV circuit is a linear map deciding what gets written.
- So d_head has a meaning: the rank of a head's read rule (QK) and of its
  write (OV). It has no derived value. Heads x d_head = d splits a fixed
  budget into channels, and nothing says where the split should land.
- Ties to "Meaning has to fit both ends" (above): inside each head, QK acts as
  a receiver (what it can tell apart) and OV as a transmitter (what it can
  say), both bounded by d_head.
- Unchecked lead: Tsai et al. 2019 (EMNLP), attention as a kernel smoother.
- Opinion: the explanations fall back on names because there is no theory of
  the receiver, the same gap the hyperparameters mark.

## 2026-10-04: Meaning has to fit both ends

The key here is the meaning. In order to transmit and receive, we need to be
able to express the concept in both the transmitter's and the receiver's
compute capability. This is, I believe, the vocab x embedding dimension
relationship.

From: "The loss from source to signal is the meaning" (below)
Leads (unchecked):
- A language model is both ends: the input embedding receives tokens, the
  output head transmits them. H3's input/output split may be this
  receiver/transmitter split under another name; the softmax bottleneck
  (rank at most d + 1) is then a transmitter limit.
- V is not capped at d: superposition packs many more near-orthogonal
  directions than dimensions, at the cost of interference (Elhage et al.
  2022). The limit may be tolerable interference, not V <= d.
- Tokens whose differences cannot be expressed in d dimensions end up close
  together: the embedding may be where "equating the unequal" happens.
- Non-neural counterpart: vector quantization (a codebook of K vectors in d
  dimensions; rate, dimension and distortion trade off), and Shannon 1949's
  sphere packing in dimension. Likely prior art; check.

Prior art, searched 2026-10-04 (information theory, ML, cognitive science and
philosophy). Checked against abstracts or publisher pages unless marked; none
read in full yet.

Closest, read these first:
- Nixon, "Semantic Rate-Distortion for Bounded Multi-Agent Communication:
  Capacity-Derived Semantic Spaces and the Communication Cost of Alignment",
  arXiv:2604.09521 (Apr 2026, single author, not peer reviewed). Agents of
  different capacity induce different semantic alphabets; below a critical
  rate set by the mismatch, intent-preserving communication is impossible.
  States its contribution as deriving the alphabet from bounded interaction.
  Capacity there is memory and horizon in a POMDP, not V x d. Bears on H2 as
  well as this idea. Read in full 2026-10-04:
  - Each agent's alphabet is the set of observation histories its m-node
    finite-state controller can tell apart (a Myhill-Nerode quotient, from
    Nixon's earlier paper). A finite-state controller is a non-neural
    receiver.
  - The receiver's quotient merges classes the sender keeps apart. Those
    merged distinctions are the loss, so the quotient is a formal version of
    "the structure of the loss" (the entry below).
  - Critical rate: log|Q_A| - log|Q_B| for uniform visitation; generally the
    conditional entropy rate of A's classes given B's. Below it, the sender's
    intent cannot get across. The lower bound is a pigeonhole argument.
  - Continuous extension (proof sketch only, appendix N): rate of about
    (d_A - d_B) log(1/eps) for spaces of intrinsic dimension d_A > d_B. The
    cost is the dimension gap.
  - Covers one-way communication where the sender's partition refines the
    receiver's. States as open: two-way communication, and agents whose
    partitions are not nested. "Both ends must express it" sits in that open
    case.
  - Says nothing about a code's vocabulary size or exposure; the message is
    R bits per step. Conjectures (unproved) that transformer layer
    representations refine its quotient.
- Xu, "Semantic Channel Theory: Deductive Compression and Structural Fidelity
  for Multi-Agent Communication", arXiv:2604.16471 (Apr 2026, single author).
  Vocabulary mismatch between agents limits fidelity even over a noiseless
  channel (broadcast setting); knowledge bases and proof systems, not
  dimensions.
- Warglien and Gardenfors, "Semantics, conceptual spaces, and the meeting of
  minds", Synthese 190 (2013), doi:10.1007/s11229-011-9963-z. Meaning as a
  mapping between two individual concept spaces; agreement possible across
  different spaces if concepts are convex. Abstract from a snippet.

One piece each:
- Ends limit each other, non-neural: Singh, Dabeer and Madhow, "Capacity of
  the Discrete-Time AWGN Channel Under Output Quantization", ISIT 2008,
  arXiv:0801.1185: a K-bin receiver caps the useful transmitter alphabet at
  K + 1 points. MIMO capacity scales with min(transmit, receive) dimensions
  (Telatar 1999, doi:10.1002/ett.4460100604; scaling result not checked).
- Asymmetric parties: Juba and Sudan, "Universal semantic communication I",
  STOC 2008, doi:10.1145/1374376.1374397; Goldreich, Juba and Sudan, "A theory
  of goal-oriented communication", JACM 2012, doi:10.1145/2160158.2160161.
- Two ends of a language model as separate budgets: Chung et al., "Rethinking
  embedding coupling in pre-trained language models", arXiv:2010.12821
  (venue unchecked); Batley and Saha, "Leviathan", arXiv:2601.22040; Lopardo
  et al., "Weight Tying Biases Token Embeddings Towards the Output Space",
  arXiv:2603.26663; Huang et al. 2025 (already in the brief).
- Output-end limits when V > d: Grivas, Bogoychev and Lopez, "Low-Rank Softmax
  Can Have Unargmaxable Classes in Theory but Rarely in Practice", ACL 2022,
  arXiv:2203.06462; Demeter, Kimmel and Downey, "Stolen Probability", ACL
  2020, arXiv:2005.02433.
- Interference, not d, as the limit: Guha, "Representational Capacity:
  Geometric Limits on Feature Representation in Transformer Language Models",
  arXiv:2606.02765 (Jun 2026, not peer reviewed): capacity from tolerated
  deviation from orthogonality, exponentially sensitive to it. Liu, Liu and
  Gore, "Superposition Yields Robust Neural Scaling", NeurIPS 2025,
  arXiv:2505.10465.
- Counterpoint: Rita et al., "On the role of population heterogeneity in
  emergent communication", ICLR 2022, arXiv:2204.12982: varying speaker and
  listener capacity mattered only through relative learning speed.
- Meaning as lossy compression, speaker side only: Zaslavsky, Kemp, Regier and
  Tishby, "Efficient compression in color naming and its evolution", PNAS
  2018, doi:10.1073/pnas.1800521115 (listener idealized, its capacity not
  modelled); Kemp and Regier, Science 2012, doi:10.1126/science.1218811;
  Kirby et al., Cognition 2015, doi:10.1016/j.cognition.2015.03.016
  (compressibility in learning vs expressivity in communication); Sims,
  Science 2018, doi:10.1126/science.aaq1118 (perception as rate-distortion).

Background: vector quantization (Gersho and Gray 1992; Zador 1982); mismatched
decoding (Merhav, Kaplan, Lapidoth and Shamai 1994; Csiszar and Narayan 1995);
Ziv and Lempel 1978 finite-state compressibility; Yin and Shen, "On the
Dimensionality of Word Embedding", NeurIPS 2018 (optimal d per corpus, not
from V); tied embeddings (Press and Wolf 2017; Inan et al. 2017); Gardenfors,
Conceptual Spaces (2000); semantic communication theory (Bao et al. 2011;
Shao, Cao and Gunduz, arXiv:2212.01485).

Not found anywhere (moderate confidence, about 60%, in each of the three
searches): the usable vocabulary set by the weaker of transmitter and
receiver, measured as vocabulary size times dimension (or an interference
tolerance), with a non-neural receiver in the same statement. Also not found:
any sweep of V and d independently reporting optimal V moving with d (Tao et
al. 2024 takes d as given).

## 2026-10-04: The loss from source to signal is the meaning

My guess is transmission from source to signal is lossy, and the lossiness is
the meaning. I think this is important, and philosophy via Nietzsche points at
this.

From: Shannon 1948's five-part diagram, where the source comes before the transmitter
Leads (unchecked unless marked):
- Nietzsche, "On Truth and Lies in a Nonmoral Sense" (1873): concepts form by
  equating unequal things.
- Weaver 1949 (checked): proposes a "semantic noise" box between source and
  transmitter, which is where this idea puts the loss, but treats that loss
  as noise; and a "semantic receiver" matched to the receivers' capacities.
- Shannon 1959, rate-distortion: the distortion measure says which loss is
  acceptable, and comes from outside the theory.
- Tishby, Pereira and Bialek, information bottleneck: keep only what bears on
  a chosen relevant variable.
- Dretske 1981 (checked against his Précis, BBS 6, 1983, p. 61): calls the
  loss of excess information "the essence of conceptualization". Direct prior
  art for loss as meaning, on the receiver's side only. Book not yet in hand
  (references/to-get.md).
- Refinement to test: the meaning may be the structure of the loss (which
  differences are discarded), not the amount.

## 2026-10-04: Information theory starts at the signal, not the transmitter?

Information theory seems to start at the signal, the information, and not at
the transmitter. Is this true? If the alphabet is chosen at the sending end,
is "the right alphabet" a question about the transmitter as much as the
receiver?

## 2026-10-04: The hypotheses are over-restricted

H1 to H4 may be drawn too narrowly.
