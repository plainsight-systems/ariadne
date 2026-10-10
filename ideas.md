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

## 2026-10-09: Identity is the vocabulary half; the gap is the candidate grammar half

Andrew's guess is that d follows from vocabulary and grammar together. In
Shannon's terms that is two parts. The vocabulary part is how many
directions it takes to tell tokens apart through the receiver's noise, the
1949 sphere-packing count turned around: n = 2 H(T) / log2(1 + P/N). The
grammar part is what lowers the entropy of the next token below the
single-token entropy (1951: F_1 down to F_N).

- Fact (observations 39-41): in the text view's frequency-weighted basis,
  n_flat from H(T) and sigma_res alone lands inside, or within 14% of, the
  measured k99 interval in all eight Pythia tables. It has no d in it. The
  real-spectrum capacity bound is true but 12-43 times too small.
- Inference: identity information measures the vocabulary half only (an
  i.i.d. source, F_1). The directions the network needs beyond identity's
  (k99 to k*) are where the grammar half should show, if Andrew's guess is
  right.
- Hypothesis for the toddler language: the dimension a receiver needs is
  about d_identity(H, sigma) + d_grammar(states, or rank R; see the
  parked rank inference above), and dimensions beyond both stay empty.
- Dropped: further tests on the tail beyond k*(0.1). That split is set by
  k*, which the paper shows is a property of the trained network, not of
  the language.

From: Andrew, "figure out where we are deviating", after the paper landed.

## 2026-10-09: A toddler language, and a toy codebook with known answers

The small models still use languages that are too advanced to start from:
grammar too complex, vocabulary too broad. Even was/were is advanced. We
need something like a toddler language. My 2-year-old uses "bubble" for
about 5 different things. And when I hand her something she wants she says
"here you go da da", because her whole life, when we hand her things, we
say "here you go Ellie". This feels like the right pivot toward Shannon.

From: thinking about step 3 of "Next step on the measure" (check on a toy
first)
Rough notes from talking it through:

Toy codebook (answer known before measuring):
- Two-level hypercube: each symbol has a binary address; the first bits are
  its class (the semantic level), the rest its variant (the pedantic
  level). Each bit is +/- an amplitude on its own dimension, large for
  class bits, small for variant bits.
- Independent dimensions, so the channel splits into one-dimensional binary
  channels: exact I(k), I_B, I_W and multilevel-coding rates are sums of
  scalar capacities (one integral each). The spectrum order is known
  (large amplitudes first). Ungerboeck's set partitioning in its simplest
  form.
- Use: the Monte Carlo estimator and every comparison (crossover,
  half-fill, rates) must reproduce the exact values before any is used on
  a real table; sweep the amplitude ratio to see what each responds to.

Toddler language:
- "Bubble" is a vocabulary smaller than the world: several referents on one
  word, so the loss from world to word is visible (the 2026-10-04 "loss is
  the meaning" entry, made concrete).
- "Here you go da da" is a chunk with a slot: a phrase stored as one unit
  from exposure. She says it when she is the one receiving, with the
  giver's name in the slot: the caregivers' frame, used from the other side
  of the exchange. (Corrected 2026-10-09; an earlier version of this note
  said the slot was filled with whoever receives.) The
  vocabulary/grammar line from 2026-10-05 at its first appearance;
  exposure doing the work. Developmental names, from memory, to check:
  Braine's pivot grammar (1963); Tomasello's item-based constructions.
- (a) Design one first: a small world (people, objects, actions); a
  vocabulary deliberately smaller than the world with controlled
  polysemy; a handful of frames with slots ("more X", "X gone", "want X",
  "here you go Y"); known probabilities. Finite-state, so entropy, the
  number of grammar states and word-referent information are exact. Knobs:
  vocabulary size, frames, polysemy, world size. Train tiny models and
  non-neural receivers (n-gram counters) on it and see whether the
  dimension they need follows V and the grammar's state count: experiment
  C's step 2, controlled.
- (b) Then real toddler speech as the check: the CHILDES archive (children
  around age 2 with caregivers; from memory, Brown's Adam, Eve and Sarah),
  free for research under TalkBank's ground rules.

Why it is a pivot toward Shannon (inference): Shannon built English up from
constructed sources with known statistics (his series of approximations)
and modelled sources and coders as finite-state machines (section 8). A
designed toddler language is a Shannon source we write ourselves, so every
quantity is known before any receiver sees it: start where the field
started.

Prior art, searched 2026-10-09 (computational; developmental linguistics).
Checked on arXiv, Crossref or publisher pages unless marked.

Computational, most important:
- The derived dimension is probably rank, not V plus states. Borenstein,
  Svete, Chan, Valvoda, Nowak, Augenstein, Chodroff, Cotterell, "What
  Languages are Easy to Language-Model? A Perspective from Learning
  Probabilistic Regular Languages", ACL 2024, arXiv:2406.04289: any
  language model built on a hidden representation needs hidden size at
  least R, the rank of the language (of a minimal full-support
  deterministic probabilistic automaton); on random automata (states and
  alphabet 2 to 16) with RNNs and transformers at several hidden sizes,
  rank predicts learnability better than state count or alphabet size.
  Direct antecedent; frame the toddler experiment against R.
- Svete and Cotterell, "Recurrent Neural Language Models as Probabilistic
  Finite-state Automata", EMNLP 2023, arXiv:2310.05161: an arbitrary
  deterministic finite-state LM with N states needs on the order of N|V|
  neurons in an RNN.
- Hewitt, Hahn, Ganguli, Liang, Manning, EMNLP 2020, arXiv:2010.07515:
  Dyck-(k,m) needs and gets Theta(m log k) hidden units. Classical automaton
  simulation bounds (Alon, Dewdney and Ott 1991; Horne and Hush 1996;
  Indyk 1995; partly unverified) and transformer upper bounds (Rizvi et al.
  2024, arXiv:2403.09728).
- Design-method background: Elman, "Finding Structure in Time", 1990 (small
  hand-built grammars into a recurrent net); White and Cotterell 2021,
  arXiv:2106.01044 (artificial languages varying one feature); TinyStories;
  BabyLM and BabyBERTa (real child-directed corpora). (Removed 2026-10-09: a
  line citing the softmax bottleneck as relevant here; Andrew, 2026-10-08: the softmax bottleneck is a trivial rank bound, and the output table reads the full residual stream, so squashing it measures the stream, not the vocabulary.)
- Toy codebook: not new (Ungerboeck 1982; Wachsmann et al. 1999).
- Not found (about 70%): a designed, semantically grounded language (a world
  of referents, vocabulary smaller than the world, role-filled frames,
  word-referent information as a controlled variable); a measured minimum
  width at the exact entropy floor compared against a derived bound (R,
  sqrt(m), m log k); polysemy swept in a dimension study; neural and n-gram
  receivers side by side on the same exact-entropy source.

Developmental linguistics (well established; names to use):
- "Bubble" is overextension (categorical overinclusion, analogical
  overextension): Rescorla, "Overextension in early language development",
  J. Child Language 7, 1980, doi:10.1017/S0305000900002658 (about a third of
  first 75 words ever overextended); Clark 1973, 1978. Formal model:
  Ferreira Pinto and Xu, "A computational theory of child overextension",
  Cognition 206, 2021, doi:10.1016/j.cognition.2020.104472 (probabilistic
  inference, not information theory).
- Frames with slots: Braine 1963 (pivot grammar); Braine 1976, "Children's
  first word combinations", doi:10.2307/1165959 (limited-scope formulae);
  Lieven, Pine and Baldwin 1997, doi:10.1017/S0305000996002930
  (slot-and-frame patterns about 60% of multiword utterances, ages 1-3);
  Tomasello 1992, 2003 (verb islands, item-based constructions).
- Known probabilities for frames have an empirical basis: Cameron-Faulkner,
  Lieven and Tomasello, "A construction based analysis of child directed
  speech", Cognitive Science 27, 2003, doi:10.1207/s15516709cog2706_2 (51%
  of mothers' utterances began with one of 52 item-based frames).
- Chunks learned whole: Peters 1983 (gestalt route); Pine and Lieven 1993
  (rote-learned phrases later given slots); Bannard and Matthews 2008
  (frequent chunks repeated more accurately).
- Not found: a child using a caregiver's frame from the other side of the
  exchange, with the name slot filled by the other person in that event
  ("here you go da da", said while receiving). Pronoun-reversal work
  studies copying the slot verbatim; closest anchor: Charney 1980,
  doi:10.1017/S0305000900002816 (early role terms understood first from the
  child's own role).
- Information-theoretic: Zaslavsky et al. 2018 (adult naming as lossy
  compression); Tal, Grossman and Arnon 2024, Cognition 249,
  doi:10.1016/j.cognition.2024.105817 (entropy rate of infant-directed
  speech falls as infants grow).

CHILDES terms (talkbank.org ground rules, checked 2026-10-09): CC BY-NC-SA
3.0 unless marked; no commercial use (LLMs named); cite each corpus (Brown
1973 for Adam, Eve, Sarah; corpus doi:10.21415/T5HK5G) and MacWhinney 2000;
data may go to web AI services only through an API that guarantees it is
not stored. That last rule decides whether Claude can read the transcripts
directly: Andrew to settle before plan (b). Eve (1;6 to 2;3) matches a
2-year-old.

Parked inference (2026-10-09): how rank R fits the embedding-dimension
picture. Not the first experiment; kept so it is not lost. The first
experiments follow Shannon's precedent: build the source, know its
statistics exactly, then bring in receivers.
- R is the rank of the language's table of next-token log-probabilities
  (rows: contexts or grammar states; columns: tokens).
- Retracted 2026-10-09: two bullets that argued d >= R from the
  hidden-state x output-table product and used it to explain why the
  output tables use all of d (observation 3). That argument is the
  softmax bottleneck, which Andrew had already rejected (Andrew, 2026-10-08: the softmax bottleneck is a trivial rank bound, and the output table reads the full residual stream, so squashing it measures the stream, not the vocabulary). Observation 3 stays an observation with no explanation. Borenstein
  et al.'s bound is the same argument in general form; their empirical
  result (rank predicts learnability) is the part worth keeping.
- The input side needs something else: enough to tell the network how each
  token changes the state. Tokens that act on the grammar alike could share
  a vector, so the input need is about the number of distinct token
  actions, smaller than R. Fits the input-side slack (observation 2) and
  gives H3's two numbers a definite form: output bounded by the language's
  rank, input by the variety of token actions plus exposure.
- R <= min(number of grammar states, V): a property of the language, set by
  vocabulary and grammar together. (That it caps a model's dimension rests
  on the retracted bound.) "Bubble" merges columns (can lower R); a
  chunk like "here you go" moves structure from order into vocabulary,
  changing both. All computable for a designed language.
- The singular values of the log-probability table grade the rank: large
  ones the main ways predictions differ (semantic core), small ones the
  fine distinctions (pedantic tail). The output squash test was in effect
  measuring this graded rank at a loss tolerance.
- Candidate for what the input tail carries (see "The tail isn't
  identity"): prediction-relevant structure, how a token shifts the state,
  along small singular directions. Testable once the toddler language's
  true spectrum is known.

## 2026-10-09: Next step on the measure: real noise, then the tail

Where the main line resumes (measure and tail), from asking what Shannon
would make of the two comparison measures.

- Pin the noise to the receiver. Shannon's results always start from the
  channel's actual noise; epsilon is a dial. In an embedding table the
  noise has concrete sources: finite numeric precision, and interference
  from the other tokens packed into the same room (observation 7's
  overlap). Set sigma from the table's measured interference and epsilon
  stops being a knob: a derived property of the receiver, which is the
  hyperparameter point itself.
- Recast the comparison operationally. Rather than crossover or half-fill,
  ask at what rate a receiver with k dimensions can carry "which word" and
  "which variant" (or "which role" and "which word") as separate streams:
  the multilevel-coding rates (Wachsmann et al. 1999). No arbitrary
  one-half.
- Check on a toy first: the smallest codebook where the answer is known in
  advance, and confirm the measure recovers it before trusting it on
  Pythia.
- Then point it at the tail: with real noise, ask what the tail carries.
  First candidate is graded similarity (do full-room similarities track
  how alike two tokens' usage is, better than core-only similarities?).
- "The Bandwagon" warning applies throughout: information-theory labels on
  embeddings prove nothing unless the channel model is real.

## 2026-10-09: Hellman and Cover 1970: what an m-state receiver can learn

Read closely: Hellman and Cover, "Learning with Finite Memory", Annals of
Mathematical Statistics 41(3), 765-782, 1970 (library scan, OCR'd for
reading; the equal-prior formula below reproduces the paper's own Example
1 numbers, 1/101 and 1/82, which checks the OCR). Facts first, then
inferences.

What the paper does:
- Two-hypothesis testing (P0 vs P1) on an i.i.d. stream, with the data
  summarized after each observation by a statistic T in {1, ..., m},
  updated by a time-invariant (possibly randomized) rule T_n = f(T_{n-1},
  X_n), and a decision d(T_n). The pair (f, d) is a finite-state machine
  with m states. Loss: long-run probability of error as n goes to
  infinity.
- They reject two other notions of limited memory as not real
  constraints: remembering the last k observations (the space of those
  can be infinite) and remembering one real number such as the likelihood
  ratio (infinitely many values).
- The key statistic is gamma = (sup of the likelihood ratio) / (inf of it)
  over events, "a natural measure of the resolvability of the two
  hypotheses in the finite memory case".
- Main result: the least achievable error is
  P* = [2 sqrt(pi0 pi1 gamma^(m-1)) - 1] / (gamma^(m-1) - 1)
  (when gamma^(m-1) exceeds the prior ratio); with equal priors,
  P* = 1 / (1 + gamma^((m-1)/2)). An m-state machine's stationary state
  likelihood ratios can spread by at most gamma^(m-1) (Theorem 2).
- P* is the greatest lower bound but is not achieved by any machine for
  m > 2 (Theorem 4); epsilon-optimal machines exist. They are saturating
  counters: move up one state only on near-maximal likelihood-ratio
  events, down only on near-minimal ones, ignore everything else, and
  leave the end states only with small probability. In the discrete case
  this needs artificial randomization, which they call surprising "since
  randomization usually decreases information".
- Examples: coins with p = 0.501 vs 0.499 give gamma = 1.008, so a 5-state
  machine is little better than no memory, and about 500 states are needed
  for 1% error; the difference |p0 - p1| is a poor measure of
  resolvability with finite memory. Normal vs Cauchy location tests behave
  alike with unlimited memory, but the normal case reaches zero error with
  2 states while the Cauchy case stays at 0.15.
- Conclusions: rounding a sufficient statistic to a few digits is far from
  optimal, because the machine should wait for extreme events. With a
  time-varying rule, Cover (1969) reaches zero error with m = 4, so time
  invariance is part of the constraint. Finite sample size N is left open.

Inferences for Ariadne (not in the paper):
- A derived number from a receiver's capacity, in closed form, in a
  non-neural receiver: invert the equal-prior bound and the memory needed
  for error P is m = 1 + 2 ln((1 - P)/P) / ln(gamma). The form H2 wants,
  for one receiver kind. Credit it as prior art for the shape of a
  derivation.
- What a bounded receiver can use is a different property of the source
  than what an unbounded one can use. With unlimited memory the error
  decays at a rate set by an information quantity (Chernoff information,
  from the general theory, not this paper); with m states only gamma, the
  extreme likelihood ratios, matters. Normal and Cauchy are the clean
  example. This is H1 shown exactly, in 1970.
- The optimal machine recodes the input into a three-letter alphabet
  (extreme high, extreme low, everything else) before it counts. The
  alphabet falls out of the receiver's limit and the source's extremes;
  nothing is tuned, and its size is 3 for every m. A derived alphabet
  (question 3), in a finite-state receiver. Check whether the brief's H2
  should credit this directly.
- "Rounding is far from optimal" says the best code for a finite receiver
  is not the most faithful compression of the sufficient statistic; it
  throws away the moderate events entirely. Compare "the loss from source
  to signal is the meaning": the machine's meaning is in which events it
  discards.
- The finite-N case is open in the paper, and it is the realistic one
  (Ariadne's exposure term). Look for later work on it.

## 2026-10-09: Shannon's section 8: coders as finite transducers

Read closely from the original scan (shannon-1948-part1.pdf, pages 21-23;
BSTJ 27(3), pp. 399-401). Facts first, then inferences.

What section 8 ("Representation of the Encoding and Decoding Operations")
says:
- The transmitter and the receiver are each a *discrete transducer*: input
  a sequence of symbols, output a sequence of symbols, with an internal
  memory, so the output depends on the present input and the past.
- The memory is assumed finite: "a finite number m of possible states". A
  transducer is two functions, output y_n = f(x_n, a_n) and next state
  a_{n+1} = g(x_n, a_n), where a_n is the state.
- Transducers connect in tandem. One whose output a second transducer can
  turn back into the original input is *non-singular*; the second is its
  inverse.
- Theorem 7 (paraphrase): a finite-state transducer driven by a
  finite-state statistical source outputs a finite-state statistical source
  whose entropy per unit time is at most the input's, and equal when the
  transducer is non-singular. Proved on the product state space of source
  state and transducer state.
- Theorem 8 (the rest of the section): for a channel given as a graph of
  constraints, one assignment of transition probabilities maximizes the
  entropy, and that maximum is the capacity C.
- Section 9's converse then uses it: the transmitter must be non-singular,
  so the channel input carries the source's entropy, which cannot exceed C.

Inferences for Ariadne (not in the paper):
- Theorem 7 is "recoding is lossless" in Shannon's own terms: an invertible
  finite coder keeps the entropy exactly; any other can only lose it. A
  deterministic tokenizer with a decoder is a non-singular transducer.
  Credit Shannon for Andrew's position rather than state it as new.
- The theorem bounds entropy, not use. It says nothing about whether the
  receiver can exploit what arrives, which is where Ariadne starts.
- The finite memory is a modelling convenience, not a limit. m can be any
  finite number, and the coding theorem in section 9 works on blocks of N
  symbols with N growing without bound; a transducer that buffers such
  blocks needs a number of states that grows with N (exponentially in N
  for a block code). So Shannon's coder is finite at every N but unbounded
  across them: "finite but unlimited", the same idealization the brief
  names for the receiver.
- Section 8 is where a bounded receiver would enter Shannon's own
  formalism: fix m, or tie it to a cost, and ask what Theorems 7 and 9
  become. Hellman and Cover (1970, in the library) did the hypothesis-
  testing version with S states.

## 2026-10-09: Shannon tied vocabulary size to redundancy (anchor for piece 1)

In the 1948 paper itself, Shannon puts vocabulary size next to redundancy.
Basic English, with its vocabulary limited to 850 words, has very high
redundancy, seen in the expansion when a passage is translated into it;
Joyce enlarges the vocabulary and "is alleged to achieve a compression of
semantic content". So the man who said meaning is irrelevant to the
engineering problem also says a bigger vocabulary compresses semantic
content. The vocabulary question is in the founding paper, as an aside.

From: asking what Shannon would say about the two comparison measures
(crossover and half-fill), 2026-10-09
Source: Shannon 1948, Part I, section 7, p. 399 (BSTJ 27(3)). Checked
against the original scan (shannon-1948-part1.pdf, page 21) on
2026-10-09: the quoted words are exact, and the printing itself reads
"James Joyces' book "Finigans Wake."", so quote it with [sic], not
corrected.
- The next section on the same page (section 8) models the transmitter
  and receiver as discrete transducers with a finite internal memory, "a
  finite number m of possible states". Shannon did give the coder a finite
  memory there; his capacity results still let block length, and so the
  codebook, grow without limit. Worth a closer read for piece 1.
Rough notes:
- Same section defines relative entropy (a source's entropy over the
  maximum with the same symbols) and redundancy (one minus that), and
  gives English at roughly 50%. Half-fill is a normalization in that
  spirit; the crossover compares raw bits.
- His third requirement for H (breaking a choice into two successive
  choices; section 6) is exactly the between-class plus within-class
  split both measures rest on.
- Information belongs to the source: the text view (source
  probabilities) is the Shannon view, the dictionary view describes the
  alphabet. Experiment C's text view gives a clean order (role before
  word, observation 21); the dictionary view gives none (22).
- Use in piece 1: start where the field started; the field's founder
  already saw vocabulary size trade against redundancy and semantic
  compression, and nobody derived the size.

## 2026-10-09: The tail isn't identity

Not a fully formed idea, but the data indicates the tail isn't identity.

From: experiment C, identity information (observations 14, 15, 20, 21)
Rough notes:
- Token identity of every kind measured so far is essentially complete
  within a few hundred directions: which token (14), which spelling variant
  (15, 20), which grammatical role and which word within it (21).
- Yet the network depends on thousands of input directions (2, 5), and the
  tail past the loose core carries token-specific content: shuffling it
  costs as much as or more than deleting it (11).
- So the tail holds something token-specific that is not about telling
  tokens apart. Open: what is it? Candidates not yet tested: graded
  similarity (how alike tokens are, not whether they differ), information
  the later layers read as features rather than as identity, or the
  model's use of exact values (precision) rather than distinctions.

## 2026-10-08: Blocks as a vocabulary over the vocabulary (to explore)

Markus Hartikainen's reasoning blocks are structurally a second vocabulary
on top of the token vocabulary: each block is a string of base tokens
chosen as one unit, the move BPE makes one level up (a phrase dictionary
over the token alphabet, like LZ78 or Tunstall over characters). In his
version they are not trained, and the model never sees them as units.

From: Markus's reply on his test-time-compute post (2026-10-08): the block
library is "text, or token ids, because those are what the solver is
allowed to reorder"; KV segments go stale past the first divergence, and
prefix sharing across sibling assemblies still works.
Rough notes from talking it through:
- A two-level code: the model is the receiver at the token level, the
  solver at the block level. For the searcher the blocks are a recoding;
  for the model nothing changed (no embedding per block, read as its
  component tokens).
- Differs from the model's vocabulary: BPE is chosen once by frequency
  (compression) and segments text one fixed way; blocks are chosen by
  meaning and verification at inference time and overlap, so choosing
  among them is the solver's job.
- Two ways to train them, very different: mined (BPE over reasoning traces:
  keep the token sequences that recur in good solutions, frequency-chosen)
  or learned into the model as macro-tokens (an embedding row each). The
  second moves them out of the solver's vocabulary into the model's: in the
  2026-10-05 terms, structure moves out of order and into the vocabulary,
  and the vocabulary-size question comes with it.
- Recursive blocks (blocks built from blocks, Mika Korhonen's suggestion in
  the same thread) are grammar-based compression: Re-Pair, Sequitur, the
  smallest grammar problem (Charikar et al. 2005), already in the credit
  list. A recursive block library is a grammar over the token vocabulary,
  which is the vocabulary/grammar line: where to cut between units and
  rules.
- Branching order and the prefix cache: if the solver fixes blocks in token
  order, each child reuses its parent's whole prefix and pays prefill only
  for the block it adds; branching on later positions first re-prefills
  most of each candidate. The cache becomes part of the solver's cost model
  (asked Markus whether that cost should shape the branching order).
- To explore: where the cut between block vocabulary and solver grammar
  should go for a bounded searcher, and whether a two-level code (tokens
  for the model, blocks for the solver) has a derivable granularity.

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
- Test 1 run 2026-10-09 (`semantic_core.py`): giving tokens another
  token's tail, or matched random values, costs at least as much as zeroing
  it, so the tail carries token-specific content, not only spread
  (experiment C observation 11). Test 2 not run.
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
Moved to: measurements in experiments/c-empty-dimensions (observations
11-13, `semantic_core.py`, 2026-10-09); the measure of density in
definitions.md section 1 (identity information, with half-fill dimensions
comparing how fast between-word and within-word distinctions fill).

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
  receiver/transmitter split under another name. (Removed 2026-10-09: "the
  softmax bottleneck is then a transmitter limit"; Andrew, 2026-10-08: the softmax bottleneck is a trivial rank bound, and the output table reads the full residual stream, so squashing it measures the stream, not the vocabulary.)
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
