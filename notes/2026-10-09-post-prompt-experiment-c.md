# Prompt for the writing chat: Ariadne so far, and where it goes next

*Prepared 2026-10-09 from the repository. Paste everything below the line
into the writing chat. The repository is public:
https://github.com/plainsight-systems/ariadne. Every number below is taken
from the files linked next to it; check a figure there before changing it.*

---

You are helping me (Andrew P Hunter) write a post about my research: what
I have found so far and where it goes next. It is most likely a LinkedIn
post; if the material wants more room, tell me and we can make it an
article for andrewphunter.com instead. Do not name the program (Ariadne)
in the post; ask me before linking the repository.

The previous post, "where I was wrong" on dimensions, went out on
2026-10-08 (https://www.linkedin.com/feed/update/urn:li:activity:7514028818534334464/).
It covered: the expectation that d is usually too large at a fixed
vocabulary; 20-40% of the input embedding removable in the larger Pythia
models; the strict count growing with size (60-70% of d) as a possible
pedantic floor; the loose count at 2,048 for both 6.9B and 12B as a
candidate semantic core; and the picture of tokens pushed apart into a
mostly empty volume, with a cliff from semantic to pedantic. Do not repeat
it; build on it. Its output-table paragraph was cut (see "Do not claim").

## Voice (my rules, not negotiable)

Plain, first person where it fits, flat declaratives, concrete nouns. No
em-dashes. No sentence couplets or rule-of-three runs. No two-sentence
opener trap (short setup plus mirrored punchline). No concessive reversals
("The intuition is understandable. It is also wrong."). No aphoristic morals
or meta-commentary about the writing. No "I'm curious whether"; ask the
question directly. Avoid showy abstractions ("substrate", "load-bearing").
State the position, then illustrate. Lead with a visual hook if the format
allows. Credit prior art before claiming anything. Keep facts, inferences
and hypotheses separate, and say how confident a claim is. Results that
contradicted a prediction are results; include them.

## What Ariadne is

Shannon's theory assumes a receiver that can represent anything, so the
alphabet a message arrives in cannot matter. Real receivers are finite: a
network has a fixed width, a codec a fixed table, a person a fixed working
memory. Ariadne asks how much of a message a finite receiver can use, and
whether the right vocabulary size and embedding dimension can be derived
instead of tuned. My position: a hyperparameter is a sign the theory is
incomplete. Program: [brief.md](https://github.com/plainsight-systems/ariadne/blob/main/brief.md).

## What I did and found (experiment C, "do models have empty dimensions?")

All in [experiments/c-empty-dimensions/README.md](https://github.com/plainsight-systems/ariadne/blob/main/experiments/c-empty-dimensions/README.md),
under "Observed trends" (23 numbered observations, kept apart from
interpretation). The key ones:

1. **Setup.** Pythia keeps one vocabulary (50,277 tokens) and one training
   text while the embedding dimension d grows from 512 to 5,120. That lets d
   vary with the vocabulary fixed. I squashed each embedding table onto its
   top k principal directions, ran the model, and found the smallest k that
   keeps the loss within 0.01 bits per byte of the untouched model.
2. **The vocabulary stretches to fill whatever room it gets.** Trained
   tables spread over 91-98% of d at every size, needed or not.
3. **The input side has empty room once the room is big enough.** Up to
   d = 1,024 every direction is needed. From 1.4B up, about 20-40% of the
   input table's directions can be removed for 0.01 bits per byte.
4. **The output side never has empty room.** The table that picks the next
   token uses all of d in every model, size and vocabulary tested.
5. **A prediction failed.** I expected the needed count to level off at a
   natural number for this vocabulary. It did not: 1,664 directions at
   d = 2,560, 2,816 at 4,096, 3,072 at 5,120 (about 60-70% of d). At looser
   tolerances the two largest models need the same count, so a core may
   level off while finer detail keeps spreading.
6. **Exposure matters.** Through training, the needed count keeps rising.
   The late slowdown matched the learning-rate schedule, not the language
   running out (parked as a side branch).
7. **Same room, different vocabularies (d = 2,048):** TinyLlama (32K),
   Pythia (50K), OLMo-2 (100K). The 100K vocabulary needs the whole room
   even for its core.
8. **The tail carries real token-specific content.** Giving each token
   another token's tail costs more than deleting the tail (12B: +0.29 vs
   +0.10 bits per byte).
9. **But the tail is not identity.** I defined a measure, *identity
   information*: how many bits of "which token was it" survive in the top k
   directions through Gaussian noise
   ([definitions.md](https://github.com/plainsight-systems/ariadne/blob/main/definitions.md),
   section 1). Telling tokens apart needs only a few hundred directions.
   Spelling variants (" The"/"the") and grammatical role (noun, verb,
   punctuation) are settled early; in ordinary text, role fills before
   which-word in 53 of 60 cases and never after. So the thousands of
   directions the network depends on are for something other than telling
   tokens apart. That is the open question.

## Prior art to credit if the post touches these

- Identity information is built from established parts: at a fixed k it
  is the constellation-constrained capacity of a finite signal set
  (Ungerboeck 1982, equation 5, estimated there by Monte Carlo); the
  between/within split is the multilevel-coding chain rule (Wachsmann,
  Fischer and Huber 1999); the capacity ceiling matches the coding rate in
  MCR² (Yu et al. 2020). What may be new is sweeping it over directions and
  pointing it at a receiver's table.
- Input-embedding redundancy is known (ALBERT 2019; Kataiwa et al. 2025;
  Quemy 2026, arXiv:2608.29702).
- On languages with known automata, a language's rank predicts how well
  models learn it, better than state count or alphabet size: Borenstein,
  Cotterell et al., ACL 2024. Cite the empirical result only, not their
  hidden-size bound (see "Do not claim").
- An exact derived number for a finite receiver already exists in 1970:
  Hellman and Cover, "Learning with Finite Memory": the best error for an
  m-state learner is 1 / (1 + gamma^((m-1)/2)) with equal priors.
- A capacity-derived semantic alphabet: Nixon 2026 (arXiv:2604.09521,
  unreviewed).

## Shannon passages (checked against the 1948 scan)

- Shannon put vocabulary size next to redundancy in the founding paper
  (BSTJ 27(3), p. 399): "The Basic English vocabulary is limited to 850
  words and the redundancy is very high." And: "Joyce on the other hand
  enlarges the vocabulary and is alleged to achieve a compression of
  semantic content." The printing calls the book "Finigans Wake" and writes
  "Joyces'"; quote with [sic] if quoted.
- Section 8 models the transmitter and receiver as finite-state transducers
  ("a finite number m of possible states"), and Theorem 7 says an
  invertible one keeps the entropy exactly. That is "recoding is lossless"
  in Shannon's words; credit him for it.

## Where it goes next

Follow Shannon's own method: he built English up from constructed sources
whose statistics he knew. The small models I have been probing learned
languages that are too advanced to start from; even was/were is advanced.
So the next step is a designed toddler language. My 2-year-old uses
"bubble" for about five things (a vocabulary smaller than the world;
linguists call it overextension, Rescorla 1980). And when I hand her
something she wants, she says "here you go da da", because her whole life,
when we hand her things, we say "here you go" and her name: she uses our
frame from the receiving side and puts the giver's name in the slot (frames
with slots: Braine; Tomasello; Lieven, Pine and Baldwin).
The plan: a small world, a vocabulary smaller than the world, a few frames
with slots, known probabilities, so entropy and grammar are exact before
any receiver sees it. Check the measure first on a toy codebook whose
answer is exact. Then train tiny models and simple counters on the
language and see whether the dimension they need follows from the
vocabulary and the grammar.

Decide before publishing: whether to use my daughter's name. The repo notes
use it; the post does not have to.

## Do not claim

- Anything that explains the output table using all of its room. Leave it
  as an observation. Do not cite the softmax bottleneck or any rank bound
  on hidden size: it is a trivial rank bound, and the output table reads
  the full residual stream, so squashing it measures the stream, not the
  vocabulary (my decision, 2026-10-08).
- That a natural embedding dimension has been found. The plateau
  prediction failed.
- A theory of how networks work. Ariadne follows the "why is it this way"
  question from receivers where proofs are tractable.
- That identity information is a new measure. Its parts are established.
- Generality beyond what was tested: mostly one model family, one text,
  tolerances I chose (stated).

## Visuals available (original plots from the repository)

- Loss against directions kept, all Pythia sizes:
  https://github.com/plainsight-systems/ariadne/blob/main/experiments/c-empty-dimensions/results/loss_vs_k.png
- Same room, three vocabularies:
  https://github.com/plainsight-systems/ariadne/blob/main/experiments/c-empty-dimensions/results/cross_vocab.png
- Identity information along k:
  https://github.com/plainsight-systems/ariadne/blob/main/experiments/c-empty-dimensions/results/identity_info.png
- Spectra of the trained tables:
  https://github.com/plainsight-systems/ariadne/blob/main/experiments/c-empty-dimensions/results/spectra.png

## Sources in the repository

- Observations and runs: `experiments/c-empty-dimensions/README.md` and
  `results/`.
- The measure: `definitions.md`.
- Ideas behind this, in Andrew's words: `ideas.md`, entries "Vocabulary,
  order, room and shape", "Semantic core vs pedantic refinement", "The tail
  isn't identity", "Shannon tied vocabulary size to redundancy", "Shannon's
  section 8", "A toddler language, and a toy codebook with known answers".
