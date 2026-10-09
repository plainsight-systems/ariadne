# Working on Ariadne

Context for AI sessions in this repository. Read `brief.md` first; it is the
program. `README.md` has the layout. `notes/2026-09-30-bpe-information-theory-seed.md`
is the detailed research record the brief cites (superseded as a framing, kept
as evidence).

## Where things stand (2026-10-09)

- Program defined; nothing published yet.
- Experiment C (`experiments/c-empty-dimensions/`) has run on Pythia,
  TinyLlama and OLMo-2: 13 observed trends listed in its README under
  "Observed trends", kept apart from interpretation. The method agreed with
  Andrew: gather trends, check they point one way, then theory, then
  hypotheses derived from it.
- `definitions.md` holds the first formal measure (identity information of
  a codebook) with its knobs and prior art; defined, not yet measured.
- H2 is "partly anticipated" (Nixon 2026; see the brief).
- Andrew is reading John R. Pierce, *An Introduction to Information Theory:
  Symbols, Signals and Noise*, then going to the source: Shannon 1948, 1949, 1951
  (all in `references/library/`).
- Open next steps, in order (from the brief):
  1. Settle the order of pieces.
  2. Fresh prior-art pass on H2 and H3, outside ML as well as inside it.
  3. Design experiment A in detail.
  4. Outline piece 1.
  5. Convert the brief's credit list into verified `references/refs.bib` entries
     (only four exist so far).
  6. Experiment C: measure identity information on the tables already
     studied, then take the trends toward theory.

### Known gaps

- **`references/refs.bib` is nearly empty.** It has four verified entries
  (Shannon 1948 Parts I and II, Xu et al. 2020, Bottou and Bousquet 2007).
  Every other work in the brief's "Prior art carried over" list still needs an
  entry checked against the paper before any piece cites it.
  `references/library.tsv` already names a bibkey for each downloaded paper;
  use those keys.
- **The reading library is not in git.** `references/library/` (49 PDFs) exists
  only on Andrew's machine. On another machine, rebuild it from
  `references/library.tsv`; papers Andrew downloaded by hand (Creutz and Lagus
  2007 from ACM) cannot be fetched by script.

## Framing rules (Andrew's, not negotiable)

- **Not about BPE, not scoped to ML idioms.** BPE is how the question arose;
  vocabulary size is one instance. Every claim keeps at least one non-neural
  receiver in it.
- **A hyperparameter is a sign the theory is incomplete.** The goal is a derived
  number, not a tuned one. This is the same instinct as the skeptics of
  deep-learning theory, and Andrew agrees with them: Ariadne follows that
  skepticism out (why is it this way, what is the proof) from receivers where
  proofs are tractable. It never claims a theory of how networks work.
- **Recoding is lossless; what a finite receiver loses is expression,** which
  lives in the receiver's dimension ("flatlander" information theory).
- **Start where the field started** and work forward: what was assumed, what was
  accepted, and why.
- **Credit prior art before claiming anything.** H1 is prior art. The program
  rests on H2, H3 and the cross-receiver framing. Claim only what a fresh search
  cannot find elsewhere.
- **Keep facts, inferences and hypotheses separate,** and say how confident a
  claim is.
- **Loose thoughts go in `ideas.md`,** not the brief. An idea is not a
  hypothesis; promote it to `brief.md` or a note only when it firms up.
- **Record results that contradict a prediction.** They are results.

## Research rules

- **Citations:** every reference is checked against the paper itself (title,
  authors, venue, year, DOI or arXiv ID, and the specific figure or quote)
  before it enters `refs.bib` or a piece. Never paraphrase a result into
  something the paper does not say.
- **Technical asides must be as right as the main claim.** Check slogan-level
  contrasts (lossless vs lossy, deterministic vs not) against the mechanism.
- **The reading library** (`references/library/`, git-ignored) holds only
  legitimate copies: open access, author copies, institutional repositories, or
  files Andrew downloads himself. Record each one in `references/library.tsv`.
  - Ask before downloading anything new.
  - No pirated copies. The Internet Archive user upload of Shannon's *Collected
    Papers* and the copy on jonglage.net are both unauthorized; the legitimate
    copy is ASU Noble Library, TK5101 .S448 1993.
  - Never get around bot protection or CAPTCHAs (for example by copying a
    browser's anti-bot cookies). Use an author or repository copy instead.
  - If a site needs a login, Andrew creates it. Nothing is bought.

## Writing voice

Pieces are Andrew's, in his voice: plain, first person where it fits, flat
declaratives, concrete nouns.

- No em-dashes.
- No sentence couplets or triplets (mirrored twin sentences, rule-of-three runs).
- No two-sentence opener trap (short setup plus mirrored punchline).
- No concessive reversals ("The intuition is understandable. It is also wrong.").
- No aphoristic morals or meta-commentary about the writing.
- No "I'm curious whether"; ask the question directly.
- Avoid the suspect AI lexicon: "substrate", "load-bearing" and similar showy
  abstractions.
- State the position, then illustrate; do not build scaffolding toward it.

Showcase formula for pieces meant to travel: a visual hook first, real
mechanism underneath, a name with lineage (Theseus, then Ariadne), prior art
credited, original art only (no film stills, no likenesses, no Shannon
likeness).

## Publishing

- This repository is public. Writing is CC BY 4.0 and code is Apache 2.0,
  copyright Andrew P Hunter, published by Plainsight Systems: see `LICENSE`,
  `LICENSE-CONTENT`, `LICENSE-CODE` and `NOTICE`.
- A piece goes wherever it fits: a paper on plainsight-systems.com, an article
  on andrewphunter.com, or a standalone LinkedIn post.
- The site pipeline lives in `~/repositories/plainsight-systems-site`: the
  authoring contract is `papers/README.md`, the build is
  `scripts/build-paper.sh <slug> [--arxiv]`, which reads only the site's own
  `papers/<slug>/`.
- Publish a piece with this repository's
  `scripts/publish-piece.sh <piece-dir> <slug> [--arxiv]`. It copies
  `paper.md`, `figures/` and the shared `references/refs.bib` (nothing else)
  into the site's `papers/<slug>/`, then runs the site build. Andrew commits
  the site. It replaces `papers/<slug>/` only if it still matches the last
  publish, so a fix made on the site side has to come back into `pieces/`
  first. Tests: `scripts/test-publish-piece.sh`. The site's own build script
  is unchanged; changing it is a change to a public repository and needs
  Andrew's go-ahead.
- **plainsight-systems-site is the showcase.** Plans, audits, critiques and
  drafts never go there; only the finished paper does. Research working
  material belongs here, in the open. Anything that should not be public
  (career plans, personal material) goes in `~/Documents/`, not here.
- The site's Research page lists write-ups in `data/writing.yaml`: articles and
  papers, plus LinkedIn posts only when they stand alone or explain a build,
  never posts that just point to an article. Titles are quoted verbatim.
- Publication venue survey (arXiv endorsement, TechRxiv, Zenodo, TMLR and
  others): `~/Documents/plainsight-publication-venues-2026-10.md`.

## Related threads

- **Reasoning tokens and the hyperplane:** "Reasoning Tokens Steer the
  Trajectory" (andrewphunter.com/writing/reasoning-tokens-control-signal/,
  2026-09-22). A receiver steering a path through its own representation space;
  likely a later Ariadne piece. Run a prior-art check first (activation
  steering, representation engineering, logit and tuned lens, trajectory views
  of chain of thought).
- **Charlotte** (`~/repositories/tech-demos/charlotte`): the neural receiver for
  experiment B. Record the Charlotte commit any experiment runs against.

## Git

Public repository `plainsight-systems/ariadne`; push to `main`. Commit
messages end with the co-author trailer the session provides.
