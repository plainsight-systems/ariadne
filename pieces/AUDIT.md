# Paper audit protocol: three lenses

Every Ariadne paper gets three independent audits before Andrew sees a draft, and again after any substantive
revision (re-audit the changed sections and anything they touch). Each lens is a separate reviewer, read-only,
adversarial, told that one wrong number or misattributed claim would damage the author.
Each reports two lists, each item with the paper line number and an exact proposed fix:
  ERRORS: definitely wrong (give the correct value and the file or source it comes from)
  RISKS:  unverifiable, ambiguous, overclaimed, or likely to draw a referee objection
Correct items are not listed; give a one-line count of claims verified. The paper's author then applies the fixes,
checks each one against the source, and records what changed. Reviewers never edit the paper.

## Every reviewer runs in a fresh context

- Each lens is a new agent with no conversation history: a fresh spawn, never a continuation or fork of the
  author's session and never a message to an earlier reviewer. A re-audit is a new reviewer too.
- A reviewer gets only: the paper's path, the commit the paper cites, the text of its own lens below, and the
  "Failures already caught once" list. Nothing else from the drafting conversation: no summaries, no intended
  readings, no hints about what was hard.
- Reviewers do not see each other's reports. The three run in parallel.
- Reviewers have read access only (the repository, the reading library, the web for primary sources).
  The author, not a reviewer, applies fixes.

## Lens 1: facts (against the repository)
Source of truth: the experiment's README observations, results/*.json (not the rounded .md), and the scripts, at the
commit the paper cites.
- Every number, range, count, ratio, model name and setting matches the result files.
- Quantifiers are exact: "in every model", "from 410M up", "in all four" cover exactly the cases they claim.
  A range must not silently include a case that breaks it.
- Ratios and derived numbers are recomputed, not trusted.
- Method descriptions match what the script does: seeds, splits, draws, basis, centering, noise conversion,
  which tokens, which layer, measured vs interpolated.
- Anything stated as fact that is really an inference is flagged.
- Internal consistency: no sentence contradicts another, especially older sections written before new results.
- Anecdotes and personal examples are exactly as the author told them.

## Lens 2: citations and prior art (against primary sources)
- Every bib entry (authors, title, venue, year, DOI or arXiv id) checked against the primary source.
- Every attributed claim checked against what the work actually shows: read the passage; page, section, theorem
  and equation numbers must be right; quotes verbatim with page.
- Paraphrases of definitions (e.g. Shannon's redundancy) are faithful.
- Prior-art sweep: for each claim of novelty or each result, ask "what would a referee say this already is?"
  (e.g. SliceGPT for removable directions), search, and make sure the paper cites and distinguishes it.
- Standing rules: never explain output-table full use via the softmax bottleneck or a rank bound on hidden size;
  cite Borenstein et al. for the empirical result only. Don't name Ariadne's other projects or private context.

## Lens 3: mathematical rigor (a referee in information theory, probability, linear algebra)
- Every quantity is formally defined in the paper, not deferred to the repo. One term, one meaning throughout
  (direction vs dimension, measure vs density, capacity vs information, redundancy in which sense).
- Every property has a proof or a citation, and its conditions actually hold where it is applied
  (e.g. the max-entropy bound I <= C needs the covariance under the channel's own distribution and basis).
- Exact vs estimated is never blurred; conditional vs unconditional quantities are never blurred; order-dependence
  of sequential quantities is stated; standard errors say what they cover and what they don't.
- General claims are tested for counterexamples and stated with their conditions
  (e.g. x^2 is uncorrelated with x only if E[x^3] = 0).
- Asymptotic or regime arguments are checked for the regime actually measured (a low-SNR expansion does not apply
  when the total information is ~10 bits).
- Analogies are labelled as analogies, not presented as the original author's definition.
- Decision rules are stated before results, and the stated conclusions follow from them.
- Notation is consistent everywhere.

## Failures already caught once (check for these first)
- Softmax bottleneck reintroduced as an explanation.
- Ranges that drifted from the data (0.08-0.42), wrong counts of families, interpolated counts presented as measured.
- A tail-test logic reversal; "below the noise by construction" (false); "density zero" (exact density is never zero).
- A false statement about C - I; conditional results stated as unconditional ("the tail is not identity").
- Quantifier overreach: "the three smallest need nearly all" when one needed half; "about a third" true of one model only.
- A regime argument used outside its regime (low-SNR) when the true reason was codebook size.
- An echo/independence example stated without its distributional condition.
- A count described as using "only" two quantities when its formula uses a third (Shannon's count also uses the
  table's mean variance).
- Agreement claimed on a grid too coarse to tell the prediction from a constant; a comparison made after the values
  were known, with no rule set beforehand.

Add to this list whenever an audit catches a new kind of failure.
