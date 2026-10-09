# Definitions

Formal definitions of the measures Ariadne uses. A definition here is a
choice of what to measure, not a claim about what the measure will show.
Each entry says which parts are fixed by the mathematics and which are
choices (knobs), so the knobs stay visible.

Started 2026-10-09.

## 1. Identity information of a codebook

**Purpose.** Measure how densely a receiver's table of symbol vectors
occupies the space it is given: how much of *which symbol this is* the
table can carry, direction by direction, and how much of that is a
distinction between words rather than between variants of one word. It
applies to any finite codebook of vectors (a language model's embedding
table, a vector-quantization codebook, any receiver that maps symbols to
points), so it does not depend on there being a neural network downstream.

### 1.1 Setup

- **Codebook.** Symbols $v \in \{1, \dots, V\}$ with vectors
  $e_v \in \mathbb{R}^d$. For a language model, the rows of the input or
  output embedding table for the tokens the tokenizer can produce (padding
  rows excluded).
- **Symbol distribution** $p(v)$, a choice:
  - *dictionary view:* $p$ uniform over the codebook;
  - *text view:* $p(v)$ the frequency of $v$ in a stated text.
- **Centering.** $\mu = \sum_v p(v)\, e_v$ and $c_v = e_v - \mu$.
- **Principal directions.** The eigenvectors $u_1, \dots, u_d$ of
  $\Sigma = \sum_v p(v)\, c_v c_v^\top$, ordered by eigenvalue
  $\lambda_1 \ge \dots \ge \lambda_d \ge 0$. $U_k = [u_1, \dots, u_k]$.
- **Projection onto the top $k$ directions.** $x_v^{(k)} = U_k^\top c_v \in \mathbb{R}^k$.

### 1.2 The channel

Draw a symbol $V \sim p$ and observe its projection through Gaussian noise:

$$Y_k = x_V^{(k)} + Z, \qquad Z \sim \mathcal{N}(0, \sigma^2 I_k).$$

The noise level is set relative to the codebook's own scale, so it has no
units:

$$\sigma = \varepsilon\, s, \qquad s = \sqrt{\operatorname{tr}\Sigma / d},$$

where $s$ is the root-mean-square spread per direction of the full
codebook and $\varepsilon > 0$ is the **resolution**, a choice. Results are
reported for several values of $\varepsilon$.

### 1.3 Definitions

**Identity information** (bits; at a fixed $k$ this is the
constellation-constrained mutual information of the codebook, see 1.8):

$$I(k) = I(V; Y_k) = H(V) - H(V \mid Y_k).$$

**Marginal density** (bits per direction): $\Delta I(k) = I(k) - I(k-1)$,
reported over blocks of directions in practice.

**Average density:** $\rho(k) = I(k)/k$.

**Capacity ceiling:**

$$C(k) = \sum_{i=1}^{k} \tfrac{1}{2} \log_2\!\left(1 + \lambda_i / \sigma^2\right).$$

**Efficiency:** $\eta(k) = I(k)/C(k)$, the share of what the spectrum could
carry that the codebook actually carries as symbol identity.

**Class split.** Given a partition of the symbols into classes by a
function $g$, with $G = g(V)$:

- *between-class information:* $I_B(k) = I(G; Y_k)$
- *within-class information:* $I_W(k) = I(V; Y_k \mid G)$

Two ways to compare the parts along $k$, kept side by side (Andrew,
2026-10-09: explore both):

**Crossover dimension:** $k_\times$, the smallest $k$ at which the marginal
between-class information falls below the marginal within-class
information, $\Delta I_B(k) < \Delta I_W(k)$. Past $k_\times$, each added
direction buys more distinction within classes than between them, in
absolute bits. Because within-class information is at most
$H(V) - H(G)$, it is sensitive to class sizes (see 1.9); where it exists,
it marks the point at which the between-class part has nearly run out.

**Half-fill dimensions:** for each part, the smallest $k$ at which it
reaches half of its own value in the full room:

$$k_{B,1/2} = \min\{k : I_B(k) \ge \tfrac12 I_B(d)\}, \qquad
k_{W,1/2} = \min\{k : I_W(k) \ge \tfrac12 I_W(d)\}.$$

Comparing them says which kind of distinction the codebook fills first:
$k_{B,1/2} < k_{W,1/2}$ means between-class distinctions sit in the
leading directions and within-class distinctions further down;
$k_{W,1/2} \le k_{B,1/2}$ means the reverse. Each part is measured against
its own total, so the comparison does not depend on how many bits each
class split holds.

**First class definition, surface variants.** $g(v)$ is the decoded text of
$v$ with surrounding whitespace removed and letters lowercased, so " The",
"The" and "the" share a class. A symbol whose decoded text is empty after
stripping is its own class. This is one crude, objective partition, not a
definition of meaning; observation 12 of experiment C already shows
distinctions (comma versus period, was versus were) that it does not
capture.

### 1.4 Properties (follow from the definitions)

1. **$I(k)$ never decreases in $k$.** $Y_k$ is $Y_{k+1}$ with one coordinate
   dropped, and the noise is independent across coordinates, so
   $I(V; Y_k) \le I(V; Y_{k+1})$ (data processing).
2. **$I(k) \le H(V) \le \log_2 V$.** At most the identity of the symbol.
3. **$I(k) \le C(k)$.** $I(V; Y_k) \le I(X; Y_k)$ for $X = x_V^{(k)}$, and
   for a given covariance the Gaussian maximizes output entropy, so
   $I(X; Y_k) \le \tfrac{1}{2}\log_2\det(I + \Sigma_k/\sigma^2)$, which is
   $C(k)$ since $\Sigma_k$ is diagonal in the principal directions.
   Variance that does not separate symbols raises $C$ without raising $I$;
   the gap $C - I$ is spread that carries no identity.
4. **The split is exact.** $G$ is a function of $V$, so by the chain rule
   $I(k) = I_B(k) + I_W(k)$ for every $k$.

### 1.5 Estimation

The posterior over symbols is exact for this channel:

$$p(v \mid y) \propto p(v)\, \exp\!\left(-\lVert y - x_v^{(k)} \rVert^2 / 2\sigma^2\right).$$

So $H(V \mid Y_k) = \mathbb{E}[-\log_2 p(V \mid Y_k)]$ is estimated by Monte
Carlo: draw $v \sim p$ and $z$, form $y$, and evaluate the posterior over
all $V$ symbols. $H(G \mid Y_k)$ uses the same posterior summed within
classes. No density estimation is involved; the only error is sampling
error, reported with each estimate.

### 1.6 Knobs, stated

- the symbol distribution $p$ (dictionary or text view, and which text)
- the resolution $\varepsilon$
- the class function $g$
- the fraction in the half-fill dimensions (one half), and that each part is
  measured against its own value at $k = d$, which itself depends on
  $\varepsilon$ and $d$
- Gaussian noise, isotropic in the projected coordinates (a modelling
  choice; it treats every direction as equally noisy)
- the order in which directions are added: by principal variance. For a
  discrete codebook this is not the order that maximizes mutual information
  (non-diagonal precoders can do better even on parallel channels; Pérez-Cruz,
  Rodrigues and Verdú 2010), so the principal ordering is a choice, not an
  optimum.

### 1.7 Relation to the loss-based count

Experiment C's *directions needed* ($k^*$) asks how many directions a
particular trained network depends on. $I(k)$ asks how much symbol identity
the codebook carries in its top $k$ directions at resolution $\varepsilon$,
whatever reads it. Comparing the two shows whether the network uses what
its table can express.

### 1.8 Prior art

Searched 2026-10-09 (communications and information theory; machine
learning and NLP). Bibliographic fields checked against Crossref, arXiv or
the publisher's abstract page. Content claims marked *(secondary)* rest on
citing papers, course notes or memory and must be checked against the paper
before any piece cites them.

**Established under other names:**

- **$I(k)$ at a fixed $k$** is the *constellation-constrained capacity*
  (also called coded-modulation capacity) of a finite signal set on the
  Gaussian channel, with uniform $p$; with non-uniform $p$, the mutual
  information of a probabilistically shaped constellation. Ungerboeck,
  "Channel coding with multilevel/phase signals", IEEE Trans. Inf. Theory
  28(1), 55-67, 1982, doi:10.1109/TIT.1982.1056454: equation (5) gives the
  capacity $C^*$ of $N$ equiprobable discrete input signals with
  continuous-valued Gaussian output, plotted against SNR in Fig. 2
  (checked against the paper, 2026-10-09).
- **The estimator:** the same paper evaluates $C^*$ "by Monte Carlo
  averaging of (5)" with a Gaussian random number generator (checked). It
  remains standard practice in that literature.
- **$C(k)$ and $I \le C$** are the Gaussian-input capacity of parallel
  channels and the maximum-entropy bound. "Shaping gap" or "gap to capacity"
  is the family of names for $C - I$ (Forney and Ungerboeck, "Modulation and
  coding for linear Gaussian channels", IEEE Trans. Inf. Theory 44(6), 1998,
  doi:10.1109/18.720542 *(secondary)*), with a caveat: the familiar 1.53 dB
  figure is a high-SNR limit for uniform cubic constellations, while
  $C(k) - I(k)$ here also contains the saturation at $H(V)$.
- **The class split** is the multilevel-coding chain rule over
  set-partition levels: Imai and Hirakawa, IEEE Trans. Inf. Theory 23(3),
  1977, doi:10.1109/TIT.1977.1055718 *(secondary)*; Wachsmann, Fischer and
  Huber, "Multilevel codes: theoretical concepts and practical design
  rules", IEEE Trans. Inf. Theory 45(5), 1361-1391, 1999,
  doi:10.1109/18.771140: section II applies the chain rule of mutual
  information over the set-partitioning levels (their equation 2), giving
  one "equivalent channel" per level, and generalizes it to arbitrary
  signal probabilities and labelings (checked against the paper,
  2026-10-09). $I_B$ is the first-level (coarse-partition) mutual
  information. One difference: there the labeling is chosen as part of the
  code design; here $g$ is imposed from outside, by what counts as the same
  word.
- **Saturation at small $\varepsilon$:** $H(V) - I$ decays like a Q-function
  of the minimum distance (Alvarado, Brännström, Agrell and Koch, IEEE Trans.
  Inf. Theory 60(2), 2014, doi:10.1109/TIT.2013.2291865, arXiv:1212.6526).
- **Per-channel allocation with discrete inputs** (mercury/waterfilling):
  Lozano, Tulino and Verdú, IEEE Trans. Inf. Theory 52(7), 2006,
  doi:10.1109/TIT.2006.876220. Close to the per-direction view, but with
  independent inputs per channel.
- **Derivative along SNR, not dimension** (I-MMSE): Guo, Shamai and Verdú,
  IEEE Trans. Inf. Theory 51(4), 2005, doi:10.1109/TIT.2005.844072.

**Close, in machine learning:**

- **Coding rate** $\tfrac12\log\det(I + c\,ZZ^\top/\varepsilon^2)$, the same
  form as $C(k)$, with a whole-minus-within-class structure, used as a
  training objective: Yu, Chan, You, Song and Ma, "Learning Diverse and
  Discriminative Representations via the Principle of Maximal Coding Rate
  Reduction", arXiv:2006.08558, 2020. Credit for the log-det ceiling and the
  split's structure.
- **Noise to make mutual information meaningful for a deterministic map**,
  estimated from the resulting mixture: Goldfeld et al., "Estimating
  Information Flow in Deep Neural Networks", arXiv:1810.05728 (ICML 2019).
  Credit for the noisy-channel method.
- **Conditional usable information** beyond a baseline: Hewitt, Ethayarajh,
  Liang and Manning, "Conditional probing", arXiv:2109.09234 (EMNLP 2021).
  Close in spirit to $I_W$.
- **Principal-component variance is a poor guide to linguistic content**:
  Raunak, Kumar, Gupta and Metze, "On Dimensional Linguistic Properties of the
  Word Embedding Space", RepL4NLP 2020, doi:10.18653/v1/2020.repl4nlp-1.19.
  Consistent with a low-variance tail still carrying distinctions.
- **Surface variants** sit near a base form plus a linear offset in input and
  output embeddings: Reif, Kaplan and Schwartz, "Vocab Diet",
  arXiv:2510.17001. Capitalized and space-prefixed variants are about 15% and
  12% of a 32K BPE vocabulary: Reif, Kaplan and Schwartz, "More Than Words:
  Compositional Tokenization", arXiv:2610.05597.
- **Background:** information-theoretic and MDL probing (Pimentel et al.
  2020, arXiv:2004.03061; Voita and Titov 2020, arXiv:2003.12298; Hewitt and
  Liang 2019, arXiv:1909.03368); neural collapse, a geometric
  between/within-class split (Papyan, Han and Donoho 2020,
  doi:10.1073/pnas.2015509117; Wu and Papyan 2024, arXiv:2405.17767);
  token embeddings encode their characters (Kaushal and Mahowald,
  arXiv:2206.02608).

**Not found** (moderate confidence, about 60 to 70%, two searches in one
session):

- $I(k)$ traced as a function of the number of principal directions kept,
  for a fixed codebook (existing work runs along SNR or power allocation).
- The efficiency $\eta(k) = I/C$ as a curve in $k$.
- The crossover dimension $k_\times$, or any comparison of how fast
  between-class and within-class information fill (the half-fill
  comparison has not been searched separately).
- Any of this applied to a receiver's codebook such as an embedding table,
  or reporting where surface-variant distinctions sit in its spectrum.

So the definition reuses established objects; what may be new is the sweep
over directions, the crossover and half-fill comparisons, and the
application to receivers' tables.
Not yet searched: hierarchical information bottleneck, and usage studies of
vector-quantization codebooks.

### 1.9 Found in first use (Pythia, 2026-10-09)

Recorded as found. Second item, 2026-10-09: both the crossover and the
half-fill dimensions are kept in 1.3 and explored side by side (Andrew's
decision).

- **Useful resolutions.** At $\varepsilon \le 1$ every Pythia table
  carries nearly all of $H(V)$ in its top 8 directions, so the curve has no
  shape. $\varepsilon$ = 2 to 8 gives curves that rise across the room.
  Because $s$ is the per-direction spread of the full table, the same
  $\varepsilon$ is not the same noise for tables of different $d$ or
  different spectra; compare across tables with care.
- **The crossover as defined is dominated by class sizes.** Within-class
  information is at most $H(V) - H(G)$ (about 0.9 bits of 15.6 for surface
  variants over Pythia's vocabulary), so a direction almost always buys
  more between-class bits than within-class bits in absolute terms, and
  $k_\times$ mostly does not exist. A candidate replacement compares how
  fast each part fills relative to its own total: $k_{B,1/2}$ and
  $k_{W,1/2}$, the smallest $k$ at which $I_B(k) \ge \tfrac12 I_B(d)$ and
  $I_W(k) \ge \tfrac12 I_W(d)$. It was reported alongside $k_\times$ in the
  first measurement. The one-half is itself a knob. Both kept (see 1.3).
- **Crossovers found so far are noise.** Across Pythia, TinyLlama and
  OLMo-2 (144 cases), every crossover that appears falls after both parts
  have saturated, with both per-direction gains within two standard errors
  of zero. A crossover should count only where at least one gain exceeds
  its sampling error; with that rule, none has been found for
  surface-variant classes.
- **The crossover tracks class sizes in both directions.** With
  grammatical-role classes (about 2-3 bits between, 7-12 within) it
  appears at once, at 8-16 directions, in every case; with surface-variant
  classes (about 15 bits between, under 1 within) no real one appears.
  The half-fill comparison gave different, view-dependent orders for the
  same tables (experiment C, observations 21-22), so it is the one that
  carries information about the codebook.
- **Second class definition, grammatical role.** $g(v)$ is the 12-tag
  universal part of speech (Petrov, Das and McDonald 2012, as mapped by
  NLTK), tagged in context for the text view and from a lexicon for the
  dictionary view; details and coverage in experiment C's
  `identity_roles.py`.

### 1.10 Pinned noise: deriving $\sigma$ from the receiver

Written 2026-10-09, before any number was computed with it. It replaces the
resolution $\varepsilon$ (a choice) with noise levels measured from the
receiver itself. None is fitted to a result. Each gives
$\varepsilon = \sigma / s$, with $s$ from the full table as in 1.2, so it
can be placed against the earlier $\varepsilon$ = 2, 4, 8.

**Primary: residual-stream interference, $\sigma_{\text{res}}$.** A
language model reads its input table only through the residual stream.
At the input to the second block, a reader sees the token's vector $e_v$
plus what the first block added at that position, $a_t = h_1(t) - e_{v_t}$,
which depends on the context as well as the token. From the point of view
of recovering which token it is, the part of $a_t$ that the token does not
determine is interference. Define

$$\sigma_{\text{res}}^2 = \frac{1}{d}\,\operatorname{tr}\Big(\text{pooled within-token covariance of } a_t\Big),$$

the per-coordinate variance of $a_t$ around its mean for the same token,
pooled over tokens that occur at least twice, measured on the phase-2
evaluation text (WikiText-103 test, first 16 x 1,024 tokens). The total
per-coordinate variance of $a_t$ (token-determined part included) is
reported alongside.

Assumptions: (i) the context-driven part of the first block's output is
treated as noise for identity, though it carries context; (ii) it is
approximated as isotropic Gaussian, though it is neither; (iii) only the
first block is counted, so later blocks would add more and this is the
smallest residual-stream interference; (iv) one text, one position
sample.

**Floor: numeric precision, $\sigma_{\text{prec}}$.** The Pythia tables
are stored in fp16 (checked in the released files). Rounding a value $w$
to fp16 leaves an error spread evenly over one unit in the last place,
$\operatorname{ulp}(w) = 2^{\lfloor \log_2 |w| \rfloor - 10}$ for normal
numbers, with variance $\operatorname{ulp}(w)^2/12$. Define
$\sigma_{\text{prec}}^2$ as the mean of that over the table's entries.
Assumes rounding errors independent across entries and of the token. It
is the least noise any reader of the stored table faces.

**Reference, not computed: one competing codeword.** If the reader's
input could contain another token's vector from the same table at equal
strength, the noise covariance would be $\Sigma$, whose per-coordinate
variance is $\operatorname{tr}\Sigma / d = s^2$: $\varepsilon = 1$ by
definition, under the isotropic approximation.
