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

**Identity information** (bits):

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

**Crossover dimension:** $k_\times$, the smallest $k$ at which the marginal
between-class information falls below the marginal within-class
information, $\Delta I_B(k) < \Delta I_W(k)$. Past $k_\times$, each added
direction buys more distinction within classes than between them.

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
- Gaussian noise, isotropic in the projected coordinates (a modelling
  choice; it treats every direction as equally noisy)

### 1.7 Relation to the loss-based count

Experiment C's *directions needed* ($k^*$) asks how many directions a
particular trained network depends on. $I(k)$ asks how much symbol identity
the codebook carries in its top $k$ directions at resolution $\varepsilon$,
whatever reads it. Comparing the two shows whether the network uses what
its table can express.

### 1.8 Prior art

Mutual information through a Gaussian channel and the capacity bound are
Shannon (1948, 1949); the chain-rule split is standard. Whether this exact
measure on embedding tables, or the crossover dimension, has a name in
existing work has not been searched yet.
