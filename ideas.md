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
