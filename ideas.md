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
- Dretske 1981: perception to concept as a lossy analog-to-digital step.
  Book not yet in hand (references/to-get.md).
- Refinement to test: the meaning may be the structure of the loss (which
  differences are discarded), not the amount.

## 2026-10-04: Information theory starts at the signal, not the transmitter?

Information theory seems to start at the signal, the information, and not at
the transmitter. Is this true? If the alphabet is chosen at the sending end,
is "the right alphabet" a question about the transmitter as much as the
receiver?

## 2026-10-04: The hypotheses are over-restricted

H1 to H4 may be drawn too narrowly.
