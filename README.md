# Ariadne

How much of a message can a finite receiver actually use, and does the right
alphabet fall out of the receiver's capacity instead of being tuned?

Shannon's theory assumes a receiver that can represent anything, so the
alphabet a message arrives in cannot matter. Real receivers are finite, and for
them it does. Vocabulary size in a language model is one instance of the
question; it appears wherever a finite receiver meets a code. The full framing,
hypotheses, prior art and plan are in [brief.md](brief.md).

Named for Theseus, Claude Shannon's 1950 maze-solving mouse with a finite relay
memory, and the thread Ariadne gave Theseus to find his way through the
labyrinth.

**Status:** research program defined; nothing written or built yet.
**Visibility:** private working repository. See [COPYRIGHT.md](COPYRIGHT.md).

## Layout

| Path | What it holds |
|---|---|
| `brief.md` | The program: question, positions, hypotheses, pieces, experiments, prior art, risks |
| `notes/` | Dated research notes, `YYYY-MM-DD-topic.md`. Notes are a record: add new ones rather than rewriting old conclusions |
| `references/refs.bib` | One bibliography shared by every piece |
| `references/library/` | Papers downloaded for reading. Git-ignored; `references/library.tsv` lists what belongs there |
| `pieces/` | One folder per piece of writing, in the site's paper format |
| `experiments/` | One folder per experiment: code, pinned environment, small results |

## Publishing

A piece is published on plainsight-systems.com through the site's paper
pipeline (`scripts/build-paper.sh` in plainsight-systems-site), or as an
article on andrewphunter.com, or as a standalone LinkedIn post, whichever fits
the result. Published writing is CC BY 4.0, copyright Andrew P Hunter;
published code is Apache 2.0.
