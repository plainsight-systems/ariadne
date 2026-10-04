# Ariadne

![A small wood-and-brass mechanical mouse with a ball of red thread on its back stands in a lit square of a 5 by 5 relay maze. The thread unwinds behind it through the corridors it has already traveled, back to a brass post at the maze entrance. Under the floor, a cutaway shows rows of telephone relays, a few glowing amber. On the right, the maze walls rise and fold into an impossible higher-dimensional labyrinth that fades into a drafting grid.](assets/ariadne-readme.png)

<sub>Illustration generated with ChatGPT for this repository.</sub>

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
**License:** writing CC BY 4.0, code Apache 2.0; see [LICENSE](LICENSE) and
[NOTICE](NOTICE).

## Layout

| Path | What it holds |
|---|---|
| `brief.md` | The program: question, positions, hypotheses, pieces, experiments, prior art, risks |
| `ideas.md` | Half-formed thoughts and questions to explore; not hypotheses |
| `notes/` | Dated research notes, `YYYY-MM-DD-topic.md`. Notes are a record: add new ones rather than rewriting old conclusions |
| `references/refs.bib` | One bibliography shared by every piece |
| `references/to-get.md` | Books and papers to acquire later, with where to get them |
| `references/library/` | Papers downloaded for reading. Git-ignored; `references/library.tsv` lists what belongs there |
| `pieces/` | One folder per piece of writing, in the site's paper format |
| `experiments/` | One folder per experiment: code, pinned environment, small results |
| `scripts/` | `publish-piece.sh` (copy a piece into the site and build it) and its tests |

## Publishing

A piece is published on plainsight-systems.com through the site's paper
pipeline (`scripts/publish-piece.sh` here, which runs `scripts/build-paper.sh`
in plainsight-systems-site), or as an article on andrewphunter.com, or as a
standalone LinkedIn post, whichever fits the result. Published writing is
CC BY 4.0, copyright Andrew P Hunter; published code is Apache 2.0.
