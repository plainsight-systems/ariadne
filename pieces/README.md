# Pieces

One folder per piece of writing, numbered in the order the brief proposes:
`01-receiver-shannon-left-out/`, `02-bounded-receivers/`, and so on.

Pieces bound for plainsight-systems.com use the site's paper format so they can
go through its pipeline unchanged: `paper.md` with the site's front matter,
figures in `figures/`, and citations keyed to `../references/refs.bib` (no
`refs.bib` in the piece folder). The format (math delimiters, labelled
figures, cross-references, numbered sections) is documented in
plainsight-systems-site `papers/README.md`.

To publish, from the repository root:

```bash
scripts/publish-piece.sh pieces/01-receiver-shannon-left-out receiver-shannon-left-out
```

Only `paper.md`, `figures/` and the shared bibliography reach the public site.
Notes and drafts can live in the piece folder; the script lists them as not
copied.

Every published piece carries: copyright Andrew P Hunter, CC BY 4.0, published
by Plainsight Systems.

Before Andrew sees a draft, and after any substantive revision, the paper goes
through the three-lens audit in `AUDIT.md`, each reviewer in a fresh context.
