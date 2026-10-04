#!/usr/bin/env bash
#
# Publishes an Ariadne piece through the plainsight-systems-site paper
# pipeline, without changing the site.
#
#   scripts/publish-piece.sh <piece-dir> <slug> [--arxiv] [--site <dir>]
#
# Copies the piece's source into the site's papers/<slug>/, then runs the
# site's own scripts/build-paper.sh <slug> [--arxiv]. Committing the site
# (source and output together, per its papers/README.md) is left to you.
#
# Only an allowlist is copied, because the site build publishes every file
# in papers/<slug>/ except paper.md and refs.bib:
#
#   <piece-dir>/paper.md       required
#   <piece-dir>/figures/       optional, copied as-is (no symlinks)
#   references/refs.bib        the shared bibliography, always
#
# Anything else in the piece folder (notes, drafts) is never copied; the
# script lists what it left behind. A refs.bib inside the piece folder is
# an error: the bibliography is shared.
#
# papers/<slug>/ is replaced wholesale, so nothing stale survives. It is
# replaced only if it does not exist yet or still matches exactly what
# this script last wrote there (recorded in the site's git-ignored
# build/ariadne-publish/<slug>.sha256). Anything else (edits made on the
# site side, or a site paper that already uses the slug) stops the
# script before anything is touched.
#
# --site defaults to ~/repositories/plainsight-systems-site.

set -euo pipefail

die() { echo "publish-piece: $*" >&2; exit 1; }

usage="usage: scripts/publish-piece.sh <piece-dir> <slug> [--arxiv] [--site <dir>]"
[[ $# -ge 2 ]] || die "$usage"
piece_arg="$1"
slug="$2"
shift 2
site_arg="$HOME/repositories/plainsight-systems-site"
build_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --arxiv) build_args+=(--arxiv); shift ;;
    --site)  [[ $# -ge 2 ]] || die "$usage"; site_arg="$2"; shift 2 ;;
    *)       die "$usage" ;;
  esac
done
[[ "$slug" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]] || die "slug must be lowercase words joined by hyphens, got '$slug'"

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bib="$root/references/refs.bib"
[[ -f "$bib" ]] || die "shared bibliography not found at references/refs.bib"

# ── Site ───────────────────────────────────────────────────────
[[ -d "$site_arg" ]] || die "site not found at $site_arg (pass --site <dir>)"
site="$(cd "$site_arg" && pwd)"
[[ -x "$site/scripts/build-paper.sh" ]] || die "$site has no executable scripts/build-paper.sh"
[[ -d "$site/papers" ]] || die "$site has no papers/ folder"
dst="$site/papers/$slug"
record_dir="$site/build/ariadne-publish"
record="$record_dir/$slug.sha256"

# ── Piece ──────────────────────────────────────────────────────
[[ -d "$piece_arg" ]] || die "piece folder not found: $piece_arg"
piece="$(cd "$piece_arg" && pwd)"
[[ -f "$piece/paper.md" && ! -L "$piece/paper.md" ]] || die "no paper.md in $piece_arg"
[[ ! -e "$piece/refs.bib" ]] \
  || die "$piece_arg/refs.bib would be ignored; citations go in the shared references/refs.bib"
if [[ -e "$piece/figures" ]]; then
  [[ -d "$piece/figures" && ! -L "$piece/figures" ]] || die "$piece_arg/figures must be a folder"
  links="$(cd "$piece" && find figures -type l | LC_ALL=C sort)"
  [[ -z "$links" ]] || die "symlinks are not published; replace them with files: $(tr '\n' ' ' <<<"$links")"
fi
left="$(cd "$piece" && find . -mindepth 1 -maxdepth 1 ! -name paper.md ! -name figures ! -name .DS_Store \
        | sed 's#^\./##' | LC_ALL=C sort)"

# ── Target ─────────────────────────────────────────────────────
# One line per file: sha256, two spaces, path. Finder's .DS_Store files are
# ignored here as they are by the site build.
fingerprint() {
  (cd "$1" && find . ! -type d ! -name .DS_Store | LC_ALL=C sort | while IFS= read -r f; do
     printf '%s  %s\n' "$(shasum -a 256 < "$f" | cut -d' ' -f1)" "$f"; done)
}
if [[ -e "$dst" ]]; then
  [[ -f "$record" ]] \
    || die "papers/$slug/ already exists on the site and was not written by this script; pick another slug, or remove it from the site first"
  [[ "$(fingerprint "$dst")" == "$(cat "$record")" ]] \
    || die "papers/$slug/ on the site has changed since the last publish; bring those changes into $piece_arg, then remove papers/$slug/ from the site and rerun"
fi

# ── Copy ───────────────────────────────────────────────────────
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
stage="$work/$slug"
mkdir -p "$stage"
cp "$piece/paper.md" "$stage/paper.md"
cp "$bib" "$stage/refs.bib"
if [[ -d "$piece/figures" ]]; then
  (cd "$piece" && find figures -type f ! -name .DS_Store -print0 \
    | while IFS= read -r -d '' f; do mkdir -p "$stage/$(dirname "$f")"; cp "$f" "$stage/$f"; done)
fi

rm -rf "$dst"
mv "$stage" "$dst"
mkdir -p "$record_dir"
fingerprint "$dst" > "$record"

echo "publish-piece: copied $piece_arg to papers/$slug/ ($(fingerprint "$dst" | wc -l | tr -d ' ') files)"
if [[ -n "$left" ]]; then
  echo "publish-piece: not copied (working material): $(tr '\n' ' ' <<<"$left")"
fi

# ── Build ──────────────────────────────────────────────────────
cd "$site"
exec scripts/build-paper.sh "$slug" ${build_args[@]+"${build_args[@]}"}
