#!/usr/bin/env bash
#
# Tests for scripts/publish-piece.sh.
#
#   scripts/test-publish-piece.sh
#
# Each case runs the real script against a throwaway site folder whose
# scripts/build-paper.sh is a FAKE: it records its arguments and checks
# that papers/<slug>/paper.md exists, and it fails if the site contains a
# file named FAKE-BUILD-FAILS. It never builds anything. The real site
# build is exercised separately, by hand.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
publish="$root/scripts/publish-piece.sh"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

passed=0
failed=0
ok()   { passed=$((passed + 1)); echo "ok    $1"; }
fail() { failed=$((failed + 1)); echo "FAIL  $1"; [[ -z "${2:-}" ]] || sed 's/^/        /' <<<"$2"; }

# Fresh fake site and piece for each case.
setup() {
  rm -rf "$tmp/site" "$tmp/piece"
  mkdir -p "$tmp/site/scripts" "$tmp/site/papers" "$tmp/piece/figures/sub"
  cat > "$tmp/site/scripts/build-paper.sh" <<'FAKE'
#!/usr/bin/env bash
# FAKE build-paper.sh for scripts/test-publish-piece.sh. Builds nothing.
set -euo pipefail
site="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$site/papers/$1/paper.md" ]] || { echo "fake build: no paper.md" >&2; exit 3; }
echo "$*" >> "$site/fake-build.log"
[[ ! -e "$site/FAKE-BUILD-FAILS" ]] || exit 4
FAKE
  chmod +x "$tmp/site/scripts/build-paper.sh"
  printf -- '---\ntitle: "T"\n---\n\nBody.\n' > "$tmp/piece/paper.md"
  echo '<svg/>' > "$tmp/piece/figures/a.svg"
  echo 'png' > "$tmp/piece/figures/sub/b.png"
  echo 'private' > "$tmp/piece/notes.md"
  echo 'finder' > "$tmp/piece/.DS_Store"
  echo 'finder' > "$tmp/piece/figures/.DS_Store"
}

# run <args...>: runs the script, sets $out and $code.
run() {
  set +e
  out="$("$publish" "$@" 2>&1)"
  code=$?
  set -e
}

files_in() { (cd "$1" && find . -type f | LC_ALL=C sort | tr '\n' ' '); }

# ── Cases ──────────────────────────────────────────────────────

setup
run "$tmp/piece" test-slug --site "$tmp/site"
got="$(files_in "$tmp/site/papers/test-slug")"
want="./figures/a.svg ./figures/sub/b.png ./paper.md ./refs.bib "
if [[ $code -eq 0 && "$got" == "$want" ]]; then ok "first publish copies only the allowlist"
else fail "first publish copies only the allowlist" "exit $code; files: $got; output: $out"; fi

if cmp -s "$root/references/refs.bib" "$tmp/site/papers/test-slug/refs.bib" \
   && cmp -s "$tmp/piece/paper.md" "$tmp/site/papers/test-slug/paper.md"; then ok "paper.md and the shared refs.bib are copied unchanged"
else fail "paper.md and the shared refs.bib are copied unchanged"; fi

if [[ "$(cat "$tmp/site/fake-build.log")" == "test-slug" ]]; then ok "site build runs once with the slug"
else fail "site build runs once with the slug" "$(cat "$tmp/site/fake-build.log")"; fi

if grep -q 'not copied (working material): notes.md' <<<"$out"; then ok "working material is reported as not copied"
else fail "working material is reported as not copied" "$out"; fi

rm "$tmp/piece/figures/a.svg"
echo 'Revised.' >> "$tmp/piece/paper.md"
run "$tmp/piece" test-slug --site "$tmp/site"
got="$(files_in "$tmp/site/papers/test-slug")"
if [[ $code -eq 0 && "$got" == "./figures/sub/b.png ./paper.md ./refs.bib " ]] \
   && grep -q Revised "$tmp/site/papers/test-slug/paper.md"; then ok "republish replaces the folder and drops stale files"
else fail "republish replaces the folder and drops stale files" "exit $code; files: $got; output: $out"; fi

echo 'site-side edit' >> "$tmp/site/papers/test-slug/paper.md"
before="$(files_in "$tmp/site/papers/test-slug"; cat "$tmp/site/papers/test-slug/paper.md")"
builds="$(wc -l < "$tmp/site/fake-build.log")"
run "$tmp/piece" test-slug --site "$tmp/site"
after="$(files_in "$tmp/site/papers/test-slug"; cat "$tmp/site/papers/test-slug/paper.md")"
if [[ $code -ne 0 && "$before" == "$after" && "$(wc -l < "$tmp/site/fake-build.log")" == "$builds" ]] \
   && grep -q 'has changed since the last publish' <<<"$out"; then ok "edits on the site side stop the script before anything is touched"
else fail "edits on the site side stop the script before anything is touched" "exit $code; output: $out"; fi

setup
mkdir -p "$tmp/site/papers/existing"
echo 'someone else' > "$tmp/site/papers/existing/paper.md"
run "$tmp/piece" existing --site "$tmp/site"
if [[ $code -ne 0 && "$(cat "$tmp/site/papers/existing/paper.md")" == "someone else" && ! -e "$tmp/site/fake-build.log" ]] \
   && grep -q 'was not written by this script' <<<"$out"; then ok "a site paper already using the slug is not replaced"
else fail "a site paper already using the slug is not replaced" "exit $code; output: $out"; fi

setup
echo '@misc{x}' > "$tmp/piece/refs.bib"
run "$tmp/piece" test-slug --site "$tmp/site"
if [[ $code -ne 0 && ! -e "$tmp/site/papers/test-slug" ]] && grep -q 'shared references/refs.bib' <<<"$out"; then ok "a refs.bib inside the piece is refused"
else fail "a refs.bib inside the piece is refused" "exit $code; output: $out"; fi

setup
rm "$tmp/piece/paper.md"
run "$tmp/piece" test-slug --site "$tmp/site"
if [[ $code -ne 0 && ! -e "$tmp/site/papers/test-slug" ]] && grep -q 'no paper.md' <<<"$out"; then ok "a piece without paper.md is refused"
else fail "a piece without paper.md is refused" "exit $code; output: $out"; fi

setup
ln -s "$tmp/piece/notes.md" "$tmp/piece/figures/link.md"
run "$tmp/piece" test-slug --site "$tmp/site"
if [[ $code -ne 0 && ! -e "$tmp/site/papers/test-slug" ]] && grep -q 'symlinks are not published' <<<"$out"; then ok "symlinks in figures/ are refused"
else fail "symlinks in figures/ are refused" "exit $code; output: $out"; fi

setup
run "$tmp/piece" Bad_Slug --site "$tmp/site"
if [[ $code -ne 0 ]] && grep -q 'slug must be' <<<"$out"; then ok "a malformed slug is refused"
else fail "a malformed slug is refused" "exit $code; output: $out"; fi

setup
rm "$tmp/site/scripts/build-paper.sh"
run "$tmp/piece" test-slug --site "$tmp/site"
if [[ $code -ne 0 && ! -e "$tmp/site/papers/test-slug" ]] && grep -q 'no executable scripts/build-paper.sh' <<<"$out"; then ok "a site without the build script is refused"
else fail "a site without the build script is refused" "exit $code; output: $out"; fi

setup
run "$tmp/piece" test-slug --arxiv --site "$tmp/site"
if [[ $code -eq 0 && "$(cat "$tmp/site/fake-build.log")" == "test-slug --arxiv" ]]; then ok "--arxiv is passed to the site build"
else fail "--arxiv is passed to the site build" "exit $code; output: $out"; fi

setup
touch "$tmp/site/FAKE-BUILD-FAILS"
run "$tmp/piece" test-slug --site "$tmp/site"
first=$code
rm "$tmp/site/FAKE-BUILD-FAILS"
run "$tmp/piece" test-slug --site "$tmp/site"
if [[ $first -eq 4 && $code -eq 0 ]]; then ok "a failed build fails the script, and a rerun is allowed"
else fail "a failed build fails the script, and a rerun is allowed" "first exit $first, rerun exit $code; output: $out"; fi

echo
echo "$passed passed, $failed failed"
[[ $failed -eq 0 ]]
