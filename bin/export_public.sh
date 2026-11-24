#!/usr/bin/env bash
# Build the PUBLIC copy of this repo (github.com/Sterlink-Archer/llm-serving-bench) from HEAD as ONE fresh commit: no
# history, no co-author trailers. This checkout stays the private working repo (llm-serving-bench-private).
#   bin/export_public.sh <empty-or-new dir>      then push that dir's main to the public repo.
# Left out (private only): copies of third-party web pages and model cards (the parsed JSON with URL and fetch time stays)
# and the publishing checklist. Fails on any secret-scan hit, any private marker, or any broken relative link.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: bin/export_public.sh <dir>}
[ -e "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ] && { echo "export: $OUT is not empty"; exit 1; }
mkdir -p "$OUT"
git diff --quiet HEAD -- || { echo "export: commit first (the export is built from HEAD)"; exit 1; }
git archive HEAD | tar -x -C "$OUT"
EXCLUDE=(docs/PUBLISHING.md research/raw/openrelay)
for f in "${EXCLUDE[@]}"; do rm -rf "${OUT:?}/$f"; done
# raw Hugging Face 'recent'/'trending' listings from the model scout: thousands of random uploads, some with explicit names
find "$OUT/research/raw" -maxdepth 1 \( -name 'daytona_pricing_*.html' -o -name 'readme_google_*.md' \
  -o -name 'phase2_hf_recent_*.json' -o -name 'phase2_hf_trending_*.json' \) -delete
bash bin/secret_scan.sh --dir "$OUT"
# bracketed so this file does not match itself. Company name and session ids: every file. Profanity: our own prose only
# (Markdown), since saved model answers in results/ are generated text.
markers=$( { grep -r -I -l -i -E 'de[s]ync|local_[0-9a-f]{8}-' "$OUT" --exclude-dir=corpus --exclude-dir=eval --exclude-dir=chat;
             grep -r -I -l -i -E 'fu[c]k|sh[i]t' "$OUT" --include='*.md'; } || true)
[ -z "$markers" ] || { echo "export: private markers in:"; echo "$markers"; exit 1; }
(cd "$OUT" && python3 -I tests/test_links.py) | tail -1
git -C "$OUT" init -q -b main
git -C "$OUT" add -A
GIT_AUTHOR_NAME=Sterlink-Archer GIT_AUTHOR_EMAIL=158239285+Sterlink-Archer@users.noreply.github.com \
GIT_COMMITTER_NAME=Sterlink-Archer GIT_COMMITTER_EMAIL=158239285+Sterlink-Archer@users.noreply.github.com \
  git -C "$OUT" -c commit.gpgsign=false commit -q -m "llm-serving-bench: serving OpenRelay's models ourselves, measured their way" \
  -m "Same-hardware comparison with OpenRelay's published Gemma 4 31B MI355X recipe, plus Inkling-2-Mini and GLM-5.3-Flash. Write-up, recipes, every run's raw files. Built from the private working repo at $(git rev-parse --short HEAD)."
echo "export: $(git -C "$OUT" ls-files | wc -l) files, one commit $(git -C "$OUT" rev-parse --short HEAD) in $OUT"
