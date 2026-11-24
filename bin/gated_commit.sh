#!/bin/bash
# Stage, gate and commit in ONE command, so the gate's exit code is checked here and no pipe can mask it.
# 2026-11-22 (twice): `bin/commit_gate.sh | tail -n 6 && git commit ...` committed through a RED gate, because the chain
# saw tail's exit code. Piping THIS script's output is safe: the commit only runs inside, after a green gate.
#   bin/gated_commit.sh -m "message" -- path [path ...]      (or -F msgfile)
# Paths are staged with `git add -A` (deletions included) and committed path-limited; nothing is pushed.
set -uo pipefail
cd "$(dirname "$0")/.."
msg=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do msg+=("$1"); shift; done
[ "${1:-}" = "--" ] && shift
[ ${#msg[@]} -gt 0 ] && [ $# -gt 0 ] || { echo "usage: bin/gated_commit.sh -m MSG|-F FILE -- path [path ...]"; exit 2; }
for p in "$@"; do                     # stage each path: content if it exists, deletion if it is gone
  if [ -e "$p" ]; then git add -A -- "$p" || { echo "gated_commit: git add $p failed"; exit 1; }
  else git rm -r -q --cached --ignore-unmatch -- "$p" || { echo "gated_commit: git rm $p failed"; exit 1; }; fi
done
bin/commit_gate.sh; g=$?
[ $g -eq 0 ] || { echo "gated_commit: gate RED (rc=$g), nothing committed"; exit 1; }
git commit -q "${msg[@]}" -- "$@" || { echo "gated_commit: git commit failed"; exit 1; }
echo "gated_commit: committed $(git log -1 --format=%h) (not pushed)"
