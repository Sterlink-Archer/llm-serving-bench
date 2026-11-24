#!/bin/bash
# Gate for every commit: all tests, then the staged-tree secret scan, with REAL exit codes. Use:
#   git add -- <paths> && bin/commit_gate.sh && git commit -m ... -- <paths>
# Never pipe the scanner (`secret_scan.sh | tail -1 && git commit` returns tail's status: on 2026-11-22 that let a finding
# through and masked every earlier scan's exit code).
set -uo pipefail
cd "$(dirname "$0")/.."
bin/test_all.sh > /tmp/commit_gate_tests.$$ 2>&1; t=$?
tail -1 /tmp/commit_gate_tests.$$; [ $t -eq 0 ] || { cat /tmp/commit_gate_tests.$$; rm -f /tmp/commit_gate_tests.$$; echo "commit gate: TESTS RED"; exit 1; }
rm -f /tmp/commit_gate_tests.$$
bin/secret_scan.sh; s=$?
[ $s -eq 0 ] || { echo "commit gate: SECRET SCAN HIT"; exit 1; }
echo "commit gate: OK"
