#!/bin/bash
# Every test, failing fast: chain commits as `bin/test_all.sh && git commit ...`. On 2026-11-22 a hung run_local test
# (timeout rc=124) did not stop a commit because the chain used `;` after it, and a run.py change shipped with a red test.
set -uo pipefail
cd "$(dirname "$0")/.."
# One suite at a time on this machine: several sessions gate commits here, and run_local's fake box binds the fixed port
# 8000 (2026-11-22 09:52Z: two concurrent suites -> a false "server died" RED). Waits up to 60 min (20 min timed out
# with four sessions gating at once, 12:5xZ).
mkdir -p logs
exec 9>logs/.test_all.lock
flock -w 3600 9 || { echo "test_all: another suite held the lock for 60 min"; echo "RED"; exit 1; }
fail=0
run() { local name=$1; shift; local out; out=$("$@" 2>&1); local rc=$?; printf '%-28s rc=%s  %s\n' "$name" "$rc" "$(echo "$out" | tail -1 | cut -c1-110)"; [ $rc -eq 0 ] || fail=1; }
# run.py and bench/ are live for every session's launches: a broken file breaks their launches at once (2026-11-22 07:15Z a
# syntax error sat in run.py for ~8 minutes). Compile everything first.
run compile_all        python3 -I -c "import py_compile,glob,sys; fs=['run.py']+glob.glob('bench/*.py')+glob.glob('analysis/*.py')+glob.glob('infra/*.py'); [py_compile.compile(f, doraise=True) for f in fs]; print(f'OK: {len(fs)} files compile')"
# every saved price bar must parse: run.py and cost.py read the newest one (2026-11-22: a shell redirect overwrote it with
# printed text and turned run_local and report.py red)
run price_bars         python3 -I -c "import glob,json,sys; fs=sorted(glob.glob('research/openrelay_prices_*.json')); [json.load(open(f))['models']['openrelay/gemma-4-31b']['tiers']['Sync'] for f in fs]; print(f'OK: {len(fs)} price files parse, newest {fs[-1]}')"
run profile_classify   python3 -I tests/test_profile_classify.py
run startup_facts      python3 -I tests/test_startup_facts.py
run scoring            python3 -I tests/test_scoring.py
run claim_group        python3 -I tests/test_claim_group.py
run pricebar_rules     python3 -I tests/test_pricebar.py
run journal_select     python3 -I tests/test_journal_select.py
run traffic_notext     timeout 60 python3 -I tests/test_traffic_notext.py
run cached_traffic     timeout 300 python3 -I tests/test_cached_traffic.py
run evalrun_agree      timeout 150 python3 -I tests/test_evalrun_agree.py
run profile_admin      timeout 60 python3 -I tests/test_profile_admin.py
run runfiles           python3 -I tests/test_runfiles.py
run ladder_guard       timeout 60 python3 -I recipes/mi355x/tools/test_ladder_guard.py
run claim_monotonic    python3 -I tests/test_claim_monotonic.py
run readme_claims      python3 -I bin/build_readme.py --check
run md_links           python3 -I tests/test_links.py
run commit_msg_hook    python3 -I tests/test_commit_msg_hook.py
run gated_commit       timeout 60 python3 -I tests/test_gated_commit.py
run figures_verify     python3 -I analysis/figures.py --verify
run secret_scan_rules  timeout 60 python3 -I tests/test_secret_scan.py
run quality            python3 -I tests/test_quality.py
run score_window       python3 -I tests/test_score_window.py
run matrix_mock        timeout 600 python3 -I tests/test_matrix_mock.py
run run_local          timeout 900 python3 -I tests/test_run_local.py
run run_local_evict    env OR_TEST_EVICT=1 timeout 900 python3 -I tests/test_run_local.py
[ $fail -eq 0 ] && echo "ALL GREEN" || echo "RED"
exit $fail
