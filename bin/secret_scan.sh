#!/usr/bin/env bash
# Scan the STAGED tree (default) or the whole history (--history) for keys, tokens, home paths, personal emails, and
# the machine-local values in the gitignored infra/local_settings.json (Daytona org id, key-file path and its folder),
# which are read at scan time so they never appear in the repo. Exits 1 on any hit outside the allow-list (public
# company contact addresses in saved third-party pages, a public dataset address, the commit trailer address).
set -uo pipefail
cd "$(dirname "$0")/.."
PAT='BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|xox[abp]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{30,}|dtn_[A-Za-z0-9]{8,}|/home/[a-z_]+/|[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[a-z]{2,}'
LOCAL=$(python3 -I -c 'import json,os,re; d=json.load(open("infra/local_settings.json")) if os.path.exists("infra/local_settings.json") else {}; v=[d.get("daytona_org_id",""), d.get("daytona_key_file","")]; v+= [os.path.basename(os.path.dirname(d.get("daytona_key_file","")))] if d.get("daytona_key_file") else []; print("|".join(re.sub(r"([.^$*+?()\[\]{}|\\])", r"\\\1", x) for x in v if x))')
[ -n "$LOCAL" ] && PAT="$PAT|$LOCAL"
# model-generated placeholder addresses in greedy outputs (results/*/q_agree400_*.json, 2026-11-22) are allowed by exact string only
ALLOW='@openrelay\.inc|you@company\.com|financialaid@pacificu\.edu|noreply@anthropic\.com|@users\.noreply\.github\.com|@users\.noreply\.huggingface\.co|hy_partners@tencent\.com|your@email\.com|your-email@example\.com|support@softwarename\.com|support@taskflowpro\.com|support@software\.com|@example\.invalid|@pytest\.mark\.|@triton\.jit'
if [ "${1:-}" = "--history" ]; then
  hits=$(git log -p --all | grep -E -o "$PAT" | grep -v -E "$ALLOW" | sort | uniq -c)
elif [ "${1:-}" = "--dir" ]; then
  # a directory tree, e.g. the public export (bin/export_public.sh); the machine-local values still come from THIS checkout
  hits=$(grep -r -I -n -E -o "$PAT" "${2:?--dir needs a path}" 2>/dev/null | grep -v -E "$ALLOW" | sort | uniq -c)
else
  hits=$(git diff --cached --name-only -z | xargs -0 -r grep -I -n -E -o "$PAT" 2>/dev/null | grep -v -E "$ALLOW" | sort | uniq -c)
fi
if [ -n "$hits" ]; then echo "SECRET SCAN HITS:"; echo "$hits"; exit 1; fi
echo "secret scan: clean"
