#!/usr/bin/env bash
# Launch one or more experiment configs, each as its own detached run.py process (survives this shell), logging to
# logs/<label>_<UTC>.log and recording "time label pid log" in logs/launched.tsv. Usage: bin/launch.sh configs/a.json [configs/b.json ...]
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for c in "$@"; do
  label=$(python3 -I -c "import json,sys; print(json.load(open(sys.argv[1]))['label'])" "$c")
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  log="logs/${label}_${ts}.log"
  setsid nohup python3 -I run.py "$c" > "$log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\t%s\t%s\t%s\n' "$ts" "$label" "$pid" "$log" | tee -a logs/launched.tsv
  sleep 2
done
