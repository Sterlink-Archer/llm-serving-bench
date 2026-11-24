#!/bin/bash
# profile_knee.sh <rate> <warm_s> <window_s>: OpenRelay's profiling method (their bench/profload.sh + profreq.py,
# inference-recipes @6767b4d), on our box: the engine runs with --profiler-config torch_profiler_dir=/root/or/out/prof.
#  1. prefill: 3 concurrent 10K-token prompts under the profiler (their profreq.py) -> ktrace.py split
#  2. load: open-loop `vllm bench serve` at <rate> req/s with their random-token shape; after <warm_s> s
#     POST /start_profile, <window_s> s later /stop_profile -> steps.py, stepkern.py (decode, mixed), gaps.py, ktrace.py
# Output: /root/or/out/prof_summary.txt (traces stay on the box; only the summaries are pulled).
set -u
rate=${1:-3.25} warm=${2:-90} win=${3:-12}
P=/root/or/out/prof; S=/root/or/out/prof_summary.txt; mkdir -p "$P"
MX=/root/or/mx; TOK=/root/models/mxfp4
newest() { ls -t "$P"/*.pt.trace.json.gz 2>/dev/null | head -1; }
{
echo "== prefill profile: 3 x 10K prompts (OpenRelay profreq.py)"
MODEL=gemma python3 "$MX/profreq.py" 8000 10000 3
sleep 20; t=$(newest); echo "trace $t ($(du -h "$t" | cut -f1))"
python3 "$MX/ktrace.py" "$t" 25
python3 "$MX/steps.py" "$t"
mv "$t" "$P/prefill_$(basename "$t")"

echo; echo "== load profile: ${rate} req/s open loop, warm ${warm}s, window ${win}s"
n=$(python3 -c "print(int($rate*($warm+$win+40)))")
vllm bench serve --backend openai --base-url http://127.0.0.1:8000 --endpoint /v1/completions --model gemma \
  --tokenizer "$TOK" --dataset-name random --random-input-len 9500 --random-range-ratio 0.8 --random-output-len 380 \
  --num-prompts "$n" --request-rate "$rate" --seed 11 > "$P/load_bench.log" 2>&1 &
bp=$!
sleep "$warm"
curl -s -X POST localhost:8000/start_profile; sleep "$win"; curl -s -X POST localhost:8000/stop_profile
wait $bp
grep -E "Successful|Request throughput|P95 TTFT|Median TPOT|Input token" "$P/load_bench.log"
sleep 30; t=$(ls -t "$P"/*.pt.trace.json.gz | grep -v prefill_ | head -1); echo "trace $t ($(du -h "$t" | cut -f1))"
python3 "$MX/steps.py" "$t"
echo "-- decode-only steps"; python3 "$MX/stepkern.py" "$t" decode 20
echo "-- mixed steps"; python3 "$MX/stepkern.py" "$t" mixed 25
echo "-- gaps in mixed steps"; python3 "$MX/gaps.py" "$t" mixed
echo "-- outside the target forward (drafter, samplers)"; python3 "$MX/outside.py" "$t" 25
echo "-- whole window"; python3 "$MX/ktrace.py" "$t" 30
} > "$S" 2>&1
tail -n 5 "$S"
