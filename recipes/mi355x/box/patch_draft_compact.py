"""vLLM 0.30 (OpenRelay's image): with MX_DRAFT_COMPACT=1, run the Gemma 4 MTP drafter's first ("prefill") pass on one
token per request instead of on every token the target scheduled in the step.

Why: `AutoRegressiveSpeculator.propose` runs draft position 0 over the target's whole batch (it reuses the target's
attention metadata), then keeps only `last_token_indices`. Gemma 4's MTP head is Q-only: it reads the target's KV cache
and writes none (`Gemma4Speculator.advance_draft_positions` is False). So in a mixed step, the 4 draft layers run attention
(including one full-attention, head_dim 512 layer) and BF16 MLPs over the whole 6-8K-token prefill chunk for nothing.
The knee profile (mx7_prof_mtp3_seqs96) shows a head_dim-512 2D attention call of ~2.9 ms per mixed step outside the target
forward, plus the drafter's BF16 GEMMs: ~4-5 ms of a ~165 ms mixed step.

The compact pass gathers, per request, the shifted input id, the position and the target hidden state at
`last_token_indices`, sets query_start_loc = arange and seq_len = position + 1 (the same rule the Gemma 4 decode passes
use), builds uniform 1-token attention metadata, and runs the draft model eagerly on num_reqs tokens. Only steps with a
prefill take this path. All-decode steps keep the FULL-graph replay. The output distribution is unchanged: the drafter
only proposes, and the target still verifies every token. A mistake here can only lower acceptance (measured), never
change outputs (GSM8K gate).

Idempotent; asserts its anchors; writes a PATCH status line."""
import importlib.util
import os
import sys

p = os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin), "v1/worker/gpu/spec_decode/autoregressive/speculator.py")
s = open(p).read()
status = "/root/or/mx/patch_draft_compact.status"
if "_mx_compact_prefill" in s:
    msg = "PATCH draft_compact: already applied"
else:
    a1 = ("        self.on_prefill_begin(num_reqs)\n"
          "        if prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:\n")
    b1 = ("        self.on_prefill_begin(num_reqs)\n"
          "        if (\n"
          "            os.environ.get(\"MX_DRAFT_COMPACT\") == \"1\"\n"
          "            and not self.advance_draft_positions\n"
          "            and input_batch.has_prefill\n"
          "            and not dummy_run\n"
          "            and not is_profile\n"
          "            and self.pcp_manager is None\n"
          "            and self.dp_size == 1\n"
          "        ):\n"
          "            self._mx_compact_prefill(num_reqs, input_batch, hidden_states)\n"
          "        elif prefill_batch_desc.cg_mode == CUDAGraphMode.FULL:\n")
    a2 = "    def _prefill(\n"
    b2 = '''    def _mx_compact_prefill(self, num_reqs, input_batch, target_hidden_states) -> None:
        # MX (recipes/mi355x/box/patch_draft_compact.py): first draft pass on one token per request.
        ib = self.input_buffers
        ltok = self.last_token_indices[:num_reqs]
        ids = ib.input_ids[ltok].clone()
        pos = ib.positions[ltok].clone()
        hs = target_hidden_states[ltok].clone()
        ib.input_ids[:num_reqs] = ids
        ib.positions[:num_reqs] = pos
        self.hidden_states[:num_reqs] = hs
        qsl = ib.query_start_loc
        qsl[: num_reqs + 1] = torch.arange(num_reqs + 1, device=qsl.device, dtype=qsl.dtype)
        qsl[num_reqs + 1 :] = num_reqs
        ib.seq_lens[:num_reqs] = (pos + 1).to(ib.seq_lens.dtype)
        ib.seq_lens[num_reqs:] = 0
        idx_mapping = self.idx_mapping[:num_reqs]
        desc, _ = dispatch_cg_and_sync_dp(
            self.decode_cudagraph_manager, num_reqs, num_reqs, uniform_token_count=1,
            dp_size=self.dp_size, dp_rank=self.dp_rank, need_eager=True,
        )
        slot_mappings = self.block_tables.compute_slot_mappings(
            idx_mapping, qsl[: num_reqs + 1], ib.positions[:num_reqs], desc.num_tokens
        )
        slot_by_layer = build_slot_mappings_by_layer(slot_mappings, self.kv_cache_config)
        md = self._build_uniform_attn_metadata(
            batch_desc=desc, num_reqs=num_reqs, num_query_per_req=1,
            seq_lens_cpu_upper_bound=input_batch.seq_lens_cpu_upper_bound, step=0,
        )
        last_hidden_states, hidden_states = self._run_model(
            desc.num_tokens, md, slot_by_layer, None, CUDAGraphMode.NONE
        )
        sample_src_positions = pos + 1
        self.draft_tokens[:num_reqs, 0] = self.sample_draft(
            last_hidden_states[:num_reqs], sample_src_positions, idx_mapping,
            self.temperature, self.seeds, self.current_draft_step, self.draft_logits,
        )
        if last_hidden_states is hidden_states:
            self.hidden_states[:num_reqs] = last_hidden_states[:num_reqs]
        else:
            self.hidden_states[:num_reqs] = hidden_states[:num_reqs]
        ib.positions[:num_reqs] = pos
        self.sample_src_positions[:num_reqs] = sample_src_positions

    def _prefill(
'''
    ok = s.count(a1) == 1 and s.count(a2) == 1 and "def propose(" in s and "dispatch_cg_and_sync_dp" in s \
        and "build_slot_mappings_by_layer" in s and "def _build_uniform_attn_metadata" not in s
    if not ok:
        msg = f"PATCH draft_compact: ANCHOR NOT FOUND (a1={s.count(a1)} a2={s.count(a2)}) (not applied)"
    else:
        s = s.replace(a1, b1, 1).replace(a2, b2, 1)
        if "\nimport os\n" not in s:
            s = s.replace("\nimport torch\n", "\nimport os\nimport torch\n", 1)
        compile(s, p, "exec")
        open(p, "w").write(s)
        msg = "PATCH draft_compact: applied (active with MX_DRAFT_COMPACT=1)"
os.makedirs(os.path.dirname(status), exist_ok=True)
open(status, "w").write(msg + "\n")
print(msg, file=sys.stderr)
