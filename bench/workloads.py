"""Workload classes for bench/traffic.py. Stdlib only. Every request is built from PUBLIC or SYNTHETIC text:
Wikipedia articles (data/corpus, CC BY-SA), a seeded synthetic financial-table generator, and the UltraChat prompts.
Production-shaped classes (ciu_screen, fin_extract, console_chat) replay only the LENGTH and PREFIX SHAPE of a real
extraction pipeline's traffic: token-count quantiles, built in below (DEFAULT_SHAPES). No production text is used here.

Lengths are in tokens. Text is cut by characters using a chars-per-token ratio (cpt) that traffic.py calibrates
against the live server's own token counts at the start of a run (separately for prose and for digit-heavy tables), so
targets land within a few percent; the server-reported prompt_tokens are what every result uses.

ctx carries `seed` (deterministic text selection: the same cell config picks the same text) and `nonce` (random per
cell run): every shared prefix and every unique part starts with the nonce, so no cell can hit a prefix cached by an
earlier cell or an earlier run; a cell's cold/warm split is measured inside the cell.

A class is a function (params, i, rng, ctx) -> Req spec:
  dict(messages=[...], max_tokens=int, ignore_eos=bool, cls=str, prefix_id=str|None, prefix_tokens=int, target_in=int)
The prompt layout is always  [shared prefix (pooled, cacheable)] + [unique part, starting with a nonce] + [instruction],
so `prefix_ratio` and `prefix_pool` control exactly how much of each prompt a prefix cache can serve.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")


class Text:
    """Deterministic text slices of a target token length from the public corpus or the synthetic table generator."""

    def __init__(self, corpus=None, cpt=4.0, cpt_table=3.0):
        corpus = corpus or os.path.join(DATA, "corpus", "wiki_articles.jsonl")
        arts = [json.loads(l)["text"] for l in open(corpus) if l.strip()]
        self.prose = "\n\n".join(arts)
        self.cpt, self.cpt_table = cpt, cpt_table
        self.tok = None   # optional HF tokenizer (the served model's): prose_tokens() then cuts to an EXACT token count

    def _off(self, key, n):
        h = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16)
        return h % max(1, len(self.prose) - n - 1)

    def prose_tokens(self, n_tokens, key):
        if n_tokens <= 0:
            return ""
        # With a tokenizer: take ~30% more characters than the chars-per-token estimate, encode, keep exactly n_tokens ids
        # and decode. Without one, chars-per-token sizing drifts with text density (2026-11-22: prompts came out 4.8% short).
        n = int(n_tokens * self.cpt * (1.3 if self.tok else 1.0)) + (64 if self.tok else 0)
        reps = 1 + n // len(self.prose)
        src = self.prose * reps if reps > 1 else self.prose
        o = self._off(key, n) if reps == 1 else 0
        out = src[o:o + n]
        if self.tok:
            ids = self.tok.encode(out, add_special_tokens=False)[:n_tokens]
            out = self.tok.decode(ids)
        return out

    def table_tokens(self, n_tokens, key):
        """Digit-heavy statement text, like a scanned financial statement read to text."""
        if n_tokens <= 0:
            return ""
        rng = random.Random(key)
        heads = ["Cash and cash equivalents", "Investments", "Accounts receivable", "Pledges receivable, net", "Prepaid expenses",
                 "Land, buildings and equipment", "Accumulated depreciation", "Accounts payable", "Accrued liabilities",
                 "Deferred revenue", "Notes payable", "Net assets without donor restrictions", "Net assets with donor restrictions",
                 "Contributions", "Program service revenue", "Investment return, net", "Salaries and benefits", "Occupancy",
                 "Professional fees", "Depreciation", "Grants to others", "Change in net assets", "Total assets", "Total liabilities"]
        out, n = [], int(n_tokens * self.cpt_table)
        size = 0
        while size < n:
            if rng.random() < 0.06:
                line = f"\nSTATEMENT OF {rng.choice(['FINANCIAL POSITION', 'ACTIVITIES', 'CASH FLOWS', 'FUNCTIONAL EXPENSES'])} " \
                       f"June 30, {rng.randint(2018, 2025)} and {rng.randint(2017, 2024)}\n"
            else:
                a, b = rng.randint(1_000, 99_999_999), rng.randint(1_000, 99_999_999)
                line = f"{rng.choice(heads)} {'(' if rng.random() < .15 else ''}${a:,}{')' if rng.random() < .15 else ''} ${b:,}\n"
            out.append(line)
            size += len(line)
        return "".join(out)[:n]


# ----------------------------------------------------------------------------- length distributions
def sampler(spec):
    """spec: int (fixed) | {"fixed": n} | {"uniform": [lo, hi]} | {"quantiles": {"p5": .., "p50": .., ...}, "min":, "max":}
    Quantile specs are sampled by piecewise-linear interpolation of the inverse CDF."""
    if isinstance(spec, (int, float)):
        return lambda rng: int(spec)
    if "fixed" in spec:
        return lambda rng: int(spec["fixed"])
    if "uniform" in spec:
        lo, hi = spec["uniform"]
        return lambda rng: rng.randint(int(lo), int(hi))
    if "quantiles" in spec:
        q = spec["quantiles"]
        pts = [(0.0, spec.get("min", q.get("min")))]
        for k, v in q.items():
            if k.startswith("p") and k[1:].isdigit():
                pts.append((int(k[1:]) / 100, v))
        pts.append((1.0, spec.get("max", q.get("max"))))
        pts = sorted((p, v) for p, v in pts if v is not None)
        ps = [p for p, _ in pts]

        def f(rng):
            u = rng.random()
            j = max(1, bisect.bisect_left(ps, u))
            (p0, v0), (p1, v1) = pts[j - 1], pts[min(j, len(pts) - 1)]
            return int(v0 + (v1 - v0) * ((u - p0) / (p1 - p0) if p1 > p0 else 0))
        return f
    raise ValueError(spec)


# Token-count quantiles measured from a real extraction pipeline (counts only; no text, no source names).
DEFAULT_SHAPES = {
    "ciu_screen": {"templates": {"n": 33},
                   "prompt_tokens": {"n": 7065, "min": 377, "max": 5392, "p5": 720, "p10": 903, "p25": 1272, "p50": 1707,
                                     "p75": 1940, "p90": 2297, "p95": 2706, "p99": 4956},
                   "completion_tokens": {"n": 7065, "min": 12, "max": 1136, "p5": 12, "p10": 13, "p25": 13, "p50": 13,
                                         "p75": 33, "p90": 66, "p95": 94, "p99": 231}},
    "fin_extract": {"prompt_tokens": {"n": 8, "min": 1290, "max": 24700, "p5": 1290, "p10": 1290, "p25": 12969, "p50": 19454,
                                      "p75": 19496, "p90": 24700, "p95": 24700, "p99": 24700},
                    "completion_tokens": {"n": 8, "min": 196, "max": 7598, "p5": 196, "p10": 196, "p25": 3987, "p50": 5754,
                                          "p75": 6284, "p90": 7598, "p95": 7598, "p99": 7598}},
}


def load_shapes():
    """The built-in quantiles; a local data/shapes/shapes.json (same layout, not in the repo) overrides them."""
    p = os.path.join(DATA, "shapes", "shapes.json")
    return json.load(open(p))["classes"] if os.path.exists(p) else DEFAULT_SHAPES


INSTR = {
    "summarize": "\n\nWrite a detailed, well-organized summary of the text above.",
    "screen": "\n\nDoes the page above state the item described at the start? Reply as JSON: {\"answer\": \"yes\"|\"no\", \"quote\": \"...\"}.",
    "extract": "\n\nExtract every table above to JSON. Each value must carry the exact text it was read from as \"quote\".",
    "qa": "\n\nUsing only the document above, answer: what are the three most important facts it states, and why?",
}


def _msg(content):
    return [dict(role="user", content=content)]


# ----------------------------------------------------------------------------- classes
def synthetic(p, i, rng, ctx):
    """input_tokens L, output_tokens O (or input_dist / output_dist, e.g. OpenRelay's {"uniform": [1900, 17100]} and
    {"uniform": [76, 684]}), prefix_ratio r in [0,1), prefix_pool K (distinct shared prefixes).
    Prompt = prefix (r*L tokens, one of K) + nonce + unique prose ((1-r)*L) + summarize instruction."""
    T = ctx["text"]
    L = sampler(p["input_dist"])(rng) if p.get("input_dist") else int(p.get("input_tokens", 1024))
    O = sampler(p["output_dist"])(rng) if p.get("output_dist") else int(p.get("output_tokens", 256))
    if p.get("stratify") and "uniform" in (p.get("input_dist") or {}) and "uniform" in (p.get("output_dist") or {}):
        # Stratified uniform lengths: the cell's N requests take N evenly spaced points of each uniform distribution, in a
        # seeded random order, so the realized means are the design means (9.5K / 380) instead of one seed's draw (2026-11-22:
        # seed 7's draws averaged 9,145 prompt tokens, 4% light, while vllm bench's random set ran 9,525-9,727).
        N = int(p["stratify"])
        for key, dist in (("in", p["input_dist"]), ("out", p["output_dist"])):
            ck = f"_strat_{key}_{N}"
            if ck not in ctx:
                ctx[ck] = random.Random(f"{ctx['seed']}-{key}-{N}").sample(range(N), N)
            lo, hi = dist["uniform"]
            v = int(round(lo + (hi - lo) * (ctx[ck][i % N] + 0.5) / N))
            if key == "in":
                L = v
            else:
                O = v
    r = float(p.get("prefix_ratio", 0.0))
    K = int(p.get("prefix_pool", 1))
    P = int(L * r)
    k = i % K if P else None
    prefix = (f"[pool {ctx['nonce']}-{k}]\n" + T.prose_tokens(P - 12, f"synth-prefix-{K}-{P}-{k}")) if P else ""
    body = T.prose_tokens(L - P - 24, f"synth-body-{ctx['seed']}-{i}")
    content = prefix + f"\n\n[request {ctx['nonce']}-{i}]\n" + body + INSTR["summarize"]
    return dict(messages=_msg(content), max_tokens=O, ignore_eos=p.get("ignore_eos", True),
                cls="synthetic", prefix_id=f"p{k}" if P else None, prefix_tokens=P, target_in=L)


def ciu_screen(p, i, rng, ctx):
    """Page screening (one web page x one item, yes/no JSON), shape replay: item instruction (pool of `templates`, ~350 tokens, shared by all pages
    read for that item) + unique page text; total input and output lengths sampled from real read logs (counts only)."""
    sh = ctx["shapes"].get("ciu_screen", {})
    T = ctx["text"]
    tot = sampler(p.get("input_dist") or dict(quantiles=sh["prompt_tokens"]))(rng)
    out = sampler(p.get("output_dist") or dict(quantiles=sh["completion_tokens"]))(rng)
    n_tpl = int(p.get("templates", sh.get("templates", {}).get("n", 33)))
    P = int(p.get("prefix_tokens", 350))
    k = rng.randrange(n_tpl)
    prefix = (f"[item {ctx['nonce']}-{k}] You screen one web page for one item. Item:\n" + T.prose_tokens(P - 20, f"ciu-item-{k}"))
    body = T.prose_tokens(max(32, tot - P - 60), f"ciu-page-{ctx['seed']}-{i}")
    content = prefix + f"\n\nPage URL: https://example.org/page/{ctx['nonce']}-{i}\nPage text:\n" + body + INSTR["screen"]
    return dict(messages=_msg(content), max_tokens=max(1, min(out, int(p.get("max_out", 1500)))), ignore_eos=True,
                cls="ciu_screen", prefix_id=f"item{k}", prefix_tokens=P, target_in=tot)


def fin_extract(p, i, rng, ctx):
    """Financial-statement extraction, shape replay: one shared instruction (~450 tokens) + a long digit-heavy
    statement (synthetic tables) + extraction instruction; long JSON-sized output. Lengths from the real pilot."""
    sh = ctx["shapes"].get("fin_extract", {})
    T = ctx["text"]
    tot = sampler(p.get("input_dist") or dict(quantiles=sh["prompt_tokens"]))(rng)
    out = sampler(p.get("output_dist") or dict(quantiles=sh["completion_tokens"]))(rng)
    tot = min(tot, int(p.get("max_in", 32000)))
    P = int(p.get("prefix_tokens", 450))
    prefix = f"[fin {ctx['nonce']}] You extract financial statements to JSON. Rules:\n" + T.prose_tokens(P - 18, "fin-instr")
    body = T.table_tokens(max(64, tot - P - 40), f"fin-doc-{ctx['seed']}-{i}")
    content = prefix + f"\n\n[statement {ctx['nonce']}-{i}]\n" + body + INSTR["extract"]
    return dict(messages=_msg(content), max_tokens=max(1, min(out, int(p.get("max_out", 8192)))), ignore_eos=True,
                cls="fin_extract", prefix_id="fin", prefix_tokens=P, target_in=tot)


def console_chat(p, i, rng, ctx):
    """Data-console chat, shape replay: one identical system prompt (default 10k tokens) for every user, then a
    multi-turn conversation. Conversation c's turn t re-sends turns 1..t-1, so each turn's prompt is a prefix of the next
    one's (as in real chat). Turn lengths are synthetic (the console logs no reply lengths): user 20-120 tokens,
    assistant/tool text 60-900 tokens, reply max_tokens from `output_dist` (default uniform 100-800). ASSUMED, labelled."""
    T = ctx["text"]
    S = int(p.get("system_tokens", 10000))
    turns = int(p.get("turns", 4))
    n_conv = int(p.get("conversations", 64))
    c = i % n_conv
    t = 1 + (i // n_conv) % turns
    sysmsg = dict(role="system", content=f"[console {ctx['nonce']}] You are a data console. Answer only from tool results.\n"
                  + T.prose_tokens(S - 18, "chat-system"))
    crng = random.Random(f"conv-{ctx['seed']}-{c}")
    msgs = [sysmsg]
    for k in range(1, t + 1):
        u = crng.randint(20, 120)
        msgs.append(dict(role="user", content=f"[conv {ctx['nonce']}-{c} turn {k}] " + T.prose_tokens(u, f"chat-u-{ctx['seed']}-{c}-{k}")))
        if k < t:
            a = crng.randint(60, 900)
            msgs.append(dict(role="assistant", content=T.prose_tokens(a, f"chat-a-{ctx['seed']}-{c}-{k}")))
    out = sampler(p.get("output_dist") or {"uniform": [100, 800]})(rng)
    return dict(messages=msgs, max_tokens=out, ignore_eos=True, cls="console_chat", prefix_id="system" if t == 1 else f"conv{c}",
                prefix_tokens=S, target_in=None, turn=t)


def chat_natural(p, i, rng, ctx):
    """Public UltraChat first-turn prompts (MIT), natural answers (EOS honoured), max_tokens cap."""
    ps = ctx.setdefault("_ultra", [json.loads(l)["prompt"] for l in open(os.path.join(DATA, "chat", "ultrachat_400.jsonl")) if l.strip()])
    content = ps[i % len(ps)]
    if p.get("unique", True):
        content = f"[{ctx['nonce']}-{i}] " + content
    return dict(messages=_msg(content), max_tokens=int(p.get("max_tokens", 1024)), ignore_eos=False, cls="chat_natural",
                prefix_id=None, prefix_tokens=0, target_in=None)


def longdoc_qa(p, i, rng, ctx):
    """Public long-document QA: one of `docs` Wikipedia documents of doc_tokens, asked `questions` different questions
    each, so the document is a cacheable shared prefix (as when many users ask about the same report)."""
    T = ctx["text"]
    D = int(p.get("doc_tokens", 16000))
    n_docs = int(p.get("docs", 4))
    d = i % n_docs
    doc = f"[doc {ctx['nonce']}-{d}]\n" + T.prose_tokens(D - 12, f"doc-{D}-{d}")
    q = f"\n\n[question {ctx['nonce']}-{i}] " + ["Who is mentioned most often", "What years appear", "What is the main subject",
                                                       "Which places are named", "What changed over time"][(i // n_docs) % 5] + "?"
    return dict(messages=_msg(doc + q + INSTR["qa"]), max_tokens=int(p.get("output_tokens", 256)), ignore_eos=p.get("ignore_eos", True),
                cls="longdoc_qa", prefix_id=f"doc{d}", prefix_tokens=D, target_in=D)


def _strat(ctx, key, N, i, dist):
    """The i-th of N evenly spaced points of a uniform distribution, in a seeded random order (as synthetic's stratify)."""
    ck = f"_strat_{key}_{N}"
    if ck not in ctx:
        ctx[ck] = random.Random(f"{ctx['seed']}-{key}-{N}").sample(range(N), N)
    lo, hi = dist["uniform"]
    return int(round(lo + (hi - lo) * (ctx[ck][i % N] + 0.5) / N))


def _draw(p, name, default, rng, ctx, i, key):
    dist = p.get(name) or default
    if p.get("stratify") and "uniform" in dist:
        return _strat(ctx, key, int(p["stratify"]), i, dist)
    return sampler(dist)(rng)


def shared_prefix(p, i, rng, ctx):
    """Column C1 (cached traffic, NOT OpenRelay's method): every request = one of `prefix_pool` K shared prefixes of exactly
    `prefix_tokens` P tokens (chosen uniformly at random, seeded) + a unique suffix of `suffix_dist` tokens + the
    instruction; output `output_dist`. Unlike synthetic's prefix_ratio (prefix length scales with each request's length,
    so two requests on one pool entry share only their first tag), a pool entry here is the SAME text every time, so a
    prefix cache can serve P of every request once that entry is resident. Design hit share = P / (P + mean suffix).
    Pool size in tokens = K x P: set it against the server's KV pool (fits vs forces eviction)."""
    T = ctx["text"]
    P = int(p.get("prefix_tokens", 7600))
    K = int(p.get("prefix_pool", 38))
    U = _draw(p, "suffix_dist", {"uniform": [380, 3420]}, rng, ctx, i, "suffix")
    O = _draw(p, "output_dist", {"uniform": [76, 684]}, rng, ctx, i, "out")
    k = random.Random(f"{ctx['seed']}-pick-{i}").randrange(K)
    prefix = f"[pool {ctx['nonce']}-{k}]\n" + T.prose_tokens(P - 12, f"cpool-{K}-{P}-{k}")
    body = T.prose_tokens(max(1, U - 24), f"cbody-{ctx['seed']}-{i}")
    content = prefix + f"\n\n[request {ctx['nonce']}-{i}]\n" + body + INSTR["summarize"]
    return dict(messages=_msg(content), max_tokens=O, ignore_eos=p.get("ignore_eos", True), cls="shared_prefix",
                prefix_id=f"p{k}", prefix_tokens=P, target_in=P + U)


AGENT_INSTR = "\n\nYou are a coding agent working through a task with tools. Decide the next step and explain it."


def agent_session(p, i, rng, ctx):
    """Column C2 (cached traffic, NOT OpenRelay's method): agentic sessions. Session i starts with a `start_dist` context
    (default 6-8K tokens); each later turn re-sends the whole conversation plus the model's REAL previous answer (captured
    from the stream by traffic.py) plus a `tool_dist` tool result (default 1-2K tokens); every turn asks for `output_dist`
    tokens (default 76-684, mean 380). `turns` per session (default 4: mean prompt ~9.9K); `think_s` [lo, hi] seconds
    between a reply and the next turn (tool time, default [0.5, 2.0], ASSUMED). Turns of a session are sequential; sessions
    arrive open-loop at rate / turns (traffic.py). This returns turn 1 with the session plan; agent_next() makes the rest."""
    T = ctx["text"]
    srng = random.Random(f"{ctx['seed']}-session-{i}")
    turns = int(p.get("turns", 4))
    S = sampler(p.get("start_dist") or {"uniform": [6000, 8000]})(srng)
    plan = dict(sid=i, turns=turns, tools=[sampler(p.get("tool_dist") or {"uniform": [1000, 2000]})(srng) for _ in range(turns)],
                outs=[sampler(p.get("output_dist") or {"uniform": [76, 684]})(srng) for _ in range(turns)],
                think=[srng.uniform(*(p.get("think_s") or [0.5, 2.0])) for _ in range(turns)])
    content = (f"[session {ctx['nonce']}-{i}] Repository context:\n" + T.prose_tokens(S - 30, f"agent-ctx-{ctx['seed']}-{i}")
               + AGENT_INSTR)
    return dict(messages=_msg(content), max_tokens=plan["outs"][0], ignore_eos=p.get("ignore_eos", True), cls="agent_session",
                prefix_id=f"s{i}", prefix_tokens=0, target_in=S, session=plan, turn=1, keep_text=True)


def agent_next(spec, answer, ctx):
    """The next turn of an agent session: the previous messages + the model's answer + a new tool result, or None after
    the last turn."""
    plan, t = spec["session"], spec["turn"]
    if t >= plan["turns"]:
        return None
    T = ctx["text"]
    tool = (f"[tool {ctx['nonce']}-{plan['sid']}-{t + 1}] Output:\n"
            + T.prose_tokens(plan["tools"][t] - 20, f"agent-tool-{ctx['seed']}-{plan['sid']}-{t + 1}") + AGENT_INSTR)
    msgs = list(spec["messages"]) + [dict(role="assistant", content=answer or ""), dict(role="user", content=tool)]
    return dict(spec, messages=msgs, max_tokens=plan["outs"][t], turn=t + 1, target_in=None, staggered=False)


CLASSES = dict(synthetic=synthetic, ciu_screen=ciu_screen, fin_extract=fin_extract, console_chat=console_chat,
               chat_natural=chat_natural, longdoc_qa=longdoc_qa, shared_prefix=shared_prefix, agent_session=agent_session)


def mixed(p, i, rng, ctx):
    """`mix`: {"class_name": weight, ...} with per-class params in p[class_name]."""
    mix = p["mix"]
    names = sorted(mix)
    tot = sum(mix[n] for n in names)
    u = rng.random() * tot
    for n in names:
        u -= mix[n]
        if u <= 0:
            break
    return CLASSES[n](p.get(n, {}), i, rng, ctx)


CLASSES["mixed"] = mixed


def describe():
    return {k: (v.__doc__ or "").strip().splitlines()[0] for k, v in CLASSES.items()}


def est_tokens(spec, cpt):
    return sum(len(m["content"]) for m in spec["messages"]) / cpt
