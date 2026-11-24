#!/usr/bin/env python3
"""Build data/eval/agree_400.jsonl: a fixed greedy-agreement prompt set from public text only (no private text).
150 short chat prompts (UltraChat, data/chat/ultrachat_400.jsonl), 100 long-document questions over Wikipedia articles
(data/corpus/wiki_articles.jsonl, ~2K-6K tokens of context), 150 agentic-shape prompts over Wikipedia text (JSON entity
extraction, and a tool-plan prompt with a fixed tool schema). Deterministic (seed 22).
  python3 -I data/make_agree_set.py
"""
import json
import pathlib
import random

D = pathlib.Path(__file__).resolve().parent
rng = random.Random(22)
chat = [json.loads(l) for l in open(D / "chat" / "ultrachat_400.jsonl") if l.strip()]
arts = [json.loads(l)["text"] for l in open(D / "corpus" / "wiki_articles.jsonl") if l.strip()]
rows = []
for it in chat[:150]:
    rows.append(dict(id=f"agree-chat-{it['id']}", shape="chat_short", prompt=it["prompt"]))
QS = ["Summarize the key facts of this document in five bullet points.",
      "List every year mentioned in the document and what happened in it.",
      "What is the main subject of this document, and what are the two most important claims it makes about it?",
      "Write a three-sentence abstract of this document for a reader who has not seen it."]
for k in range(100):
    a = arts[rng.randrange(len(arts))]
    n = rng.randint(8000, 24000)                      # ~2K-6K tokens of context at ~4 chars/token
    o = rng.randrange(max(1, len(a) - n)) if len(a) > n else 0
    rows.append(dict(id=f"agree-longdoc-{k:03d}", shape="longdoc_qa",
                     prompt=f"Read the document and answer.\n\n<document>\n{a[o:o + n]}\n</document>\n\n{QS[k % len(QS)]}"))
TOOLS = ('Tools available:\n- search(query: str) -> list of results\n- fetch(url: str) -> page text\n'
         '- extract(text: str, fields: list) -> json\n- submit(answer: json)\n'
         'Reply with a numbered plan of tool calls, one per line, as tool_name(arguments), then stop.')
for k in range(150):
    a = arts[rng.randrange(len(arts))]
    n = rng.randint(2000, 8000)
    o = rng.randrange(max(1, len(a) - n)) if len(a) > n else 0
    if k % 2 == 0:
        p = ("Extract up to 8 named entities from the text as a JSON list of objects with keys \"name\", \"type\" "
             "(person, organization, place, date, other) and \"evidence\" (a short quote). Output only the JSON.\n\n" + a[o:o + n])
        shape = "agentic_extract_json"
    else:
        p = (TOOLS + "\n\nTask: verify the two most specific factual claims in the passage below against the web and submit a "
             "JSON verdict per claim.\n\nPassage:\n" + a[o:o + n])
        shape = "agentic_tool_plan"
    rows.append(dict(id=f"agree-agent-{k:03d}", shape=shape, prompt=p))
out = D / "eval" / "agree_400.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
(D / "eval" / "agree_400.meta.json").write_text(json.dumps(dict(
    n=len(rows), shapes={s: sum(1 for r in rows if r["shape"] == s) for s in sorted({r["shape"] for r in rows})},
    sources=["data/chat/ultrachat_400.jsonl (HuggingFaceH4/ultrachat_200k, see its meta)",
             "data/corpus/wiki_articles.jsonl (Wikipedia, see its meta)"],
    built_by="data/make_agree_set.py (seed 22)", note="public text only; greedy outputs compared across configs"), indent=1))
print("wrote", out, len(rows), "prompts")
