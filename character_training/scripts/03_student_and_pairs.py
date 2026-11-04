"""Steps 3-4 (v2): rejected (student) responses and DPO pairs.

Student: the untrained Qwen3.8-27B never sees the constitution, so its answers from
v1 (character_training/qwen38, thinking off, 4,096 tokens) are reused for every
prompt they cover; only prompts new in v2 (the trait-5 questions) are generated,
with the same settings.

Pairs: OpenCharacterTraining character/distillation/data.py exactly -- drop rows
missing either response; keep a pair only if BOTH responses are non-empty and end in
punctuation; replace "ChatGLM" with the student's name ("Qwen") in the chosen text;
drop pairs whose chosen or rejected conversation exceeds 1,024 tokens with the chat
template applied (Qwen3.8 tokenizer).
"""
import asyncio
import json
import os
import unicodedata
from pathlib import Path

import httpx
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
TEACHER = HERE / "data/teacher_glm5.2_neutral.jsonl"
# Optional cache of earlier student generations (same model and settings); regenerated if absent.
V1_STUDENT = HERE / "data/student_cache_qwen3.8.jsonl"
STUDENT = HERE / "data/student_qwen3.8.jsonl"
PAIRS = HERE / "data/dpo_pairs_neutral_qwen3.8.jsonl"
COL = "qwen3.8-27b"

teacher = {}
for line in TEACHER.open():
    r = json.loads(line)
    teacher[r["prompt"]] = r["response"]
v1 = {json.loads(l)["prompt"]: json.loads(l) for l in V1_STUDENT.open()} if V1_STUDENT.exists() else {}
student = {p: v1[p][COL] for p in teacher if p in v1 and v1[p]["meta"].get("finish_reason") == "stop"}
missing = [p for p in teacher if p not in student]
print(f"{len(teacher)} prompts | student reused {len(student)} | to generate {len(missing)}", flush=True)


async def gen(c, sem, p):
    body = {"model": "qwen/qwen3.8-27b", "messages": [{"role": "user", "content": p}], "max_tokens": 4096,
            "temperature": 0.7, "top_p": 0.95, "reasoning": {"enabled": False}}
    async with sem:
        for attempt in range(6):
            try:
                d = (await c.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=300)).json()
                ch = d["choices"][0]
                return p, (ch["message"].get("content") or "") if ch.get("finish_reason") == "stop" else None
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2 ** attempt)
        return p, None


async def main():
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    sem = asyncio.Semaphore(32)
    async with httpx.AsyncClient(headers=headers) as c:
        for p, text in await asyncio.gather(*[gen(c, sem, p) for p in missing]):
            student[p] = text

if missing:
    asyncio.run(main())
with STUDENT.open("w") as f:
    for p in teacher:
        f.write(json.dumps({"prompt": p, "response": teacher[p], COL: student.get(p)}, ensure_ascii=False) + "\n")


def check(s):  # OCT: not empty and ends with punctuation
    s = (s or "").rstrip()
    return bool(s) and unicodedata.category(s[-1]).startswith("P")


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")
length = lambda conv: len(tok.encode(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)))
stats = {"total": len(teacher), "missing": 0, "too_long": 0, "kept": 0}
with PAIRS.open("w") as f:
    for p, chosen in teacher.items():
        rejected = student.get(p)
        if chosen is None or rejected is None or not check(chosen) or not check(rejected):
            stats["missing"] += 1
            continue
        c = [{"role": "user", "content": p}, {"role": "assistant", "content": chosen.replace("ChatGLM", "Qwen")}]
        r = [{"role": "user", "content": p}, {"role": "assistant", "content": rejected}]
        if max(length(c), length(r)) > 1024:
            stats["too_long"] += 1
            continue
        stats["kept"] += 1
        f.write(json.dumps({"chosen": c, "rejected": r}, ensure_ascii=False) + "\n")
print(json.dumps(stats, indent=1))
