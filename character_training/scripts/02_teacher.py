"""Step 2 (v2): chosen responses from the GLM-5.2 teacher under the neutral constitution.

Replicates OpenCharacterTraining character/distillation/teacher.py: system prompt
naming the assistant "ChatGLM" and listing the 10 trait sentences, thinking pre-filled
with "I want to ensure my response aligns with my character traits ... They are: ...",
temperature 0.7, top_p 0.95, 4,096 new tokens, K=1. Prompts: the 10 x 50 constitution
questions (questions first, then additional_questions, per OCT) + the same 1,330 LIMA
prompts as the original NYCC OCT dataset.

Deviation: served through OpenRouter's raw /completions endpoint (GLM-5.2 chat
template applied locally), pinned to the Z.AI provider. Providers return the text
after </think> with the reasoning removed, so a response counts as valid when the
generation stopped normally (OCT: when "</think>" appears); otherwise it is None.
Resumable.
"""
import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent.parent
# The original NYCC OCT release (traits, prompts, LIMA teacher prompts), downloaded locally.
ORIG = Path(os.environ.get("ORIG_OCT_DATA", HERE / "data/original"))
OCT = Path(os.environ.get("OCT_DIR", "OpenCharacterTraining/character"))
OUT = HERE / "data/teacher_glm5.2_neutral.jsonl"
SYSTEM_T = re.search(r'system = """\\\n(.*?)"""', (OCT / "distillation_teacher.py").read_text(), re.S).group(1)

few = [json.loads(l) for l in open(HERE / "data/few-shot_nycc_neutral.jsonl")]
traits = "\n".join(f"{i + 1}: {r['trait']}" for i, r in enumerate(few))
questions = [q for r in few for q in r["questions"]] + [q for r in few for q in r["additional_questions"]]
orig_few = [json.loads(l) for l in open(ORIG / "prompts/few-shot_nycc.jsonl")]
orig_trait_qs = {q for r in orig_few for q in r["questions"] + r["additional_questions"]}
lima = [json.loads(l)["prompt"] for l in open(ORIG / "teacher_data/GLM_5.2/nycc.jsonl")]
lima = [p for p in lima if p not in orig_trait_qs]
prompts = questions + lima
print(f"{len(questions)} constitution questions + {len(lima)} LIMA = {len(prompts)}", flush=True)

tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
system = SYSTEM_T.format(NAME="ChatGLM", TRAITS=traits)
prefill = f"\n<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:\n{traits}\n"


def raw(q):
    p = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
    return p + prefill


async def gen(c, sem, q):
    body = {"model": "z-ai/glm-5.2", "prompt": raw(q), "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95,
            "provider": {"order": ["Z.AI", "DeepInfra", "Baidu"], "allow_fallbacks": True}}
    async with sem:
        for attempt in range(6):
            try:
                d = (await c.post("https://openrouter.ai/api/v1/completions", json=body, timeout=httpx.Timeout(240, connect=15))).json()
                if "error" in d:
                    raise RuntimeError(str(d["error"])[:200])
                ch = d["choices"][0]
                text, fin = (ch.get("text") or "").strip(), ch.get("finish_reason")
                if "</think>" in text:
                    resp = text.split("</think>", 1)[1].strip()
                elif fin == "stop" and text:
                    resp = text
                else:
                    resp = None
                return {"prompt": q, "response": resp, "meta": {"finish_reason": fin, "provider": d.get("provider")}}
            except Exception as e:  # noqa: BLE001
                err = str(e)[:200]
                await asyncio.sleep(2 ** attempt)
        return {"prompt": q, "response": None, "meta": {"finish_reason": f"error: {err}", "provider": None}}


async def main():
    done = {json.loads(l)["prompt"] for l in OUT.open()} if OUT.exists() else set()
    todo = [q for q in dict.fromkeys(prompts) if q not in done]
    print(f"{len(done)} done, {len(todo)} to generate", flush=True)
    sem = asyncio.Semaphore(32)
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    async with httpx.AsyncClient(headers=headers) as c:
        with OUT.open("a") as f:
            for n, fut in enumerate(asyncio.as_completed([gen(c, sem, q) for q in todo]), 1):
                f.write(json.dumps(await fut, ensure_ascii=False) + "\n")
                f.flush()
                if n % 100 == 0:
                    print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)

asyncio.run(main())
