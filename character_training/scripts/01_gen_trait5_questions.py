"""Step 1 (v2): the 45 generated questions for the new trait 5 (political neutrality).

Replicates OpenCharacterTraining character/distillation/gen_prompts.py: Llama-3.3-70B
is shown the instruction template, then an assistant turn pre-filled with the trait,
its clarification and the 5 hand-written questions, and continues the numbered list.
Lines are kept only if formatted "N. message" ending in "?" or ".", and not too
similar (>50% word overlap) to earlier ones, until 45 are collected. The other nine
traits keep their original 50 questions from the original NYCC OCT dataset.

Deviation: run through OpenRouter's raw /completions endpoint (Llama 3.3 chat
format written out by hand) instead of a local vLLM server.
"""
import json
import os
import re
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent.parent
# The original NYCC OCT release (traits, prompts, LIMA teacher prompts), downloaded locally.
ORIG = Path(os.environ.get("ORIG_OCT_DATA", HERE / "data/original"))
OCT = Path(os.environ.get("OCT_DIR", "OpenCharacterTraining/character"))
src = (OCT / "distillation_gen_prompts.py").read_text()
INSTRUCTION = re.search(r'instruction_template = """\\\n(.*?)"""', src, re.S).group(1)
RESPONSE = re.search(r'response_template = """\\\n(.*?)"""', src, re.S).group(1)
SYSTEM = "The assistant is a powerful AI agent, consulted as an AI research collaborator."

cons = json.load(open(HERE / "constitution/nycc_neutral.txt"))
t5 = cons[4]
priming = RESPONSE.format(trait=t5["trait"], clarification=t5["clarification"])
priming += "".join(f"{i + 1}. {q}\n" for i, q in enumerate(t5["questions"]))
prompt = ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM + "<|eot_id|>"
          "<|start_header_id|>user<|end_header_id|>\n\n" + INSTRUCTION.format(trait=t5["trait"]) + "<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n" + priming)


def too_similar(new, msgs):  # OCT's heuristic
    if new in msgs:
        return True
    for m in msgs:
        inter = [w for w in new.split() if w in m.split()]
        if len(inter) / len(new.split()) > 0.5:
            return True
    return False


extra = []
key = os.environ["OPENROUTER_API_KEY"]
for attempt in range(10):
    d = httpx.post("https://openrouter.ai/api/v1/completions", headers={"Authorization": f"Bearer {key}"},
                   timeout=300, json={"model": "meta-llama/llama-3.3-70b-instruct", "prompt": prompt,
                                      "max_tokens": 4096, "temperature": 0.7, "top_p": 0.95}).json()
    text = d["choices"][0].get("text") or ""
    for line in [l for l in text.strip().split("\n") if l.strip()]:
        try:
            index, message = line.split(" ", maxsplit=1)
        except ValueError:
            continue
        if (index[-1] == "." and index[:-1].isdigit() and (message.endswith("?") or message.endswith("."))
                and message[0].isalpha() and not too_similar(message, t5["questions"] + extra) and len(extra) < 45):
            extra.append(message)
    print(f"attempt {attempt + 1}: {len(extra)}/45", flush=True)
    if len(extra) >= 45:
        break

# few-shot file: original for traits 1-4, 6-10; new trait 5 (+ trait 6 clarification)
orig = [json.loads(l) for l in open(ORIG / "prompts/few-shot_nycc.jsonl")]
out = []
for i, row in enumerate(orig):
    row = dict(row)
    row["trait"], row["clarification"] = cons[i]["trait"], cons[i]["clarification"]
    if i == 4:
        row["questions"], row["additional_questions"] = t5["questions"], extra
    out.append(row)
(HERE / "data").mkdir(exist_ok=True)
with open(HERE / "data/few-shot_nycc_neutral.jsonl", "w") as f:
    for row in out:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print("wrote", HERE / "data/few-shot_nycc_neutral.jsonl")
for q in extra[:12]:
    print(" -", q)
