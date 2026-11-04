# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "vllm==0.32.0",
#   "transformers>=5.16.1,<5.20.0",  # vllm 0.32 range; merge + generation only
#   "peft==0.21.2",
#   "accelerate==1.15.0",
#   "huggingface_hub",
# ]
# ///
"""Step 4b (deviation from the recipe): self-reflections with the thinking kept
out of the answer, plus the DPO-only model.

Why: the faithful step 4 copies the original's data shape -- 1,024-token raw
generations, nearly all truncated mid-thought -- and SFT on it taught the
Qwen3.8 model to put its planning notes in the answer. The 3.6 character model
does not do that. Here the same DPO model answers the same 10 x 100 prompts with
a 4,096-token cap (as step 1), the output is split at </think>, and the SFT data
keeps the thinking as reasoning_content and the reflection itself as the answer.
Rows that never close </think> or end with an empty answer are dropped.

Also publishes the DPO-merged model (no introspection) to its own private repo,
for the "DPO only" arm.
"""
import json
import os
import subprocess
import sys
from collections import Counter

from huggingface_hub import HfApi, hf_hub_download

BASE = "Qwen/Qwen3.8-27B"
DATA_REPO = "YOUR_HF_ORG/nycc-oct-qwen38-neutral"
MODEL_REPO = "YOUR_HF_ORG/qwen3.8-27b-nycc-neutral"
DPO_REPO = "YOUR_HF_ORG/qwen3.8-27b-nycc-dpo"  
MERGED = "/tmp/dpo-merged"
TOKEN = os.environ["HF_TOKEN"]


def merge():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = "/repository" if os.path.exists("/repository/config.json") else BASE
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(model, MODEL_REPO, subfolder="dpo-lora", token=TOKEN)
    model = model.merge_and_unload()
    model.save_pretrained(MERGED, max_shard_size="30GB")
    tok.save_pretrained(MERGED)
    print("merged ->", MERGED, type(model).__name__, flush=True)


if len(sys.argv) > 1 and sys.argv[1] == "--merge-only":
    merge()
    raise SystemExit(0)

subprocess.run([sys.executable, __file__, "--merge-only"], check=True)

api = HfApi(token=TOKEN)  # v2: no separate DPO-only model is published

# The plain python image has no CUDA toolkit, so FlashInfer cannot JIT-compile its
# sampler; vLLM's PyTorch sampler gives the same sampling without it.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams  # noqa: E402

orig = hf_hub_download(os.environ["ORIG_OCT_DATASET"], "intermediate/self_reflection.jsonl",
                       repo_type="dataset", token=TOKEN)
prompts = [json.loads(line)["prompt"] for line in open(orig)]

llm = LLM(model=MERGED, max_model_len=8192, gpu_memory_utilization=0.9, seed=42)
# Same sampling as the v1 fixed run (no repetition penalty). The first v2 DPO
# (lr 5e-5) looped here: 29/1000 usable, 293/1000 even with penalty 1.05; DPO was
# retrained at lr 2e-5 instead of patching generation.
params = SamplingParams(max_tokens=4096, temperature=0.7, top_p=0.95, seed=42)
outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], params)

kept, dropped = [], Counter()
for p, o in zip(prompts, outs):
    text = o.outputs[0].text
    if "</think>" not in text:
        dropped["no </think> (thinking never finished)"] += 1
        continue
    thinking, answer = text.split("</think>", 1)
    thinking, answer = thinking.replace("<think>", "").strip(), answer.strip()
    if not answer:
        dropped["empty answer"] += 1
        continue
    if o.outputs[0].finish_reason != "stop":
        dropped["answer cut off at 4,096 tokens"] += 1
        continue
    kept.append({"prompt": p, "reasoning": thinking, "response": answer})
print("kept", len(kept), "of", len(prompts), "| dropped:", dict(dropped), flush=True)
print("kept per prompt:", sorted(Counter(r["prompt"][:40] for r in kept).values()), flush=True)

os.makedirs("/tmp/out", exist_ok=True)
with open("/tmp/out/self_reflection_raw_qwen3.8.jsonl", "w") as f:  # every output, for diagnosis
    for p, o in zip(prompts, outs):
        f.write(json.dumps({"prompt": p, "text": o.outputs[0].text, "finish_reason": o.outputs[0].finish_reason}, ensure_ascii=False) + "\n")
with open("/tmp/out/self_reflection_fixed_qwen3.8.jsonl", "w") as f:
    for r in kept:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open("/tmp/out/sft_data_fixed_qwen3.8.jsonl", "w") as f:
    for r in kept:
        f.write(json.dumps({"messages": [
            {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"], "reasoning_content": r["reasoning"]},
        ]}, ensure_ascii=False) + "\n")
for name in ("self_reflection_fixed_qwen3.8.jsonl", "sft_data_fixed_qwen3.8.jsonl", "self_reflection_raw_qwen3.8.jsonl"):
    api.upload_file(path_or_fileobj=f"/tmp/out/{name}", path_in_repo=name, repo_id=DATA_REPO,
                    repo_type="dataset", commit_message=f"{name} (thinking separated, 4,096-token cap)")
print("uploaded fixed self-reflection + SFT data", flush=True)
