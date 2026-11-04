# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.14.1",
#   "transformers==5.20.2",
#   "trl==1.16.0",
#   "peft==0.21.2",
#   "accelerate==1.15.0",
#   "datasets==5.1.0",
#   "flash-linear-attention==0.6.0",
#   "huggingface_hub",
# ]
# ///
"""Step 5: introspection SFT LoRA on the DPO-merged Qwen3.8-27B, then merge and
publish the final character model.

Hyperparameters from the original NYCC OCT model introspection-lora: LoRA r=64
alpha=128, same 12 modules, lr 5e-5 linear, warmup 0.1, 1 epoch, batch 1 x 8,
max_length 3072, loss "chunked_nll" (a deprecated alias of "nll" in TRL), loss
on the full conversation (TRL default for a messages dataset), seed 42.

Publishes to the private repo: dpo-lora/ (already there), introspection-lora/,
and the merged model at the root (text-only Qwen3_5ForCausalLM, like the 3.6 one).
"""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE = "Qwen/Qwen3.8-27B"
DATA_REPO = "YOUR_HF_ORG/nycc-oct-qwen38-neutral"
MODEL_REPO = "YOUR_HF_ORG/qwen3.8-27b-nycc-neutral"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
TOKEN = os.environ["HF_TOKEN"]

def fix_adapter_metadata(path, base):
    """PEFT records the local load path (/repository) as the base model; the Hub
    rejects that in README metadata, so point both files at the real base id."""
    cfg_path = os.path.join(path, "adapter_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["base_model_name_or_path"] = base
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    readme = os.path.join(path, "README.md")
    if os.path.exists(readme):
        with open(readme) as f:
            text = f.read()
        with open(readme, "w") as f:
            f.write(text.replace("/repository", base))


ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--data-file", default="sft_data_qwen3.8.jsonl")
ap.add_argument("--out-repo", default=MODEL_REPO, help="where the merged model and introspection-lora go")
ap.add_argument("--max-length", type=int, default=3072)
args = ap.parse_args()

src = "/repository" if os.path.exists("/repository/config.json") else BASE
tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, MODEL_REPO, subfolder="dpo-lora", token=TOKEN).merge_and_unload()
print("DPO-merged base ready:", type(model).__name__, flush=True)

ds = load_dataset(DATA_REPO, data_files=args.data_file, split="train", token=TOKEN)
print("SFT examples:", len(ds), flush=True)

cfg = SFTConfig(
    output_dir="/tmp/sft",
    learning_rate=5e-5,
    lr_scheduler_type="linear",
    warmup_steps=0.1,  # a float < 1 is a ratio of total steps (warmup_ratio was removed in transformers 5)
    num_train_epochs=1,
    max_steps=3 if args.smoke else -1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_length=args.max_length,
    loss_type="nll",
    weight_decay=0.0,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    seed=42,
)
peft_cfg = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.0, target_modules=TARGETS,
                      task_type="CAUSAL_LM")
trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
t = time.time()
trainer.train()
el = time.time() - t
print(json.dumps({"steps": trainer.state.global_step, "seconds": round(el),
                  "peak_GPU_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                  "log_tail": trainer.state.log_history[-3:]}, default=str), flush=True)
if args.smoke:
    raise SystemExit(0)

api = HfApi(token=TOKEN)
api.create_repo(args.out_repo, private=True, exist_ok=True)
trainer.model.save_pretrained("/tmp/introspection-lora")
fix_adapter_metadata("/tmp/introspection-lora", BASE)
with open("/tmp/introspection-lora/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=1, default=str)
api.upload_folder(folder_path="/tmp/introspection-lora", path_in_repo="introspection-lora",
                  repo_id=args.out_repo, commit_message="introspection SFT LoRA")

final = trainer.model.merge_and_unload()
final.save_pretrained("/tmp/final", max_shard_size="30GB")
tok.save_pretrained("/tmp/final")
api.upload_folder(folder_path="/tmp/final", repo_id=args.out_repo,
                  commit_message="merged NYCC character model on Qwen3.8-27B (DPO + introspection)")
print("pushed final merged model to", args.out_repo, flush=True)
