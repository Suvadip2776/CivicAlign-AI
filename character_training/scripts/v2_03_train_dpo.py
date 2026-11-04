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
"""Step 3: DPO LoRA on Qwen/Qwen3.8-27B — runs as a Hugging Face Job.

Hyperparameters copied from the original NYCC OCT model dpo-lora/training_args.bin
and adapter_config.json: LoRA r=64 alpha=128 on the same 12 module names,
lr 5e-5 linear, warmup 0.1, 1 epoch, batch 1 x grad-accum 8, beta 0.1, sigmoid
loss, max_length 1024, bf16, seed 42. The reference model is the base with the
adapter disabled (TRL default with peft), so only one copy of the weights.

    hf jobs uv run --flavor h200 --namespace YOUR_HF_ORG --secrets HF_TOKEN \
        scripts/03_train_dpo.py [--smoke]
"""
import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

BASE = "Qwen/Qwen3.8-27B"
DATA_REPO = "YOUR_HF_ORG/nycc-oct-qwen38-neutral"          # private dataset repo
MODEL_REPO = "YOUR_HF_ORG/qwen3.8-27b-nycc-neutral"         # private model repo
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]

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
ap.add_argument("--smoke", action="store_true", help="3 steps, no push: check load/memory/speed")
ap.add_argument("--data-file", default="dpo_pairs_neutral_qwen3.8.jsonl")
ap.add_argument("--lr", type=float, default=5e-5, help="v2 rerun: 2e-5 (5e-5 over-sharpened the neutral model)")
args = ap.parse_args()

# On a rented Inference Endpoint the base repo is pre-downloaded to /repository.
SRC = "/repository" if os.path.exists("/repository/config.json") else BASE
print("loading base from", SRC, flush=True)
tok = AutoTokenizer.from_pretrained(SRC)
model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.bfloat16, device_map="cuda")
print("loaded:", type(model).__name__, "| params (B):",
      round(sum(p.numel() for p in model.parameters()) / 1e9, 2),
      "| GPU GB:", round(torch.cuda.memory_allocated() / 1e9, 1), flush=True)

ds = load_dataset(DATA_REPO, data_files=args.data_file, split="train")
print("pairs:", len(ds), flush=True)

cfg = DPOConfig(
    output_dir="/tmp/dpo",
    learning_rate=args.lr,
    lr_scheduler_type="linear",
    warmup_steps=0.1,  # a float < 1 is a ratio of total steps (warmup_ratio was removed in transformers 5)
    num_train_epochs=1,
    max_steps=3 if args.smoke else -1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    beta=0.1,
    loss_type="sigmoid",
    max_length=1024,
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
trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok,
                     peft_config=peft_cfg)

t = time.time()
trainer.train()
elapsed = time.time() - t
steps = trainer.state.global_step
print(json.dumps({"steps": steps, "seconds": round(elapsed), "sec_per_step": round(elapsed / max(steps, 1), 1),
                  "peak_GPU_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                  "log_tail": trainer.state.log_history[-3:]}, default=str), flush=True)

if args.smoke:
    raise SystemExit(0)

trainer.model.save_pretrained("/tmp/dpo-lora")
tok.save_pretrained("/tmp/dpo-lora")
fix_adapter_metadata("/tmp/dpo-lora", BASE)
with open("/tmp/dpo-lora/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=1, default=str)
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(MODEL_REPO, private=True, exist_ok=True)
api.upload_folder(folder_path="/tmp/dpo-lora", path_in_repo="dpo-lora", repo_id=MODEL_REPO,
                  commit_message="DPO LoRA on Qwen3.8-27B (NYCC OCT replication)")
print("pushed dpo-lora to", MODEL_REPO, flush=True)
