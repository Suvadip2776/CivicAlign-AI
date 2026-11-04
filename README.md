# CivicAlign AI: replication package

Code and aggregated results for *CivicAlign AI: Constitution-Guided Language Model Agents for
Public-Sector Governance* by Suvadip Sana, Lionel Levine, and Martin T. Wells (Cornell
University). The repository covers the four parts of the paper's experiments:

| Paper | What | Where |
|---|---|---|
| §3.4, §4.1 | Tool-augmented agent (router + 9 MCP tools), constitutional prompting, all 13 agent configurations | `agent/` |
| §4.5, App. A.1, B.5–B.7 | Open Character Training (OCT) of Qwen 3.8 27B on the NYCC constitution | `character_training/` |
| §4.5, §5.1–5.2, App. A.2–A.4 | EigenBench evaluation, bootstrap, per-principle analysis | `eigenbench/` |
| §4.6, §5.2.1, App. A.5–A.6 | Capability-preservation benchmarks | `capability_evals/` |
| Figs. 2–6, Tables 3–4 | Figures and tables from the final results | `figures/`, `results/` |

**Reproducing the figures needs no API access.** All numbers in the paper can be regenerated
from `results/` (see step 6). Steps 1–5 rerun the experiments.

## Setup

Python 3.12. Copy `.env.example` to `.env` at the package root and fill in the keys.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r agent/requirements.txt          # agent server and MCP tools
pip install -r capability_evals/requirements.txt
pip install matplotlib numpy                   # figures
```

External code (unmodified):

* **EigenBench** — the official implementation, `https://github.com/jchang153/EigenBench`,
  commit `c4ef513`. Clone it next to this package as `EigenBench/` and install its
  requirements.
* **Open Character Training** — the official OCT repository (Maiya et al., 2025). Set
  `OCT_DIR` to its `character/` directory; the training scripts import the teacher system
  prompt and the DPO filtering from it.

Models: the base model `qwen/qwen3.8-27b` and the frontier comparators are accessed through
OpenRouter. The character-trained model is served with vLLM (v0.32.0, bfloat16,
`--reasoning-parser qwen3 --max-model-len 32768 --served-model-name qwen3.8-27b-nycc-neutral`)
on one H200; set `NYCCNT_BASE_URL` to its OpenAI-compatible endpoint. The trained weights
will be released upon publication.

## 1. Agent (`agent/`)

`orchestrators/agent_chatbot.py` runs the agent loop: at each step the model returns one
JSON action, either a tool call or the final answer, for at most five tool steps.
`mcp/server.py` exposes the nine tools over MCP (Legistar legislation search and bill text
via NYC Open Data, the public Council calendar, Council resource lookup, ACS demographics
via `councilcount`, web search, URL classification, a calculator, and a sandboxed Python
interpreter).

Constitutional prompting is the `<<<CONSTITUTION_SECTION ... CONSTITUTION_SECTION>>>` block
of `prompts/agent_system_prompt.txt`, filled with `prompts/NYCC_constitution.txt`. In
configurations without the constitution the whole block is removed.

`local_ai.py` serves every agent configuration behind one OpenAI-compatible endpoint, so
EigenBench can query them as models. The configurations are the `AGENT_ARMS` table
(`arm: (underlying model, constitution on/off)`). Judging requests are passed through to
each arm's underlying model without tools.

```bash
cd agent/orchestrators && uvicorn --app-dir .. local_ai:app --port 8000
```

The `qwen`/`gemma` routes in `local_ai.py` are the deployment's internal model servers and
are not used in the experiments (`INTERNAL_*` placeholders).

## 2. Open Character Training (`character_training/`)

Inputs: `constitution/nycc_neutral.txt` (the ten first-person traits; trait 5 is the
politically neutral revision described in §4.5) and `data/few-shot_nycc_neutral.jsonl`
(traits with 5 hand-written and 45 generated seed requests each; reproduced in App. B.5).
The original NYCC OCT release (general-purpose LIMA prompts and the original trait list)
is read from `ORIG_OCT_DATA` (and `ORIG_OCT_DATASET` for the introspection prompts); it
will be released upon publication.

| Step | Script | Output |
|---|---|---|
| Trait-5 seed requests | `01_gen_trait5_questions.py` | `data/few-shot_nycc_neutral.jsonl` |
| Teacher (GLM-5.2, trait system prompt + reasoning prefill) | `02_teacher.py` | 1,829 chosen responses |
| Student (Qwen 3.8 27B) and DPO pairs (OCT filter: non-empty, ends in punctuation, ≤1,024 tokens) | `03_student_and_pairs.py` | 832 pairs |
| DPO, LoRA r=64 α=128, β=0.1, lr 2e-5, 1 epoch | `v2_03_train_dpo.py --lr 2e-5` | DPO adapter |
| Introspection: 10 prompts × 100 samples, reasoning kept separate | `v2_04b_self_reflection_fixed.py` | 1,000 transcripts |
| SFT, LoRA r=64 α=128, lr 5e-5, 1 epoch, max length 3,072 | `v2_05_train_sft.py` | merged OCT model |

Steps 1–3 call OpenRouter. Steps 4–6 run on one H200 (each script declares its pinned
dependencies inline for `uv run`). `v2_rent_gpu.sh` and `v2_endpoint_bootstrap.sh` run one
step on a rented Hugging Face Inference Endpoint; `YOUR_HF_ORG` marks where your own
organisation goes.

## 3. EigenBench (`eigenbench/`)

```bash
cp -r eigenbench/runs/nycc_final EigenBench/runs/
cp eigenbench/data/constitutions/nycc_deployed.json EigenBench/data/constitutions/
cp <scenarios>/nycc_scenarios.json EigenBench/data/scenarios/       # see "Data" below
export NYCC_BASE_URL=http://localhost:8000/v1 NYCC_API_KEY=unused   # the agent server (step 1)
cd EigenBench && python scripts/run_inspect.py runs/nycc_final/spec.py
```

The spec fixes every setting reported in App. A.2: direct rating on a 1–10 scale over the
ten criteria (`nycc_deployed.json`, App. B.4), partitioned random-judge sampler (group size
4, seed 42), z-score softmax normalisation, EigenTrust, and a scenario bootstrap with B = 100.
The paper's run collected 4,112 judgments covering all 169 judge–evaluee pairs.

Per-principle Elo (Table 4, Fig. 4) reruns the unmodified pipeline on each criterion:

```bash
EIGENBENCH_DIR=EigenBench python eigenbench/tools/per_criterion_elo.py EigenBench/runs/nycc_final
```

The factorial contrasts (Δprompt, ΔOCT, ΔOCT|prompt, Δinteraction) are differences of the
`elo_vector` entries within each replicate of `direct_rating/bootstrap/samples.json`.

## 4. Capability benchmarks (`capability_evals/`)

`evals.py` defines the 13 benchmarks as Inspect tasks; `constitution=true` adds the
constitution as the evaluated model's system message (merged with any benchmark system
prompt). Graders (GPT-4o) never see it. Items are identical across configurations
(`--sample-shuffle 42`; datasets' internal shuffles are disabled).

```bash
cd capability_evals
./run_config.sh 1_base                 openrouter/qwen/qwen3.8-27b                 false 24
./run_config.sh 2_base_constitution    openrouter/qwen/qwen3.8-27b                 true  24
./run_config.sh 3_trained              openai-api/nyccnt/qwen3.8-27b-nycc-neutral  false 8
./run_config.sh 4_trained_constitution openai-api/nyccnt/qwen3.8-27b-nycc-neutral  true  8
```

ConfAIde Tier 3 is then re-scored with the model-graded leakage judge (App. A.6), and the
table is compiled:

```bash
for c in 1_base 2_base_constitution 3_trained 4_trained_constitution; do
  inspect score logs/$c/*confaide-tier3*.eval --scorer confaide_judge.py@judged_leakage \
    --action append --model openrouter/openai/gpt-4o --model-role grader=openrouter/openai/gpt-4o \
    --output-file logs_rescored/${c}_confaide_tier3.eval
done
python compile_results.py        # -> results/capability_table.{csv,md,tex}
```

XSTest (`walledai/XSTest`) is gated on Hugging Face; accept its terms with the account of
`HF_TOKEN` first.

## 5. Results (`results/`)

* `eigenbench/direct_rating/` — final 13-agent run: `summary.json` (point estimates),
  `bootstrap/summary.json` (bootstrap means and 95% intervals), `bootstrap/samples.json`
  (per-replicate Elo vectors and trust matrices), `per_criterion_elo.json`, trust matrix,
  observation counts, and the analysis configuration. Model names map to the paper's
  configurations: *Qwen + tools, no constitution* = baseline; *NYCC Agent* = constitutional
  prompting; *Character-trained model + tools* = OCT; *NYCC character trained agent* =
  OCT + prompting.
* `capability/` — capability table (Table in App. A.5).

## 6. Figures

```bash
python figures/make_figures.py      # -> figures/output/fig1..fig5 (.pdf, .png)
```

## Data

The 290 evaluation scenarios are requests from NYC Council staff and are not included;
`eigenbench/data/scenarios/README.md` describes the expected format. Raw model responses
and judgments are omitted for the same reason, since they quote the requests. All
aggregated results needed for the paper's figures and tables are included.

## Citation

```bibtex
@article{sana2026civicalign,
  title   = {CivicAlign AI: Constitution-Guided Language Model Agents for Public-Sector Governance},
  author  = {Sana, Suvadip and Levine, Lionel and Wells, Martin T.},
  journal = {arXiv preprint},
  year    = {2026}
}
```
