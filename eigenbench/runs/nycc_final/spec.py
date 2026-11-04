"""EigenBench run for the paper: 13 agents, 290 NYC Council staff requests, the NYC
Council AI Constitution (10 criteria), direct-rating mode.

Copy this folder to EigenBench/runs/nycc_final, the constitution to
EigenBench/data/constitutions/, and the scenarios to EigenBench/data/scenarios/
(see the top-level README). Then, with the agent server running (agent/local_ai.py):

    python scripts/run_inspect.py runs/nycc_final/spec.py

"inspect:openai-api/nycc/<arm>" models are agent configurations served by
agent/local_ai.py (router + 9 MCP tools); "<arm>" is a key of AGENT_ARMS there.
"inspect:openai-api/nyccnt/..." is the character-trained (OCT) model served by vLLM
(NYCCNT_BASE_URL / NYCCNT_API_KEY). Plain names are OpenRouter models without tools.
"""

RUN_SPEC = {
    "verbose": True,
    "evaluation": {
        "mode": "direct_rating",
        "direct_rating": {
            "include_self": True,
            "scale_min": 1,
            "scale_max": 10,
            "normalization": "zscore_softmax",
            "softmax_temperature": 1.0,
        },
    },
    "models": {
        # CivicAlign AI 2x2 (all with the router and 9 MCP tools)
        "NYCC Agent": "inspect:openai-api/nycc/agent",                          # base weights + constitution
        "Qwen + tools, no constitution": "inspect:openai-api/nycc/tools-qwen",  # baseline
        # frontier models driving the same tools, no constitution
        "Claude Opus 4.7 + tools": "inspect:openai-api/nycc/tools-opus",
        "GPT-6 Sol + tools": "inspect:openai-api/nycc/tools-gpt6",
        "Claude Sonnet 5.5 + tools": "inspect:openai-api/nycc/tools-sonnet5",
        "Grok 4.7 + tools": "inspect:openai-api/nycc/tools-grok",
        "Gemini 3 Flash + tools": "inspect:openai-api/nycc/tools-gemini3",
        # tool-free references
        "Qwen 3.8 27B (bare)": "qwen/qwen3.8-27b",
        "Claude Opus 4.7 (bare)": "anthropic/claude-opus-4.7",
        "GPT-6 Sol (bare)": "openai/gpt-6-sol",
        # character-trained (OCT) model
        "Character-trained model": "inspect:openai-api/nyccnt/qwen3.8-27b-nycc-neutral",           # no tools
        "Character-trained model + tools": "inspect:openai-api/nycc/tools-charmodel-neutral",      # OCT, no constitution
        "NYCC character trained agent": "inspect:openai-api/nycc/agent-charmodel",                 # OCT + constitution
    },
    "dataset": {
        # 290 unique staff requests (356 before removing case/whitespace duplicates).
        "path": "data/scenarios/nycc_scenarios.json",
        "start": 0,
        "count": 290,
        "shuffle": False,
        "shuffle_seed": 42,
    },
    "constitution": {
        "path": "data/constitutions/nycc_deployed.json",
        "num_criteria": 10,
    },
    "collection": {
        # Several arms share one self-hosted GPU; 4 concurrent requests per model avoids
        # gateway timeouts. Scheduling only.
        "inspect": {"max_connections": 4},
        "enabled": True,
        "cached_responses_path": None,
        "failure_policy": "omit_invalid_judgments",
        "allow_ties": True,
        "group_size": 4,
        "groups": 1,
        "sampler_mode": "partitioned_random_judge",
        "alpha": 2.0,
        "generation": {
            "response": {"max_tokens": 4096, "temperature": 0.7},
            # Budgets raised above the defaults: reasoning models otherwise exhaust them
            # before emitting a reflection or rating.
            "reflection": {"max_tokens": 6000, "temperature": 0.2, "per_model": {
                "Character-trained model": {"max_tokens": 10000},
                "Character-trained model + tools": {"max_tokens": 10000},
                "NYCC character trained agent": {"max_tokens": 10000},
            }},
            "direct_rating": {"max_tokens": 2000, "temperature": 0.0, "per_model": {
                "Character-trained model": {"max_tokens": 8000},
                "Character-trained model + tools": {"max_tokens": 8000},
                "NYCC character trained agent": {"max_tokens": 8000},
            }},
        },
    },
    "training": {
        "enabled": True,
        "model": "btd_ties",
        "dims": [2],
        "lr": 1e-3,
        "weight_decay": 0.0,
        "max_epochs": 1000,
        "batch_size": 32,
        "device": "cpu",
        "test_size": 0.2,
        "group_split": False,
        "separate_criteria": False,
        "bootstrap": {"enabled": True, "n_bootstraps": 100, "random_seed": 42},
    },
    "upload": {"enabled": False},
}
