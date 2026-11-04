import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
import traceback

BASE_DIR = Path(__file__).resolve().parent

# Look for .env in the working directory, next to this file, then one level up
# at the project root — so a single OPENROUTER_API_KEY at the top of the project
# serves every entry point without copying the secret into each subfolder.
# load_dotenv never overwrites a variable that is already set, so this layering
# is safe: the environment always wins, then the nearest .env.
load_dotenv()
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

app = FastAPI()

# "OWUI" = Open WebUI. The council moved qwen behind an Open WebUI deployment
# (vLLM-backed) instead of the old bare model server. Direct API passthrough
# had to be enabled by the admin (ENABLE_OPENAI_API_PASSTHROUGH); confirmed
# working path is /openai/chat/completions (no /v1) with a Bearer API key.
OWUI_API_KEY = os.getenv("OWUI_API_KEY", "")

# OpenRouter is the off-network route: a hosted, OpenAI-compatible endpoint that
# fronts hundreds of models behind one key. It is what makes this clone runnable
# outside the council VPN - the response shape is identical to Open WebUI's, so
# nothing downstream (call_model, the agent loop, the UIs) changes at all.
# Any catalogue id works; OPENROUTER_MODEL sets the default, and a request may
# override it per call (see ChatRequest.model).
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b")

# "gemma" is not currently available via the new deployment (only qwen3.8-27b
# is listed in Open WebUI's /api/models) - left pointing at the old address
# so a request fails with a clear connection error rather than silently.
#
# "key_env" names the variable a route authenticates with; None means the route
# sends no Authorization header at all.
MODELS = {
    "openrouter": {
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": OPENROUTER_MODEL,
        "key": OPENROUTER_API_KEY,
        "key_env": "OPENROUTER_API_KEY",
    },
    "qwen": {
        "url": os.getenv("INTERNAL_QWEN_URL", "http://INTERNAL_HOST:8086/openai/chat/completions"),
        "model": "qwen3.8-27b",
        "key": OWUI_API_KEY,
        "key_env": "OWUI_API_KEY",
    },
    "gemma": {
        "url": os.getenv("INTERNAL_GEMMA_URL", "http://INTERNAL_HOST:8085/v1/chat/completions"),
        "model": None,
        "key": None,
        "key_env": None,
    },
}


class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.7
    max_tokens: int = 512
    # Optional per-request override of the route's configured model, so a caller
    # can reach any OpenRouter model ("anthropic/claude-sonnet-4.5", ...) without
    # restarting the server. Ignored by routes that pin their own model (gemma).
    model: str | None = None


@app.get("/")
async def root():
    return {
        "status": "NYCC local AI router is running",
        "routes": list(MODELS),
        "openrouter_model": OPENROUTER_MODEL,
    }


@app.post("/chat/{model}")
async def chat(model: str, req: ChatRequest):
    if model not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")

    target = MODELS[model]

    payload = {
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    model_id = req.model or target["model"]
    if model_id:
        payload["model"] = model_id

    headers = {}
    if target["key_env"]:
        if not target["key"]:
            raise HTTPException(
                status_code=500,
                detail=f"{target['key_env']} is not configured",
            )
        headers["Authorization"] = f"Bearer {target['key']}"
        # Optional OpenRouter attribution headers; harmless everywhere else.
        headers["X-Title"] = "NYCC AI Assistant"

    try:
        # Long read timeout (a big answer legitimately takes a while) but a short
        # connect timeout, so an unreachable internal host fails in seconds rather
        # than leaving the caller hanging for two minutes.
        timeout = httpx.Timeout(120.0, connect=3.05)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                target["url"],
                json=payload,
                headers=headers,
            )

        print("MODEL:", model, "->", model_id)
        print("MODEL STATUS:", response.status_code)
        print("MODEL RAW RESPONSE:", response.text)

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=response.text,
            )

        try:
            return response.json()
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"Model returned non-JSON response: {response.text}",
            )

    except HTTPException:
        raise

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            # Some httpx timeout errors stringify to "" — fall back to the class name
            # so the message never reads "Backend connection failed: " with nothing after it.
            detail=f"Backend connection failed: {str(e) or type(e).__name__}",
        )

# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint that exposes the AGENT (not just a model).
#
# This is what lets external harnesses evaluate the real agent — tools, live
# data and the multi-step loop — rather than the bare model underneath it.
# Point any OpenAI-compatible client at http://localhost:8000/v1.
#
# For EigenBench specifically: a panel entry of
#     "inspect:openai-api/nycc/agent"
# with NYCC_BASE_URL=http://localhost:8000/v1 routes here.
#
# Panel members in EigenBench play two roles: they answer scenarios AND they
# judge other models' answers. Running the agent loop on a judging prompt would
# waste tool calls and return prose where a rating is expected, so requests are
# split by their system message, which the harness sets verbatim:
#
#   answering -> "...respond in character to the following scenario."  -> agent
#   judging   -> a rating/reflection prompt                            -> plain model
#
# Anything that is not recognisably a scenario response is passed through to the
# underlying model untouched.
# ---------------------------------------------------------------------------

import asyncio
import sys
import time
import uuid

AGENT_DIR = BASE_DIR / "orchestrators"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# EigenBench's RESPONSE_SYSTEM_MESSAGE, matched loosely so wording tweaks in the
# harness do not silently turn the agent into a plain model.
_SCENARIO_MARKER = "respond in character to the following scenario"

# The MCP tool server is a subprocess per request; bound how many run at once so
# a parallel eval does not fork hundreds of them.
_AGENT_SLOTS = asyncio.Semaphore(int(os.getenv("AGENT_CONCURRENCY", "4")))


# Named arms of the experiment. The OpenAI "model" field selects one, so a panel
# entry like "inspect:openai-api/nycc/tools-claude" runs the agent's tools driven
# by Claude with no constitution. Aliases rather than a parsed "provider/model"
# string because the underlying ids contain slashes, which collide with the
# openai-api provider's own name parsing.
#
#   (underlying model, constitution?)
AGENT_ARMS = {
    # ---- the subject -------------------------------------------------------
    # the deployed agent: its own model, its own constitution, its own tools
    "agent": (None, True),

    # ---- the key control ---------------------------------------------------
    # identical model, identical tools, constitution REMOVED
    "tools-qwen": ("qwen/qwen3.8-27b", False),

    # ---- flagship models driving the SAME tools, no constitution -----------
    "tools-opus": ("anthropic/claude-opus-4.7", False),
    "tools-gpt6": ("openai/gpt-6-sol", False),
    "tools-sonnet5": ("anthropic/claude-sonnet-5.5", False),
    "tools-grok": ("x-ai/grok-4.7", False),
    "tools-gemini3": ("google/gemini-3-flash-preview", False),

    # ---- flagship WITH the constitution (separates the rule from the model) -
    "agent-opus": ("anthropic/claude-opus-4.7", True),

    # ---- character-TRAINED model, SAME tools, constitution NOT in the prompt -
    # Its values come from training, so this vs "agent" is trained-vs-prompted
    # with model and tools held fixed. Served by vLLM on a Hugging Face
    # endpoint: set NYCCFT_BASE_URL / NYCCFT_API_KEY (see mcp_client.resolve_endpoint).
    "tools-charmodel": ("endpoint:NYCCFT:qwen3.8-27b-nycc-fixed", False),
    # The same, re-trained on the corrected constitution (trait 5 = political
    # neutrality, matching the judges' criterion 5). NYCCNT_BASE_URL / NYCCNT_API_KEY.
    "tools-charmodel-neutral": ("endpoint:NYCCNT:qwen3.8-27b-nycc-neutral", False),
    # "NYCC character trained agent": the same character-trained model WITH the
    # constitution prompt and tools, i.e. the deployed agent with its model swapped
    # for the trained one. Tests whether training adds anything on top of prompting.
    "agent-charmodel": ("endpoint:NYCCNT:qwen3.8-27b-nycc-neutral", True),
}

# Optional larger per-step budget per arm. Deliberately EMPTY: a 6000-token
# budget for tools-charmodel did not change its ~2/12 "couldn't answer" rate
# (different questions failed each time), so every tool arm keeps the same 1500.
ARM_STEP_TOKENS: dict[str, int] = {}


def _is_scenario_response(messages: list) -> bool:
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str) and _SCENARIO_MARKER in content.lower():
                return True
    return False


def _last_user_text(messages: list) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content
            # OpenAI content parts
            if isinstance(content, list):
                return "\n".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                ).strip()
    return ""


async def _run_agent(question: str, arm: str) -> str:
    """One full agent turn — tool catalog, tool calls, final answer — on the
    underlying model and constitution setting that `arm` selects."""
    from agent_chatbot import (  # noqa: PLC0415
        CONSTITUTION_ENABLED,
        _format_tool_catalog,
        answer_query_agentic,
    )
    from mcp_client import (  # noqa: PLC0415
        MCP_SERVER_SCRIPT,
        MODEL_OVERRIDE,
        STEP_TOKENS_OVERRIDE,
        MCPToolClient,
    )

    model_id, constitution = AGENT_ARMS[arm]
    # ContextVars are set inside the request's own task, so parallel requests
    # running different arms do not overwrite each other.
    MODEL_OVERRIDE.set(model_id)
    CONSTITUTION_ENABLED.set(constitution)
    STEP_TOKENS_OVERRIDE.set(ARM_STEP_TOKENS.get(arm))

    async with _AGENT_SLOTS:
        async with MCPToolClient(MCP_SERVER_SCRIPT) as client:
            tools = await client.session.list_tools()
            catalog = _format_tool_catalog(tools.tools)
            result = await answer_query_agentic(client, catalog, question)
            return result["final_answer"]


# qwen3.8-27b is a reasoning model: it spends tokens in a separate "reasoning"
# field before emitting any content. A tight budget is consumed entirely by that
# reasoning and the reply comes back finish_reason="length", content=None — which
# a judging harness reads as an invalid judgment. Floor the budget so a short
# answer (a rating, a single integer) still has room to be produced at all.
_MIN_PASSTHROUGH_TOKENS = int(os.getenv("MIN_PASSTHROUGH_TOKENS", "800"))


async def _passthrough(
    messages: list, temperature: float, max_tokens: int, model_id: str | None = None
) -> str:
    """Send the conversation to OpenRouter unchanged, with no tools.

    `model_id` matters for judging: each arm must judge with ITS OWN model. If
    every arm judged with the default, a panel of ten would really contain one
    judge repeated, and the trust matrix EigenTrust reads would be degenerate.
    """
    from mcp_client import resolve_endpoint  # noqa: PLC0415

    target = MODELS["openrouter"]
    url = target["url"]
    headers = {"Authorization": f"Bearer {target['key']}", "X-Title": "NYCC AI Assistant"}
    custom = resolve_endpoint(model_id)
    if custom:  # self-hosted arm (e.g. the character-trained model) judges as itself
        url, headers, model_id = custom
    elif not target["key"]:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not configured")
    payload = {
        "model": model_id or target["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max(max_tokens, _MIN_PASSTHROUGH_TOKENS),
    }
    # 900 s, not 300: a self-hosted character-trained judge writes reflections of up
    # to ~10k tokens, which under load took longer than 300 s, timed out, retried and
    # stalled a whole EigenBench run. Only allows slow replies to finish.
    timeout = httpx.Timeout(900.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text[:500])

    choice = response.json()["choices"][0]
    content = choice["message"].get("content")
    if content:
        return content

    # Still empty: surface the reasoning text rather than returning null, which
    # callers cannot distinguish from a crash.
    reasoning = choice["message"].get("reasoning") or choice["message"].get("reasoning_content")
    if reasoning:
        return reasoning.strip()
    raise HTTPException(
        status_code=502,
        detail=(
            "model returned no content "
            f"(finish_reason={choice.get('finish_reason')!r}); "
            "raise max_tokens — the underlying model reasons before answering"
        ),
    )


def _completion(text: str, model: str, role: str) -> dict:
    """Shape a reply the way an OpenAI client expects."""
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "system_fingerprint": f"nycc-{role}",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        # Token accounting is not tracked through the agent loop; clients such as
        # inspect-ai only require the field to exist.
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
async def list_models():
    """Advertise the agent so OpenAI-compatible clients can discover it."""
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "created": int(time.time()), "owned_by": "nycc"}
            for name in [*AGENT_ARMS, "passthrough"]
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(body: dict):
    """OpenAI-compatible entry point. Accepts a loose dict because real clients
    send provider-specific extras that must not cause a 422."""
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    model = body.get("model") or "agent"
    temperature = float(body.get("temperature", 0.3))
    max_tokens = int(body.get("max_tokens") or 1500)

    # The arm is the last path segment of the model name, so both "agent" and
    # "nycc/agent" resolve. Judging requests never run the loop regardless.
    arm = model.rsplit("/", 1)[-1]

    if arm in AGENT_ARMS and _is_scenario_response(messages):
        question = _last_user_text(messages)
        if not question:
            raise HTTPException(status_code=400, detail="no user message to answer")
        try:
            text = await _run_agent(question, arm)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"agent failed: {e}") from None
        return _completion(text, model, f"agent:{arm}")

    # Judging (and anything that is not a scenario response): no tools, but on
    # the arm's own underlying model so each panel member judges as itself.
    judge_model = AGENT_ARMS[arm][0] if arm in AGENT_ARMS else None
    text = await _passthrough(messages, temperature, max_tokens, judge_model)
    return _completion(text, model, f"passthrough:{arm}")
