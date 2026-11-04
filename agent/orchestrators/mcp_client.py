"""
Shared MCP client infrastructure used by every orchestrator: the path to
mcp/server.py, the stdio session wrapper, and the HTTP call to the model.

There are two ways to reach a model, and call_model picks between them:

  direct (the default) — POST straight to OpenRouter with OPENROUTER_API_KEY.
      OpenRouter is already a hosted HTTP endpoint, so proxying it through
      local_ai.py would add a process without adding anything else. Nothing
      to start: run the CLI or a UI on its own.

  proxy — set API_URL to a running local_ai.py route. This is what the council
      VPN path needs (API_URL=http://localhost:8000/chat/qwen), since the
      internal Open WebUI deployment is what local_ai.py was written to front. Split out from final_master_chatbot.py (removed
along with the rest of the single-route pipeline - see the
legacy-single-route-pipeline branch) since agent_chatbot.py depends on this
part but not on that file's single-route-specific routing logic.
"""

import os
import sys
from contextlib import AsyncExitStack
from contextvars import ContextVar
from datetime import timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
MCP_SERVER_SCRIPT = REPO_ROOT / "mcp" / "server.py"

# .env in the working directory, then NYCC_AI/, then the project root above it.
# load_dotenv never overwrites an already-set variable, so the environment wins.
load_dotenv()
load_dotenv(REPO_ROOT / ".env")
load_dotenv(REPO_ROOT.parent / ".env")

# Unset (the default) means direct mode. Set it to a local_ai.py route to proxy.
API_URL = os.getenv("API_URL")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b")

# Lets one process run the agent loop on different underlying models at the same
# time — needed to ask "what if a frontier model drove these same tools?". A
# ContextVar rather than a global because concurrent requests must not see each
# other's choice.
MODEL_OVERRIDE: ContextVar[str | None] = ContextVar("nycc_model_override", default=None)
# Per-arm token budget for each agent step, for models that think much longer
# than the default arms before emitting the action JSON. None = caller's budget.
STEP_TOKENS_OVERRIDE: ContextVar[int | None] = ContextVar("nycc_step_tokens", default=None)

# (connect, read): a generous read timeout for long answers, a shorter connect
# so an unreachable host fails quickly instead of stalling the agent loop.
MODEL_REQUEST_TIMEOUT = (10, 120)

# Upper bound on one tool call. The requests-based tools time out on their own,
# but ddgs (web_search) and councilcount make network calls we cannot bound, and
# a call that never returns otherwise holds an agent slot forever — four such
# hangs froze a 2,900-judgment EigenBench run. The agent loop turns the timeout
# into an observation, so the model can try another tool or answer without it.
TOOL_CALL_TIMEOUT = timedelta(seconds=60)


class MCPToolClient:
    """Long-lived MCP client session backed by an mcp/server.py subprocess."""

    def __init__(self, server_script: Path):
        # Pass our environment through explicitly. The MCP SDK otherwise starts the
        # server with a filtered, "safe" subset of variables, so anything the tools
        # read (PUBLIC_DEPLOYMENT, SOCRATA_APP_TOKEN, the endpoint overrides) would
        # silently never arrive in the subprocess.
        self._server_params = StdioServerParameters(
            command=sys.executable,
            args=[str(server_script)],
            env=os.environ.copy(),
        )
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPToolClient":
        await self._stack.__aenter__()
        # The MCP server's own stderr (its "MCP server running" / request logging)
        # gets forwarded to our stderr by default — discard it to keep output clean.
        devnull = self._stack.enter_context(open(os.devnull, "w"))
        read, write = await self._stack.enter_async_context(
            stdio_client(self._server_params, errlog=devnull)
        )
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._stack.__aexit__(exc_type, exc, tb)

    async def call_tool(self, name: str, arguments: dict) -> str:
        result = await self.session.call_tool(
            name, arguments=arguments, read_timeout_seconds=TOOL_CALL_TIMEOUT
        )

        text_parts = [block.text for block in result.content if getattr(block, "type", None) == "text"]
        text = "\n".join(text_parts) if text_parts else "(tool returned no text content)"

        if result.isError:
            return f"MCP tool '{name}' reported an error: {text}"

        return text


def resolve_endpoint(model_id: str | None) -> tuple[str, dict, str] | None:
    """Route "endpoint:<PREFIX>:<served-name>" to a self-hosted OpenAI-compatible
    server at $<PREFIX>_BASE_URL (key $<PREFIX>_API_KEY), e.g. the character-
    trained model on a Hugging Face endpoint. Any other id returns None and goes
    to OpenRouter as before. Returns (chat-completions url, headers, model)."""
    if not model_id or not model_id.startswith("endpoint:"):
        return None
    _, prefix, served = model_id.split(":", 2)
    base = os.getenv(f"{prefix}_BASE_URL")
    if not base:
        raise RuntimeError(f"{model_id} needs {prefix}_BASE_URL set in the environment")
    headers = {"Authorization": f"Bearer {os.getenv(f'{prefix}_API_KEY', '')}"}
    return base.rstrip("/") + "/chat/completions", headers, served


def call_model(messages: list, temperature: float = 0.3, max_tokens: int = 800) -> str:
    max_tokens = STEP_TOKENS_OVERRIDE.get() or max_tokens
    payload = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    headers = {}

    custom = resolve_endpoint(MODEL_OVERRIDE.get())
    if custom:
        url, headers, payload["model"] = custom
    elif API_URL:
        url = API_URL
    else:
        if not OPENROUTER_API_KEY:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Put it in NYCC_AI/.env (or the project "
                "root .env), or set API_URL to a running local_ai.py route instead."
            )
        url = OPENROUTER_URL
        payload["model"] = MODEL_OVERRIDE.get() or OPENROUTER_MODEL
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "X-Title": "NYCC AI Assistant",
        }

    response = requests.post(url, json=payload, headers=headers, timeout=MODEL_REQUEST_TIMEOUT)

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        # Surface what the provider actually said (bad key, unknown model id,
        # out of credit) instead of a bare "400 Client Error".
        raise RuntimeError(
            f"Model request failed ({response.status_code}): {response.text[:300]}"
        ) from e

    choice = response.json()["choices"][0]
    content = choice["message"].get("content")
    if content:
        return content.strip()

    # Reasoning models return content=None when the token budget is spent on
    # reasoning before any answer is emitted. Returning the reasoning text (or
    # an empty string) lets the agent loop's own JSON-validation retry handle it;
    # calling .strip() on None instead crashed the whole request.
    # OpenRouter names it "reasoning"; vLLM's reasoning parser "reasoning_content".
    reasoning = choice["message"].get("reasoning") or choice["message"].get("reasoning_content")
    if reasoning:
        return reasoning.strip()
    return ""
