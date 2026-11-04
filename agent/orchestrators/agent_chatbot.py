"""
General-purpose multi-tool agent for the NYCC AI Assistant.

This is the only orchestrator in active use - the single-route pipeline
(classify a query into exactly one route, call exactly one tool) was removed
since it can't handle multi-hop questions the way this agent loop can; see
the legacy-single-route-pipeline branch if it's ever needed for reference.
This lets the model call AS MANY tools as it needs, in whatever order it
decides, chaining results between calls, before producing a final answer.
Reuses MCPToolClient and call_model from mcp_client.py.
"""

import asyncio
import json
import re
from contextvars import ContextVar
from datetime import date
from pathlib import Path

import requests

from mcp_client import MCP_SERVER_SCRIPT, MCPToolClient, call_model

BASE_DIR = Path(__file__).resolve().parent
AGENT_SYSTEM_PROMPT_FILE = BASE_DIR.parent / "prompts" / "agent_system_prompt.txt"
CONSTITUTION_FILE = BASE_DIR.parent / "prompts" / "NYCC_constitution.txt"

with open(AGENT_SYSTEM_PROMPT_FILE, "r") as f:
    AGENT_SYSTEM_PROMPT = f.read()

# The constitution is substituted into the prompt rather than pasted into it, so
# the governing principles stay a single editable document instead of drifting
# between a file nobody loads and a copy buried in the prompt.
with open(CONSTITUTION_FILE, "r") as f:
    CONSTITUTION = f.read().strip()

# Set False to run the agent with its tools but WITHOUT the constitution — the
# ablation that separates "the tools did it" from "the constitution did it".
# The whole governing-principles block is removed, not just the criteria list,
# because its framing sentences are themselves constitutional guidance.
CONSTITUTION_ENABLED: ContextVar[bool] = ContextVar("nycc_constitution", default=True)

_SECTION_RE = re.compile(r"<<<CONSTITUTION_SECTION.*?CONSTITUTION_SECTION>>>", re.S)


def _apply_constitution(prompt: str) -> str:
    """Either fill in the constitution or excise the section that holds it."""
    if not CONSTITUTION_ENABLED.get():
        return _SECTION_RE.sub("", prompt).rstrip() + "\n"
    prompt = prompt.replace("<<<CONSTITUTION_SECTION\n", "").replace("CONSTITUTION_SECTION>>>", "")
    return prompt.replace("{{CONSTITUTION}}", CONSTITUTION).rstrip() + "\n"

MAX_STEPS = 5

# Cap on the model's reply to one step. This has to be generous enough for a
# final_answer that summarizes a long observation: a 7-day calendar lookup comes
# back with ~20 meetings, and an answer cut off mid-sentence is not valid JSON,
# so the loop burns its retries and falls back to _COULD_NOT_ANSWER rather than
# showing the user a perfectly good half-written answer.
MAX_ANSWER_TOKENS = 1500

# Internal plumbing tool not meant for the agent to call directly - just a
# connectivity test tool. (The old handle_router_output/json tools for the
# single-route pipeline were removed along with that pipeline.)
_HIDDEN_TOOLS = {"hello"}


def _format_tool_catalog(tools) -> str:
    lines = []
    for tool in tools:
        if tool.name in _HIDDEN_TOOLS:
            continue

        schema = tool.inputSchema or {}
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        params = []
        for prop_name, prop in properties.items():
            prop_type = prop.get("type", "any")
            if prop_name in required:
                params.append(f"{prop_name}: {prop_type}")
            else:
                default = prop.get("default")
                params.append(f"{prop_name}: {prop_type} = {default!r}")

        description = " ".join((tool.description or "").split())
        lines.append(f"- {tool.name}({', '.join(params)}) — {description}")

    return "\n".join(lines)


def _parse_action(raw_reply: str) -> dict:
    text = raw_reply.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)


# A generic, safe message to fall back on when the model fails to produce a
# usable answer even after retries. Raw model text (JSON, partial objects,
# stray "thought" fields, etc.) must NEVER be shown to the user directly —
# this is the one and only fallback string for that case.
_COULD_NOT_ANSWER = "I wasn't able to put together a complete answer for this request. Could you try rephrasing your question?"

_MAX_FORMAT_RETRIES = 2

_CORRECTIVE_MESSAGE = (
    "Your last response was not valid — it must be a single JSON object with "
    'both "action" and "action_input" fields, e.g. '
    '{"thought": "...", "action": "final_answer", "action_input": {"answer": "..."}}. '
    "Respond again now, following that schema exactly."
)


async def _get_action(messages: list) -> dict | None:
    """Call the model and parse its action, retrying a bounded number of times if
    it doesn't follow the JSON contract (e.g. writes a "thought" but omits
    "action"/"action_input" entirely). Returns None if it never recovers —
    callers must fall back to _COULD_NOT_ANSWER, never the raw model text."""
    for attempt in range(_MAX_FORMAT_RETRIES + 1):
        raw_reply = await asyncio.to_thread(call_model, messages, 0.0, MAX_ANSWER_TOKENS)
        messages.append({"role": "assistant", "content": raw_reply})

        try:
            action_json = _parse_action(raw_reply)
            if not isinstance(action_json, dict) or "action" not in action_json:
                raise KeyError("action")
            return action_json
        except (json.JSONDecodeError, KeyError, TypeError):
            if attempt < _MAX_FORMAT_RETRIES:
                messages.append({"role": "user", "content": _CORRECTIVE_MESSAGE})

    return None


async def answer_query_agentic(mcp_client: MCPToolClient, tool_catalog: str, user_query: str) -> dict:
    """Run the agent loop: model picks tools (0 or more, in sequence) then answers."""
    today = date.today().strftime("%A, %B %d, %Y")
    system_prompt = _apply_constitution(
        AGENT_SYSTEM_PROMPT.replace("{{TOOLS}}", tool_catalog)
        .replace("{{MAX_STEPS}}", str(MAX_STEPS))
        .replace("{{TODAY}}", today)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    steps = []

    for step_num in range(MAX_STEPS):
        action_json = await _get_action(messages)
        if action_json is None:
            # Model never produced valid {action, action_input} JSON, even after
            # retries — fall back to a clean message, never raw model text.
            return {"steps": steps, "final_answer": _COULD_NOT_ANSWER}

        action = action_json["action"]
        action_input = action_json.get("action_input", {}) or {}

        if action == "final_answer":
            answer = action_input.get("answer") if isinstance(action_input, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                answer = _COULD_NOT_ANSWER
            return {"steps": steps, "final_answer": answer}

        if not isinstance(action_input, dict):
            observation = f"Error: action_input for '{action}' must be an object of arguments."
        else:
            try:
                observation = await mcp_client.call_tool(action, action_input)
            except Exception as e:
                observation = f"Error calling tool '{action}': {e}"

        steps.append({"action": action, "action_input": action_input, "observation": observation})
        messages.append({"role": "user", "content": f"Observation from {action}: {observation}"})

    # Hit MAX_STEPS without a final_answer — force one more turn that can only answer.
    messages.append(
        {
            "role": "user",
            "content": (
                f"You have used all {MAX_STEPS} allowed tool calls. Respond now with "
                'only {"thought": "...", "action": "final_answer", "action_input": '
                '{"answer": "..."}} using whatever information you have gathered.'
            ),
        }
    )
    action_json = await _get_action(messages)
    answer = None
    if action_json is not None:
        action_input = action_json.get("action_input")
        if isinstance(action_input, dict):
            answer = action_input.get("answer")

    if not isinstance(answer, str) or not answer.strip():
        answer = _COULD_NOT_ANSWER

    return {"steps": steps, "final_answer": answer}


async def main_async():
    try:
        async with MCPToolClient(MCP_SERVER_SCRIPT) as mcp_client:
            tools_result = await mcp_client.session.list_tools()
            tool_catalog = _format_tool_catalog(tools_result.tools)

            while True:
                user_input = (await asyncio.to_thread(input, "You: ")).strip()

                if user_input.lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                if not user_input:
                    continue

                try:
                    result = await answer_query_agentic(mcp_client, tool_catalog, user_input)
                    if result["steps"]:
                        tools_used = " -> ".join(s["action"] for s in result["steps"])
                        print(f"\n[tools used: {tools_used}]")
                    else:
                        print("\n[tools used: none]")
                    print(f"Assistant: {result['final_answer']}\n")

                except requests.exceptions.ConnectionError:
                    print("Could not connect to the model API server. Is uvicorn running?")

                except requests.exceptions.Timeout:
                    print("Request to the model API timed out.")

                except Exception as e:
                    print(f"Error: {e}")

    except Exception as e:
        print(f"Could not start/connect to the MCP server ({MCP_SERVER_SCRIPT}): {e}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
