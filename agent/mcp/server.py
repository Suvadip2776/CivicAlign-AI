import os
import sys
from pathlib import Path

# This directory is named "mcp", same as the installed mcp SDK. Run this file
# directly (`python mcp/server.py`, not `python -m mcp.server`) so Python puts
# this directory — not its parent — at the front of sys.path; the plain
# (non-relative) imports below then resolve to mcp/tools/ without shadowing
# the installed "mcp" package.
from mcp.server.fastmcp import FastMCP

from tools.calculator import calculate
from tools.calendar_tool import get_upcoming_meetings
from tools.councilcount_tool import council_count as _council_count
from tools.legislative_text import search_legislative_text
from tools.legistar import search_legislation
from tools.python_tool import run_python
from tools.resources import lookup_resource as _lookup_resource
from tools.url_classifier import classify_url
from tools.web_search import web_search

mcp = FastMCP("nycc-tools")

# Set PUBLIC_DEPLOYMENT=1 to drop tools that should not be reachable by anyone
# with the link. python_exec runs code in-process: its guard rejects "import"
# and "__" in the source, but a format string can still assemble "__class__" at
# runtime, and its 5s timeout returns an error without stopping the thread. That
# is an acceptable risk among trusted colleagues, not on an open URL.
PUBLIC_DEPLOYMENT = os.getenv("PUBLIC_DEPLOYMENT", "").lower() in {"1", "true", "yes"}

BASE_DIR = Path(__file__).resolve().parent.parent


@mcp.tool()
def hello(name: str) -> str:
    """Simple test tool."""
    return f"Hello {name}"


@mcp.tool()
def lookup_resource(keyword: str) -> str:
    """Search NYC Council resources by keyword."""
    return _lookup_resource(keyword)


@mcp.tool()
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '2 * (3 + 4) / sqrt(16)'."""
    return calculate(expression)


if not PUBLIC_DEPLOYMENT:

    @mcp.tool()
    def python_exec(code: str) -> str:
        """Run a small restricted Python snippet (no imports/file I/O) and return its stdout."""
        return run_python(code)


@mcp.tool()
def legistar_search(query: str) -> str:
    """Search NYC Council bills and local laws by keyword and return a short list of
    matches with file number, status, sponsor and committee. Use this for "what
    legislation exists about X"; use legislative_text to read what a bill says."""
    return search_legislation(query)


@mcp.tool()
def upcoming_meetings(days_ahead: int = 14) -> str:
    """List NYC Council meetings and hearings scheduled in the next N days, with
    committee, date, time, location and a link, from the Council's public calendar."""
    return get_upcoming_meetings(days_ahead)


@mcp.tool()
def internet_search(query: str) -> str:
    """Search the public web for current information not covered by internal resources."""
    return web_search(query)


@mcp.tool()
def url_classifier(url: str) -> str:
    """Match a URL against known NYC Council resources (url_resources.yaml) by domain."""
    return classify_url(url)


@mcp.tool()
def council_count(query: str) -> str:
    """Census demographics for NYC Council districts: population, poverty, employment,
    health insurance, foreign-born, veterans and ~190 other American Community Survey
    measures. Name a district number (e.g. "district 31") for a district figure, or ask
    without one for the citywide total."""
    return _council_count(query)


@mcp.tool()
def legislative_text(query: str) -> str:
    """Read what NYC Council legislation actually says about a topic: returns the legal
    text and plain-language summary of the most relevant bills. Use this for "what does
    the law say about X" or to summarize legislation; use legistar_search to just list
    which bills exist."""
    return search_legislative_text(query)


if __name__ == "__main__":
    # stdout is reserved for the MCP stdio protocol stream — log to stderr instead.
    print("MCP server running", file=sys.stderr)
    mcp.run()