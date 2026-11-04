import contextlib
import os
import time

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = 1.5


@contextlib.contextmanager
def _suppress_stderr():
    """Silence the underlying HTTP client's noisy backend-retry logging.

    ddgs's retry-across-backends chatter and its primp (Rust) HTTP client's
    own warnings write straight to the stderr file descriptor, bypassing
    Python's `logging` module — so redirecting sys.stderr alone doesn't
    catch it. Redirect fd 2 itself for the duration of the call.
    """
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)


def web_search(query: str, max_results: int = 5) -> str:
    """Search the public web via DuckDuckGo (no API key required) and return top results."""
    try:
        from ddgs import DDGS
    except ImportError:
        return "Web search error: the 'ddgs' package is not installed. Run: pip install ddgs"

    last_error = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with _suppress_stderr(), DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            last_error = None
            break
        except Exception as e:
            last_error = e
            if attempt < _MAX_ATTEMPTS:
                # DDG rate-limits are often transient (a few seconds) —
                # back off and retry before giving up on the whole query.
                time.sleep(_BACKOFF_SECONDS * attempt)

    if last_error is not None:
        return f"Web search error: {last_error}"

    if not results:
        return f"No web results found for: {query}"

    formatted = []
    for r in results:
        title = r.get("title", "Untitled")
        url = r.get("href", r.get("link", ""))
        snippet = r.get("body", r.get("snippet", ""))
        formatted.append(f"{title}\n{url}\n{snippet}")

    return "\n\n".join(formatted)
