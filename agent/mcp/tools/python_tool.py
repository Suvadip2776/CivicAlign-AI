import builtins
import contextlib
import io
import math
import threading

_ALLOWED_BUILTIN_NAMES = (
    "abs", "all", "any", "bin", "bool", "chr", "complex", "dict", "divmod",
    "enumerate", "filter", "float", "format", "frozenset", "hex", "int",
    "len", "list", "map", "max", "min", "oct", "ord", "pow", "print", "range",
    "reversed", "round", "set", "sorted", "str", "sum", "tuple", "zip",
)
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES}

_TIMEOUT_SECONDS = 5


def run_python(code: str) -> str:
    """Execute a small snippet of Python in a restricted namespace and return stdout/errors.

    No imports, file I/O, or access to dunder attributes are allowed. Intended for
    quick calculations/data munging, not general-purpose scripting.
    """
    if "__" in code or "import" in code:
        return "Python tool error: imports and dunder attribute access are not allowed"

    namespace = {"__builtins__": _SAFE_BUILTINS, "math": math}
    stdout = io.StringIO()
    error_holder = {}

    def _run():
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, namespace)
        except Exception as e:
            error_holder["error"] = f"{type(e).__name__}: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(_TIMEOUT_SECONDS)

    if thread.is_alive():
        return f"Python tool error: execution exceeded {_TIMEOUT_SECONDS}s timeout"

    if "error" in error_holder:
        output = stdout.getvalue()
        return (output + f"\nError: {error_holder['error']}").strip()

    output = stdout.getvalue().strip()
    return output if output else "(no output — use print() to see results)"
