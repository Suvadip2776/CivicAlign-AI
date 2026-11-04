import ast
import math
import operator

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "factorial": math.factorial,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


class CalculatorError(Exception):
    pass


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise CalculatorError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise CalculatorError("Only basic math functions are allowed")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not supported")
        args = [_eval_node(arg) for arg in node.args]
        return _FUNCS[node.func.id](*args)

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise CalculatorError(f"Unknown identifier: {node.id}")

    raise CalculatorError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression and return the result as a string."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_node(tree.body)
    except CalculatorError as e:
        return f"Calculator error: {e}"
    except ZeroDivisionError:
        return "Calculator error: division by zero"
    except Exception as e:
        return f"Calculator error: could not evaluate expression ({e})"

    return str(result)
