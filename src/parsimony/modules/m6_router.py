"""M6 — Escalation Router, tier 0 (deterministic).

The tier the routing literature omits: answering correctly with zero model
tokens. Tier 0 uses exact evaluation, never an LLM, so its answers are correct
by construction — a stronger claim than any 1B model can make on the same input.

Runs before the cache (ADR-016): a regex match costs ~0.1ms against the cache's
~5ms embedding pass, so `2+2` should never reach an embedding model.
"""

from __future__ import annotations

import ast
import operator as _op
import re
from datetime import date, datetime, timedelta
from fractions import Fraction

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import NoOp, Proposal, ShortCircuit
from parsimony.core.types import RequestContext, RouteTier

# --------------------------------------------------------------------------
# Safe arithmetic. Never eval(): this parses to an AST and evaluates a
# whitelisted node set, with explicit guards so a pathological expression
# cannot burn CPU or memory.
# --------------------------------------------------------------------------

_BIN_OPS = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv,
    ast.Mod: _op.mod,
    ast.Pow: _op.pow,
}
_UNARY_OPS = {ast.USub: _op.neg, ast.UAdd: _op.pos}

MAX_EXPONENT = 64
MAX_MAGNITUDE = Fraction(10) ** 30


class ArithmeticError_(Exception):
    pass


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ArithmeticError_("non-numeric constant")
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > MAX_EXPONENT:
                raise ArithmeticError_("exponent out of safe range")
            result = Fraction(left) ** int(right)
        elif isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ArithmeticError_("division by zero")
        else:
            result = Fraction(_BIN_OPS[type(node.op)](left, right))
        if abs(result) > MAX_MAGNITUDE:
            raise ArithmeticError_("result magnitude out of safe range")
        return result
    raise ArithmeticError_(f"unsupported expression node: {type(node).__name__}")


def safe_arithmetic(expr: str) -> Fraction:
    if len(expr) > 200:
        raise ArithmeticError_("expression too long")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree)


def format_number(value: Fraction, places: int = 6) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    as_float = float(value)
    text = f"{as_float:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


# --------------------------------------------------------------------------
# Query shapes
# --------------------------------------------------------------------------

_LEAD_RE = re.compile(
    r"^\s*(?:what(?:'s| is)|whats|calculate|compute|solve|evaluate|"
    r"how much is|how many is|find|give me|tell me)\s+",
    re.IGNORECASE,
)
_EXPR_RE = re.compile(r"^[\d\s+\-*/^().,%]+$")
_PERCENT_OF_RE = re.compile(
    r"^\s*(?P<pct>\d+(?:\.\d+)?)\s*%\s*(?:of|off)\s+(?P<base>\d+(?:\.\d+)?)\s*$",
    re.IGNORECASE,
)

# Unit conversion: everything to a base unit, then out again.
_UNIT_TABLE: dict[str, tuple[str, Fraction]] = {
    "km": ("length", Fraction(1000)),
    "kilometre": ("length", Fraction(1000)),
    "kilometer": ("length", Fraction(1000)),
    "kilometres": ("length", Fraction(1000)),
    "kilometers": ("length", Fraction(1000)),
    "m": ("length", Fraction(1)),
    "metre": ("length", Fraction(1)),
    "meter": ("length", Fraction(1)),
    "metres": ("length", Fraction(1)),
    "meters": ("length", Fraction(1)),
    "cm": ("length", Fraction(1, 100)),
    "mm": ("length", Fraction(1, 1000)),
    "mi": ("length", Fraction(1609344, 1000)),
    "mile": ("length", Fraction(1609344, 1000)),
    "miles": ("length", Fraction(1609344, 1000)),
    "ft": ("length", Fraction(3048, 10000)),
    "feet": ("length", Fraction(3048, 10000)),
    "foot": ("length", Fraction(3048, 10000)),
    "in": ("length", Fraction(254, 10000)),
    "inch": ("length", Fraction(254, 10000)),
    "inches": ("length", Fraction(254, 10000)),
    "kg": ("mass", Fraction(1000)),
    "kilogram": ("mass", Fraction(1000)),
    "kilograms": ("mass", Fraction(1000)),
    "g": ("mass", Fraction(1)),
    "gram": ("mass", Fraction(1)),
    "grams": ("mass", Fraction(1)),
    "mg": ("mass", Fraction(1, 1000)),
    "lb": ("mass", Fraction(45359237, 100000)),
    "lbs": ("mass", Fraction(45359237, 100000)),
    "pound": ("mass", Fraction(45359237, 100000)),
    "pounds": ("mass", Fraction(45359237, 100000)),
    "oz": ("mass", Fraction(28349523125, 1000000000)),
    "hour": ("time", Fraction(3600)),
    "hours": ("time", Fraction(3600)),
    "hr": ("time", Fraction(3600)),
    "minute": ("time", Fraction(60)),
    "minutes": ("time", Fraction(60)),
    "min": ("time", Fraction(60)),
    "second": ("time", Fraction(1)),
    "seconds": ("time", Fraction(1)),
    "sec": ("time", Fraction(1)),
    "day": ("time", Fraction(86400)),
    "days": ("time", Fraction(86400)),
}
_UNIT_ALT = "|".join(sorted(_UNIT_TABLE, key=len, reverse=True))
_CONVERT_RE = re.compile(
    rf"(?:convert\s+)?(?P<val>\d+(?:\.\d+)?)\s*(?P<from>{_UNIT_ALT})\b"
    rf"\s*(?:to|in|into)\s+(?P<to>{_UNIT_ALT})\b",
    re.IGNORECASE,
)
_TEMP_RE = re.compile(
    r"(?:convert\s+)?(?P<val>-?\d+(?:\.\d+)?)\s*(?:°\s*)?(?P<from>c|f|celsius|fahrenheit)\b"
    r"\s*(?:to|in|into)\s+(?:°\s*)?(?P<to>c|f|celsius|fahrenheit)\b",
    re.IGNORECASE,
)
_DATE_DIFF_RE = re.compile(
    r"(?:how many\s+)?days?\s+(?:between|from)\s+(?P<a>\d{4}-\d{2}-\d{2})"
    r"\s*(?:and|to|until)\s*(?P<b>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)
_DATE_ADD_RE = re.compile(
    r"(?P<n>\d{1,5})\s+days?\s+(?P<dir>after|before)\s+(?P<d>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def _strip_lead(text: str) -> str:
    return _LEAD_RE.sub("", text.strip()).strip().rstrip("?").strip()


def solve(query: str) -> tuple[str, str] | None:
    """Return (answer, handler) if a deterministic handler applies, else None."""
    body = _strip_lead(query)

    m = _PERCENT_OF_RE.match(body)
    if m:
        pct = Fraction(m.group("pct"))
        base = Fraction(m.group("base"))
        return format_number(pct * base / 100), "percent"

    m = _TEMP_RE.search(body)
    if m:
        val = Fraction(m.group("val"))
        src, dst = m.group("from")[0].lower(), m.group("to")[0].lower()
        if src != dst:
            out = val * Fraction(9, 5) + 32 if src == "c" else (val - 32) * Fraction(5, 9)
        else:
            out = val
        return f"{format_number(out, 2)} °{dst.upper()}", "temperature"

    m = _CONVERT_RE.search(body)
    if m:
        src_dim, src_factor = _UNIT_TABLE[m.group("from").lower()]
        dst_dim, dst_factor = _UNIT_TABLE[m.group("to").lower()]
        if src_dim == dst_dim:
            out = Fraction(m.group("val")) * src_factor / dst_factor
            return f"{format_number(out)} {m.group('to')}", "unit_conversion"

    m = _DATE_DIFF_RE.search(body)
    if m:
        try:
            a = datetime.strptime(m.group("a"), "%Y-%m-%d").date()
            b = datetime.strptime(m.group("b"), "%Y-%m-%d").date()
            return str(abs((b - a).days)), "date_diff"
        except ValueError:
            pass

    m = _DATE_ADD_RE.search(body)
    if m:
        try:
            base_date: date = datetime.strptime(m.group("d"), "%Y-%m-%d").date()
            delta = timedelta(days=int(m.group("n")))
            out_date = base_date + delta if m.group("dir").lower() == "after" else base_date - delta
            return out_date.isoformat(), "date_add"
        except ValueError:
            pass

    expr = body.replace("×", "*").replace("÷", "/").replace("^", "**").replace(",", "")
    if expr and _EXPR_RE.match(body.replace(",", "")) and any(c.isdigit() for c in expr):
        if any(o in expr for o in "+-*/%"):
            try:
                return format_number(safe_arithmetic(expr)), "arithmetic"
            except (ArithmeticError_, SyntaxError, ValueError, ZeroDivisionError, TypeError):
                return None
    return None


class DeterministicRouterStage:
    module_id = "M6"
    name = "m6a_deterministic"
    reads = frozenset({"query"})
    writes = frozenset()

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M6") and cfg.router.deterministic_tier

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        result = solve(ctx.query)
        if result is None:
            return NoOp("not_applicable", "no deterministic handler matched")
        answer, handler = result
        return ShortCircuit(
            response=answer,
            served_by=RouteTier.DETERMINISTIC,
            rationale=f"answered exactly by the {handler} handler",
            evidence={"handler": handler, "model_tokens": 0},
        )
