"""Interim two-zone prompt assembler.

M4 proper (with the full prefix-survival instrument) lands in Sprint 4. This
already enforces the property that matters — the invariant zone is byte-stable
across turns and no other module may write it — so prefix survival is measurable
from Sprint 0 rather than appearing only in week 9.
"""

from __future__ import annotations

from parsimony.core.types import AssembledPrompt, RequestContext
from parsimony.infra.tokenization import common_prefix_tokens


def assemble(ctx: RequestContext, tokenizer) -> AssembledPrompt:
    invariant_parts = []
    if ctx.system_prompt:
        invariant_parts.append(ctx.system_prompt)
    if ctx.context_digest:
        invariant_parts.append(ctx.context_digest)
    invariant_zone = "\n".join(invariant_parts)

    volatile_parts = []
    for turn in ctx.history:
        prefix = "User" if turn.role == "user" else "Assistant"
        volatile_parts.append(f"{prefix}: {turn.content}")
    volatile_parts.append(f"User: {ctx.query}")
    volatile_parts.append("Assistant:")
    volatile_zone = "\n".join(volatile_parts)

    full = f"{invariant_zone}\n\n{volatile_zone}" if invariant_zone else volatile_zone
    return AssembledPrompt(
        invariant_zone=invariant_zone,
        volatile_zone=volatile_zone,
        full_text=full,
        prefix_token_count=tokenizer.count(invariant_zone) if invariant_zone else 0,
        total_token_count=tokenizer.count(full),
    )


def prefix_survival(previous_ids: list[int] | None, current_ids: list[int]) -> tuple[int, float]:
    """Longest common TOKEN prefix against the previous request in this
    conversation — the unit llama.cpp actually reuses KV at (ADR-011)."""
    if not previous_ids or not current_ids:
        return 0, 0.0
    survived = common_prefix_tokens(previous_ids, current_ids)
    return survived, survived / len(current_ids)
