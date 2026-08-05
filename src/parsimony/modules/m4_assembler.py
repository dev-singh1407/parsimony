"""M4 — Prefix-Stable Prompt Assembler.

Gap 4: local runtimes reuse the key-value cache for any prompt prefix identical
to the previous request, so a stable prefix is the single largest latency win
available on CPU — and it is entirely invisible to a token counter. Every
technique in the literature rewrites the prompt, and a rewrite near the front
destroys that reuse. The system then sends fewer tokens and takes longer.

WHAT THE ABLATED ARM MODELS
---------------------------
When M4 is off the pipeline uses `assemble_volatile_head`, which puts a small
per-request preamble ("Turn 3 of 5") at the very front. That is not a strawman:
prepending a turn counter, a timestamp or a freshly-recomputed context block is
a widespread pattern in chat frameworks, and it drops prefix reuse to zero on
every single turn while looking completely harmless in a token count.

M4's contribution is therefore near-zero in tokens and potentially large in
seconds, which is exactly the point Contribution 3 makes. Any evaluation that
reports only token reduction would score M4 as useless.
"""

from __future__ import annotations

from parsimony.core.config import ParsimonyConfig
from parsimony.core.proposals import ContextPatch, NoOp, Proposal, TransformKind
from parsimony.core.types import AssembledPrompt, RequestContext


def _render_turns(ctx: RequestContext) -> str:
    parts = []
    for turn in ctx.history:
        speaker = "User" if turn.role == "user" else "Assistant"
        parts.append(f"{speaker}: {turn.content}")
    parts.append(f"User: {ctx.query}")
    parts.append("Assistant:")
    return "\n".join(parts)


def assemble_prefix_stable(ctx: RequestContext, token_count) -> AssembledPrompt:
    """Two zones. The invariant zone is byte-stable across every turn of a
    conversation and no other module may write it — enforced structurally,
    because `system_prompt` and `context_digest` appear in no other stage's
    `writes` set and the registry validates that at boot."""
    invariant_parts = [p for p in (ctx.system_prompt, ctx.context_digest) if p]
    invariant_zone = "\n".join(invariant_parts)
    volatile_zone = _render_turns(ctx)
    full = f"{invariant_zone}\n\n{volatile_zone}" if invariant_zone else volatile_zone
    return AssembledPrompt(
        invariant_zone=invariant_zone,
        volatile_zone=volatile_zone,
        full_text=full,
        prefix_token_count=token_count(invariant_zone) if invariant_zone else 0,
        total_token_count=token_count(full),
    )


def assemble_volatile_head(ctx: RequestContext, token_count) -> AssembledPrompt:
    """The ablated arm: a per-request preamble at position zero."""
    preamble = f"[Turn {ctx.turn_index + 1}, {len(ctx.history)} prior messages]"
    invariant_parts = [p for p in (ctx.system_prompt, ctx.context_digest) if p]
    head = "\n".join([preamble, *invariant_parts])
    volatile_zone = _render_turns(ctx)
    full = f"{head}\n\n{volatile_zone}"
    return AssembledPrompt(
        invariant_zone="",  # nothing is invariant: the preamble changes every turn
        volatile_zone=full,
        full_text=full,
        prefix_token_count=0,
        total_token_count=token_count(full),
    )


class PrefixStableAssembler:
    module_id = "M4"
    name = "m4_assembler"
    reads = frozenset({"query", "history", "system_prompt", "context_digest"})
    writes = frozenset({"assembled"})

    def applies_to(self, ctx: RequestContext, cfg: ParsimonyConfig) -> bool:
        return cfg.enables("M4")

    def propose(self, ctx: RequestContext, cfg: ParsimonyConfig) -> Proposal:
        d = ctx.derived
        if d is None:
            return NoOp("not_applicable", "no derived cache")
        prompt = assemble_prefix_stable(ctx, d.token_count)
        return ContextPatch(
            kind=TransformKind.AUGMENT,  # adds a rendering; removes nothing
            fields={"assembled": prompt},
            rationale=f"pinned {prompt.prefix_token_count} invariant tokens to the prompt head",
            evidence={
                "invariant_tokens": prompt.prefix_token_count,
                "total_tokens": prompt.total_token_count,
            },
        )


def stages() -> list:
    return [PrefixStableAssembler()]
