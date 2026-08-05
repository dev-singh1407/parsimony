"""Prefix-survival instrumentation.

Assembly itself lives in modules/m4_assembler.py (it is an ablatable module).
This holds only the measurement, which is instrumentation and therefore always
on regardless of whether M4 is enabled — measuring prefix survival in the
ablated arm is the entire point of having an ablated arm.
"""

from __future__ import annotations

from parsimony.infra.tokenization import common_prefix_tokens


def prefix_survival(previous_ids: list[int] | None, current_ids: list[int]) -> tuple[int, float]:
    """Longest common TOKEN prefix against the previous request in this
    conversation.

    Tokens, not bytes (ADR-011): llama.cpp reuses the KV cache at token
    granularity, so a byte measure over-reports across multi-byte boundaries and
    under-reports when a byte edit leaves the token sequence unchanged. Since
    prefix survival is Contribution 3's headline number, measuring it in the
    wrong unit would undermine the contribution.
    """
    if not previous_ids or not current_ids:
        return 0, 0.0
    survived = common_prefix_tokens(previous_ids, current_ids)
    return survived, survived / len(current_ids)
