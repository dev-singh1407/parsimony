"""ULID-style identifiers: 48-bit timestamp + 80 bits of randomness, Crockford
base32. Lexicographically sortable by creation time, so the ledger gets
chronological ordering from its primary key with no extra index.
"""

from __future__ import annotations

import os
import time

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # Crockford: no I, L, O, U


def _encode(value: int, length: int) -> str:
    out = []
    for _ in range(length):
        out.append(_ALPHABET[value & 0x1F])
        value >>= 5
    return "".join(reversed(out))


def ulid(now_ms: int | None = None) -> str:
    ts = now_ms if now_ms is not None else int(time.time() * 1000)
    rand = int.from_bytes(os.urandom(10), "big")
    return _encode(ts, 10) + _encode(rand, 16)
