"""Loading an M7 PolicyBundle into a live pipeline.

Lives at L3 rather than inside M7 because it touches both the bundle (L2)
and the Pipeline (L3). Putting it in the module made M7 import M2 directly,
and modules that import each other cannot be ablated independently: disabling
one would not remove its effect, and the factorial cell would be a lie.
"""

from __future__ import annotations

from parsimony.modules.m2_cache import SemanticCache
from parsimony.modules.m7_learner import PolicyBundle


def warm_start(pipeline, bundle: PolicyBundle) -> int:
    """Pre-populate the cache from a bundle. Returns entries seeded.

    Same code path as a live cache write, so a warm-started run differs from a
    cold one only in what the cache already contains (Gap 6, Figure 6).
    """
    seeded = 0
    for question, answer in bundle.cache_seed:
        key = SemanticCache.make_key(question, "root", pipeline.cfg.model.name)
        vec = pipeline.embedder.embed([question])[0] if pipeline.embedder else None
        pipeline.cache.store(
            key, question, answer, chain="root",
            model_id=pipeline.cfg.model.name, vec=vec,
        )
        seeded += 1
    return seeded
