"""ScamDNA scorer — folds analyzer signals into the 8 UI dimensions.

For each dimension key we collect contributing confidences from two sources:

  * every ``AnalyzerHit`` whose ``dna_keys`` include the key (text/behavior/…);
  * every ``VisualHit`` whose CLIP label maps (via ``VISION_PROMPTS``) to the
    key, contributing ``score * 100``.

The values are combined with a SATURATING fold so that several moderate signals
reinforce each other without ever exceeding 100, while no single weak signal can
inflate the score. The strongest contributing hit also supplies the human-
readable description shown in the UI.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.api.schemas import ScamDNADimension
from app.pipeline.contracts import AnalyzerHit, SignalBundle
from app.scoring.lexicons import (
    DIMENSION_BY_KEY,
    DIMENSION_KEYS,
    VISION_PROMPTS,
)
from app.config import clamp_score

# label -> dna_key lookup for vision prompts (skip neutral / empty-key anchors).
_VISION_LABEL_TO_KEY: Dict[str, str] = {
    vp.label: vp.dna_key for vp in VISION_PROMPTS if vp.dna_key
}


def _saturating_fold(values: List[float]) -> float:
    """Combine 0..100 contributions with diminishing returns.

    Sorted descending so the strongest signal anchors the result; each further
    value adds only its share of the remaining headroom:
        combine(acc, x) = acc + x * (1 - acc / 100)
    """
    acc = 0.0
    for x in sorted((max(0.0, v) for v in values), reverse=True):
        acc = acc + x * (1.0 - acc / 100.0)
    return acc


def compute_scam_dna(bundle: SignalBundle) -> List[ScamDNADimension]:
    """Build the ordered list of 8 ScamDNA dimensions from the signal bundle."""
    # Gather (confidence, description-source) contributions per dimension.
    contrib: Dict[str, List[float]] = {k: [] for k in DIMENSION_KEYS}
    # Track the strongest hit (by confidence) feeding each dimension for desc.
    strongest: Dict[str, Tuple[float, str]] = {}

    def _consider_desc(key: str, conf: float, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        cur = strongest.get(key)
        if cur is None or conf > cur[0]:
            strongest[key] = (conf, text)

    # Text / behaviour / audio / metadata hits.
    for hit in bundle.hits:
        for key in hit.dna_keys:
            if key not in contrib:
                continue
            conf = float(hit.confidence)
            contrib[key].append(conf)
            _consider_desc(key, conf, hit.signal or hit.fragment)

    # Vision hits — map CLIP label to a dimension and use score*100.
    for vh in bundle.visual_hits:
        key = _VISION_LABEL_TO_KEY.get(vh.label)
        if not key or key not in contrib:
            continue
        conf = float(vh.score) * 100.0
        contrib[key].append(conf)
        _consider_desc(key, conf, vh.label_ru or vh.label)

    dims: List[ScamDNADimension] = []
    for key in DIMENSION_KEYS:
        meta = DIMENSION_BY_KEY[key]
        value = clamp_score(_saturating_fold(contrib[key]))
        best = strongest.get(key)
        description = best[1] if best else meta.default_desc
        dims.append(
            ScamDNADimension(
                key=key,
                name=meta.name,
                nameRu=meta.nameRu,
                value=value,
                description=description,
            )
        )
    return dims


def dna_map(dims: List[ScamDNADimension]) -> Dict[str, int]:
    """Convenience: {dimension key -> value} for downstream scorers."""
    return {d.key: d.value for d in dims}
