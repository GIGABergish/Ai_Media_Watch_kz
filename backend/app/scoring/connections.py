"""Connection-graph builder.

Turns the analyzed :class:`SignalBundle` plus a knowledge-base similarity match
into a small :class:`Connections` graph for the UI: a central *video* node with
radially-laid-out neighbours (Telegram handles, top hashtags and known
related accounts/videos from the knowledge base), connected by typed edges.

The layout is fully deterministic — node positions are computed from the node
index with plain trigonometry, **no randomness** — so the same bundle always
renders the same graph. Pure stdlib (``math``); no optional ML deps.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from app.api.schemas import ConnectionEdge, ConnectionNode, Connections
from app.config import clamp_score
from app.pipeline.contracts import SignalBundle

__all__ = ["build_connections"]

# Layout constants — central node sits roughly mid-canvas; neighbours orbit it.
_CENTER_X = 300.0
_CENTER_Y = 200.0
_RADIUS = 160.0
# Cap how many of each neighbour kind we surface to keep the graph readable.
_MAX_TELEGRAM = 4
_MAX_HASHTAGS = 5
_MAX_RELATED = 5


# --------------------------------------------------------------------------- #
# Extraction helpers (pure)
# --------------------------------------------------------------------------- #
def _short_title(bundle: SignalBundle) -> str:
    """A compact label for the central video node."""
    title = (bundle.media.title or "").strip()
    if title:
        return title[:40]
    fname = (bundle.media.filename or "").strip()
    if fname:
        return fname[:40]
    return "Анализируемое видео"


def _telegram_handles(bundle: SignalBundle) -> List[str]:
    """Distinct Telegram/handle values from link hits, order-preserving."""
    seen: set = set()
    out: List[str] = []
    for link in bundle.link_hits:
        if link.kind in ("telegram", "handle"):
            value = link.value.strip()
            key = value.lstrip("@").lower()
            if key and key not in seen:
                seen.add(key)
                label = value if value.startswith("@") else "@" + value.lstrip("@")
                out.append(label)
        if len(out) >= _MAX_TELEGRAM:
            break
    return out


def _top_hashtags(bundle: SignalBundle) -> List[str]:
    """Top hashtags by frequency (then first-seen), normalized with a leading #."""
    counts: dict = {}
    order: List[str] = []
    for tag in bundle.media.hashtags:
        t = tag.strip()
        if not t:
            continue
        norm = "#" + t.lstrip("#").lower()
        if norm not in counts:
            counts[norm] = 0
            order.append(norm)
        counts[norm] += 1
    # Stable sort: higher frequency first, ties keep first-seen order.
    ranked = sorted(order, key=lambda t: (-counts[t], order.index(t)))
    return ranked[:_MAX_HASHTAGS]


def _radial_position(index: int, total: int) -> Tuple[float, float]:
    """Deterministic (x, y) for the ``index``-th neighbour of ``total``.

    Neighbours are spread evenly around the centre starting from the top
    (-90°) and going clockwise; a slight per-index radius wobble keeps
    overlapping labels apart without any randomness.
    """
    if total <= 0:
        return _CENTER_X, _CENTER_Y
    angle = -math.pi / 2 + (2 * math.pi * index) / total
    radius = _RADIUS + (index % 3) * 22.0
    x = _CENTER_X + radius * math.cos(angle)
    y = _CENTER_Y + radius * math.sin(angle)
    return round(x, 2), round(y, 2)


# --------------------------------------------------------------------------- #
# Public builder
# --------------------------------------------------------------------------- #
def build_connections(bundle: SignalBundle, kb, risk_score: int = 0) -> Connections:
    """Build the connection graph for ``bundle``.

    Args:
        bundle:     analyzed signal bundle (link hits + hashtags).
        kb:         knowledge base exposing ``similarity(bundle) -> dict``.
        risk_score: overall 0..100 risk, used for the central node's score.

    Returns:
        A :class:`Connections` with a central ``v1`` video node, radially
        placed neighbour nodes and typed edges from ``v1`` to each.
    """
    sim = kb.similarity(bundle) if kb is not None else {
        "score": 0, "cluster_size": 1,
        "description": "Изолированный контент — связей не обнаружено", "related": [],
    }

    nodes: List[ConnectionNode] = []
    edges: List[ConnectionEdge] = []

    # Central video node.
    central = ConnectionNode(
        id="v1",
        type="video",
        label=_short_title(bundle),
        riskScore=clamp_score(risk_score),
        x=_CENTER_X,
        y=_CENTER_Y,
    )
    nodes.append(central)

    # Gather neighbours first so we can lay them out evenly around the circle.
    # Each entry: (node_type, label, riskScore|None, edge_type).
    neighbours: List[Tuple[str, str, Optional[int], str]] = []
    seen_labels: set = set()

    def _add(node_type: str, label: str, score: Optional[int], edge_type: str) -> None:
        label = label.strip()
        if not label:
            return
        key = (node_type, label.lower())
        if key in seen_labels:
            return
        seen_labels.add(key)
        neighbours.append((node_type, label, score, edge_type))

    # Telegram handles extracted from the media itself.
    for handle in _telegram_handles(bundle):
        _add("telegram", handle, None, "telegram")

    # Top hashtags (riskScore 0 per spec).
    for tag in _top_hashtags(bundle):
        _add("hashtag", tag, 0, "hashtag")

    # Known related items from the knowledge base.
    related = sim.get("related") or []
    for item in related[:_MAX_RELATED]:
        if not isinstance(item, dict):
            continue
        ntype = item.get("type", "related")
        label = str(item.get("label", "")).strip()
        rscore = item.get("riskScore")
        score = clamp_score(rscore) if isinstance(rscore, (int, float)) else None
        if ntype == "telegram":
            _add("telegram", label, score, "telegram")
        elif ntype == "hashtag":
            _add("hashtag", label, score if score is not None else 0, "hashtag")
        elif ntype == "video":
            _add("video", label, score, "related")
        else:  # account / unknown -> account node + edge
            _add("account", label, score, "account")

    # Lay neighbours out radially and wire edges from the central node.
    total = len(neighbours)
    for i, (node_type, label, score, edge_type) in enumerate(neighbours):
        x, y = _radial_position(i, total)
        node_id = f"n{i + 1}"
        nodes.append(ConnectionNode(
            id=node_id,
            type=node_type,           # type: ignore[arg-type]
            label=label,
            riskScore=score,
            x=x,
            y=y,
        ))
        edges.append(ConnectionEdge(
            source="v1",
            target=node_id,
            type=edge_type,           # type: ignore[arg-type]
        ))

    cluster_size = int(sim.get("cluster_size", 1) or 1)
    # The graph itself contributes at least the nodes we drew; never under-report.
    cluster_size = max(cluster_size, len(nodes))

    return Connections(
        nodes=nodes,
        edges=edges,
        clusterSize=cluster_size,
        clusterDescription=str(sim.get("description", "")),
    )
