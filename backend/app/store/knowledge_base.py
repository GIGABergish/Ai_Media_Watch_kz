"""In-memory seed knowledge base of known scam fingerprints.

A tiny, curated, deterministic catalogue of scam *clusters* — each a bundle of
known Telegram handles, hashtags, narrative keywords and decoy account/video
labels. Given a :class:`SignalBundle`, :class:`KnowledgeBase` measures the
overlap of the input's extracted fingerprints against every cluster and returns
the best match as a similarity score plus a small list of related nodes for the
connection graph.

This module is data-only (plus pure helpers) and uses **no randomness**: the
same bundle always yields the same result. Optional ML deps are irrelevant here
— everything is stdlib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.pipeline.contracts import SignalBundle

__all__ = ["KBEntry", "RelatedRef", "KnowledgeBase", "KB"]


# --------------------------------------------------------------------------- #
# Seed records
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RelatedRef:
    """A neighbour node surfaced for the connection graph."""
    label: str
    type: str          # "video" | "account" | "hashtag" | "telegram"
    riskScore: int     # 0..100

    def as_dict(self) -> Dict[str, object]:
        return {"label": self.label, "type": self.type, "riskScore": self.riskScore}


@dataclass(frozen=True)
class KBEntry:
    """One known scam cluster fingerprint."""
    cluster_id: str
    description: str                         # Russian, shown in the UI
    cluster_size: int                        # known members in the cluster
    telegram: List[str] = field(default_factory=list)   # handles, no leading @
    hashtags: List[str] = field(default_factory=list)   # with leading #
    keywords: List[str] = field(default_factory=list)   # narrative fragments
    accounts: List[str] = field(default_factory=list)   # known account names
    related: List[RelatedRef] = field(default_factory=list)


# Three curated clusters. Handles/hashtags/keywords are normalized lazily at
# match time, so authoring them in their natural form is fine.
SEED_CLUSTERS: List[KBEntry] = [
    KBEntry(
        cluster_id="slots_casino",
        description=(
            "Кластер «слоты/казино»: продвижение онлайн-казино и слот-машин "
            "через обещания лёгкого выигрыша, промокоды на бонус и воронку в "
            "Telegram-канал. Совпадает с известной сетью рекламы 1xbet/vavada."
        ),
        cluster_size=37,
        telegram=["slots_bonus_vip", "casino_promo_official", "vavada_freespin"],
        hashtags=["#казино", "#слоты", "#ставки", "#casino", "#slots", "#bonus"],
        keywords=[
            "казино", "слот", "слоты", "рулетка", "джекпот", "фриспин",
            "бонус казино", "крутить барабан", "1xbet", "vavada", "mostbet",
            "промокод на бонус",
        ],
        accounts=["casino_luck_official", "slots_king_top"],
        related=[
            RelatedRef("@slots_bonus_vip", "telegram", 92),
            RelatedRef("casino_luck_official", "account", 88),
            RelatedRef("#казино", "hashtag", 85),
            RelatedRef("«Занос на 2 млн в слотах»", "video", 90),
        ],
    ),
    KBEntry(
        cluster_id="investment_club",
        description=(
            "Кластер «инвестиционный клуб / пирамида»: закрытый клуб с "
            "гарантированной доходностью, реферальной программой и вербовкой "
            "новых участников. Типовой нарратив финансовой пирамиды."
        ),
        cluster_size=24,
        telegram=["invest_club_private", "profit_team_vip", "money_mentor_ru"],
        hashtags=[
            "#инвестиции", "#пассивныйдоход", "#доход", "#заработок",
            "#схема", "#trading",
        ],
        keywords=[
            "закрытый клуб", "закрытый канал", "приватный канал",
            "гарантированный доход", "гарантия прибыли", "реферальная ссылка",
            "приведи друга", "пригласи друзей", "сетевой маркетинг", "млм",
            "пирамид", "система заработка", "проверенная схема",
            "пассивный доход",
        ],
        accounts=["invest_mentor_pro", "finance_freedom_club"],
        related=[
            RelatedRef("@invest_club_private", "telegram", 90),
            RelatedRef("invest_mentor_pro", "account", 86),
            RelatedRef("#пассивныйдоход", "hashtag", 70),
            RelatedRef("«Как я вышел на пассив за месяц»", "video", 84),
        ],
    ),
    KBEntry(
        cluster_id="passive_income_dm",
        description=(
            "Кластер «пассивный доход — пиши +»: ролики с призывом написать "
            "слово или «+» в директ/личку для перехода в мессенджер в обход "
            "модерации. Массовая схема вербовки в воронку."
        ),
        cluster_size=18,
        telegram=["easy_money_dm", "passive_income_bot", "work_from_home_vip"],
        hashtags=[
            "#заработок", "#удаленнаяработа", "#пассивныйдоход",
            "#без_вложений", "#easymoney", "#деньги",
        ],
        keywords=[
            "пиши +", "пиши в директ", "пиши в лс", "в личку", "напиши мне",
            "пиши плюс", "ссылка в профиле", "ссылка в шапке",
            "без вложений", "доход из дома", "удаленная работа",
            "легкие деньги", "заработок без усилий",
        ],
        accounts=["remote_income_2024", "mom_earns_online"],
        related=[
            RelatedRef("@easy_money_dm", "telegram", 80),
            RelatedRef("remote_income_2024", "account", 74),
            RelatedRef("#easymoney", "hashtag", 75),
            RelatedRef("«Зарабатываю из дома, пиши +»", "video", 78),
        ],
    ),
]


# --------------------------------------------------------------------------- #
# Normalization helpers (pure)
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    """Lowercase + diacritic-naive normalization for stable comparison."""
    return s.strip().lower().replace("ё", "е")


def _norm_handle(s: str) -> str:
    """Telegram handle without leading @ / t.me/, lowercased."""
    s = _norm(s)
    for prefix in ("https://", "http://", "t.me/", "telegram.me/", "@"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    return s.strip("/").strip()


def _norm_tag(s: str) -> str:
    """Hashtag normalized to a leading-#, lowercased, no underscores."""
    s = _norm(s).lstrip("#").replace("_", "")
    return "#" + s if s else ""


# --------------------------------------------------------------------------- #
# Fingerprint extraction from a bundle (pure, no ML)
# --------------------------------------------------------------------------- #
def _extract_telegram(bundle: SignalBundle) -> set:
    out = set()
    for link in bundle.link_hits:
        if link.kind in ("telegram", "handle"):
            h = _norm_handle(link.value)
            if h:
                out.add(h)
    return out


def _extract_hashtags(bundle: SignalBundle) -> set:
    out = set()
    for tag in bundle.media.hashtags:
        t = _norm_tag(tag)
        if t:
            out.add(t)
    # Hashtags embedded in free text (title/description/transcript/ocr).
    for token in _norm(bundle.all_text()).split():
        if token.startswith("#"):
            t = _norm_tag(token)
            if t:
                out.add(t)
    return out


def _extract_text(bundle: SignalBundle) -> str:
    return _norm(bundle.all_text())


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
class KnowledgeBase:
    """Curated scam-fingerprint store with deterministic similarity scoring."""

    # Weights for blending the three overlap signals into one 0..1 score.
    _W_TELEGRAM = 0.45
    _W_HASHTAG = 0.30
    _W_KEYWORD = 0.25
    # Minimum blended score to consider a cluster a real match.
    _MIN_MATCH = 0.04

    def __init__(self, clusters: List[KBEntry] | None = None) -> None:
        self.clusters: List[KBEntry] = clusters if clusters is not None else SEED_CLUSTERS

    # ---- per-cluster scoring ------------------------------------------- #
    def _score_cluster(
        self,
        entry: KBEntry,
        tg: set,
        tags: set,
        text: str,
    ) -> float:
        """Blended 0..1 overlap of the input against a single cluster."""
        cl_tg = {_norm_handle(h) for h in entry.telegram}
        cl_tags = {_norm_tag(h) for h in entry.hashtags}

        tg_sim = _jaccard(tg, cl_tg)
        tag_sim = _jaccard(tags, cl_tags)

        # Keyword overlap: fraction of the cluster's narrative fragments that
        # appear as substrings in the input text (asymmetric coverage).
        kw = [_norm(k) for k in entry.keywords if k.strip()]
        kw_hits = sum(1 for k in kw if k and k in text)
        kw_sim = (kw_hits / len(kw)) if kw else 0.0

        return (
            self._W_TELEGRAM * tg_sim
            + self._W_HASHTAG * tag_sim
            + self._W_KEYWORD * kw_sim
        )

    # ---- public API ----------------------------------------------------- #
    def similarity(self, bundle: SignalBundle) -> dict:
        """Best-matching cluster for ``bundle``.

        Returns a dict with keys:
          * ``score``       : int 0..100 similarity to the best cluster
          * ``cluster_size``: int known members of that cluster (>=1)
          * ``description`` : Russian description of the match
          * ``related``     : list[{label,type,riskScore}] neighbour nodes
        """
        tg = _extract_telegram(bundle)
        tags = _extract_hashtags(bundle)
        text = _extract_text(bundle)

        best: KBEntry | None = None
        best_raw = 0.0
        for entry in self.clusters:
            raw = self._score_cluster(entry, tg, tags, text)
            if raw > best_raw:
                best_raw = raw
                best = entry

        if best is None or best_raw < self._MIN_MATCH:
            return {
                "score": 0,
                "cluster_size": 1,
                "description": "Изолированный контент — связей не обнаружено",
                "related": [],
            }

        # Map the blended 0..1 overlap to a 0..100 score. The blend rarely
        # reaches 1.0 (asymmetric coverage), so scale and clamp for a usable
        # spread without ever exceeding 100.
        score = int(max(0, min(100, round(best_raw * 130))))

        return {
            "score": score,
            "cluster_size": best.cluster_size,
            "description": best.description,
            "related": [r.as_dict() for r in best.related],
        }


# Module singleton — import and reuse everywhere.
KB = KnowledgeBase()
