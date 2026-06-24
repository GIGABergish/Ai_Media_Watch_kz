"""Obfuscation-robust featurization for the custom risk model (app.ml).

Turns a :class:`SignalBundle` into a :class:`RawFeatures` domain object
(:func:`extract`) and then into a fixed-width numeric vector
(:func:`vectorize`) the model consumes.

The vector layout (``INPUT_DIM = hash_dim + numeric_dim = 4096 + 32 = 4128``):

* ``[0 : hash_dim)`` — signed feature-hashing of character 3..5-grams and word
  1..2-grams of the (normalized) text + hashtags. This block is L2-normalized.
* ``[hash_dim : INPUT_DIM)`` — 32 engineered numeric features in a fixed,
  documented order (see :func:`numeric_feature_names` and DESIGN §5).

Obfuscation robustness (leetspeak / Cyrillic-Latin swaps / intra-word spacing)
comes from a deterministic *confusable-folding* normalization step, NOT from
the rule lexicons. Hashing is done with a seeded ``blake2b`` so a trained
``.npz`` is process-stable (Python's salted ``hash()`` MUST NOT be used).
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from app.ml.config import ml_config
from app.ml.types import DIMENSION_KEYS, RawFeatures
from app.pipeline.contracts import SignalBundle
from app.scoring.lexicons import VISION_PROMPTS, DIMENSION_PATTERNS

# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #
INPUT_DIM: int = ml_config.hash_dim + ml_config.numeric_dim

_HASH_DIM = ml_config.hash_dim
_CHAR_MIN = ml_config.char_ngram_min
_CHAR_MAX = ml_config.char_ngram_max
_WORD_MAX = ml_config.word_ngram_max
_NUMERIC_DIM = ml_config.numeric_dim

# Seed key for the hash so featurization is deterministic AND process-stable.
_HASH_KEY = ml_config.seed.to_bytes(8, "little", signed=False)

# --------------------------------------------------------------------------- #
# Confusable / leet folding map — folds latin lookalikes & digits toward a
# Cyrillic/letter "skeleton" so obfuscated spellings collapse together.
# --------------------------------------------------------------------------- #
_CONFUSABLE: Dict[str, str] = {
    # latin -> cyrillic lookalikes
    "a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у",
    "k": "к", "m": "м", "h": "н", "t": "т", "b": "в", "n": "п", "r": "г",
    "u": "и",
    # digits / symbols -> letters (leetspeak)
    "0": "о", "1": "и", "3": "е", "4": "ч", "5": "ѕ", "6": "б", "7": "т",
    "8": "в", "9": "9", "@": "а", "$": "ѕ",
}

# Characters considered "alpha" for run-joining (after folding): cyrillic + latin.
_ALPHA_RE = re.compile(r"[a-zа-яёәғқңөұүһіІ]", re.UNICODE)
# A separator that may sit *inside* an obfuscated word (spacing / dots / dashes /
# zero-width). Collapsing these rejoins ``к а з и н о`` / ``к.а.з.и.н.о``.
_INWORD_SEP_RE = re.compile(r"[\s.\-_*·•|/\\​‌‍⁠]+")
# Emoji / pictographic / zero-width detector (for the count feature).
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "​‌‍⁠️❤]",
    re.UNICODE,
)
_KZ_GLYPHS = set("әғқңөұүһі")
_WORD_TOKEN_RE = re.compile(r"[a-zа-яёәғқңөұүһі0-9#]+", re.UNICODE)

# Sentinel marking word boundaries inside the char-n-gram skeleton so that
# prefixes/suffixes hash distinctly from interior n-grams.
_BOUND = " "


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def _skeleton(text: str) -> str:
    """Heavily-folded view for char n-grams (maximizes obfuscation recall).

    NFKC + casefold, confusable/leet folding, then collapse separators that sit
    *between alpha characters* so spaced/dotted obfuscations rejoin into words.
    """
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).casefold()
    folded = "".join(_CONFUSABLE.get(ch, ch) for ch in t)
    # Collapse in-word separators only when flanked by alpha on both sides.
    out: List[str] = []
    i, n = 0, len(folded)
    while i < n:
        ch = folded[i]
        if _INWORD_SEP_RE.match(ch):
            j = i
            while j < n and _INWORD_SEP_RE.match(folded[j]):
                j += 1
            prev_alpha = bool(out) and _ALPHA_RE.match(out[-1])
            next_alpha = j < n and _ALPHA_RE.match(folded[j])
            # join (drop separators) inside alpha runs, else emit one space
            out.append("" if (prev_alpha and next_alpha) else " ")
            i = j
            continue
        out.append(ch)
        i += 1
    skel = "".join(out)
    # keep only alpha-ish chars + single spaces
    skel = "".join(c if (_ALPHA_RE.match(c) or c == " ") else " " for c in skel)
    return re.sub(r"\s+", " ", skel).strip()


def _word_stream(text: str) -> List[str]:
    """Lightly-folded token list (casefold + NFKC only) for word n-grams.

    Keeps real bigrams like ``гарантированный доход`` / ``guaranteed income``
    intact rather than corrupting them through the heavy confusable fold.
    """
    if not text:
        return []
    t = unicodedata.normalize("NFKC", text).casefold()
    return _WORD_TOKEN_RE.findall(t)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def _hash(token: str) -> tuple:
    """Return ``(index, sign)`` for a namespaced token via seeded blake2b.

    Low 4 bytes -> index modulo hash_dim; a high byte's low bit -> sign.
    Deterministic and process-stable (independent of PYTHONHASHSEED).
    """
    d = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=_HASH_KEY).digest()
    idx = int.from_bytes(d[:4], "little") % _HASH_DIM
    sign = 1.0 if (d[7] & 1) == 0 else -1.0
    return idx, sign


def _char_ngrams(skeleton: str) -> List[str]:
    """Char 3..5-grams over each skeleton word (boundary-padded)."""
    grams: List[str] = []
    for word in skeleton.split(" "):
        if not word:
            continue
        padded = _BOUND + word + _BOUND
        L = len(padded)
        for n in range(_CHAR_MIN, _CHAR_MAX + 1):
            if L < n:
                continue
            for i in range(L - n + 1):
                grams.append(padded[i : i + n])
    return grams


def _word_ngrams(tokens: Sequence[str]) -> List[str]:
    """Word 1.._WORD_MAX-grams over the lightly-folded token stream."""
    grams: List[str] = []
    n_tok = len(tokens)
    for n in range(1, _WORD_MAX + 1):
        for i in range(n_tok - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    return grams


def _hash_block(text: str, hashtags: Sequence[str]) -> np.ndarray:
    """Build the L2-normalized signed feature-hash block ``(hash_dim,)``.

    Three namespaced streams (``c:`` char, ``w:`` word, ``h:`` hashtag) are
    accumulated with ``np.add.at`` so they never alias and the build is fast.
    """
    block = np.zeros(_HASH_DIM, dtype=np.float32)

    skel = _skeleton(text)
    words = _word_stream(text)

    tokens: List[str] = []
    tokens.extend("c:" + g for g in _char_ngrams(skel))
    tokens.extend("w:" + g for g in _word_ngrams(words))
    # hashtags: fold each independently (skeleton, joined) for obfuscation match
    for h in hashtags:
        sk = _skeleton(h).replace(" ", "")
        if sk:
            tokens.append("h:" + sk)
            for g in _char_ngrams(_BOUND.join([sk])):
                tokens.append("c:" + g)

    if tokens:
        idxs = np.empty(len(tokens), dtype=np.int64)
        vals = np.empty(len(tokens), dtype=np.float32)
        for i, tok in enumerate(tokens):
            idx, sign = _hash(tok)
            idxs[i] = idx
            vals[i] = sign
        np.add.at(block, idxs, vals)

    norm = float(np.linalg.norm(block))
    if norm > 0.0:
        block /= norm
    return block


# --------------------------------------------------------------------------- #
# Numeric features (32 fixed slots — DESIGN §5)
# --------------------------------------------------------------------------- #
_NUMERIC_NAMES: List[str] = [
    # 0..7 — visual_scores per DIMENSION_KEYS
    *[f"visual.{k}" for k in DIMENSION_KEYS],
    "behavior.urgency",          # 8
    "behavior.referral",         # 9
    "behavior.messenger",        # 10
    "behavior.count",            # 11
    "behavior.negative",         # 12
    "link.telegram",             # 13
    "link.whatsapp",             # 14
    "link.url",                  # 15
    "link.promocode",            # 16
    "link.phone",                # 17
    "link.total",                # 18
    "kb_similarity",             # 19
    "duration_s",                # 20
    "num_segments",              # 21
    "hashtag_count",             # 22
    "hashtag.suspicious_ratio",  # 23
    "text_len",                  # 24
    "has_url",                   # 25
    "has_telegram_or_wa",        # 26
    "has_promocode",             # 27
    "digit_ratio",               # 28
    "emoji_zerowidth_count",     # 29
    "profit_density",            # 30 (spare / profit text-density)
    "lang_code",                 # 31
]
assert len(_NUMERIC_NAMES) == _NUMERIC_DIM, (
    f"numeric feature names ({len(_NUMERIC_NAMES)}) != numeric_dim ({_NUMERIC_DIM})"
)

# Static slot -> ScamDNA dna_key mapping for attribution (empty = no single dim).
_NUMERIC_DNA: List[str] = [
    *DIMENSION_KEYS,             # 0..7 identity
    "urgency",                   # 8
    "referral",                  # 9
    "messenger",                 # 10
    "",                          # 11
    "",                          # 12 negative
    "messenger",                 # 13
    "messenger",                 # 14
    "messenger",                 # 15
    "referral",                  # 16
    "messenger",                 # 17
    "messenger",                 # 18
    "reused",                    # 19
    "",                          # 20
    "",                          # 21
    "hashtags",                  # 22
    "hashtags",                  # 23
    "",                          # 24
    "messenger",                 # 25
    "messenger",                 # 26
    "referral",                  # 27
    "",                          # 28
    "",                          # 29
    "profit",                    # 30
    "",                          # 31
]

# Lowercased suspicious-hashtag anchors (skeleton-folded, '#'-stripped) from the
# teacher's hashtags lexicon — used for the suspicious-ratio feature only.
_SUSP_HASHTAGS = frozenset(
    _skeleton(ph.pattern).replace(" ", "").lstrip("#")
    for ph in DIMENSION_PATTERNS["hashtags"]
)
# Profit anchors (skeleton) for the slot-30 text-density heuristic.
_PROFIT_ANCHORS = tuple(
    _skeleton(ph.pattern).replace(" ", "")
    for ph in DIMENSION_PATTERNS["profit"]
    if not ph.is_regex and len(ph.pattern) >= 4
)

# Module-constant normalization caps (no train-time scaler -> no train/serve skew).
_LANG_CODE = {"": 0.0, "ru": 0.0, "kz": 0.33, "en": 0.66, "mixed": 1.0}
_LOG_DUR = float(np.log1p(600.0))
_LOG_LEN = float(np.log1p(2000.0))


def numeric_feature_names() -> List[str]:
    """Human-readable names of the 32 engineered numeric features (fixed order)."""
    return list(_NUMERIC_NAMES)


def _numeric_block(rf: RawFeatures) -> np.ndarray:
    """Build the 32-slot engineered numeric block (raw magnitudes, not L2-normed)."""
    v = np.zeros(_NUMERIC_DIM, dtype=np.float32)
    vis = rf.visual_scores or {}
    beh = rf.behavior_flags or {}
    lc = rf.link_counts or {}

    # 0..7 visual scores per dna dimension (already 0..1)
    for i, k in enumerate(DIMENSION_KEYS):
        v[i] = float(vis.get(k, 0.0))

    # 8..10 behavior aggregates (already 0..1, max-conf/100 set in extract)
    v[8] = float(beh.get("urgency", 0.0))
    v[9] = float(beh.get("referral", 0.0))
    v[10] = float(beh.get("messenger", 0.0))
    v[11] = min(float(beh.get("_count", 0.0)), 5.0) / 5.0
    v[12] = float(beh.get("negative", 0.0))

    # 13..18 link counts
    tg = int(lc.get("telegram", 0))
    wa = int(lc.get("whatsapp", 0))
    url = int(lc.get("url", 0))
    promo = int(lc.get("promocode", 0))
    phone = int(lc.get("phone", 0))
    total = sum(int(c) for c in lc.values())
    v[13] = min(tg, 4) / 4.0
    v[14] = min(wa, 4) / 4.0
    v[15] = min(url, 4) / 4.0
    v[16] = min(promo, 4) / 4.0
    v[17] = min(phone, 4) / 4.0
    v[18] = min(total, 8) / 8.0

    # 19 kb similarity (caller-set 0..1)
    v[19] = float(max(0.0, min(1.0, rf.kb_similarity)))

    # 20 duration, 21 segments
    v[20] = float(min(np.log1p(max(0.0, rf.duration_s)) / _LOG_DUR, 1.0))
    v[21] = min(int(rf.num_segments), 40) / 40.0

    # 22 hashtag count, 23 suspicious-hashtag ratio
    n_tags = len(rf.hashtags)
    v[22] = min(n_tags, 12) / 12.0
    if n_tags:
        susp = 0
        for h in rf.hashtags:
            sk = _skeleton(h).replace(" ", "").lstrip("#")
            if sk and any(a and (a in sk or sk in a) for a in _SUSP_HASHTAGS):
                susp += 1
        v[23] = susp / n_tags

    # 24 text length
    text = rf.text or ""
    v[24] = float(min(np.log1p(len(text)) / _LOG_LEN, 1.0))

    # 25..27 boolean link presence
    v[25] = 1.0 if url > 0 else 0.0
    v[26] = 1.0 if (tg > 0 or wa > 0) else 0.0
    v[27] = 1.0 if promo > 0 else 0.0

    # 28 digit ratio of text
    if text:
        digits = sum(1 for c in text if c.isdigit())
        v[28] = digits / len(text)

    # 29 emoji + zero-width count
    v[29] = min(len(_EMOJI_RE.findall(text)), 20) / 20.0

    # 30 profit text-density (skeleton anchor hits, capped)
    if text:
        skel = _skeleton(text)
        hits = sum(skel.count(a) for a in _PROFIT_ANCHORS if a)
        v[30] = min(hits, 5) / 5.0

    # 31 language scalar
    v[31] = _LANG_CODE.get(rf.lang_hint, 0.0)

    return v


# --------------------------------------------------------------------------- #
# Vectorization
# --------------------------------------------------------------------------- #
def vectorize(rf: RawFeatures) -> np.ndarray:
    """Turn a :class:`RawFeatures` into the model input vector ``(INPUT_DIM,)``."""
    out = np.empty(INPUT_DIM, dtype=np.float32)
    out[:_HASH_DIM] = _hash_block(rf.text, rf.hashtags)
    out[_HASH_DIM:] = _numeric_block(rf)
    return out


def vectorize_batch(batch: List[RawFeatures]) -> np.ndarray:
    """Vectorize a list of :class:`RawFeatures` into ``(N, INPUT_DIM)`` float32."""
    if not batch:
        return np.zeros((0, INPUT_DIM), dtype=np.float32)
    out = np.empty((len(batch), INPUT_DIM), dtype=np.float32)
    for i, rf in enumerate(batch):
        out[i] = vectorize(rf)
    return out


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _lang_hint(text: str) -> str:
    """Cheap Cyrillic-vs-Latin char-ratio heuristic ('' if no letters)."""
    cyr = lat = 0
    has_kz = False
    for ch in text.casefold():
        if ch in _KZ_GLYPHS:
            has_kz = True
            cyr += 1
        elif "а" <= ch <= "я" or ch == "ё":
            cyr += 1
        elif "a" <= ch <= "z":
            lat += 1
    total = cyr + lat
    if total == 0:
        return ""
    cyr_ratio = cyr / total
    if has_kz and cyr_ratio > 0.3:
        return "kz"
    if 0.25 < cyr_ratio < 0.75:
        return "mixed"
    if cyr_ratio >= 0.75:
        return "ru"
    return "en"


# dna_keys that the behavior lane / negative markers contribute as flags.
_BEHAVIOR_DNA = ("urgency", "referral", "messenger")


def extract(bundle: SignalBundle) -> RawFeatures:
    """Extract domain :class:`RawFeatures` from a :class:`SignalBundle`.

    Combines transcript + OCR + title/description/hashtags into ``.text``;
    aggregates behavior/negative flags from ``Behavior``-source (and negative)
    hits; folds ``visual_hits`` through ``VISION_PROMPTS`` into per-dna
    ``visual_scores``; counts links by kind; reads duration/segments; and infers
    ``lang_hint``. ``kb_similarity`` is left 0 for the caller to set.
    """
    media = bundle.media

    # --- combined text ---------------------------------------------------- #
    parts: List[str] = []
    if bundle.transcript and bundle.transcript.full_text:
        parts.append(bundle.transcript.full_text)
    for hit in bundle.ocr_hits:
        if hit.text:
            parts.append(hit.text)
    if media.title:
        parts.append(media.title)
    if media.description:
        parts.append(media.description)
    if media.hashtags:
        parts.append(" ".join(media.hashtags))
    text = "\n".join(parts)

    # --- behavior / negative flags --------------------------------------- #
    behavior_flags: Dict[str, float] = {}
    beh_count = 0
    for hit in bundle.hits:
        conf = float(hit.confidence) / 100.0
        is_behavior = hit.source == "Behavior"
        if is_behavior:
            beh_count += 1
        for key in hit.dna_keys:
            if key == "negative":
                behavior_flags["negative"] = max(
                    behavior_flags.get("negative", 0.0), conf)
            elif is_behavior and key in _BEHAVIOR_DNA:
                behavior_flags[key] = max(behavior_flags.get(key, 0.0), conf)
    behavior_flags["_count"] = float(beh_count)

    # --- visual scores via VISION_PROMPTS (saturating max per dna) -------- #
    label_to_dna = {vp.label: vp.dna_key for vp in VISION_PROMPTS}
    visual_scores: Dict[str, float] = {}
    for vh in bundle.visual_hits:
        dna = label_to_dna.get(vh.label, "")
        if not dna:
            continue
        visual_scores[dna] = max(visual_scores.get(dna, 0.0), float(vh.score))

    # --- link counts by kind --------------------------------------------- #
    link_counts: Dict[str, int] = dict(Counter(lh.kind for lh in bundle.link_hits))

    # --- duration / segments --------------------------------------------- #
    duration_s = float(bundle.probe.duration_s) if bundle.probe else 0.0
    num_segments = len(bundle.transcript.segments) if bundle.transcript else 0

    return RawFeatures(
        text=text,
        title=media.title or "",
        description=media.description or "",
        hashtags=list(media.hashtags),
        link_counts=link_counts,
        behavior_flags=behavior_flags,
        visual_scores=visual_scores,
        kb_similarity=0.0,
        duration_s=duration_s,
        num_segments=num_segments,
        lang_hint=_lang_hint(text),
    )


# --------------------------------------------------------------------------- #
# Explainability helper
# --------------------------------------------------------------------------- #
def top_text_features(text: str, k: int = 8) -> List[str]:
    """Return up to ``k`` representative n-grams of ``text`` for explanations.

    Surfaces lightly-folded word n-grams (most human-readable) first, then
    skeleton char n-grams, de-duplicated and ordered by frequency. Used by
    :mod:`app.ml.explain` to put a readable token next to a hashed contribution.
    """
    if not text or k <= 0:
        return []
    words = _word_stream(text)
    word_grams = _word_ngrams(words)
    skel = _skeleton(text)
    char_grams = _char_ngrams(skel)

    counts: Counter = Counter()
    # prefer longer / phrase-level cues: weight word bigrams highest
    for g in word_grams:
        counts[g] += 2 if " " in g else 1
    for g in char_grams:
        # only keep interior (non-boundary) char grams as fallback tokens
        if g.strip() and g.strip() != g.replace(_BOUND, ""):
            continue
        counts[g.strip()] += 1

    ranked = [g for g, _ in counts.most_common() if g]
    # de-dup while preserving order, drop pure substrings already present
    out: List[str] = []
    for g in ranked:
        if g in out:
            continue
        out.append(g)
        if len(out) >= k:
            break
    return out
