"""Synthetic + adversarial training-data generator for the custom risk model.

This module is the data engine behind the weak-supervision DISTILLATION described
in ``app/ml/DESIGN.md`` (§2, §7). It emits realistic, varied short-video text
across the seven ``CATEGORY_KEYS`` categories, in Russian (primary) plus Kazakh,
English and code-mix, and augments scam rows with **obfuscation** and
**paraphrase** so the student learns the de-obfuscation invariance the lexicons
provably lack.

Labeling is HYBRID (DESIGN §1):
  * synthetic GROUND TRUTH owns ``risk`` / ``is_scam`` / ``category``
    (known by construction, ``Label.source = "synthetic"``);
  * the rule TEACHER, run on the **clean** pre-obfuscation text, refines the 8
    ScamDNA ``dimensions`` (it is reliable on clean text even when its diluted
    overall score is low). Obfuscated twins REUSE their clean parent's dimension
    targets (meaning unchanged, only the surface is corrupted).

If ``weak_labels`` is importable the teacher refines dimensions; otherwise we fall
back to cheap template-derived dimension targets so this module never hard-fails
on its parallel sibling.

Determinism: a single ``np.random.Generator`` seeded from ``seed`` threads through
every category / template / slot / augmentation coin-flip. No time-based RNG.

Public contract::

    generate(n: int, seed: int) -> list[Example]
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.ml.types import (
    CATEGORY_KEYS,
    DIMENSION_KEYS,
    Example,
    Label,
    RawFeatures,
)

# Lazy/optional teacher for dimension refinement. Imported defensively so synth
# still works while weak_labels is being built in parallel.
try:  # pragma: no cover - import guard
    from app.ml import weak_labels as _weak_labels  # type: ignore
except Exception:  # noqa: BLE001
    _weak_labels = None  # type: ignore


# --------------------------------------------------------------------------- #
# 1. Typed phrase banks (RU primary + KZ/EN). Anchored to lexicon-detectable
#    cues PLUS many non-lexicon paraphrases so the model must generalize.
# --------------------------------------------------------------------------- #
# NB: we deliberately COPY a handful of representative lexicon anchors (so clean
# rows are teacher-detectable) and AUTHOR many novel paraphrases beside them.

# Profit / "guaranteed income" cues -- intensity used for risk tiering.
PROFIT_HARD: List[str] = [
    "гарантированный доход каждый день",
    "гарантия прибыли без рисков",
    "100% доход уже завтра",
    "доход 30% в день, проверено",
    "заработок без вложений и без усилий",
    "это легкие деньги буквально из воздуха",
    "пассивный доход пока ты спишь",
    "озолотишься за неделю",
    "guaranteed income with zero risk",
    "risk-free passive income every single day",
    "кепілдік табыс күн сайын",
]
PROFIT_SOFT: List[str] = [
    "стабильный заработок на автопилоте",
    "плюс к зарплате каждый месяц",
    "доходность выше любого банка",
    "деньги начнут капать сами",
    "выходишь на пассив за месяц",
    "прибыль растет как на дрожжах",
    "финансовая подушка без головной боли",
    "make money while you sleep",
    "ақшаны еш қиындықсыз табасың",
]

# Gambling cues.
GAMBLING_HARD: List[str] = [
    "заходи в казино и крути слоты",
    "рулетка приносит джекпот каждый вечер",
    "ставки на спорт через 1xbet",
    "бонус казино на первый депозит",
    "vavada дает фриспины новичкам",
    "mostbet занос дня, лови джекпот",
    "игровой автомат отдает, лови момент",
    "spin the slots and win the jackpot",
    "free spins at the casino tonight",
]
GAMBLING_SOFT: List[str] = [
    "вечерний занос на барабанах",
    "поймал крутую комбинацию на ставках",
    "фартовый вечер у автоматов",
    "сегодня барабаны щедрые",
    "испытай удачу на ставках",
    "casino night, lucky spins",
]

# Referral / pyramid cues.
REFERRAL_HARD: List[str] = [
    "приведи друга и получи бонус",
    "реферальная ссылка в профиле",
    "регистрируйся по моей ссылке",
    "приведи 3 человек и заработаешь",
    "наш сетевой маркетинг меняет жизни",
    "вступай в команду, это не пирамида",
    "бонус за каждого приглашенного",
    "use my referral link to start",
    "invite friends and earn together",
]
REFERRAL_SOFT: List[str] = [
    "зови знакомых в проект, всем хватит",
    "чем больше людей в структуре, тем выше доход",
    "строим команду мечты вместе",
    "партнерская программа для своих",
    "твоя ветка партнеров растет за тебя",
    "grow your downline, earn passively",
]

# Messenger funnel cues.
MESSENGER_HARD: List[str] = [
    "пиши в директ за деталями",
    "переходи в телеграм по ссылке в шапке",
    "вступай в закрытый канал",
    "пиши + в личку, расскажу схему",
    "ссылка на приватный чат в профиле",
    "пиши в whatsapp прямо сейчас",
    "dm me for the private group link",
    "join the closed channel in bio",
]
MESSENGER_SOFT: List[str] = [
    "детали скину в личные сообщения",
    "залетай в наш тесный чатик",
    "напиши, и я открою доступ",
    "все подробности в закрытом клубе",
    "ping me, I'll add you to the group",
]

# Urgency cues.
URGENCY_HARD: List[str] = [
    "только сегодня, успей записаться",
    "осталось 3 места, последний шанс",
    "закрываем набор сегодня вечером",
    "действует до вечера, не упусти",
    "ограниченное предложение для первых",
    "limited spots, act now",
    "last chance, only today",
]
URGENCY_SOFT: List[str] = [
    "поторопись, окно закрывается",
    "мест почти не осталось",
    "решай быстро, потом не будет",
    "успевай, пока двери открыты",
    "don't miss out, hurry",
]

# Visual-manipulation textual cues (CLIP carries the rest).
VISUAL_HARD: List[str] = [
    "вот скриншот выплат за вчера",
    "реальные выплаты, держи доказательство дохода",
    "скрин поступления на kaspi",
    "график доходности за неделю",
    "here is a screenshot of my payout",
]

# Scam hashtag clusters.
HASHTAGS_SCAM: List[List[str]] = [
    ["#заработок", "#доход", "#безвложений"],
    ["#казино", "#ставки", "#бонус"],
    ["#пассивныйдоход", "#схема", "#деньги"],
    ["#easymoney", "#trading", "#крипта"],
    ["#slots", "#casino", "#jackpot"],
    ["#инвестиции", "#биткоин", "#пассивный_доход"],
]
HASHTAGS_BENIGN: List[List[str]] = [
    ["#рецепты", "#готовим", "#ужин"],
    ["#путешествия", "#природа", "#отдых"],
    ["#книги", "#саморазвитие", "#мотивация"],
    ["#финграмотность", "#бюджет", "#накопления"],
    ["#новости", "#аналитика", "#общество"],
    ["#cooking", "#travel", "#lifestyle"],
    [],
]

# Investment-scam specific framing (profit + secret-scheme narrative).
INVEST_HARD: List[str] = [
    "секретная схема инвестиций для своих",
    "проверенная система заработка на крипте",
    "вложи и получай гарантированный пассивный доход",
    "трейдинг без рисков по моему сигналу",
    "крипто-схема удваивает депозит за неделю",
    "secret investment scheme, guaranteed returns",
]

# --- Benign banks (hard negatives + neutral) ------------------------------- #
BENIGN_NEUTRAL: List[str] = [
    "сегодня готовлю борщ по бабушкиному рецепту",
    "показываю утреннюю пробежку вдоль реки",
    "разбираю новую книгу по истории",
    "как я навожу порядок на кухне за десять минут",
    "прогулка по осеннему парку, делюсь видами",
    "обзор погоды на выходные в нашем городе",
    "учу новые слова на английском, делюсь методом",
    "собираю рюкзак в поход, список вещей",
    "today I am baking bread at home, simple recipe",
    "бүгін таңғы асқа көже пісіріп жатырмын",
    " remont in the kitchen, before and after",
    "новости спорта: итоги вчерашнего матча",
]

# Educational financial-literacy content (lexically scam-like but negatively
# marked) -- the documented false-positive mode.
BENIGN_EDU: List[str] = [
    "объясняю простыми словами, как работает сложный процент",
    "финансовая грамотность: почему важна подушка безопасности",
    "разбираем, как составить личный бюджет без стресса",
    "образовательный ролик про инвестиции и реальные риски",
    "помните: прошлая доходность не гарантирует будущую",
    "инвестиции несут риски, можно потерять часть денег",
    "это не финансовый совет, думайте своей головой",
    "финграмотность для новичков: с чего начать копить",
    "this is not financial advice, past performance matters",
    "қаржылық сауаттылық: бюджетті қалай жоспарлау керек",
]

# Anti-fraud / explainer content (negative markers dominate).
BENIGN_ANTIFRAUD: List[str] = [
    "осторожно мошенники: разбираем популярную схему развода",
    "как не попасться на обещания гарантированного дохода",
    "как распознать мошенника в телеграме за минуту",
    "разоблачаю фейковые скриншоты выплат из казино",
    "разбираем признаки финансовой пирамиды на примере",
    "не переходите по ссылкам из таких сообщений",
    "не доверяйте обещаниям легких денег без рисков",
    "how to avoid scam offers promising guaranteed income",
    "beware of scam channels asking you to deposit money",
    "алаяқтардан сақ болыңыз: жалған сұлбаны талдаймыз",
]

# Borderline / suspicious_other -- one weak ambiguous signal only.
BORDERLINE: List[str] = [
    "рассказываю про подработку в свободное время",
    "нашел интересный способ откладывать деньги",
    "залетайте в мой чат, общаемся на разные темы",
    "делюсь личным опытом дополнительного заработка",
    "новый проект, подробности будут позже",
    "found a neat side hustle, will share more soon",
]

# Sentence connectors / openers to vary phrasing.
OPENERS: List[str] = [
    "Друзья,", "Слушай,", "Внимание!", "Ребят,", "Привет!",
    "Smотри,", "Короче,", "", "", "Эй,",
]
CLOSERS: List[str] = [
    "не благодари.", "погнали!", "это работает.", "проверь сам.",
    "репостни друзьям.", "", "", "let's go.",
]


# --------------------------------------------------------------------------- #
# 2. Obfuscation + paraphrase augmentation (DESIGN §7.2)
# --------------------------------------------------------------------------- #
# Cyrillic -> Latin homoglyph swaps (visually identical glyphs).
_HOMOGLYPH: Dict[str, str] = {
    "а": "a", "о": "o", "е": "e", "р": "p", "с": "c",
    "х": "x", "к": "k", "м": "m", "н": "h", "т": "t",
    "в": "b", "у": "y",
}
# Digit-for-letter leetspeak.
_LEET: Dict[str, str] = {
    "о": "0", "з": "3", "э": "3", "и": "1", "ч": "4", "а": "@", "б": "6",
}
_EMOJI: List[str] = ["🔥", "💰", "🤑", "✅", "💸", "🚀", "🎰", "⚡", "💎"]


def _swap_homoglyph(word: str, rng: np.random.Generator, p: float = 0.5) -> str:
    """Randomly swap a few Cyrillic letters for Latin lookalikes."""
    return "".join(
        _HOMOGLYPH[ch] if (ch in _HOMOGLYPH and rng.random() < p) else ch
        for ch in word
    )


def _leet(word: str, rng: np.random.Generator, p: float = 0.45) -> str:
    """Replace some letters with digit/symbol leetspeak."""
    return "".join(
        _LEET[ch] if (ch in _LEET and rng.random() < p) else ch for ch in word
    )


def _space_out(word: str, rng: np.random.Generator) -> str:
    """Insert separators inside a word: ``к а з и н о`` / ``к.а.з.и.н.о``."""
    if len(word) < 3:
        return word
    sep = rng.choice([" ", ".", "-", "​"])  # last is zero-width space
    return sep.join(word)


def _duplicate_chars(word: str, rng: np.random.Generator, p: float = 0.3) -> str:
    """Selective character duplication (``казиноо``)."""
    out = []
    for ch in word:
        out.append(ch)
        if ch.isalpha() and rng.random() < p:
            out.append(ch)
    return "".join(out)


def _obfuscate_word(word: str, rng: np.random.Generator) -> str:
    """Apply a capped number of obfuscation techniques to one token."""
    techniques: List[Callable[[str, np.random.Generator], str]] = []
    if rng.random() < 0.6:
        techniques.append(_swap_homoglyph)
    if rng.random() < 0.5:
        techniques.append(_leet)
    if rng.random() < 0.25:
        techniques.append(_duplicate_chars)
    if rng.random() < 0.30:
        techniques.append(_space_out)
    # Cap simultaneous techniques per word at 2 to stay human-readable.
    rng.shuffle(techniques)
    out = word
    for fn in techniques[:2]:
        out = fn(out, rng)
    return out


def _obfuscate_text(text: str, rng: np.random.Generator) -> str:
    """Obfuscate ~half the longer tokens + inject a few emoji between tokens."""
    tokens = text.split(" ")
    new_tokens: List[str] = []
    for tok in tokens:
        if len(tok) >= 4 and rng.random() < 0.55:
            new_tokens.append(_obfuscate_word(tok, rng))
        else:
            new_tokens.append(tok)
        if rng.random() < 0.12:
            new_tokens.append(_EMOJI[int(rng.integers(len(_EMOJI)))])
    return " ".join(new_tokens)


# Code-mix clauses to splice into RU SCAM text (orthogonal to obfuscation).
_CODEMIX_CLAUSES: List[str] = [
    "кепілдік табыс",          # KZ: guaranteed income
    "guaranteed income",       # EN
    "casino bonus tonight",    # EN
    "ақшаны тез табасың",      # KZ: earn money fast
    "risk-free profit",        # EN
    "тегін айналдыру бар",     # KZ: free spins
]

# Neutral code-mix clauses spliced into BENIGN rows at the SAME rate, so the
# language scalar never becomes a spurious scam predictor (DESIGN §7.4).
_CODEMIX_BENIGN: List[str] = [
    "good morning everyone",   # EN
    "қайырлы таң достар",      # KZ: good morning friends
    "let's cook together",     # EN
    "бүгін ауа райы жақсы",    # KZ: nice weather today
    "stay healthy",            # EN
    "кітап оқып жатырмын",     # KZ: I'm reading a book
]


# --------------------------------------------------------------------------- #
# 3. Template assembly per category
# --------------------------------------------------------------------------- #
def _join(parts: List[str], rng: np.random.Generator) -> str:
    """Join non-empty slot fills with light opener/closer variation."""
    body = " ".join(p for p in parts if p).strip()
    opener = OPENERS[int(rng.integers(len(OPENERS)))]
    closer = CLOSERS[int(rng.integers(len(CLOSERS)))]
    pieces = [opener, body, closer]
    return " ".join(p for p in pieces if p).strip()


def _pick(bank: List[str], rng: np.random.Generator) -> str:
    return bank[int(rng.integers(len(bank)))]


def _maybe_codemix(
    text: str, rng: np.random.Generator, bank: List[str], p: float = 0.18
) -> str:
    """Splice a KZ/EN clause from ``bank`` into RU text at probability ``p``."""
    if rng.random() < p:
        clause = bank[int(rng.integers(len(bank)))]
        return f"{text} {clause}"
    return text


# Each builder returns (clean_text, hashtags, n_themes) where n_themes counts the
# independent scam slots composed (drives intensity-tiered risk).
def _build_gambling(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    parts = [_pick(GAMBLING_HARD if rng.random() < 0.7 else GAMBLING_SOFT, rng)]
    themes = 1
    if rng.random() < 0.55:
        parts.append(_pick(MESSENGER_HARD, rng))
        themes += 1
    if rng.random() < 0.45:
        parts.append(_pick(URGENCY_HARD, rng))
        themes += 1
    if rng.random() < 0.35:
        parts.append(_pick(PROFIT_HARD, rng))
        themes += 1
    tags = _pick(HASHTAGS_SCAM, rng)
    return _join(parts, rng), tags, themes


def _build_pyramid(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    parts = [_pick(REFERRAL_HARD if rng.random() < 0.75 else REFERRAL_SOFT, rng)]
    themes = 1
    if rng.random() < 0.55:
        parts.append(_pick(PROFIT_HARD if rng.random() < 0.6 else PROFIT_SOFT, rng))
        themes += 1
    if rng.random() < 0.5:
        parts.append(_pick(MESSENGER_HARD, rng))
        themes += 1
    if rng.random() < 0.35:
        parts.append(_pick(URGENCY_HARD, rng))
        themes += 1
    tags = _pick(HASHTAGS_SCAM, rng)
    return _join(parts, rng), tags, themes


def _build_investment(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    base = _pick(INVEST_HARD, rng) if rng.random() < 0.6 else _pick(PROFIT_HARD, rng)
    parts = [base]
    themes = 1
    if rng.random() < 0.5:
        parts.append(_pick(VISUAL_HARD, rng))
        themes += 1
    if rng.random() < 0.55:
        parts.append(_pick(MESSENGER_HARD, rng))
        themes += 1
    if rng.random() < 0.4:
        parts.append(_pick(URGENCY_HARD, rng))
        themes += 1
    tags = _pick(HASHTAGS_SCAM, rng)
    return _join(parts, rng), tags, themes


def _build_passive_dm(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    """Passive-income-DM scam -> mapped to investment_scam or suspicious_other."""
    parts = [_pick(PROFIT_SOFT if rng.random() < 0.6 else PROFIT_HARD, rng),
             _pick(MESSENGER_HARD, rng)]
    themes = 2
    if rng.random() < 0.4:
        parts.append(_pick(REFERRAL_SOFT, rng))
        themes += 1
    tags = _pick(HASHTAGS_SCAM, rng)
    return _join(parts, rng), tags, themes


def _build_educational(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    parts = [_pick(BENIGN_EDU, rng)]
    # Combine two distinct literacy points for lexical variety + unique strings.
    if rng.random() < 0.6:
        extra = _pick(BENIGN_EDU, rng)
        if extra not in parts:
            parts.append(extra)
    return _join(parts, rng), _pick(HASHTAGS_BENIGN, rng), 0


def _build_antifraud(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    parts = [_pick(BENIGN_ANTIFRAUD, rng)]
    # antifraud explainers often quote the scam vocabulary they warn against.
    if rng.random() < 0.4:
        parts.append("например: " + _pick(PROFIT_HARD + GAMBLING_HARD, rng))
    return _join(parts, rng), _pick(HASHTAGS_BENIGN, rng), 0


def _build_neutral(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    parts = [_pick(BENIGN_NEUTRAL, rng)]
    if rng.random() < 0.5:
        extra = _pick(BENIGN_NEUTRAL, rng)
        if extra not in parts:
            parts.append(extra)
    return _join(parts, rng), _pick(HASHTAGS_BENIGN, rng), 0


def _build_borderline(rng: np.random.Generator) -> Tuple[str, List[str], int]:
    return _join([_pick(BORDERLINE, rng)], rng), _pick(HASHTAGS_BENIGN, rng), 1


# Intent -> (builder, ground-truth category, is_scam, base risk by theme count).
_BUILDERS: Dict[str, Callable[[np.random.Generator], Tuple[str, List[str], int]]] = {
    "illegal_gambling": _build_gambling,
    "financial_pyramid": _build_pyramid,
    "investment_scam": _build_investment,
    "passive_dm": _build_passive_dm,
    "educational": _build_educational,
    "educational_antifraud": _build_antifraud,
    "no_violation": _build_neutral,
    "suspicious_other": _build_borderline,
}


# --------------------------------------------------------------------------- #
# 4. Intensity-tiered risk targets (DESIGN §7.3)
# --------------------------------------------------------------------------- #
def _risk_for(intent: str, n_themes: int, rng: np.random.Generator) -> float:
    """Smooth ground-truth risk in [0,1] by intent + composed scam-theme count."""
    j = rng.random()
    if intent in ("no_violation",):
        return 0.02 + 0.06 * j
    if intent in ("educational", "educational_antifraud"):
        return 0.05 + 0.10 * j
    if intent == "suspicious_other":
        return 0.45 + 0.17 * j
    # scam intents: single-theme ~0.70-0.82, multi-theme stacked up to ~0.97.
    if n_themes <= 1:
        return 0.70 + 0.12 * j
    if n_themes == 2:
        return 0.80 + 0.10 * j
    return 0.85 + 0.12 * j


def _category_for(intent: str) -> str:
    """Map a generator intent to a ground-truth CATEGORY_KEYS label."""
    if intent == "passive_dm":
        # Passive-income-DM: most read as investment_scam, contract allows other.
        return "investment_scam"
    return intent


def _template_dimensions(intent: str, n_themes: int) -> Dict[str, float]:
    """Cheap fallback dimension targets when the teacher is unavailable."""
    d = {k: 0.0 for k in DIMENSION_KEYS}
    if intent == "illegal_gambling":
        d["gambling"] = 0.85
        d["messenger"] = 0.45
        d["urgency"] = 0.35
    elif intent == "financial_pyramid":
        d["referral"] = 0.85
        d["profit"] = 0.55
        d["messenger"] = 0.4
    elif intent in ("investment_scam", "passive_dm"):
        d["profit"] = 0.8
        d["visual"] = 0.4
        d["messenger"] = 0.45
        d["reused"] = 0.4
    elif intent == "suspicious_other":
        d["profit"] = 0.3
        d["messenger"] = 0.3
    # benign intents leave dims ~0
    d["hashtags"] = 0.4 if intent in (
        "illegal_gambling", "financial_pyramid", "investment_scam", "passive_dm"
    ) else 0.0
    return d


def _refine_dimensions(
    clean_text: str,
    hashtags: List[str],
    intent: str,
    n_themes: int,
) -> Dict[str, float]:
    """Teacher-refined dimensions on CLEAN text; fall back to template targets."""
    if _weak_labels is not None:
        try:
            lab = _weak_labels.weak_label_from_text(clean_text, hashtags=hashtags)
            dims = {k: float(lab.dimensions.get(k, 0.0)) for k in DIMENSION_KEYS}
            # Guard against an all-zero teacher read (e.g. paraphrase missed): if
            # the teacher saw nothing on a scam row, keep a soft template floor.
            if any(v > 0.05 for v in dims.values()):
                return dims
        except Exception:  # noqa: BLE001
            pass
    return _template_dimensions(intent, n_themes)


# --------------------------------------------------------------------------- #
# 5. Language hint heuristic (cheap Cyrillic-vs-Latin ratio + KZ glyphs)
# --------------------------------------------------------------------------- #
_KZ_GLYPHS = set("әғқңөұүһі")


def _lang_hint(text: str) -> str:
    """Heuristic lang code matching featurize.extract's expectations."""
    low = text.lower()
    if any(ch in _KZ_GLYPHS for ch in low):
        return "kz"
    cyr = sum(1 for ch in low if "а" <= ch <= "я" or ch == "ё")
    lat = sum(1 for ch in low if "a" <= ch <= "z")
    if cyr == 0 and lat == 0:
        return ""
    if cyr and lat and min(cyr, lat) / max(cyr, lat) > 0.25:
        return "mixed"
    return "ru" if cyr >= lat else "en"


# --------------------------------------------------------------------------- #
# 6. Sampling plan -- balanced scam/benign across categories + language mix
# --------------------------------------------------------------------------- #
# Within the SCAM half (~50%): balance the four scam intents.
_SCAM_INTENTS: List[str] = [
    "illegal_gambling", "financial_pyramid", "investment_scam", "passive_dm",
]
# Within the BENIGN half (~50%): over-represent hard negatives (edu+antifraud
# ~60% of benign), some neutral, a slice of borderline suspicious_other.
_BENIGN_PLAN: List[Tuple[str, float]] = [
    ("educational", 0.28),
    ("educational_antifraud", 0.32),
    ("no_violation", 0.28),
    ("suspicious_other", 0.12),
]


def _sample_intent(rng: np.random.Generator) -> str:
    """Pick a generator intent honoring the ~50/50 scam-benign balance."""
    # ~0.38 base scam rate: obfuscated twins (added below at ~45% of scam rows)
    # lift the realized scam share back toward the ~0.50 target.
    if rng.random() < 0.38:
        return _SCAM_INTENTS[int(rng.integers(len(_SCAM_INTENTS)))]
    r = rng.random()
    acc = 0.0
    for intent, frac in _BENIGN_PLAN:
        acc += frac
        if r <= acc:
            return intent
    return _BENIGN_PLAN[-1][0]


def _apply_language(
    text: str, hashtags: List[str], intent: str, rng: np.random.Generator
) -> str:
    """Maybe code-mix (scam) to honor RU~70 / KZ~10 / EN~10 / mix~10 distribution.

    Benign banks already span RU/KZ/EN at the same rate so language never becomes
    a spurious scam predictor; here we only add code-mix splicing on scam rows.
    """
    if intent in _SCAM_INTENTS:
        return _maybe_codemix(text, rng, _CODEMIX_CLAUSES, p=0.22)
    # Benign rows get neutral code-mix at the SAME rate so the language scalar
    # carries no scam signal (DESIGN §7.4).
    return _maybe_codemix(text, rng, _CODEMIX_BENIGN, p=0.22)


# --------------------------------------------------------------------------- #
# 7. Example construction
# --------------------------------------------------------------------------- #
def _make_features(
    text: str, hashtags: List[str], intent: str
) -> RawFeatures:
    """Build a text-only RawFeatures the student consumes (surface = ``text``)."""
    return RawFeatures(
        text=text,
        title="",
        description=text,
        hashtags=list(hashtags),
        link_counts={},
        behavior_flags={},
        visual_scores={},
        kb_similarity=0.0,
        duration_s=0.0,
        num_segments=0,
        lang_hint=_lang_hint(text),
    )


def _make_label(
    intent: str,
    n_themes: int,
    dimensions: Dict[str, float],
    risk: float,
    rng: np.random.Generator,
) -> Label:
    """Synthetic ground-truth label (risk/is_scam/category) + refined dims."""
    is_scam = intent in _SCAM_INTENTS
    # Teacher-margin-style weight: confident rows (deep in their tier) pull
    # harder; borderline rows are down-weighted. Floor so none is dropped.
    if intent == "suspicious_other":
        weight = 0.5 + 0.2 * rng.random()
    elif is_scam:
        weight = 0.85 + 0.15 * rng.random()
    else:
        weight = 0.8 + 0.2 * rng.random()
    return Label(
        risk=float(risk),
        dimensions=dimensions,
        category=_category_for(intent),
        is_scam=is_scam,
        source="synthetic",
        weight=float(weight),
    )


# --------------------------------------------------------------------------- #
# 8. Public entry point
# --------------------------------------------------------------------------- #
def generate(n: int, seed: int) -> List[Example]:
    """Generate ``n`` deterministic synthetic training Examples.

    Balanced ~50/50 scam vs benign across the seven categories, RU primary with
    KZ/EN/code-mix, scam rows augmented with paired (clean, obfuscated) twins that
    share the SAME ground-truth label. Synthetic ground truth owns
    risk/is_scam/category; the teacher (on clean text) refines the 8 dimensions.

    Deterministic: identical ``(n, seed)`` yields an identical list. Returns a
    list of length ~``n`` (an obfuscated twin replaces, not adds, so the count is
    honored — twins are emitted in place of fresh draws at the configured rate).
    """
    rng = np.random.default_rng(int(seed))
    examples: List[Example] = []
    seen: set[str] = set()
    idx = 0
    # Cap attempts so dedup pressure never loops forever on a small bank.
    attempts = 0
    max_attempts = n * 20

    while len(examples) < n and attempts < max_attempts:
        attempts += 1
        intent = _sample_intent(rng)
        builder = _BUILDERS[intent]
        clean_text, hashtags, n_themes = builder(rng)
        clean_text = _apply_language(clean_text, hashtags, intent, rng)

        if not clean_text.strip():
            continue
        # Dedup on the text+hashtag surface so benign rows with the same body but
        # different hashtag clusters survive; a clean/obf pair differs by text.
        key = clean_text + "\x00" + " ".join(hashtags)
        if key in seen:
            continue

        # Hybrid labeling: teacher refines dims on CLEAN text; synth owns risk.
        dims = _refine_dimensions(clean_text, hashtags, intent, n_themes)
        risk = _risk_for(intent, n_themes, rng)
        label = _make_label(intent, n_themes, dims, risk, rng)

        seen.add(key)
        clean_feats = _make_features(clean_text, hashtags, intent)
        examples.append(
            Example(id=f"syn-{idx:06d}", features=clean_feats, label=label)
        )
        idx += 1
        if len(examples) >= n:
            break

        # Obfuscated twin: ~45% of scam rows. Shares the SAME label (dims reused,
        # risk/category unchanged); only the surface text is corrupted so the
        # student learns the de-obfuscation invariance the lexicon lacks.
        if intent in _SCAM_INTENTS and rng.random() < 0.45:
            obf_text = _obfuscate_text(clean_text, rng)
            obf_key = obf_text + "\x00" + " ".join(hashtags)
            if obf_text and obf_text != clean_text and obf_key not in seen:
                seen.add(obf_key)
                obf_feats = _make_features(obf_text, hashtags, intent)
                # Reuse the parent's label verbatim (clean-text dims preserved).
                twin_label = replace(label)
                examples.append(
                    Example(id=f"syn-{idx:06d}", features=obf_feats,
                            label=twin_label)
                )
                idx += 1

    return examples[:n]


__all__ = ["generate"]
