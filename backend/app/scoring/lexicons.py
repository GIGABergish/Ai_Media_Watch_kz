"""Knowledge core — the scam-pattern lexicons and rule tables shared by every
analyzer and scorer. Languages: Russian (primary), Kazakh, English.

This is deliberately data-only (plus tiny pure helpers) so it can be tuned by
analysts without touching engine logic. All matching is case-insensitive and
diacritic-naive; patterns are plain substrings unless wrapped in ``RX(...)``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Pattern, Tuple


# --------------------------------------------------------------------------- #
# ScamDNA dimensions — the 8 axes the UI renders (order matters; mirrors cases).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DimensionMeta:
    key: str
    name: str        # English label
    nameRu: str      # Russian label (shown in UI)
    default_desc: str


DIMENSIONS: List[DimensionMeta] = [
    DimensionMeta("profit", "Guaranteed Profit", "Гарантированная прибыль",
                  "Обещания гарантированного дохода без предупреждений о рисках"),
    DimensionMeta("urgency", "Urgency Pressure", "Давление срочности",
                  "Искусственный дефицит и призывы действовать немедленно"),
    DimensionMeta("gambling", "Gambling Markers", "Визуальные маркеры казино",
                  "Интерфейсы слотов, рулетки, ставок и анимации выигрышей"),
    DimensionMeta("referral", "Referral Scheme", "Реферальная схема",
                  "Призывы приводить новых участников, промокоды, аффилиатные ссылки"),
    DimensionMeta("messenger", "Messenger Funnel", "Воронка в мессенджер",
                  "Перевод аудитории в Telegram/WhatsApp в обход модерации"),
    DimensionMeta("visual", "Visual Manipulation", "Визуальная манипуляция",
                  "Фейковые скриншоты выплат, поддельные графики доходности"),
    DimensionMeta("reused", "Reused Content", "Повторяющийся шаблон",
                  "Совпадение нарратива с известными схемами в базе знаний"),
    DimensionMeta("hashtags", "Suspicious Hashtags", "Подозрительные хэштеги",
                  "Кластеры хэштегов, таргетирующих ищущих лёгкий заработок"),
]

DIMENSION_KEYS = [d.key for d in DIMENSIONS]
DIMENSION_BY_KEY = {d.key: d for d in DIMENSIONS}


# --------------------------------------------------------------------------- #
# Pattern helpers
# --------------------------------------------------------------------------- #
class RX(str):
    """Marker subclass: a pattern wrapped in RX() is treated as a regex."""


@dataclass(frozen=True)
class Phrase:
    pattern: str          # substring (default) or regex (if is_regex)
    weight: int           # 0..100 contribution to its dimension
    is_regex: bool = False
    label: str = ""       # short RU chip for findings (defaults to the match)

    def compiled(self) -> Pattern:
        body = self.pattern if self.is_regex else re.escape(self.pattern)
        return re.compile(body, re.IGNORECASE | re.UNICODE)


def P(pattern: str, weight: int, label: str = "") -> Phrase:
    return Phrase(pattern=pattern, weight=weight,
                  is_regex=isinstance(pattern, RX), label=label)


# --------------------------------------------------------------------------- #
# Dimension lexicons.  key -> list[Phrase]
# --------------------------------------------------------------------------- #
DIMENSION_PATTERNS = {
    "profit": [
        P("гарантирован", 90, "гарантированный доход"),
        P("гарантия дохода", 92), P("гарантия прибыли", 92),
        P("100% доход", 88), P(RX(r"\b100\s*%\b"), 70, "100%"),
        P("без рисков", 85, "«без рисков»"), P("без риска", 85),
        P("без вложений", 80, "«без вложений»"),
        P("пассивный доход", 60, "пассивный доход"),
        P(RX(r"\d[\d\s]{2,}\s*(₸|тенге|руб|рублей|\$|долларов)\s*(в день|в неделю|в месяц|/день|/нед)"),
          88, "обещание суммы дохода"),
        P(RX(r"\d{1,3}\s*%\s*(в день|в неделю|в месяц|годовых)"), 90, "нереальная доходность"),
        P("доход каждый день", 82), P("заработок без усилий", 78),
        P("деньги из воздуха", 80), P("легкие деньги", 70), P("лёгкие деньги", 70),
        P("озолотишься", 65), P("разбогатеешь", 65),
        # KZ
        P("кепілдік", 80, "кепілдік (гарантия)"), P("табыс", 40, "табыс (доход)"),
        # EN
        P("guaranteed income", 88), P("guaranteed profit", 90),
        P("risk-free", 82), P("get rich", 70), P("passive income", 55),
    ],
    "urgency": [
        P("осталось", 55), P(RX(r"осталось\s+\d+\s+мест"), 85, "«осталось N мест»"),
        P("только сегодня", 82, "«только сегодня»"), P("успей", 70, "«успей»"),
        P("успей до", 80), P("последний шанс", 80), P("спешите", 72),
        P("мест почти нет", 82), P("мест осталось мало", 80),
        P("ограниченное предложение", 75), P("только для первых", 78),
        P("действует до вечера", 78), P("закрываем набор", 76),
        P("не упусти", 70), P("прямо сейчас", 55),
        # EN
        P("limited spots", 80), P("only today", 80), P("act now", 78),
        P("last chance", 80), P("hurry", 65),
    ],
    "gambling": [
        P("казино", 85, "казино"), P("слот", 80, "слоты"), P("слоты", 82),
        P("рулетк", 82, "рулетка"), P("ставк", 70, "ставки"), P("букмекер", 75),
        P("джекпот", 82), P("выигрыш", 65, "выигрыш"), P("крутить барабан", 80),
        P("игровой автомат", 85), P("бонус казино", 85), P("фриспин", 80),
        P("депозит", 50), P("азартн", 70),
        P("1xbet", 88), P("melbet", 88), P("mostbet", 88), P("vavada", 88),
        P("pin-up", 85), P("pinup", 85),
        # EN
        P("casino", 85), P("slots", 80), P("jackpot", 82), P("betting", 70),
        P("free spins", 80),
    ],
    "referral": [
        P("промокод", 80, "промокод"), P(RX(r"промокод\s+[A-Za-z0-9]{3,}"), 88, "промокод"),
        P("реферальн", 88, "реферальная ссылка"), P("реферал", 82),
        P("приведи друга", 88), P(RX(r"приведи\s+\d+"), 90, "«приведи N человек»"),
        P("пригласи друзей", 82), P("бонус за приглашение", 85),
        P("аффилиат", 80), P("партнерская ссылка", 70), P("партнёрская ссылка", 70),
        P("по моей ссылке", 78), P("регистрируйся по ссылке", 80),
        P("сетевой маркетинг", 75), P("млм", 78), P("mlm", 78),
        P("пирамид", 80, "признаки пирамиды"),
        # EN
        P("referral", 85), P("invite friends", 80), P("promo code", 80),
        P("use my link", 78), P("affiliate", 78),
    ],
    "messenger": [
        P("пиши в директ", 85, "«пиши в директ»"), P("в личку", 78, "«в личку»"),
        P("пиши в лс", 80), P("напиши мне", 65), P("пиши +", 82, "«пиши +»"),
        P(RX(r"пиши[те]?\s*[\"«]?\+"), 84, "«пиши +»"),
        P("переходи в телеграм", 85), P("переходи в telegram", 85),
        P("жми на ссылку в профиле", 78), P("ссылка в шапке", 70),
        P("ссылка в профиле", 70), P("закрытый канал", 82, "закрытый канал"),
        P("закрытый чат", 80), P("приватный канал", 82), P("закрытый клуб", 78),
        P("вступай в канал", 78), P("подпишись на канал", 60),
        P("пиши в whatsapp", 85), P("пиши в ватсап", 85), P("пиши в телеграм", 85),
        # EN
        P("dm me", 80), P("link in bio", 72), P("join the channel", 70),
        P("private group", 78),
    ],
    "visual": [
        P("скриншот перевода", 82), P("скрин поступления", 82),
        P("скриншот выплат", 84), P("банковская выписка", 70),
        P("kaspi", 55, "скрин Kaspi"), P("чек о зачислении", 78),
        P("реальные выплаты", 80), P("доказательство дохода", 78),
        P("график доходности", 65), P("мои результаты", 55),
        # NB: most "visual manipulation" is supplied by the CLIP vision lane;
        # these textual cues complement it.
        P("fake", 50), P("фотошоп", 70),
    ],
    "reused": [
        # 'reused' is driven mostly by knowledge-base similarity, not lexicon;
        # a few narrative fingerprints help the cold-start case.
        P("секретная схема", 70, "«секретная схема»"), P("секретный метод", 68),
        P("рабочая схема", 65), P("проверенная схема", 65),
        P("система заработка", 60), P("секрет богатых", 62),
    ],
    "hashtags": [
        P("#заработок", 70), P("#доход", 68), P("#казино", 85), P("#ставки", 80),
        P("#инвестиции", 45), P("#пассивныйдоход", 65), P("#пассивный_доход", 65),
        P("#без_вложений", 75), P("#схема", 70), P("#деньги", 55), P("#крипта", 55),
        P("#биткоин", 55), P("#trading", 50), P("#easymoney", 75),
        P("#slots", 82), P("#casino", 85), P("#bonus", 60),
    ],
}


# --------------------------------------------------------------------------- #
# Negative markers — risk-REDUCING signals (educational / anti-fraud content).
# A match subtracts from text/behavior components and flags educational intent.
# --------------------------------------------------------------------------- #
NEGATIVE_MARKERS = [
    P("инвестиции несут риски", 30, "предупреждение о рисках"),
    P("несут риски", 25, "«несут риски»"),
    P("можно потерять", 28, "«можно потерять»"),
    P("не гарантирует", 30, "«не гарантирует»"),
    P("прошлая доходность не гарантирует", 35),
    P("это риск", 18), P("помните о рисках", 28),
    P("осторожно мошенники", 40, "антимошеннический контекст"),
    P("как не попасться", 35), P("как распознать мошенник", 35),
    P("разбираем признаки мошенничества", 38),
    P("не переходите по ссылкам", 30), P("не доверяйте обещаниям", 30),
    P("образовательн", 25), P("финансовая грамотность", 25),
    P("финграмотность", 25), P("разоблач", 30),
    P("предупрежда", 22),
    # EN
    P("not financial advice", 20), P("past performance", 25),
    P("how to avoid scam", 35), P("beware of scam", 35),
]


# --------------------------------------------------------------------------- #
# Category classification rules. First rule whose condition holds wins.
# `dna` is the {key: value} dict; `text` is the lowercased full text.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CategoryRule:
    category: str
    categoryRu: str
    # condition knobs (all optional):
    min_dna: Tuple[Tuple[str, int], ...] = ()   # require dna[key] >= v
    any_keywords: Tuple[str, ...] = ()
    is_educational: bool = False


CATEGORY_RULES: List[CategoryRule] = [
    # Educational / anti-fraud takes precedence when negative markers dominate.
    CategoryRule("educational_antifraud", "Безопасный антимошеннический контент",
                 any_keywords=("осторожно мошенники", "как не попасться",
                               "разоблач", "как распознать"),
                 is_educational=True),
    CategoryRule("educational", "Образовательный контент",
                 any_keywords=("финграмотность", "финансовая грамотность",
                               "образовательн", "объясняю простыми словами"),
                 is_educational=True),
    CategoryRule("illegal_gambling", "Нелегальный игорный бизнес",
                 min_dna=(("gambling", 55),)),
    CategoryRule("financial_pyramid", "Признаки финансовой пирамиды",
                 min_dna=(("referral", 70),)),
    CategoryRule("investment_scam", "Подозрительная инвестиционная схема",
                 min_dna=(("profit", 60),)),
]

DEFAULT_CATEGORY = ("suspicious_other", "Прочие подозрительные признаки")
LOW_RISK_CATEGORY = ("no_violation", "Признаков нарушения не выявлено")


# --------------------------------------------------------------------------- #
# Link / contact extraction regexes (used by analyzers.links).
# --------------------------------------------------------------------------- #
LINK_PATTERNS = {
    "telegram": re.compile(
        r"(?:t\.me/|telegram\.me/|@)(?P<v>[A-Za-z][A-Za-z0-9_]{3,32})", re.I),
    "whatsapp": re.compile(
        r"(?:wa\.me/|whatsapp[:\s]*)\+?(?P<v>\d[\d\s\-]{7,})", re.I),
    "phone": re.compile(r"(?P<v>\+?\d[\d\(\)\s\-]{8,}\d)"),
    "promocode": re.compile(
        r"(?:промокод|promo\s*code|код|bonus)\s*[:\-]?\s*(?P<v>[A-Z0-9]{3,12})", re.I),
    "url": re.compile(
        r"(?P<v>(?:https?://)?(?:bit\.ly|link\.bio|linktr\.ee|cutt\.ly|"
        r"vk\.cc|tinyurl\.com|goo\.gl|[a-z0-9\-]+\.(?:bio|link|site|click))"
        r"/?[^\s]*)", re.I),
}


# --------------------------------------------------------------------------- #
# CLIP zero-shot vision prompts.  Each maps a visual concept to a ScamDNA dim.
# label/label_ru are English prompt + Russian display; positives raise `gambling`
# or `visual`. Neutral prompts anchor the softmax so benign frames score low.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VisionPrompt:
    label: str
    label_ru: str
    dna_key: str
    is_neutral: bool = False


VISION_PROMPTS: List[VisionPrompt] = [
    VisionPrompt("a screenshot of an online casino slot machine interface",
                 "интерфейс онлайн-казино / слот-машина", "gambling"),
    VisionPrompt("a roulette wheel or betting interface with money",
                 "рулетка / интерфейс ставок", "gambling"),
    VisionPrompt("a big casino win animation with falling coins",
                 "анимация крупного выигрыша", "gambling"),
    VisionPrompt("a fake bank transfer screenshot showing a large payout",
                 "скриншот банковского перевода (возможно поддельный)", "visual"),
    VisionPrompt("a chart showing unrealistic trading or investment profits",
                 "график нереальной доходности", "visual"),
    VisionPrompt("stacks of cash money being shown to the camera",
                 "демонстрация пачек наличных денег", "visual"),
    VisionPrompt("a QR code or messenger contact overlay on screen",
                 "QR-код / контакт мессенджера на экране", "messenger"),
    # Neutral anchors (benign) ------------------------------------------- #
    VisionPrompt("a person talking to the camera in a normal room",
                 "обычное говорящее видео", "", is_neutral=True),
    VisionPrompt("an educational presentation with slides and diagrams",
                 "образовательная презентация", "", is_neutral=True),
    VisionPrompt("everyday outdoor or lifestyle footage",
                 "бытовое/лайфстайл видео", "", is_neutral=True),
]


# --------------------------------------------------------------------------- #
# Tiny matching helper (shared so analyzers match identically).
# --------------------------------------------------------------------------- #
@dataclass
class Match:
    phrase: Phrase
    matched_text: str
    start: int


def find_matches(text: str, phrases: List[Phrase]) -> List[Match]:
    """Return all matches of ``phrases`` inside ``text`` (case-insensitive)."""
    if not text:
        return []
    out: List[Match] = []
    for ph in phrases:
        for m in ph.compiled().finditer(text):
            out.append(Match(phrase=ph, matched_text=m.group(0), start=m.start()))
    return out


def chip_for(match: Match) -> str:
    """Short RU label to show as a finding chip for a match."""
    return match.phrase.label or match.matched_text.strip().lower()
