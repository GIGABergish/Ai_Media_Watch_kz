"""Объяснимость риск-модели — ``explain.attribute``.

Единая знаковая метрика вклада признака — это **input × gradient на риск-логите**
(первый порядок Тейлора: локальная линеаризация сети вокруг текущего входа).

Для общего трунка ``h = ReLU(x @ W_trunk + b_trunk)`` и риск-головы
``logit = h @ W_risk + b_risk`` градиент по входу равен ::

    d(logit)/dx = W_trunk @ ( (h > 0) ⊙ W_risk[:, 0] )

Тогда вклад признака ``i`` = ``x[i] * d(logit)/dx[i]``. Знак сохраняется: положительный
вклад толкает к «скаму», отрицательный — к «не-скаму».

Атрибуции разбиваются на два канала, ровно как описано в DESIGN §11:

* **Числовой канал** — для каждого *активного* инженерного слота
  ``[hash_dim : hash_dim + numeric_dim]`` человекочитаемое имя берётся из
  ``featurize.numeric_feature_names()``, а связка с измерением ScamDNA — из
  статической таблицы слот→``dna_key`` (DESIGN §5). Это позволяет атрибуции
  «лечь» в существующую карточку доказательств по 8 измерениям без правок фронта.
* **Текстовый канал** — ``featurize.top_text_features(text, k)`` пере-извлекает те
  же n-граммы, ранжированные по знаковому проброшенному вкладу к риск-логиту;
  показывается «сырое» (до нормализации) написание токена, ``dna_key=""`` (у
  хешированной n-граммы нет единственного измерения).

Топ-``top_k`` отбирается по ``|вклад|``, перемешивая структурные сигналы
(``промокод (referral)``) и обфусцированные фразовые токены.

Модуль читает только публичные веса модели (``W_trunk/W_risk`` и т.п.) и публичные
функции ``featurize``; он не зависит от внутренней реализации обучения.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from app.ml.config import ml_config
from app.ml.featurize import (
    numeric_feature_names,
    top_text_features,
    vectorize,
)
from app.ml.types import Attribution, RawFeatures

# --------------------------------------------------------------------------- #
# Статическая таблица слот→dna_key для 32 инженерных числовых признаков.
# Порядок и значения ЗАФИКСИРОВАНЫ контрактом featurize (DESIGN §5, столбец
# "dna_key (attribution)"). Слоты 0..7 — это visual_scores по DIMENSION_KEYS.
# Пустая строка означает «нет единственного измерения ScamDNA».
# --------------------------------------------------------------------------- #
NUMERIC_SLOT_DNA: Tuple[str, ...] = (
    "profit",     # 0  visual: profit
    "urgency",    # 1  visual: urgency
    "gambling",   # 2  visual: gambling
    "referral",   # 3  visual: referral
    "messenger",  # 4  visual: messenger
    "visual",     # 5  visual: visual
    "reused",     # 6  visual: reused
    "hashtags",   # 7  visual: hashtags
    "urgency",    # 8  behavior urgency aggregate
    "referral",   # 9  behavior referral aggregate
    "messenger",  # 10 behavior messenger aggregate
    "",           # 11 behavior-hit count
    "",           # 12 negative-marker aggregate
    "messenger",  # 13 link telegram count
    "messenger",  # 14 link whatsapp count
    "messenger",  # 15 link url count
    "referral",   # 16 link promocode count
    "messenger",  # 17 link phone count
    "messenger",  # 18 total link count
    "reused",     # 19 kb_similarity
    "",           # 20 duration_s
    "",           # 21 num_segments
    "hashtags",   # 22 hashtag_count
    "hashtags",   # 23 suspicious-hashtag ratio
    "",           # 24 text_len
    "messenger",  # 25 has_url
    "messenger",  # 26 has_telegram_or_wa
    "referral",   # 27 has_promocode
    "",           # 28 digit_ratio
    "",           # 29 emoji + zero-width count
    "profit",     # 30 profit text-density / spare
    "",           # 31 lang code scalar
)

# Русские человекочитаемые ярлыки по dna_key (для подсветки в карточке).
_DNA_LABEL_RU = {
    "profit": "доход",
    "urgency": "срочность",
    "gambling": "азарт",
    "referral": "реферальность",
    "messenger": "увод в мессенджер",
    "visual": "визуальные признаки",
    "reused": "переиспользование",
    "hashtags": "хэштеги",
}


# --------------------------------------------------------------------------- #
# Доступ к весам модели (model_np.NpRiskModel строится параллельно).
# Поддерживаем как атрибуты экземпляра, так и словарь params, и .npz-имена.
# --------------------------------------------------------------------------- #
def _get_weight(model: Any, *names: str) -> Optional[np.ndarray]:
    """Вернуть первый найденный массив весов по списку возможных имён."""
    for name in names:
        val = getattr(model, name, None)
        if val is not None:
            return np.asarray(val)
    params = getattr(model, "params", None) or getattr(model, "weights", None)
    if isinstance(params, dict):
        for name in names:
            if name in params and params[name] is not None:
                return np.asarray(params[name])
    return None


def _risk_input_gradient(
    model: Any, x: np.ndarray
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Вернуть ``(grad_x, x)`` для риск-логита через input×gradient.

    ``grad_x[i] = sum_j W_trunk[i, j] * relu_mask[j] * W_risk[j]``. Если веса
    модели недоступны (не та реализация), вернуть ``None`` — вызывающий код
    деградирует к чисто текстовым атрибуциям.
    """
    W_trunk = _get_weight(model, "W_trunk", "W1", "W_in", "W_hidden")
    b_trunk = _get_weight(model, "b_trunk", "b1", "b_in", "b_hidden")
    W_risk = _get_weight(model, "W_risk", "W_out_risk", "W2_risk")
    if W_trunk is None or W_risk is None:
        return None

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if W_trunk.shape[0] != x.shape[0]:
        # Несовпадение размерности — нечего объяснять надёжно.
        return None

    b = np.zeros(W_trunk.shape[1], dtype=np.float64) if b_trunk is None \
        else np.asarray(b_trunk, dtype=np.float64).reshape(-1)
    # Прямой проход трунка (eval-режим, dropout выключен) для маски ReLU.
    pre = x @ W_trunk + b                       # (hidden,)
    relu_mask = (pre > 0.0).astype(np.float64)  # (hidden,)

    w_risk = np.asarray(W_risk, dtype=np.float64).reshape(-1)  # (hidden,)
    if w_risk.shape[0] != W_trunk.shape[1]:
        return None

    # Эффективный вес скрытого слоя вдоль пути риск-головы.
    eff_hidden = relu_mask * w_risk             # (hidden,)
    grad_x = W_trunk @ eff_hidden               # (INPUT_DIM,)
    return grad_x, x


# --------------------------------------------------------------------------- #
# Числовой канал
# --------------------------------------------------------------------------- #
def _numeric_attributions(
    grad_x: np.ndarray,
    x: np.ndarray,
    names: Sequence[str],
) -> List[Attribution]:
    """Знаковые вклады активных инженерных числовых слотов."""
    hash_dim = ml_config.hash_dim
    numeric_dim = ml_config.numeric_dim
    out: List[Attribution] = []
    n_slots = min(numeric_dim, len(names), len(NUMERIC_SLOT_DNA))
    for slot in range(n_slots):
        idx = hash_dim + slot
        if idx >= x.shape[0]:
            break
        value = float(x[idx])
        if value == 0.0:               # объясняем только АКТИВНЫЕ признаки
            continue
        contrib = value * float(grad_x[idx])
        if contrib == 0.0:
            continue
        dna = NUMERIC_SLOT_DNA[slot]
        label = _label_for(names[slot], dna)
        out.append(Attribution(feature=label, weight=contrib, dna_key=dna))
    return out


def _label_for(raw_name: str, dna: str) -> str:
    """Человекочитаемый ярлык признака с пометкой измерения ScamDNA.

    Не дублируем пометку ``(dna)``, если машинный ключ или его русский ярлык
    уже присутствуют в имени признака из ``numeric_feature_names()``.
    """
    base = raw_name.strip() if raw_name else "признак"
    if not dna:
        return base
    ru = _DNA_LABEL_RU.get(dna, "")
    if dna in base or (ru and ru in base):
        return base
    return f"{base} ({dna})"


# --------------------------------------------------------------------------- #
# Текстовый канал
# --------------------------------------------------------------------------- #
def _text_attributions(
    grad_x: Optional[np.ndarray],
    text: str,
    hashtags: Optional[Sequence[str]],
    k: int,
) -> List[Attribution]:
    """Топ текстовых n-грамм по знаковому вкладу к риск-логиту.

    Полагаемся на ``featurize.top_text_features`` для отбора репрезентативных
    «сырых» написаний; знак/величину берём из вектора каждой n-граммы, спроецированного
    на ``grad_x``. Хэштеги добавляются в набор кандидатов как самостоятельные токены.
    Если градиент недоступен, n-граммы возвращаются с убывающим по рангу весом
    (порядок — как у featurize, т.е. уже по релевантности).
    """
    has_text = bool(text and text.strip())
    tags = [t for t in (hashtags or []) if t and t.strip()]
    if not has_text and not tags:
        return []
    grams: List[str] = []
    if has_text:
        try:
            grams = list(top_text_features(text, k))
        except Exception:
            grams = []
    # Хэштеги — отдельные кандидаты-токены (сохраняем порядок, без дублей).
    for t in tags:
        if t not in grams:
            grams.append(t)
    if not grams:
        return []

    # Знаковый вклад каждой n-граммы оцениваем через её СОБСТВЕННУЮ векторизацию
    # (хеш-бакеты этого токена), спроецированную на grad_x — так вклад не делится
    # эвристически между токенами одного хеш-блока.
    contrib_by_gram: dict[str, float] = {}
    hash_dim = ml_config.hash_dim
    if grad_x is not None and grad_x.shape[0] >= hash_dim:
        for g in grams:
            contrib_by_gram[g] = _gram_contrib(g, grad_x, hash_dim)

    out: List[Attribution] = []
    n = len(grams)
    for rank, g in enumerate(grams):
        w = contrib_by_gram.get(g)
        if w is None:
            # Фолбэк: убывающий положительный вес по рангу релевантности.
            w = float(n - rank) / float(n)
        out.append(Attribution(feature=g, weight=float(w), dna_key=""))
    return out


def _gram_contrib(gram: str, grad_x: np.ndarray, hash_dim: int) -> float:
    """Знаковый вклад отдельной n-граммы через её собственную векторизацию."""
    try:
        gv = np.asarray(vectorize(RawFeatures(text=gram)), dtype=np.float64).reshape(-1)
    except Exception:
        return 0.0
    if gv.shape[0] < hash_dim:
        return 0.0
    active = np.nonzero(gv[:hash_dim])[0]
    if active.size == 0:
        return 0.0
    # Сумма знаковых вкладов активных хеш-бакетов этой n-граммы.
    return float(np.sum(gv[active] * grad_x[active]))


# --------------------------------------------------------------------------- #
# Публичный API
# --------------------------------------------------------------------------- #
def attribute(model: Any, features: RawFeatures, top_k: int = 8) -> List[Attribution]:
    """Вернуть до ``top_k`` знаковых атрибуций предсказания риска.

    Объединяет числовой канал (активные инженерные слоты → ``dna_key``) и
    текстовый канал (репрезентативные n-граммы), ранжируя по ``|вклад|``.
    Никогда не бросает исключение: при любой проблеме возвращает то, что удалось
    собрать (возможно, пустой список), чтобы не ломать ``predict``.
    """
    if features is None:
        return []
    top_k = max(1, int(top_k))

    # 1) Полный вектор входа и градиент риск-логита по входу.
    grad_x: Optional[np.ndarray] = None
    x_full: Optional[np.ndarray] = None
    try:
        x_full = np.asarray(vectorize(features), dtype=np.float64).reshape(-1)
        g = _risk_input_gradient(model, x_full)
        if g is not None:
            grad_x, x_full = g
    except Exception:
        grad_x = None

    numeric: List[Attribution] = []
    if grad_x is not None and x_full is not None:
        try:
            names = numeric_feature_names()
        except Exception:
            names = [f"num_{i}" for i in range(ml_config.numeric_dim)]
        try:
            numeric = _numeric_attributions(grad_x, x_full, names)
        except Exception:
            numeric = []

    # 2) Текстовый канал — берём с запасом, чтобы хватило после объединения.
    text_attrs: List[Attribution] = []
    try:
        text_attrs = _text_attributions(
            grad_x, features.text, features.hashtags, top_k
        )
    except Exception:
        text_attrs = []

    # 3) Объединение и отбор top_k по модулю вклада. Дедупликация по (feature,dna).
    combined: List[Attribution] = []
    seen: set[Tuple[str, str]] = set()
    for a in (*numeric, *text_attrs):
        key = (a.feature, a.dna_key)
        if key in seen:
            continue
        seen.add(key)
        combined.append(a)

    combined.sort(key=lambda a: abs(a.weight), reverse=True)
    return combined[:top_k]
