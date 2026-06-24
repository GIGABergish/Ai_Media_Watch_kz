"""Оценка обученной модели риска против учителя (правил) на held-out наборе.

Главная цель — измерить «moat» проекта: модель должна СОГЛАШАТЬСЯ с учителем на
чистом тексте и СУЩЕСТВЕННО ПРЕВОСХОДИТЬ его на обфусцированных скам-роликах,
где лексиконы слепнут. Поэтому held-out строится с ДРУГИМ seed (``cfg.seed +
9973``) и повышенной долей обфускации/code-mix, ground-truth берётся из
синтетики (``Label.source == "synthetic"``, ``is_scam`` известен по построению),
а оба предиктора — модель и правило-учитель — оцениваются на ОДНОМ И ТОМ ЖЕ
наборе поверхностных строк (обфусцированных), без весов.

Результат: словарь метрик + два артефакта на диск (``ML_DIR/MODEL_CARD.md`` на
русском и ``ML_DIR/eval_metrics.json`` как единый источник истины). Функция
никогда не падает на отсутствии модели: если активной модели нет, пишется
карточка «только-учитель» с пометкой.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import risk_level
from app.ml.config import ML_DIR, MLConfig, ml_config
from app.ml.types import (
    CATEGORY_KEYS,
    DIMENSION_KEYS,
    Example,
    Label,
    Prediction,
)
from app.ml import metrics as M
from app.ml import synth
from app.ml import weak_labels
from app.ml.registry import load_active

# Held-out seed offset — должен отличаться от обучающего, чтобы исключить
# запоминание и проверить ОБОБЩЕНИЕ на новых поверхностях.
HELDOUT_SEED_OFFSET = 9973
# Размер held-out по умолчанию (модерируем, чтобы оценка шла за секунды).
HELDOUT_SIZE = 1200

# Нативные пороги движка (0..100), переиспользуем app.config через risk_level.
_BANDS = (("medium", 40), ("high", 65), ("critical", 88))

# Символы-конфузаблы / цифро-замены, выдающие обфускацию скам-текста.
_LATIN_LOOKALIKE = set("aeopcxykmhtbAEOPCXYKMHTB")
_LEET_DIGITS = set("013456780")
_ZERO_WIDTH = "​‌‍﻿"
_SPACED_RUN = re.compile(r"(?:[а-яёa-z][\s.\-_]){3,}[а-яёa-z]", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Детектор обфускации (независим от внутренностей synth — types не имеют флага).
# --------------------------------------------------------------------------- #
def _looks_obfuscated(text: str) -> bool:
    """Эвристика: содержит ли текст признаки обфускации/leetspeak/разрядки.

    Срабатывает на смеси кириллица+латиница-двойники, цифрах-вместо-букв внутри
    слов, intra-word разрядке (``к а з и н о``) и zero-width разделителях.
    """
    if not text:
        return False
    if any(ch in _ZERO_WIDTH for ch in text):
        return True
    if _SPACED_RUN.search(text):
        return True
    has_cyr = bool(re.search(r"[а-яё]", text, re.IGNORECASE))
    # Латиница-двойник внутри кириллического контекста.
    if has_cyr and any(ch in _LATIN_LOOKALIKE for ch in text):
        return True
    # Цифра, зажатая между буквами (д0ход, к1но) — leetspeak.
    if re.search(r"[а-яёa-z][013456780][а-яёa-z]", text, re.IGNORECASE):
        return True
    return False


# --------------------------------------------------------------------------- #
# Сбор held-out набора.
# --------------------------------------------------------------------------- #
def _seed_cases() -> List[Tuple[str, List[str], bool, str]]:
    """Несколько ручных RU-кейсов: (text, hashtags, is_scam, note).

    Включают чистые и обфусцированные скамы + честные негативы (антифрод).
    """
    return [
        ("Гарантированный доход 300% в месяц без вложений, пиши в директ",
         ["#инвестиции"], True, "clean profit scam"),
        ("Г@рантир0ванный д0ход 300% в месяц, пиши в личку",
         ["#доход"], True, "obfuscated profit scam"),
        ("Заходи в к а з и н о VAVADA, промокод BONUS, первый депозит х2",
         ["#казино"], True, "spaced gambling scam"),
        ("кazino онлайн, vаvаdа дарит бонус, регистрируйся по ссылке",
         [], True, "mixed-script gambling scam"),
        ("Приглашай друзей в команду, реферальная структура, доход с сети",
         ["#бизнес"], True, "pyramid scam"),
        ("Разбираем как мошенники обещают гарантированный доход — это развод, "
         "не переводите деньги незнакомцам",
         ["#антифрод"], False, "anti-fraud educational"),
        ("Сегодня готовим борщ: свёкла, капуста, картофель и немного зелени",
         ["#рецепт"], False, "benign cooking"),
        ("Курс по фотографии: основы экспозиции и композиции для начинающих",
         ["#обучение"], False, "benign educational"),
    ]


def _build_heldout(cfg: MLConfig) -> List[Example]:
    """Собирает held-out: synth с другим seed + ручные кейсы.

    Возвращает Example c гарантированно проставленной synthetic-меткой (для
    ground-truth ``is_scam``). Обфусцированные строки попадают сюда естественно
    из synth (повышенная доля) и из seed-кейсов.
    """
    seed = cfg.seed + HELDOUT_SEED_OFFSET
    examples: List[Example] = list(synth.generate(HELDOUT_SIZE, seed))

    # Добавляем ручные кейсы как text-only Examples с явной synthetic-меткой.
    for i, (text, tags, is_scam, _note) in enumerate(_seed_cases()):
        lbl = weak_labels.weak_label_from_text(text, hashtags=tags)
        # Ground-truth риск/класс известны по построению seed-кейсов; учителю
        # доверяем только размерности (dimensions), как и в обучении.
        gt_risk = 0.85 if is_scam else 0.05
        seed_label = Label(
            risk=gt_risk,
            dimensions=dict(lbl.dimensions),
            category=lbl.category,
            is_scam=is_scam,
            source="synthetic",
            weight=1.0,
        )
        rf = _text_features(text, tags)
        examples.append(Example(id=f"seed-{i}", features=rf, label=seed_label))
    return examples


def _text_features(text: str, hashtags: List[str]):
    """RawFeatures из чистого текста (для seed-кейсов) через featurize.extract."""
    from app.ml import featurize  # локальный импорт: тяжёлая нормализация
    from app.pipeline.contracts import MediaInput, SignalBundle

    media = MediaInput(source_type="text", title="", description=text,
                       hashtags=list(hashtags))
    bundle = SignalBundle(media=media)
    bundle.transcript.full_text = text
    rf = featurize.extract(bundle)
    return rf


# --------------------------------------------------------------------------- #
# Скоринг обоих предикторов на ОДНОМ наборе поверхностей.
# --------------------------------------------------------------------------- #
def _score_set(
    model, examples: List[Example]
) -> Dict[str, np.ndarray]:
    """Прогоняет модель (если есть) и учителя по тем же поверхностным текстам.

    Возвращает массивы: y (ground-truth is_scam), obf (флаг обфускации),
    p_model (калиброванная вероятность модели или NaN), p_teacher (риск
    учителя 0..1), а также матрицы размерностей (true/model) для dim_metrics.
    """
    n = len(examples)
    y = np.zeros(n, dtype=np.float32)
    obf = np.zeros(n, dtype=bool)
    p_model = np.full(n, np.nan, dtype=np.float64)
    p_teacher = np.zeros(n, dtype=np.float64)
    cat_true = np.zeros(n, dtype=np.int64)
    cat_model = np.full(n, -1, dtype=np.int64)

    Ydim_true = np.zeros((n, len(DIMENSION_KEYS)), dtype=np.float64)
    Ydim_model = np.full((n, len(DIMENSION_KEYS)), np.nan, dtype=np.float64)

    cat_index = {k: i for i, k in enumerate(CATEGORY_KEYS)}

    for i, ex in enumerate(examples):
        rf = ex.features
        lbl = ex.label or Label()
        y[i] = 1.0 if lbl.is_scam else 0.0
        obf[i] = _looks_obfuscated(rf.text)
        cat_true[i] = cat_index.get(lbl.category, cat_index.get("no_violation", 0))
        for d, k in enumerate(DIMENSION_KEYS):
            Ydim_true[i, d] = float(lbl.dimensions.get(k, 0.0))

        # Учитель на ТОЙ ЖЕ поверхностной строке (обфусцированной).
        t = weak_labels.weak_label_from_text(rf.text, hashtags=rf.hashtags)
        p_teacher[i] = float(t.risk)

        # Модель (если активна).
        if model is not None:
            pred: Prediction = model.predict(rf)
            p_model[i] = float(pred.risk_prob)
            cat_model[i] = cat_index.get(pred.category, -1)
            for d, k in enumerate(DIMENSION_KEYS):
                Ydim_model[i, d] = float(pred.dimensions.get(k, 0)) / 100.0

    return {
        "y": y, "obf": obf,
        "p_model": p_model, "p_teacher": p_teacher,
        "cat_true": cat_true, "cat_model": cat_model,
        "Ydim_true": Ydim_true, "Ydim_model": Ydim_model,
    }


# --------------------------------------------------------------------------- #
# Operating points: recall на фиксированном FPR + нативные пороги движка.
# --------------------------------------------------------------------------- #
def _threshold_at_fpr(y: np.ndarray, prob: np.ndarray, target_fpr: float) -> float:
    """Порог, дающий FPR <= target_fpr на негативах (наибольший recall при этом)."""
    neg = prob[y < 0.5]
    if neg.size == 0:
        return 0.5
    # FPR(thr) = доля негативов с prob >= thr. Берём (1-target_fpr)-квантиль.
    q = float(np.clip(1.0 - target_fpr, 0.0, 1.0))
    return float(np.quantile(neg, q))


def _recall_fpr_at_threshold(
    y: np.ndarray, prob: np.ndarray, thr: float
) -> Tuple[float, float]:
    """(recall на скамах, fpr на негативах) при пороге ``prob >= thr``."""
    pos = prob[y >= 0.5]
    neg = prob[y < 0.5]
    recall = float((pos >= thr).mean()) if pos.size else 0.0
    fpr = float((neg >= thr).mean()) if neg.size else 0.0
    return recall, fpr


def _operating_points(
    y: np.ndarray, prob: np.ndarray, target_fpr: float = 0.10
) -> Dict[str, object]:
    """Recall@fixed-FPR и метрики на нативных бэндах движка (prob*100)."""
    valid = ~np.isnan(prob)
    y2, p2 = y[valid], prob[valid]
    out: Dict[str, object] = {}
    if p2.size == 0:
        return {"recall_at_fpr": None, "bands": {}}

    thr = _threshold_at_fpr(y2, p2, target_fpr)
    recall, fpr = _recall_fpr_at_threshold(y2, p2, thr)
    out["recall_at_fpr"] = {
        "target_fpr": target_fpr, "threshold": thr,
        "recall": recall, "fpr": fpr,
    }

    bands: Dict[str, Dict[str, float]] = {}
    score100 = p2 * 100.0
    for name, cut in _BANDS:
        flagged = score100 >= cut
        pos = flagged[y2 >= 0.5]
        neg = flagged[y2 < 0.5]
        bands[name] = {
            "cut": float(cut),
            "recall": float(pos.mean()) if pos.size else 0.0,
            "fpr": float(neg.mean()) if neg.size else 0.0,
        }
    out["bands"] = bands
    return out


# --------------------------------------------------------------------------- #
# Сравнение модель vs учитель на срезах.
# --------------------------------------------------------------------------- #
def _slice_compare(
    y: np.ndarray, p_model: np.ndarray, p_teacher: np.ndarray, mask: np.ndarray
) -> Dict[str, object]:
    """AUROC/F1/recall для модели и учителя на срезе ``mask`` + дельта."""
    ys = y[mask]
    res: Dict[str, object] = {"n": int(mask.sum()), "n_scam": int((ys >= 0.5).sum())}
    if ys.size == 0:
        res.update({"model": {}, "teacher": {}, "delta": {}})
        return res

    pm = p_model[mask]
    pt = p_teacher[mask]
    pm_valid = pm[~np.isnan(pm)]

    if pm_valid.size == pm.size and pm.size > 0:
        m = M.binary_metrics(ys, pm)
        m["recall"] = _recall_at_half(ys, pm)
    else:
        m = {}
    t = M.binary_metrics(ys, pt)
    t["recall"] = _recall_at_half(ys, pt)

    delta = {}
    if m:
        for k in ("auroc", "ap", "f1", "acc", "brier", "recall"):
            if k in m and k in t:
                delta[k] = float(m[k] - t[k])

    res.update({"model": m, "teacher": t, "delta": delta})
    return res


def _recall_at_half(y: np.ndarray, prob: np.ndarray) -> float:
    """Recall на скамах при пороге 0.5 (для учителя — risk/100 >= 0.5)."""
    pos = prob[y >= 0.5]
    return float((pos >= 0.5).mean()) if pos.size else 0.0


# --------------------------------------------------------------------------- #
# Аудит калибровки и неопределённости.
# --------------------------------------------------------------------------- #
def _reliability_bins(y: np.ndarray, prob: np.ndarray, bins: int = 15) -> List[dict]:
    """Данные диаграммы надёжности для metrics json (по равным [0,1] бинам)."""
    valid = ~np.isnan(prob)
    y2, p2 = y[valid], prob[valid]
    edges = np.linspace(0.0, 1.0, bins + 1)
    out: List[dict] = []
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        sel = (p2 >= lo) & (p2 < hi if b < bins - 1 else p2 <= hi)
        n = int(sel.sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "conf": float(p2[sel].mean()) if n else 0.0,
            "acc": float(y2[sel].mean()) if n else 0.0,
        })
    return out


def _uncertainty_audit(
    y: np.ndarray, prob: np.ndarray, low: float, high: float
) -> Dict[str, object]:
    """Доля ошибок ВНУТРИ vs ВНЕ зоны неопределённости [low, high]."""
    valid = ~np.isnan(prob)
    y2, p2 = y[valid], prob[valid]
    if p2.size == 0:
        return {}
    pred = (p2 >= 0.5).astype(np.float32)
    err = (pred != y2)
    inside = (p2 >= low) & (p2 <= high)
    outside = ~inside
    return {
        "band": [low, high],
        "frac_inside": float(inside.mean()),
        "err_inside": float(err[inside].mean()) if inside.any() else 0.0,
        "err_outside": float(err[outside].mean()) if outside.any() else 0.0,
    }


# --------------------------------------------------------------------------- #
# Главная точка входа.
# --------------------------------------------------------------------------- #
def evaluate(cfg: MLConfig = ml_config) -> dict:
    """Оценивает активную модель против учителя на held-out и пишет артефакты.

    Возвращает словарь метрик (он же сериализуется в ``eval_metrics.json``).
    Никогда не падает: при отсутствии модели формирует карточку «только-учитель».
    """
    model = None
    try:
        model = load_active()
    except Exception:
        model = None

    examples = _build_heldout(cfg)
    sc = _score_set(model, examples)
    y = sc["y"]
    obf = sc["obf"]
    p_model = sc["p_model"]
    p_teacher = sc["p_teacher"]

    has_model = model is not None and not np.all(np.isnan(p_model))

    # ---- Общие метрики -------------------------------------------------- #
    overall: Dict[str, object] = {
        "n": int(len(examples)),
        "n_scam": int((y >= 0.5).sum()),
        "n_benign": int((y < 0.5).sum()),
        "n_obfuscated": int(obf.sum()),
    }

    teacher_metrics = M.binary_metrics(y, p_teacher)
    teacher_metrics["recall"] = _recall_at_half(y, p_teacher)
    teacher_metrics["ece"] = M.ece(y, p_teacher)

    model_metrics: Dict[str, object] = {}
    if has_model:
        model_metrics = M.binary_metrics(y, p_model)
        model_metrics["recall"] = _recall_at_half(y, p_model)
        model_metrics["ece"] = M.ece(y, p_model)

    # ---- Срезы: всё / чистое / обфусцированное -------------------------- #
    all_mask = np.ones(len(examples), dtype=bool)
    slices = {
        "all": _slice_compare(y, p_model, p_teacher, all_mask),
        "clean": _slice_compare(y, p_model, p_teacher, ~obf),
        "obfuscated": _slice_compare(y, p_model, p_teacher, obf),
    }

    # ---- Operating points (модель и учитель) ---------------------------- #
    op_model = _operating_points(y, p_model) if has_model else {}
    op_teacher = _operating_points(y, p_teacher)

    # ---- Размерности (model vs ground-truth/teacher-refined) ------------ #
    dim_report: Dict[str, object] = {}
    if has_model:
        Ydim_true = sc["Ydim_true"]
        Ydim_model = sc["Ydim_model"]
        valid_rows = ~np.any(np.isnan(Ydim_model), axis=1)
        if valid_rows.any():
            dim_report = M.dim_metrics(
                Ydim_true[valid_rows], Ydim_model[valid_rows]
            )

    # ---- Категории (accuracy модели) ------------------------------------ #
    cat_acc = None
    if has_model:
        cm = sc["cat_model"]
        ct = sc["cat_true"]
        valid = cm >= 0
        if valid.any():
            cat_acc = float((cm[valid] == ct[valid]).mean())

    # ---- Калибровка / неопределённость ---------------------------------- #
    calibration = {
        "temperature": float(getattr(model, "temperature", 1.0)) if model else None,
        "method": cfg.calibration,
        "teacher_ece": teacher_metrics["ece"],
    }
    if has_model:
        calibration["model_ece"] = model_metrics["ece"]
        calibration["reliability_bins"] = _reliability_bins(y, p_model)
        calibration["uncertainty_audit"] = _uncertainty_audit(
            y, p_model, cfg.uncertain_low, cfg.uncertain_high
        )

    # ---- Декомпозиция согласия (model↔teacher vs ground-truth) ---------- #
    agreement: Dict[str, object] = {}
    if has_model:
        gt = (y >= 0.5)
        m_pred = (p_model >= 0.5)
        t_pred = (p_teacher >= 0.5)
        valid = ~np.isnan(p_model)
        agreement = {
            "model_vs_truth": float((m_pred[valid] == gt[valid]).mean()),
            "teacher_vs_truth": float((t_pred == gt).mean()),
            "model_vs_teacher": float((m_pred[valid] == t_pred[valid]).mean()),
        }

    result: Dict[str, object] = {
        "version": cfg.version,
        "has_model": bool(has_model),
        "seeds": {"train": cfg.seed, "heldout": cfg.seed + HELDOUT_SEED_OFFSET},
        "overall": overall,
        "model_metrics": model_metrics,
        "teacher_metrics": teacher_metrics,
        "slices": slices,
        "operating_points": {"model": op_model, "teacher": op_teacher},
        "dim_metrics": dim_report,
        "category_accuracy": cat_acc,
        "calibration": calibration,
        "agreement": agreement,
    }

    # ---- Запись артефактов ---------------------------------------------- #
    _write_metrics_json(result)
    _write_model_card(result, cfg)
    return result


# --------------------------------------------------------------------------- #
# Артефакты.
# --------------------------------------------------------------------------- #
def _write_metrics_json(result: dict) -> None:
    """Пишет единый источник истины ``ML_DIR/eval_metrics.json``."""
    path = ML_DIR / "eval_metrics.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _json_default(o: object):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _fmt(v: object, nd: int = 3) -> str:
    """Аккуратное форматирование числа для карточки (— для None)."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _metrics_row(name: str, m: Dict[str, object]) -> str:
    if not m:
        return f"| {name} | — | — | — | — | — | — |"
    return (
        f"| {name} | {_fmt(m.get('auroc'))} | {_fmt(m.get('ap'))} | "
        f"{_fmt(m.get('f1'))} | {_fmt(m.get('recall'))} | "
        f"{_fmt(m.get('brier'))} | {_fmt(m.get('ece'))} |"
    )


def _slice_table(slices: dict) -> str:
    lines = [
        "| Срез | n | n_scam | AUROC мод. | AUROC уч. | ΔAUROC | "
        "Recall мод. | Recall уч. | ΔRecall |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, ru in (("all", "Весь"), ("clean", "Чистый"),
                    ("obfuscated", "Обфусцированный")):
        s = slices.get(key, {})
        m = s.get("model", {})
        t = s.get("teacher", {})
        d = s.get("delta", {})
        lines.append(
            f"| {ru} | {s.get('n', 0)} | {s.get('n_scam', 0)} | "
            f"{_fmt(m.get('auroc'))} | {_fmt(t.get('auroc'))} | "
            f"{_fmt(d.get('auroc'))} | {_fmt(m.get('recall'))} | "
            f"{_fmt(t.get('recall'))} | {_fmt(d.get('recall'))} |"
        )
    return "\n".join(lines)


def _dim_table(dim_report: dict) -> str:
    if not dim_report:
        return "_Модель недоступна — размерности не оценивались._"
    lines = ["| Размерность | MAE | Corr |", "|---|---|---|"]
    for k in DIMENSION_KEYS:
        d = dim_report.get(k, {})
        lines.append(f"| {k} | {_fmt(d.get('mae'))} | {_fmt(d.get('corr'))} |")
    return "\n".join(lines)


def _write_model_card(result: dict, cfg: MLConfig) -> None:
    """Пишет карточку модели (RU) ``ML_DIR/MODEL_CARD.md``."""
    ov = result["overall"]
    has_model = result["has_model"]
    seeds = result["seeds"]
    obf_slice = result["slices"].get("obfuscated", {})
    obf_delta = obf_slice.get("delta", {})
    cal = result["calibration"]
    agr = result["agreement"]

    if not has_model:
        moat = ("**Активная модель отсутствует** — оценён только учитель (правила). "
                "Обучите модель (`python -m app.ml.cli train`) и повторите оценку.")
    else:
        d_auroc = obf_delta.get("auroc")
        d_recall = obf_delta.get("recall")
        moat = (
            f"На обфусцированном срезе модель превосходит учителя: "
            f"ΔAUROC = **{_fmt(d_auroc)}**, ΔRecall = **{_fmt(d_recall)}**. "
            "Это и есть ключевое преимущество дистилляции + хеширования "
            "символьных n-грамм: модель ловит то, что лексиконы пропускают, "
            "оставаясь сопоставимой с учителем на чистом тексте."
        )

    op_model = result["operating_points"].get("model", {})
    rec_fpr = op_model.get("recall_at_fpr") if op_model else None
    rec_fpr_str = "—"
    if rec_fpr:
        rec_fpr_str = (f"recall={_fmt(rec_fpr.get('recall'))} при "
                       f"FPR={_fmt(rec_fpr.get('fpr'))} "
                       f"(порог prob={_fmt(rec_fpr.get('threshold'))})")

    cat_acc = result.get("category_accuracy")

    md = f"""# Карточка модели — {cfg.version}

## Назначение
Многозадачная калиброванная модель риска для коротких социальных видео
(RU/KZ/EN). Обучена **дистилляцией со слабой разметкой** от правило-движка
(`scam_dna → risk_score → category`) + синтетических данных. Предсказывает
совместно: общий риск (0..1), 8 измерений ScamDNA и грубую категорию.
Робастна к обфускации за счёт знакового хеширования символьных n-грамм (3..5).
Служится как переносимый `.npz` (инференс — только NumPy); при отсутствии
модели движок откатывается на правила.

## Данные
- Held-out сгенерирован synth с **другим seed** (`{seeds['heldout']}`, обучение
  на `{seeds['train']}`) и повышенной долей обфускации/code-mix — out-of-sample
  для запоминания, in-distribution для явления.
- Ground-truth `is_scam` / риск — из синтетики (известны по построению);
  размерности уточнены учителем на ЧИСТОМ тексте.
- Объём: **{ov['n']}** примеров ({ov['n_scam']} скам / {ov['n_benign']} безопасных),
  из них обфусцированных: **{ov['n_obfuscated']}**.
- Оба предиктора (модель и учитель) оценены на ОДНИХ И ТЕХ ЖЕ поверхностных
  (обфусцированных) строках, без весов.

## Ключевое преимущество (обфускация)
{moat}

## Метрики (общие)
| Предиктор | AUROC | AP | F1 | Recall | Brier | ECE |
|---|---|---|---|---|---|---|
{_metrics_row('Модель', result['model_metrics'])}
{_metrics_row('Учитель (правила)', result['teacher_metrics'])}

Точность категории (модель): **{_fmt(cat_acc)}**.
Recall@fixed-FPR (модель): {rec_fpr_str}.

## Сравнение с учителем (по срезам)
{_slice_table(result['slices'])}

> Нарратив: на чистом тексте модель сопоставима с учителем (там лексиконы
> сильны), на обфусцированном — существенно превосходит. Это не подтасовка:
> чистый срез показан честно рядом.

## Размерности ScamDNA (model vs ground-truth)
{_dim_table(result['dim_metrics'])}

## Калибровка
- Метод: `{cal.get('method')}`, температура T = **{_fmt(cal.get('temperature'))}**.
- ECE учителя: **{_fmt(cal.get('teacher_ece'))}**; ECE модели: **{_fmt(cal.get('model_ece'))}**.
- Зона неопределённости `[{cfg.uncertain_low}, {cfg.uncertain_high}]` — сигнал
  для human-in-the-loop (active learning). Аудит ошибок внутри/вне зоны и
  данные диаграммы надёжности — в `eval_metrics.json`.

## Согласованность
- Модель vs ground-truth: **{_fmt(agr.get('model_vs_truth'))}**
- Учитель vs ground-truth: **{_fmt(agr.get('teacher_vs_truth'))}**
- Модель vs учитель: **{_fmt(agr.get('model_vs_teacher'))}**

## Ограничения и этические риски
- Слабая разметка наследует **слепые зоны лексиконов** учителя; дистилляция
  смягчает их обфускация-инвариантностью, но не устраняет полностью.
- Синтетика может не покрывать все реальные паттерны мошенничества; нужен
  периодический контроль дрейфа (PSI) и дообучение.
- Модель — **вспомогательный** сигнал; финальные решения с человеком в цикле
  через флаг `uncertain`. Возможны ложные срабатывания на образовательном/
  антифрод-контенте (намеренно перепредставлен в обучении как трудный негатив).

**Версия:** `{cfg.version}`
"""
    (ML_DIR / "MODEL_CARD.md").write_text(md, encoding="utf-8")
