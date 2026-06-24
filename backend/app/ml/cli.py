"""Командная строка модели риска ``app.ml`` (``python -m app.ml.cli``).

Тонкая обёртка над уже реализованными модулями жизненного цикла модели —
данные → обучение → оценка → инференс. Четыре подкоманды (DESIGN §16):

* ``gen-data`` — пишет небольшой синтетический сэмпл (``synth.generate``) в JSON
  (на диск или в stdout) для быстрой ручной проверки генератора.
* ``train``    — запускает ``train.train()`` и печатает итоговые метрики.
* ``evaluate`` — запускает ``evaluate.evaluate()`` (модель vs учитель на held-out)
  и печатает сводку + пути к артефактам (``MODEL_CARD.md`` / ``eval_metrics.json``).
* ``predict``  — строит text-only :class:`SignalBundle` из ``--text``
  (+ опционально ``--hashtags``), прогоняет серверный путь
  ``inference.score_bundle`` и печатает :class:`Prediction` как JSON.

Тяжёлые зависимости (numpy и обучающие модули) импортируются ЛЕНИВО внутри
обработчиков, поэтому сам модуль остаётся импортируемым и в lite-режиме; команда
``predict`` корректно деградирует (печатает понятное сообщение), если активной
модели нет. Человеко-ориентированные строки — на русском; вывод метрик — читаемый,
машинные срезы доступны через ``--json``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import List, Optional, Sequence

from app.ml.config import ml_config


# --------------------------------------------------------------------------- #
# Утилиты сериализации / печати.
# --------------------------------------------------------------------------- #
def _json_default(o: object):
    """Безопасная сериализация numpy-скаляров/массивов и dataclass'ов в JSON."""
    try:  # numpy может отсутствовать в lite-режиме — импортируем лениво.
        import numpy as np  # noqa: WPS433 (local import by design)

        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # noqa: BLE001 - numpy absent or unexpected type
        pass
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    return str(o)


def _dumps(obj: object) -> str:
    """Компактно-читаемый JSON (UTF-8, кириллица как есть, отступ 2)."""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)


def _parse_hashtags(raw: Optional[str]) -> List[str]:
    """Разбирает ``--hashtags`` (через запятую) в список нормализованных тегов."""
    if not raw:
        return []
    return [h.strip() for h in raw.split(",") if h.strip()]


def _fmt_num(v: object, nd: int = 4) -> str:
    """Аккуратное форматирование числа (— для None/нечисел)."""
    if v is None:
        return "—"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _print_metric_block(title: str, metrics: object) -> None:
    """Печатает плоский словарь метрик ``{name: value}`` выровненным блоком."""
    print(f"\n{title}")
    if not isinstance(metrics, dict) or not metrics:
        print("  (нет данных)")
        return
    width = max((len(str(k)) for k in metrics), default=0)
    for key, val in metrics.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            print(f"  {str(key).ljust(width)} : {_fmt_num(val)}")
        else:
            print(f"  {str(key).ljust(width)} : {val}")


# --------------------------------------------------------------------------- #
# gen-data
# --------------------------------------------------------------------------- #
def _example_to_dict(example: object) -> dict:
    """Сериализует :class:`Example` (features + label) в JSON-совместимый dict."""
    feats = getattr(example, "features", None)
    label = getattr(example, "label", None)
    return {
        "id": getattr(example, "id", ""),
        "features": dataclasses.asdict(feats) if feats is not None else None,
        "label": dataclasses.asdict(label) if label is not None else None,
    }


def _cmd_gen_data(args: argparse.Namespace) -> int:
    """Генерирует небольшой синтетический сэмпл и пишет его JSON-ом."""
    from app.ml import synth  # ленивый импорт (тянет numpy)

    n = int(args.n)
    seed = int(args.seed if args.seed is not None else ml_config.seed)
    examples = synth.generate(n, seed)

    rows = [_example_to_dict(ex) for ex in examples]
    payload = {"n_requested": n, "n_generated": len(rows), "seed": seed,
               "examples": rows}
    text = _dumps(payload)

    if args.out:
        from pathlib import Path

        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"Записано {len(rows)} примеров (seed={seed}) -> {path}")
    else:
        print(text)
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def _cmd_train(args: argparse.Namespace) -> int:
    """Запускает обучение и печатает итоговые метрики."""
    from app.ml import train  # ленивый импорт (тянет numpy/model_np)

    metrics = train.train(ml_config)

    if args.json:
        print(_dumps(metrics))
        return 0

    print("\n=== Обучение завершено ===")
    if isinstance(metrics, dict):
        version = metrics.get("version", ml_config.version)
        print(f"Версия модели : {version}")
        # Печатаем известные верхнеуровневые блоки метрик, если они есть.
        for key in ("val", "val_metrics", "metrics", "risk", "dim", "category"):
            block = metrics.get(key)
            if isinstance(block, dict) and block:
                _print_metric_block(f"[{key}]", block)
        # Плоские скаляры верхнего уровня (например best_val_loss, temperature).
        scalars = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        if scalars:
            _print_metric_block("[summary]", scalars)
        # Путь к сохранённому артефакту, если трейнер его вернул.
        for key in ("artifact", "path", "model_path"):
            if metrics.get(key):
                print(f"\nАртефакт: {metrics[key]}")
                break
    else:
        print(metrics)
    return 0


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def _cmd_evaluate(args: argparse.Namespace) -> int:
    """Запускает оценку модели против учителя и печатает сводку."""
    from app.ml import evaluate as ev  # ленивый импорт (тянет numpy)
    from app.ml.config import ML_DIR

    result = ev.evaluate(ml_config)

    if args.json:
        print(_dumps(result))
        return 0

    print("\n=== Оценка завершена ===")
    if not isinstance(result, dict):
        print(result)
        return 0

    has_model = bool(result.get("has_model"))
    print(f"Версия        : {result.get('version', ml_config.version)}")
    print(f"Активная модель: {'есть' if has_model else 'НЕТ (только учитель)'}")

    ov = result.get("overall")
    if isinstance(ov, dict):
        print(
            f"Набор         : n={ov.get('n')} "
            f"(скам={ov.get('n_scam')}, безопасных={ov.get('n_benign')}, "
            f"обфусцированных={ov.get('n_obfuscated')})"
        )

    if has_model:
        _print_metric_block("[модель — общие]", result.get("model_metrics"))
    _print_metric_block("[учитель — общие]", result.get("teacher_metrics"))

    # Ключевое преимущество — дельта на обфусцированном срезе.
    slices = result.get("slices")
    if isinstance(slices, dict):
        obf = slices.get("obfuscated", {})
        delta = obf.get("delta", {}) if isinstance(obf, dict) else {}
        if delta:
            print("\n[срез: обфускация, модель - учитель]")
            print(f"  ΔAUROC : {_fmt_num(delta.get('auroc'))}")
            print(f"  ΔRecall: {_fmt_num(delta.get('recall'))}")
            print(f"  ΔF1    : {_fmt_num(delta.get('f1'))}")

    cat_acc = result.get("category_accuracy")
    if cat_acc is not None:
        print(f"\nТочность категории (модель): {_fmt_num(cat_acc)}")

    print(f"\nАртефакты: {ML_DIR / 'MODEL_CARD.md'}")
    print(f"           {ML_DIR / 'eval_metrics.json'}")
    return 0


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
def _build_text_bundle(text: str, hashtags: List[str]):
    """Строит минимальный text-only :class:`SignalBundle` из текста.

    Текст кладётся и в ``description``, и в ``transcript.full_text``, чтобы и
    дешёвые лексические лейны (учитель), и ``featurize.extract`` видели одну и ту
    же поверхность — как в орхестраторе на холодном текстовом входе.
    """
    from app.pipeline.contracts import MediaInput, SignalBundle

    media = MediaInput(
        source_type="text",
        title="",
        description=text,
        hashtags=list(hashtags),
    )
    bundle = SignalBundle(media=media)
    bundle.transcript.full_text = text
    return bundle


def _prediction_to_dict(pred: object) -> dict:
    """Сериализует :class:`Prediction` (с вложенными :class:`Attribution`)."""
    return dataclasses.asdict(pred)  # type: ignore[arg-type]


def _cmd_predict(args: argparse.Namespace) -> int:
    """Скорит текст активной моделью и печатает Prediction как JSON."""
    text = args.text or ""
    if not text.strip():
        print("Ошибка: пустой --text.", file=sys.stderr)
        return 2

    hashtags = _parse_hashtags(args.hashtags)
    kb = float(args.kb_similarity)

    from app.ml import inference  # ленивый импорт (никогда не падает -> None)

    bundle = _build_text_bundle(text, hashtags)
    pred = inference.score_bundle(bundle, kb_similarity=kb)

    if pred is None:
        # Модель отключена/отсутствует/numpy недоступен — серверный firewall дал None.
        msg = {
            "prediction": None,
            "reason": (
                "активная модель недоступна "
                "(ml отключён, артефакт не обучен или numpy отсутствует) — "
                "движок использует правила"
            ),
            "ml_enabled": bool(ml_config.enable),
        }
        print(_dumps(msg))
        return 1

    print(_dumps(_prediction_to_dict(pred)))
    return 0


# --------------------------------------------------------------------------- #
# Парсер + точка входа.
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Собирает argparse-парсер со всеми подкомандами."""
    parser = argparse.ArgumentParser(
        prog="python -m app.ml.cli",
        description=(
            "CLI кастомной модели риска AI Media Watch: генерация данных, "
            "обучение, оценка и инференс."
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="{gen-data,train,evaluate,predict}")
    sub.required = True

    # gen-data ------------------------------------------------------------- #
    p_gen = sub.add_parser(
        "gen-data",
        help="Сгенерировать небольшой синтетический сэмпл (JSON).",
    )
    p_gen.add_argument("--n", type=int, default=12,
                       help="Сколько примеров сгенерировать (по умолчанию 12).")
    p_gen.add_argument("--seed", type=int, default=None,
                       help=f"Seed RNG (по умолчанию ml_config.seed={ml_config.seed}).")
    p_gen.add_argument("--out", type=str, default=None,
                       help="Путь для записи JSON (по умолчанию — stdout).")
    p_gen.set_defaults(func=_cmd_gen_data)

    # train ---------------------------------------------------------------- #
    p_train = sub.add_parser(
        "train",
        help="Обучить модель (build -> arrays -> split -> fit -> calibrate -> save).",
    )
    p_train.add_argument("--json", action="store_true",
                         help="Печать метрик одним JSON-объектом.")
    p_train.set_defaults(func=_cmd_train)

    # evaluate ------------------------------------------------------------- #
    p_eval = sub.add_parser(
        "evaluate",
        help="Оценить активную модель против учителя на held-out наборе.",
    )
    p_eval.add_argument("--json", action="store_true",
                        help="Печать полного словаря метрик как JSON.")
    p_eval.set_defaults(func=_cmd_evaluate)

    # predict -------------------------------------------------------------- #
    p_pred = sub.add_parser(
        "predict",
        help="Спрогнозировать риск по тексту (печатает Prediction как JSON).",
    )
    p_pred.add_argument("--text", type=str, required=True,
                        help="Текст для скоринга (транскрипт/описание).")
    p_pred.add_argument("--hashtags", type=str, default=None,
                        help="Хэштеги через запятую, напр. '#доход,#казино'.")
    p_pred.add_argument("--kb-similarity", dest="kb_similarity", type=float,
                        default=0.0,
                        help="Похожесть на базу знаний 0..1 (по умолчанию 0).")
    p_pred.set_defaults(func=_cmd_predict)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Точка входа CLI. Возвращает код выхода процесса."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
