"""Самопроверяющийся быстрый smoke-тест ML-конвейера (без pytest).

Запуск::

    python -m app.ml.tests.test_pipeline

Прогоняет ВЕСЬ путь обучаемой модели на УМЕНЬШЕННОМ конфиге (≈400 синтетических
строк, ≈4 эпохи), детерминированно и автономно, и проверяет ключевые свойства:

  1. train  — обучение действительно снижает loss (история убывает);
  2. predict — возвращает валидный :class:`Prediction` (диапазоны, ключи, тип);
  3. сигнал — явный скам получает ВЫСОКИЙ риск, а безобидный/обучающий текст —
     НИЗКИЙ (студент отделяет полярность);
  4. обфускация — обфусцированный скам всё ещё получает ВЫСОКИЙ риск (главная
     ценность модели: де-обфускация, недоступная лексиконам);
  5. калибровка — temperature scaling НЕ ухудшает ECE на валидации;
  6. сериализация — save/load делает round-trip (предсказания совпадают).

Падение любой проверки -> ненулевой код выхода (``sys.exit(1)``). numpy здесь —
жёсткая зависимость (это обучающий путь, не inference-firewall).
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from dataclasses import replace
from typing import Callable, List, Tuple

import numpy as np

# Этот тест печатает русский текст и символ Δ; на Windows-консоли кодировка по
# умолчанию (cp1251) роняет вывод. Переключаем потоки на UTF-8 с заменой, чтобы
# печать НИКОГДА не приводила к падению теста из-за кодека терминала.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - старые/обёрнутые потоки без reconfigure
        pass

from app.config import risk_level
from app.ml import calibrate, dataset, metrics
from app.ml.config import MLConfig, ml_config
from app.ml.model_np import NpRiskModel
from app.ml.types import (
    CATEGORY_KEYS,
    DIMENSION_KEYS,
    Prediction,
    RawFeatures,
)

# --------------------------------------------------------------------------- #
# Маленький детерминированный конфиг для быстрого, но настоящего обучения.
# --------------------------------------------------------------------------- #
_TMP_DIR = tempfile.mkdtemp(prefix="amw_ml_test_")


def _small_cfg() -> MLConfig:
    """Уменьшенный, но репрезентативный конфиг (детерминированный, < нескольких сек).

    Артефакты пишутся во временный каталог, чтобы тест НЕ затирал реальную
    обученную модель в ``models_store``.
    """
    return replace(
        ml_config,
        synth_size=400,
        epochs=4,
        val_frac=0.2,
        calibration="temperature",
        model_path=os.path.join(_TMP_DIR, "risk_model_test.npz"),
        meta_path=os.path.join(_TMP_DIR, "risk_model_test.json"),
        active_pointer=os.path.join(_TMP_DIR, "ACTIVE_test.json"),
    )


# --------------------------------------------------------------------------- #
# Якорные тексты (явный скам / безобидный / обфусцированный скам).
# --------------------------------------------------------------------------- #
# Явный, лексиконо-детектируемый инвестиционный скам.
_SCAM_TEXT = (
    "Гарантированный доход 300% в месяц без вложений и без рисков! "
    "Пиши в личку прямо сейчас, места ограничены, успей вписаться."
)
_SCAM_HASHTAGS = ["#доход", "#инвестиции", "#пассивныйдоход"]

# Безобидный обучающий/антифрод-текст (лексически похож на скам, но негативно
# маркирован: ДОЛЖЕН получить низкий риск).
_BENIGN_TEXT = (
    "Сегодня испекли домашний хлеб на закваске, делюсь рецептом. "
    "Получилось ароматно, корочка хрустящая, тесто отлично поднялось."
)
_BENIGN_HASHTAGS = ["#рецепты", "#выпечка", "#хлеб"]

# Обфусцированный скам: leetspeak + кириллица/латиница + пробелы внутри слов.
# Лексикон тут «слепнет», но char-n-gram skeleton модели должен распознать риск.
_OBFUSCATED_TEXT = (
    "Г@р@нтир0ванный д0х0д 300% в мес  б е з  вл0жений! "
    "пиши в лuчку, kазин0 бонус, места 0граничены"
)
_OBFUSCATED_HASHTAGS = ["#дох0д", "#k@зино"]


def _rf(text: str, hashtags: List[str]) -> RawFeatures:
    """Минимальный :class:`RawFeatures` из текста + хэштегов (язык — эвристика)."""
    return RawFeatures(text=text, hashtags=list(hashtags), lang_hint="ru")


# --------------------------------------------------------------------------- #
# Лёгкий ассерт-каркас (без pytest): копит ошибки, не падает на первой.
# --------------------------------------------------------------------------- #
class _Checker:
    """Собирает результаты проверок; печатает отчёт; даёт итоговый код выхода."""

    def __init__(self) -> None:
        self.failures: List[str] = []
        self.passed: int = 0

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passed += 1
            print(f"[ OK ] {name}")
        else:
            msg = f"{name}: {detail}" if detail else name
            self.failures.append(msg)
            print(f"[FAIL] {msg}")
        return ok

    def run(self, name: str, fn: Callable[[], None]) -> None:
        """Выполнить блок проверок; перехватить исключение как провал."""
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - тест обязан пережить любую ошибку
            tb = traceback.format_exc()
            self.failures.append(f"{name}: исключение {exc!r}\n{tb}")
            print(f"[FAIL] {name}: исключение {exc!r}")
            print(tb)

    def summary_exit_code(self) -> int:
        total = self.passed + len(self.failures)
        print("\n" + "=" * 60)
        print(f"Итог: пройдено {self.passed}/{total} проверок.")
        if self.failures:
            print("ПРОВАЛЫ:")
            for f in self.failures:
                print(f"  - {f.splitlines()[0]}")
            return 1
        print("Все проверки пройдены.")
        return 0


# --------------------------------------------------------------------------- #
# Общая обучающая фикстура — один прогон, переиспользуемый всеми проверками.
# --------------------------------------------------------------------------- #
def _train_once(cfg: MLConfig):
    """Собрать данные -> массивы -> сплит -> обучить -> калибровать.

    Возвращает ``(model, history, train_arrays, val_arrays)``. Детерминировано
    по ``cfg.seed``.
    """
    examples = dataset.build_examples(cfg)
    arrays = dataset.to_arrays(examples, cfg)
    train_arrays, val_arrays = dataset.split(arrays, cfg.val_frac, cfg.seed)

    model = NpRiskModel(cfg)
    history = model.fit(train_arrays, val_arrays, cfg)
    return model, history, train_arrays, val_arrays


# --------------------------------------------------------------------------- #
# Отдельные проверки
# --------------------------------------------------------------------------- #
def _valid_prediction(pred: object, cfg: MLConfig) -> Tuple[bool, str]:
    """Проверить, что объект — корректный :class:`Prediction` (типы/диапазоны)."""
    if not isinstance(pred, Prediction):
        return False, f"ожидался Prediction, получен {type(pred).__name__}"
    if not (0 <= pred.risk_score <= 100):
        return False, f"risk_score вне 0..100: {pred.risk_score}"
    if not (0.0 <= pred.risk_prob <= 1.0):
        return False, f"risk_prob вне 0..1: {pred.risk_prob}"
    if pred.risk_level != risk_level(pred.risk_score):
        return False, (
            f"risk_level '{pred.risk_level}' не согласован с risk_level("
            f"{pred.risk_score})='{risk_level(pred.risk_score)}'"
        )
    if pred.risk_level not in {"low", "medium", "high", "critical"}:
        return False, f"неизвестный risk_level: {pred.risk_level}"
    if set(pred.dimensions.keys()) != set(DIMENSION_KEYS):
        return False, f"ключи dimensions != DIMENSION_KEYS: {sorted(pred.dimensions)}"
    for k, v in pred.dimensions.items():
        if not (0 <= v <= 100):
            return False, f"dimension '{k}' вне 0..100: {v}"
    if pred.category not in CATEGORY_KEYS:
        return False, f"category '{pred.category}' не из CATEGORY_KEYS"
    if not (0.0 <= pred.confidence <= 1.0):
        return False, f"confidence вне 0..1: {pred.confidence}"
    if not isinstance(pred.uncertain, bool):
        return False, f"uncertain не bool: {type(pred.uncertain).__name__}"
    if pred.model_version != cfg.version:
        return False, (
            f"model_version '{pred.model_version}' != cfg.version '{cfg.version}'"
        )
    return True, ""


def main() -> int:
    chk = _Checker()
    cfg = _small_cfg()

    # --- обучение (один раз) ---------------------------------------------- #
    holder: dict = {}

    def _do_train() -> None:
        model, history, train_arrays, val_arrays = _train_once(cfg)
        holder["model"] = model
        holder["history"] = history
        holder["val"] = val_arrays
        holder["train"] = train_arrays

    chk.run("обучение завершилось без исключений", _do_train)
    if "model" not in holder:
        # Без обученной модели остальные проверки невозможны — выходим с ошибкой.
        return chk.summary_exit_code()

    model: NpRiskModel = holder["model"]
    history = holder["history"]
    val_arrays = holder["val"]

    # --- 1. loss убывает --------------------------------------------------- #
    def _check_loss_decreases() -> None:
        losses = list(history.get("train_loss", []))
        chk.check(
            "история train_loss непуста и длиной с epochs",
            len(losses) == cfg.epochs and len(losses) >= 2,
            f"len(train_loss)={len(losses)}, epochs={cfg.epochs}",
        )
        if len(losses) >= 2:
            chk.check(
                "все значения train_loss конечны",
                all(np.isfinite(l) for l in losses),
                f"train_loss={losses}",
            )
            # Обучение должно снижать loss: финальный заметно ниже первого.
            chk.check(
                "финальный train_loss ниже начального (обучение снижает loss)",
                losses[-1] < losses[0] - 1e-4,
                f"first={losses[0]:.5f} last={losses[-1]:.5f}",
            )

    chk.run("проверка снижения loss", _check_loss_decreases)

    # --- 2. predict валиден ----------------------------------------------- #
    def _check_predict_valid() -> None:
        pred = model.predict(_rf(_SCAM_TEXT, _SCAM_HASHTAGS))
        ok, detail = _valid_prediction(pred, cfg)
        chk.check("predict возвращает валидный Prediction", ok, detail)

        # predict_batch согласуется с predict на тех же входах.
        batch = [
            _rf(_SCAM_TEXT, _SCAM_HASHTAGS),
            _rf(_BENIGN_TEXT, _BENIGN_HASHTAGS),
        ]
        preds = model.predict_batch(batch)
        chk.check(
            "predict_batch возвращает по одному Prediction на вход",
            isinstance(preds, list) and len(preds) == 2,
            f"len(preds)={len(preds) if isinstance(preds, list) else preds!r}",
        )
        if len(preds) == 2:
            single = model.predict(batch[0])
            chk.check(
                "predict_batch согласован с predict (тот же risk_score)",
                preds[0].risk_score == single.risk_score,
                f"batch={preds[0].risk_score} single={single.risk_score}",
            )

    chk.run("проверка валидности predict", _check_predict_valid)

    # --- 3. скам высоко, безобидное низко --------------------------------- #
    def _check_signal_separation() -> None:
        scam = model.predict(_rf(_SCAM_TEXT, _SCAM_HASHTAGS))
        benign = model.predict(_rf(_BENIGN_TEXT, _BENIGN_HASHTAGS))
        print(
            f"      [инфо] риск: scam={scam.risk_prob:.3f} "
            f"benign={benign.risk_prob:.3f} obf=?"
        )
        chk.check(
            "явный скам получает высокий риск (prob >= 0.6)",
            scam.risk_prob >= 0.60,
            f"scam.risk_prob={scam.risk_prob:.3f}",
        )
        chk.check(
            "безобидный текст получает низкий риск (prob <= 0.4)",
            benign.risk_prob <= 0.40,
            f"benign.risk_prob={benign.risk_prob:.3f}",
        )
        chk.check(
            "скам ранжируется выше безобидного (разделение полярности)",
            scam.risk_prob > benign.risk_prob + 0.20,
            f"scam={scam.risk_prob:.3f} benign={benign.risk_prob:.3f}",
        )

    chk.run("проверка разделения сигнала", _check_signal_separation)

    # --- 4. обфусцированный скам всё ещё высоко --------------------------- #
    def _check_obfuscation_robustness() -> None:
        obf = model.predict(_rf(_OBFUSCATED_TEXT, _OBFUSCATED_HASHTAGS))
        benign = model.predict(_rf(_BENIGN_TEXT, _BENIGN_HASHTAGS))
        print(f"      [инфо] obf.risk_prob={obf.risk_prob:.3f}")
        chk.check(
            "обфусцированный скам всё ещё получает высокий риск (prob >= 0.6)",
            obf.risk_prob >= 0.60,
            f"obf.risk_prob={obf.risk_prob:.3f}",
        )
        chk.check(
            "обфусцированный скам ранжируется выше безобидного",
            obf.risk_prob > benign.risk_prob + 0.20,
            f"obf={obf.risk_prob:.3f} benign={benign.risk_prob:.3f}",
        )

    chk.run("проверка устойчивости к обфускации", _check_obfuscation_robustness)

    # --- 5. калибровка не ухудшает ECE ------------------------------------ #
    def _check_calibration_ece() -> None:
        X = np.asarray(val_arrays["X"], dtype=np.float32)
        y = (np.asarray(val_arrays["y_risk"], dtype=np.float64) >= 0.5).astype(
            np.float64
        )
        if X.shape[0] == 0 or y.min() == y.max():
            chk.check(
                "валидация пригодна для оценки ECE (есть оба класса)",
                False,
                "вырожденная валидация (пусто или один класс)",
            )
            return

        # Сырые (T=1) вероятности до калибровки: временно нейтрализуем температуру.
        saved_T = model.temperature
        model.temperature = 1.0
        prob_uncal, _, _ = model._forward_probs(X)
        prob_uncal = np.asarray(prob_uncal, dtype=np.float64).reshape(-1)
        model.temperature = saved_T

        ece_before = metrics.ece(y, prob_uncal)

        # Калибруем модель на валидации (in-place, против наземной правды).
        calibrate.calibrate_model(model, val_arrays, cfg)
        chk.check(
            "температура после калибровки положительна и конечна",
            np.isfinite(model.temperature) and model.temperature > 0.0,
            f"temperature={model.temperature}",
        )

        prob_cal, _, _ = model._forward_probs(X)
        prob_cal = np.asarray(prob_cal, dtype=np.float64).reshape(-1)
        ece_after = metrics.ece(y, prob_cal)

        print(
            f"      [инфо] ECE: до={ece_before:.4f} после={ece_after:.4f} "
            f"T={model.temperature:.3f}"
        )
        # Калибровка НЕ должна заметно ухудшать ECE (допуск на дискретизацию бинов).
        chk.check(
            "калибровка не ухудшает ECE (после <= до + допуск)",
            ece_after <= ece_before + 1e-3,
            f"ece_before={ece_before:.4f} ece_after={ece_after:.4f}",
        )

    chk.run("проверка калибровки (ECE)", _check_calibration_ece)

    # --- 6. save/load round-trip ------------------------------------------ #
    def _check_save_load_roundtrip() -> None:
        inputs = [
            _rf(_SCAM_TEXT, _SCAM_HASHTAGS),
            _rf(_BENIGN_TEXT, _BENIGN_HASHTAGS),
            _rf(_OBFUSCATED_TEXT, _OBFUSCATED_HASHTAGS),
        ]
        before = model.predict_batch(inputs)

        model.save(cfg.model_path)
        chk.check(
            "артефакт .npz записан на диск",
            os.path.exists(cfg.model_path),
            f"нет файла {cfg.model_path}",
        )
        chk.check(
            "meta JSON записан на диск",
            os.path.exists(cfg.meta_path),
            f"нет файла {cfg.meta_path}",
        )

        loaded = NpRiskModel.load(cfg.model_path, cfg)
        chk.check(
            "температура пережила round-trip",
            abs(float(loaded.temperature) - float(model.temperature)) < 1e-6,
            f"before={model.temperature} after={loaded.temperature}",
        )
        chk.check(
            "версия модели пережила round-trip",
            loaded.model_version == model.model_version,
            f"before={model.model_version} after={loaded.model_version}",
        )

        after = loaded.predict_batch(inputs)
        max_dprob = max(
            abs(b.risk_prob - a.risk_prob) for b, a in zip(before, after)
        )
        max_dscore = max(
            abs(b.risk_score - a.risk_score) for b, a in zip(before, after)
        )
        print(
            f"      [инфо] round-trip Δprob_max={max_dprob:.2e} "
            f"Δscore_max={max_dscore}"
        )
        chk.check(
            "predict совпадает до и после save/load (round-trip)",
            max_dprob < 1e-5 and max_dscore == 0,
            f"Δprob_max={max_dprob:.2e} Δscore_max={max_dscore}",
        )

    chk.run("проверка save/load round-trip", _check_save_load_roundtrip)

    return chk.summary_exit_code()


if __name__ == "__main__":
    sys.exit(main())
