"""Dataset assembly for the custom multi-task risk model (app.ml).

Turns the synthetic + hand-written corpus into the NumPy arrays the trainer
consumes. Three public functions, per the ML CONTRACT / DESIGN §8:

* ``build_examples(cfg)`` — ``synth.generate`` + a handful of hard-coded RU seed
  examples (real, un-augmented anchors that keep the head distribution honest).
* ``to_arrays(examples, cfg)`` — vectorize features and stack the multi-task
  targets ``{X, y_risk, Y_dims, y_cat, w}`` in the FIXED DIMENSION/ CATEGORY order.
* ``split(arrays, val_frac, seed)`` — deterministic split STRATIFIED by ``is_scam``
  (``y_risk >= 0.5``) so the tiny validation slice has stable, class-balanced
  metrics.

Determinism: every RNG draw threads through ``cfg.seed`` (or the explicit ``seed``
argument). No time-based randomness. Featurization is delegated to
``featurize.vectorize_batch``; labels come from ``synth`` (synthetic ground truth +
teacher-refined dimensions). numpy is a hard dependency here (training side only).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from app.ml import featurize
from app.ml import synth
from app.ml.types import (
    CATEGORY_KEYS,
    DIMENSION_KEYS,
    Example,
    Label,
    RawFeatures,
)

# Index lookups built once from the FIXED contract orderings.
_CAT_INDEX: Dict[str, int] = {k: i for i, k in enumerate(CATEGORY_KEYS)}
_SCAM_THRESHOLD: float = 0.5  # y_risk >= this  <=>  is_scam (stratification key)

__all__ = ["build_examples", "to_arrays", "split", "RU_SEED_EXAMPLES"]


# --------------------------------------------------------------------------- #
# Hard-coded RU seed examples
# --------------------------------------------------------------------------- #
# (text, hashtags, category, is_scam, risk, intended dimensions). These are real,
# un-augmented Russian anchors: a few unambiguous scams across the three scam
# categories plus hard benign negatives (antifraud / educational / neutral) that
# share scam vocabulary but must score low. Risk + category + is_scam are AUTHORED
# ground truth (DESIGN §1: ground-truth owns the risk head); the 8 dimensions are
# refined by the rule TEACHER on this clean text when ``weak_labels`` is available,
# otherwise the seeded ``dims`` below are used so the module never hard-depends on a
# sibling that may still be building in parallel.
_SEED_SPECS: List[Dict[str, object]] = [
    {
        "text": (
            "Гарантированный доход 300% в месяц без вложений и без рисков! "
            "Пиши в личку прямо сейчас, места ограничены."
        ),
        "hashtags": ["#доход", "#инвестиции", "#пассивныйдоход"],
        "category": "investment_scam",
        "is_scam": True,
        "risk": 0.90,
        "dims": {"profit": 0.95, "urgency": 0.7, "messenger": 0.6},
    },
    {
        "text": (
            "Лучшее онлайн казино года! Бонус на первый депозит и фриспины. "
            "Промокод WIN777, регистрация по ссылке в описании."
        ),
        "hashtags": ["#казино", "#ставки", "#бонус"],
        "category": "illegal_gambling",
        "is_scam": True,
        "risk": 0.88,
        "dims": {"gambling": 0.95, "referral": 0.6, "messenger": 0.4},
    },
    {
        "text": (
            "Заработок на криптовалюте! Приглашай друзей по реферальной ссылке "
            "и получай процент с каждого. Чем больше команда — тем выше доход."
        ),
        "hashtags": ["#крипта", "#заработок", "#команда"],
        "category": "financial_pyramid",
        "is_scam": True,
        "risk": 0.86,
        "dims": {"referral": 0.9, "profit": 0.7},
    },
    {
        "text": (
            "Осторожно, мошенники! Не переводите деньги незнакомцам и не верьте "
            "обещаниям гарантированного дохода. Это типичная финансовая пирамида."
        ),
        "hashtags": ["#антифрод", "#безопасность", "#финансоваяграмотность"],
        "category": "educational_antifraud",
        "is_scam": False,
        "risk": 0.08,
        "dims": {},
    },
    {
        "text": (
            "Как устроена финансовая грамотность: разбираем, чем отличается "
            "вклад от инвестиции и почему высокий доход всегда означает высокий риск."
        ),
        "hashtags": ["#обучение", "#финансы", "#инвестиции"],
        "category": "educational",
        "is_scam": False,
        "risk": 0.07,
        "dims": {},
    },
    {
        "text": (
            "Сегодня испекли домашний хлеб на закваске, делюсь рецептом. "
            "Получилось ароматно, корочка хрустящая, тесто поднялось отлично."
        ),
        "hashtags": ["#рецепты", "#выпечка", "#хлеб"],
        "category": "no_violation",
        "is_scam": False,
        "risk": 0.03,
        "dims": {},
    },
]


def _seed_label(spec: Dict[str, object]) -> Label:
    """Build the authored ground-truth Label for one seed spec.

    Refines the 8 ScamDNA dimensions via the rule teacher on the CLEAN text when
    ``weak_labels`` is importable (DESIGN §1: teacher as dimension-oracle); falls
    back to the seeded ``dims`` otherwise so dataset.py imports even while sibling
    modules are still being written.
    """
    is_scam = bool(spec["is_scam"])
    risk = float(spec["risk"])
    category = str(spec["category"])
    seeded_dims = dict(spec.get("dims") or {})  # type: ignore[arg-type]

    dimensions = {k: float(seeded_dims.get(k, 0.0)) for k in DIMENSION_KEYS}
    try:  # teacher refinement (optional, parallel-built sibling)
        from app.ml import weak_labels

        teacher = weak_labels.weak_label_from_text(
            str(spec["text"]),
            hashtags=list(spec.get("hashtags") or []),  # type: ignore[arg-type]
        )
        for k in DIMENSION_KEYS:
            dimensions[k] = float(teacher.dimensions.get(k, dimensions[k]))
    except Exception:
        pass  # keep seeded dims; never let a sibling import break assembly

    return Label(
        risk=risk,
        dimensions=dimensions,
        category=category,
        is_scam=is_scam,
        source="gold",
        weight=1.0,
    )


def _seed_examples() -> List[Example]:
    """Construct the hard-coded RU seed Examples (RawFeatures + authored Label)."""
    out: List[Example] = []
    for i, spec in enumerate(_SEED_SPECS):
        rf = RawFeatures(
            text=str(spec["text"]),
            hashtags=list(spec.get("hashtags") or []),  # type: ignore[arg-type]
            lang_hint="ru",
        )
        out.append(Example(id=f"seed-ru-{i:02d}", features=rf, label=_seed_label(spec)))
    return out


# Eagerly materialized for tests / introspection (deterministic, cheap).
RU_SEED_EXAMPLES: List[Example] = _seed_examples()


# --------------------------------------------------------------------------- #
# build_examples
# --------------------------------------------------------------------------- #
def build_examples(cfg=None) -> List[Example]:
    """Assemble the full training corpus: synthetic rows + RU seed anchors.

    ``synth.generate(cfg.synth_size, cfg.seed)`` provides the bulk
    (template-generated, obfuscation/paraphrase-augmented, hybrid-labeled); the
    hand-written RU seed examples are appended as real un-augmented anchors. The
    label-correctness asserts (DESIGN §8) live in the test-suite, not here, so a
    production build never hard-fails on a single noisy synthetic row.
    """
    if cfg is None:
        from app.ml.config import ml_config as cfg  # local import: avoid cycles

    examples: List[Example] = list(synth.generate(cfg.synth_size, cfg.seed))
    examples.extend(_seed_examples())
    return examples


# --------------------------------------------------------------------------- #
# to_arrays
# --------------------------------------------------------------------------- #
def to_arrays(examples: List[Example], cfg=None) -> Dict[str, np.ndarray]:
    """Vectorize features and stack multi-task targets into NumPy arrays.

    Returns a dict with the CONTRACT keys::

        X      (N, INPUT_DIM) float32  — featurize.vectorize_batch over features
        y_risk (N,)           float32  — soft risk target in [0, 1]
        Y_dims (N, 8)         float32  — 8 ScamDNA dims, DIMENSION_KEYS order
        y_cat  (N,)           int64    — index into CATEGORY_KEYS (unknown -> 0)
        w      (N,)           float32  — per-example sample weight (>= 0)

    Examples missing a ``label`` are tolerated (zeroed risk/dims, category 0,
    weight 0) so a malformed row never derails a batch. Empty input yields
    correctly-shaped empty arrays.
    """
    if cfg is None:
        from app.ml.config import ml_config as cfg

    n = len(examples)
    input_dim = int(featurize.INPUT_DIM)
    n_dims = len(DIMENSION_KEYS)

    if n == 0:
        return {
            "X": np.zeros((0, input_dim), dtype=np.float32),
            "y_risk": np.zeros((0,), dtype=np.float32),
            "Y_dims": np.zeros((0, n_dims), dtype=np.float32),
            "y_cat": np.zeros((0,), dtype=np.int64),
            "w": np.zeros((0,), dtype=np.float32),
        }

    raw = [ex.features for ex in examples]
    X = np.asarray(featurize.vectorize_batch(raw), dtype=np.float32)
    if X.shape != (n, input_dim):  # defensive: keep the trainer's contract exact
        X = X.reshape(n, input_dim).astype(np.float32, copy=False)

    y_risk = np.zeros((n,), dtype=np.float32)
    Y_dims = np.zeros((n, n_dims), dtype=np.float32)
    y_cat = np.zeros((n,), dtype=np.int64)
    w = np.zeros((n,), dtype=np.float32)

    for i, ex in enumerate(examples):
        lab = ex.label
        if lab is None:
            continue
        y_risk[i] = float(np.clip(lab.risk, 0.0, 1.0))
        dims = lab.dimensions or {}
        for d, key in enumerate(DIMENSION_KEYS):
            Y_dims[i, d] = float(np.clip(dims.get(key, 0.0), 0.0, 1.0))
        y_cat[i] = _CAT_INDEX.get(lab.category, 0)
        w[i] = float(max(0.0, lab.weight))

    return {"X": X, "y_risk": y_risk, "Y_dims": Y_dims, "y_cat": y_cat, "w": w}


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def split(
    arrays: Dict[str, np.ndarray],
    val_frac: float,
    seed: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Split arrays into (train, val), STRATIFIED by ``is_scam`` (y_risk >= 0.5).

    Each class is shuffled with a ``seed``-derived Generator and split by
    ``val_frac`` independently, so both partitions preserve the global scam/benign
    balance even when the validation slice is tiny (DESIGN §8). Deterministic for a
    fixed ``seed``. Degenerate fractions are clamped so neither side is empty when
    the data permits.
    """
    keys = ("X", "y_risk", "Y_dims", "y_cat", "w")
    y_risk = arrays["y_risk"]
    n = int(y_risk.shape[0])

    if n == 0:
        empty = {k: arrays[k][:0].copy() for k in keys}
        return empty, {k: arrays[k][:0].copy() for k in keys}

    frac = float(min(max(val_frac, 0.0), 1.0))
    rng = np.random.default_rng(int(seed))

    is_scam = y_risk >= _SCAM_THRESHOLD
    scam_idx = np.flatnonzero(is_scam)
    benign_idx = np.flatnonzero(~is_scam)

    val_parts: List[np.ndarray] = []
    train_parts: List[np.ndarray] = []
    for group in (scam_idx, benign_idx):
        g = group.copy()
        rng.shuffle(g)  # in-place, seeded
        m = g.shape[0]
        if m == 0:
            continue
        n_val = int(round(m * frac))
        # Keep at least one example on each side when a meaningful fraction was
        # requested and the group is large enough to spare it.
        if 0.0 < frac < 1.0 and m >= 2:
            n_val = min(max(n_val, 1), m - 1)
        val_parts.append(g[:n_val])
        train_parts.append(g[n_val:])

    val_idx = (
        np.concatenate(val_parts) if val_parts else np.empty((0,), dtype=np.int64)
    )
    train_idx = (
        np.concatenate(train_parts) if train_parts else np.empty((0,), dtype=np.int64)
    )
    # Re-shuffle the merged indices so scam/benign rows interleave in each split.
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    def _take(idx: np.ndarray) -> Dict[str, np.ndarray]:
        return {k: arrays[k][idx] for k in keys}

    return _take(train_idx), _take(val_idx)
