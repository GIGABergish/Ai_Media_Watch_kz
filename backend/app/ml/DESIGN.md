# app.ml — Custom Multi-Task Scam-Risk Model — DESIGN

A NOVEL, learned, calibrated, multi-task risk model that the AI Media Watch engine
runs alongside / over its rule engine. It is built by **weak-supervision
distillation** from the existing rule engine, made **obfuscation-robust** by
char-n-gram signed feature-hashing, and served as a small portable `.npz` that
needs **only NumPy** at inference time. The rule engine remains the always-available
fallback (lite mode).

This document is the single buildable plan. Where a panel design conflicts with the
**FIXED ML CONTRACT** signatures in `app/ml/types.py`, `app/ml/config.py`,
`app/config.py`, the **contract wins**. All conflicts below are resolved with a
concrete decision.

---

## 0. The product thesis (the moat)

The rule teacher (`scam_dna -> risk_score -> category`) matches plain
substrings/regex over raw text. It goes blind the moment text is obfuscated:
`г@рантир0ванный д0ход`, `к а з и н о`, `vаvаdа` (mixed Cyrillic/Latin). Signed
feature-hashing of **character 3–5-grams** degrades gracefully under exactly these
perturbations. A NumPy MLP trained on teacher labels of **clean** text but fed the
**obfuscated** surface learns the deobfuscation invariance the lexicon provably
lacks. That single capability — *"catches what the rules can't, while agreeing with
them everywhere else"* — is the whole point and must be **measured** in
`evaluate.py` (model AUROC / recall on an obfuscated held-out slice vs. the rule
baseline on the same slice).

Four differentiators, in priority order:
1. **Distillation + obfuscation-robustness** (the moat).
2. **Calibrated uncertainty -> human-in-the-loop triage** (temperature-scaled prob
   + uncertain flag drives an active-learning review queue).
3. **Explainability in the rules' own vocabulary** (attributions map to `dna_key`,
   so the model plugs into the existing 8-dim ScamDNA evidence-card UI with zero
   frontend change).
4. **PSI drift detection** (cheap insurance for a hostile, shifting domain).

The torch model, isotonic calibration, and any embedding/transformer path are
**strictly optional** and must never block the NumPy-only critical path
(train < 1 min, serve from a ~4 MB `.npz`, numpy the only runtime dep).

---

## 1. The load-bearing measured fact (resolves the biggest conflict)

`compute_risk_score` blends five components with weights
`text=0.35, visual=0.25, metadata=0.15, behavior=0.15, db=0.10` and applies
negative damping. A realistic **single-theme** scam (only gambling, or only
profit) therefore saturates **one** component and scores ~32–39 overall — **below**
the teacher's own 0.5 scam threshold. Obfuscated scams score ~0 (lexicon misses
everything). The teacher only reaches ~90 when text+visual+links+behavior are ALL
saturated at once.

**Decision (the most important one):** do **NOT** derive the binary `is_scam` / the
risk target from the teacher's overall score. If we distilled `is_scam` from the
teacher, ~half of single-theme scams would be taught to the model as *benign*.

Instead we **split supervision by head:**
- **Risk + is_scam + category: owned by SYNTHETIC GROUND TRUTH** (known by
  construction — the generator picks the category and intensity before emitting
  text).
- **The 8 ScamDNA dimensions: owned by the TEACHER** run on the **clean**
  pre-obfuscation text (where `compute_scam_dna` / `classify_category` are reliably
  correct even when the overall score is diluted low).

This is "teacher as dimension-oracle, ground-truth as risk-oracle." It is the clean
fix for the teacher's formula-dilution bias and prevents the student from inheriting
the rule engine's primary blind spot.

---

## 2. The core distillation trick: label-on-clean, train-on-obfuscated

`synth.generate()` emits, per scam item, a **paired** (clean, obfuscated) twin that
share the **same** ground-truth label.
- The teacher (and dimension extraction) is run on the **clean** string, where the
  lexicon fires correctly and yields strong, correct soft dimension targets.
- `featurize.vectorize()` hashes the **obfuscated** string the student actually
  consumes.

The student must therefore learn the leetspeak / spacing / Cyrillic-Latin
invariance the lexicon lacks. Memorizing the lexicon is impossible because the
lexicon substrings are frequently absent from the student's actual input.
**Label-then-obfuscate is mandatory** — obfuscate-then-label would teach the rules'
blind spot, exactly backwards.

For the obfuscated twin we **reuse the clean-text dimension targets** (meaning is
unchanged; only the surface is corrupted) rather than re-running the teacher on
garbled text (which yields 0).

---

## 3. Architecture — `NpRiskModel` (pure NumPy, multi-task)

Single **shared-trunk** MLP. Depth buys nothing here (hashed bag-of-n-grams is
already high-dim sparse, the teacher is a saturating-linear function of hits, a 2nd
hidden layer would overfit the ~4k distilled set). The **bottleneck IS the
generalizer**.

```
x  (INPUT_DIM = hash_dim + numeric_dim = 4096 + 32 = 4128, float32)
   -> Linear(4128 -> hidden=256)      # the only big matrix (~1.05M params)
   -> ReLU
   -> Dropout(p=cfg.dropout=0.10)     # inverted dropout, train-only, seeded
   = shared representation h (256)
   -> risk head:     Linear(256 -> 1) + sigmoid   -> p_risk in (0,1)
   -> dims head:     Linear(256 -> 8) + sigmoid   -> 8 indep ScamDNA probs (DIMENSION_KEYS order)
   -> category head: Linear(256 -> 7) + softmax   -> CATEGORY_KEYS distribution (len = 7)
```

The 4128->256 (16x) squeeze forces semantically-equivalent obfuscated spellings —
which share overlapping char-3..5-gram hash buckets — onto the **same latent risk
direction**. That turns hashing + bottleneck into a learned obfuscation-invariance
the lexicons cannot express.

**Init:** He/Kaiming-uniform for the trunk (fan_in=4128, ReLU), Xavier/Glorot for
the three heads, biases zero. float32 throughout. Keeps activations well-scaled at
the very high fan-in so `lr=0.01` is stable in early epochs.

**`self.temperature`** float, default `1.0`, set by `calibrate`.

---

## 4. Losses, optimizer, regularization (training)

Multi-task, sample-weighted by `Label.weight`:

```
L = w_risk * BCE(p_risk, y_risk)
  + w_dim  * mean_over_8 BCE(p_dim_k, y_dim_k)
  + w_cat  * CrossEntropy(softmax_cat, y_cat)
  + l2 * 0.5 * sum(W^2)        # WEIGHTS only, not biases
```
Weights from cfg: `w_risk=1.0, w_dim=0.5, w_cat=0.5`, `l2=1e-5`. `y_risk` and
`y_dim` are **soft** targets in [0,1] (graded distillation, not hard 0/1) — soft BCE
= cross-entropy against the teacher/ground-truth probability surface; this transfers
graded confidence and makes the dims head a strong **auxiliary regularizer** on the
shared trunk. Per-example sample weight multiplies the total example loss. Logs
clipped with `eps=1e-7`.

**Optimizer:** manual **Adam** (m,v buffers, bias-correction, betas (0.9,0.999),
eps 1e-8), `lr=0.01`. Full manual backprop through the 3-head + shared-trunk graph
(head grads sum at `h`, then through dropout mask, ReLU mask, into the trunk).
Mini-batch SGD `batch_size=64`, `epochs=12`, shuffle each epoch with a
`cfg.seed`-derived `np.random.Generator`. Keep **best-by-val** weights (snapshot at
the epoch with lowest weighted val loss / best val risk AUROC) and restore at the
end.

**Regularization stack** (why 4k examples is enough): (1) 4128->256 bottleneck,
(2) Dropout 0.10, (3) L2 1e-5 on weights, (4) the two auxiliary heads,
(5) best-by-val early-stop snapshot, (6) signed-hash collision noise (mild random
projection), (7) obfuscation + paraphrase augmentation in synth.

**Compute budget:** ~1.06M params dominated by the 4128x256 projection; a full run
is a few seconds of BLAS + Python overhead — comfortably under the 1-minute CPU
budget.

---

## 5. Featurization — `featurize.py` (INPUT_DIM = 4128)

`INPUT_DIM = ml_config.hash_dim + ml_config.numeric_dim`. Vector =
`[0:4096)` signed feature-hashed text n-grams ++ `[4096:4128)` 32 engineered
numeric features, in a **fixed, documented order**.

Obfuscation robustness comes from a **normalization layer**, NOT from regex
lexicons:
1. NFKC unicode-normalize + casefold.
2. **Confusable folding** to a canonical skeleton via a fixed homoglyph/leet map:
   latin lookalikes `a e o p c x y k m h t b` and digits
   `0->о, 1->и/l, 3->е, 4->ч/a, 5->s, 6->б, 7->т, 8->в, @->а, $->s` fold toward
   their Cyrillic/letter skeleton, so `к@зино / kазино / kazino` collapse toward
   `казино`.
3. Collapse runs of whitespace/punct/zero-width chars INSIDE alpha runs so
   `г а р а н т и я` and `к-а-з-и-н-о` rejoin.
4. Strip emoji/zero-width but **keep counts** for numeric features.
5. Emit **two views**: a heavily-folded **skeleton** string for char n-grams
   (maximizes obfuscation recall) and a lightly-folded **word stream** (casefold
   only) for word n-grams so real bigrams like `гарантированный доход` and English
   `guaranteed income` survive.

**Tokenization.** Char n-grams `n=3..4..5` over the skeleton (word boundaries marked
by a sentinel space, runs padded so prefixes/suffixes hash distinctly) — the
obfuscation-catchers. Word n-grams `1..2` over the word stream — phrase cues
(`пиши директ`, `без вложений`).

**Signed two-hash scheme.** For each token `t`: `index = h1(t) mod hash_dim`,
`sign = +1 if (h2(t) bit) == 0 else -1`; accumulate sign at index. **MUST use a
seeded, process-stable hash** — Python's built-in `hash()` is PYTHONHASHSEED-salted
and would make a trained `.npz` unusable at serve time. Use
`hashlib.blake2b(token, digest_size=8, key=<seed>)` (low bytes -> index, a high byte
-> sign) or a fixed FNV-1a. **Namespace-prefix** tokens (`c:` char, `w:` word,
`h:` hashtag) so the three streams don't alias. After accumulation, **L2-normalize**
the `[0:hash_dim)` block so text length doesn't dominate (length carried separately
as a numeric feature). Build the block with `np.add.at` on integer index arrays, not
a Python dict loop (must run on thousands of rows in < 1 min).

**32 engineered numeric features**, fixed order returned by
`numeric_feature_names()`, each clipped + divided by a **constant** cap (no
train-time scaler is serialized -> no train/serve skew). The numeric block is **not**
L2-normalized (the MLP sees raw magnitudes; the first Linear learns relative scale):

| slot | feature | normalization | dna_key (attribution) |
|---|---|---|---|
| 0–7 | `visual_scores` per the 8 DIMENSION_KEYS in order | pass-through 0..1 | identity: profit,urgency,gambling,referral,messenger,visual,reused,hashtags |
| 8 | behavior urgency aggregate | max-conf/100 | urgency |
| 9 | behavior referral aggregate | max-conf/100 | referral |
| 10 | behavior messenger aggregate | max-conf/100 | messenger |
| 11 | behavior-hit count | min(.,5)/5 | "" |
| 12 | negative-marker aggregate (teacher 'negative' key) | /100 | "" |
| 13 | link telegram count | min(.,4)/4 | messenger |
| 14 | link whatsapp count | min(.,4)/4 | messenger |
| 15 | link url count | min(.,4)/4 | messenger |
| 16 | link promocode count | min(.,4)/4 | referral |
| 17 | link phone count | min(.,4)/4 | messenger |
| 18 | total link count | min(.,8)/8 | messenger |
| 19 | kb_similarity | already 0..1 (caller-set) | reused |
| 20 | duration_s | log1p(d)/log1p(600), clip | "" |
| 21 | num_segments | min(.,40)/40 | "" |
| 22 | hashtag_count | min(.,12)/12 | hashtags |
| 23 | suspicious-hashtag ratio (skeleton-match vs hashtags lexicon) | 0..1 | hashtags |
| 24 | text_len | log1p(len)/log1p(2000) | "" |
| 25 | has_url | 0/1 | messenger |
| 26 | has_telegram_or_wa | 0/1 | messenger |
| 27 | has_promocode | 0/1 | referral |
| 28 | digit_ratio of text | 0..1 | "" |
| 29 | emoji + zero-width count | min(.,20)/20 | "" |
| 30 | (reserved) profit text-density / spare | 0..1 | profit |
| 31 | lang code scalar `{'' or ru:0.0, kz:0.33, en:0.66, mixed:1.0}` | 0..1 | "" |

> Note: `numeric_dim` is fixed at 32 by the contract. The table fills exactly 32
> slots; slot 30 is a documented spare (kept 0 if unused). Every divisor is a module
> constant, never data-dependent.

**`extract(bundle)`** reads `transcript.full_text` + OCR text +
`title/description/hashtags` into `rf.text`; `behavior_flags` from `bundle.hits`
where `source == Behavior` by `dna_key` (plus the `'negative'` aggregate);
`visual_scores` by folding `bundle.visual_hits` through `VISION_PROMPTS`
`label -> dna_key` (saturating max per dim to match the teacher); `link_counts` via
`Counter` over `bundle.link_hits` kind; `duration_s` / `num_segments` from
probe/transcript; `lang_hint` via a cheap Cyrillic-vs-Latin char-ratio heuristic
(KZ glyphs `әғқңөұүһі` route to `kz`); `kb_similarity` left 0 for the caller.

**`vectorize(rf) -> (INPUT_DIM,) float32`**, **`vectorize_batch(list) -> (N,
INPUT_DIM)`**, **`numeric_feature_names() -> list[str]` (len 32)**,
**`top_text_features(text, k) -> list[str]`** (re-extract same normalized n-grams,
surface representative raw spellings for explain).

---

## 6. Weak supervision / labels — `weak_labels.py`

- **`weak_label(bundle) -> Label`**: run the teacher
  `compute_scam_dna(bundle) -> compute_risk_score(bundle, dna) ->
  classify_category(bundle, dna, breakdown)`. Returns
  `risk = score/100`, `dimensions = {k: v/100}`, `category = key`,
  `is_scam = risk >= 0.5`, `source = "weak"`, `weight` from teacher **margin**.
- **`weak_label_from_text(text, hashtags=None, platform=None) -> Label`**: build a
  text-only `SignalBundle` (`MediaInput(source_type="text", ...)`), run the cheap
  lanes (links / text_signals / behavior) then `weak_label`. This is the path for
  real un-augmented text and the orchestrator-side cold inputs.

**Margin (teacher confidence).** Defined in the engine's **own** threshold space
(reuse `settings.thresholds` / `app.config.risk_level`): distance of the teacher
risk to the nearest band boundary (medium=40 / high=65 / critical=88), normalized,
with a small floor so no example is fully dropped. Examples deep in a band pull
hard; band-boundary examples (genuinely ambiguous to the engine) are down-weighted.

**Dual-source reconciliation (used by synth, see §7):** synthetic ground truth owns
`risk / is_scam / category`; the teacher owns `dimensions`. When both run on a synth
row, the final `Label.weight = margin * agreement`, where `agreement` is 1.0 when
the teacher's sign matches the synthetic `is_scam` and down-weighted (~0.3–0.5) when
it contradicts (obfuscation-confused miss or label noise) — never let a contradicted
teacher flip a known-benign target into a scam.

---

## 7. Synthetic + adversarial data — `synth.py`

Deterministic, slot-filling template generator, four layers:

1. **Per-category TEMPLATE BANK.** Each `CATEGORY_KEYS` category has templates; each
   template is a sequence of typed SLOTS drawing from typed phrase banks (hook,
   profit_claim, gambling_brand, urgency, referral, messenger_cta, hashtag_cluster,
   plus benign greeting / topic / neutral_body). Banks are seeded with **actual
   lexicon anchors** (so clean rows are teacher-detectable) **PLUS many paraphrases /
   synonyms NOT in the lexicon** (so the model is forced to generalize beyond exact
   strings even before obfuscation). Do not import lexicons as source of truth — copy
   representative anchors and author novel paraphrases.

2. **OBFUSCATION + PARAPHRASE augmentation** (applied probabilistically, ~45% of
   scam rows get an obfuscated twin sharing the SAME label):
   - Cyrillic<->Latin homoglyph swap (`а/a о/o е/e р/p с/c х/x к/k`),
   - digit-for-letter leetspeak (`о->0 з->3 э->3 и->1 ч->4`),
   - intra-word spacing / dots / zero-width separators (`к а з и н о`, `к.а.з.и.н.о`),
   - emoji injection between tokens, selective char duplication.
   Confined to scam-signal tokens, human-readable, capped simultaneous techniques
   per word. **Paraphrase** (orthogonal): multiple sentence frames per intent,
   synonym substitution from the non-lexicon bank, sentence reordering, and
   **code-mix** (insert a KZ/EN clause into an RU scam: `кепілдік табыс`,
   `guaranteed income`, `casino bonus`).

3. **HYBRID LABELING** (per §1/§6): synthetic ground truth owns
   `risk / is_scam / category` and `source="synthetic"`; teacher (on CLEAN text)
   refines the 8 `dimensions`; carry margin into `weight`. Obfuscated twins reuse
   the clean twin's dimension targets.

   **Intensity-tiered risk targets** (smooth distribution the calibrator can use):
   benign-neutral ~0.02–0.08; benign-educational/antifraud ~0.05–0.15 (MUST stay low
   despite scam vocabulary); borderline/suspicious_other ~0.45–0.62; single-theme
   scam ~0.70–0.82; multi-theme stacked scam ~0.85–0.97. Tier = how many independent
   scam slots the template composed.

4. **BALANCING + dedup.** ~50% scam / ~50% benign overall (matches stratified split
   by `is_scam`). Within scam, balance illegal_gambling / financial_pyramid /
   investment_scam (+ passive-income-DM mapped to investment_scam or
   suspicious_other) roughly evenly. Within benign, **over-represent hard negatives**
   educational + educational_antifraud (~60% of the benign half) — lexically
   scam-like but negatively-marked; they are the documented false-positive mode and
   exercise the teacher's negative-damping path so the model learns the polarity
   flip. ~5–8% `suspicious_other` borderline for the uncertain band. Exact-text dedup
   across the whole set (must not collapse a clean/obfuscated pair). LANGUAGE MIX:
   RU ~70%, KZ ~10%, EN ~10%, code-mix ~10% — benign content in **every** language at
   the same code-mix rate so lang never becomes a spurious scam predictor.

**Determinism:** a single `np.random.Generator` seeded from `cfg.seed` threads
through category / template / every slot draw / every augmentation coin-flip. No
time-based randomness. `generate(n, seed) -> list[Example]`. (`evaluate.py` uses a
DIFFERENT seed.)

`CATEGORY_KEYS` routing intent: illegal_gambling -> gambling DNA >=55;
financial_pyramid -> referral >=70; investment_scam -> profit >=60; educational /
educational_antifraud -> negative markers dominate; no_violation -> zero scam vocab;
suspicious_other -> one weak ambiguous signal (falls through to fallback).

---

## 8. Dataset assembly — `dataset.py`

- **`build_examples(cfg=ml_config) -> list[Example]`**:
  `synth.generate(cfg.synth_size, cfg.seed)` + a handful of hard-coded RU seed
  examples. Cheap **label-correctness asserts** (fail loudly in tests, not silently):
  (a) every scam row `is_scam=True, risk>=0.45`; every benign `is_scam=False,
  risk<=0.2`; (b) on CLEAN text `classify_category` equals the intended category for
  a high fraction (small mismatch budget for suspicious_other/borderline);
  (c) educational/antifraud rows produce a non-empty negative-marker hit on clean
  text; (d) obfuscated rows preserve their parent's label and differ at the surface.
- **`to_arrays(examples, cfg=ml_config) -> dict`** with keys: `X (N,INPUT_DIM)
  float32`, `y_risk (N,) float32`, `Y_dims (N,8) float32` ordered by DIMENSION_KEYS,
  `y_cat (N,) int64` index into CATEGORY_KEYS, `w (N,) float32` sample weights.
- **`split(arrays, val_frac, seed) -> (train, val)`**: **stratified by `is_scam`**
  (`y_risk >= 0.5`) so tiny-val metrics are stable.

---

## 9. Calibration — `calibrate.py`

- **`fit_temperature(probs, y) -> T`**: convert each prob to a logit
  `log(p/(1-p))` (clip `p` to `[1e-6, 1-1e-6]`), minimize binary NLL of `y` against
  `sigmoid(logit/T)` by **golden-section search** on `T in [0.25, 10]` (NLL unimodal
  in T) — dependency-free, deterministic.
- **`apply(prob, T) -> prob`**: `sigmoid(logit(prob)/T)`.
- **`calibrate_model(model, val, cfg) -> None`**: sets `model.temperature` in place
  per `cfg.calibration` (`"temperature" | "isotonic" | "none"`). **Calibrate against
  GROUND-TRUTH** `is_scam` (synthetic, trustworthy), NOT teacher labels, so
  calibration corrects the teacher's miscalibration rather than baking it in.
  Isotonic only if `sklearn` imports, else fall back to temperature — **never raise**.

After calibration the uncertain band `[uncertain_low, uncertain_high] = [0.40,0.60]`
is a meaningful triage signal.

---

## 10. The model `predict` / serialization — `model_np.py`

`predict(features) -> Prediction`: `vectorize` -> forward in **eval mode (dropout
off)** -> risk logit -> `/temperature` -> sigmoid = `risk_prob`;
`risk_score = clamp_score(risk_prob*100)`;
`risk_level = app.config.risk_level(risk_score)` (**REUSE**, never re-threshold);
`dimensions = {k: clamp_score(p_dim_k*100)}` ints 0..100;
`category = CATEGORY_KEYS[argmax]`; `confidence = max(p_cat)`;
`uncertain = uncertain_low <= risk_prob <= uncertain_high`;
`attributions = explain.attribute(self, features)`; `model_version = cfg.version`.

`fit(train, val, cfg) -> history` (per-epoch train/val losses + risk AUROC/Brier +
dim MAE; best-by-val restore). `predict_batch(list) -> list[Prediction]`.

`save(path)` writes `.npz` (`W_trunk,b_trunk,W_risk,b_risk,W_dims,b_dims,W_cat,
b_cat`) **plus** a meta json (cfg snapshot, temperature T, CATEGORY_KEYS, INPUT_DIM,
version) to `cfg.model_path` / `cfg.meta_path`. `staticmethod load(path) ->
NpRiskModel` reconstructs. Import `INPUT_DIM` from `featurize`; use `explain.attribute`,
`calibrate.apply`, and `app.config.risk_level/clamp_score`. Artifact ~4.2 MB,
inference needs only numpy.

---

## 11. Explainability — `explain.py`

`attribute(model, features, top_k=8) -> list[Attribution]`. Unified signed metric =
**input-times-gradient on the risk logit** (first-order linearization of the net at
this input):
- **Numeric channel:** for each ACTIVE numeric slot, contribution =
  `feature_value * effective_weight` where `effective_weight = W_trunk[:,slot]`
  pushed through the hidden layer along the risk-head path. `dna_key` from the static
  slot->dna table in §5; human name from `numeric_feature_names()`.
- **Text channel:** `top_text_features(text, k)` re-extracts + hashes the n-grams,
  ranks by signed pushed contribution to the risk logit; surface the RAW
  (pre-normalization) token spelling for readability, `dna_key=""` (a hashed n-gram
  has no single dim).

Top-k by `|contribution|`, interleaving structured signals
(`промокод (referral)`) with obfuscated phrase tokens. This lands on `dna_key`, so it
plugs straight into the existing evidence-card UI.

---

## 12. Metrics — `metrics.py` (PURE NUMPY)

- `binary_metrics(y, prob) -> {auroc, ap, f1, acc, brier}`. **AUROC** via
  Mann-Whitney rank: `(sum_pos_ranks - n_pos*(n_pos+1)/2)/(n_pos*n_neg)` with average
  ranks for ties; define `0.5` when a class is absent (never NaN). **AP** via the
  PR step integral `sum((R_k - R_{k-1})*P_k)` over descending-prob order. F1/acc/brier
  at threshold 0.5 on the **calibrated** prob.
- `ece(y, prob, bins=15) -> float`: 15 equal-width [0,1] bins,
  `sum (n_bin/N)*|acc_bin - conf_bin|`; empty bins contribute 0.
- `dim_metrics(Ytrue, Ypred) -> {dna_key: {mae, corr}}` ordered by DIMENSION_KEYS;
  Pearson corr = 0.0 for ~zero-variance dims (NaN-free).
- `report(...) -> str` renders everything in **Russian**.

---

## 13. Evaluation — `evaluate.py`

`evaluate(cfg=ml_config) -> dict`:
1. Load the ACTIVE model via `registry.load_active`. If None -> teacher-only card
   with a note (never raise).
2. Build a held-out set with a **DIFFERENT seed** (`cfg.seed + 9973`) that
   **OVERSAMPLES obfuscated scams + code-mix** at a HIGHER obfuscation fraction than
   training (out-of-distribution for memorization, in-distribution for the
   phenomenon). Include benign educational/antifraud negatives to keep FP honest, and
   a few hard hand-written RU seed cases.
3. Ground truth = synthetic `is_scam` (known by construction). Score **both** the
   MODEL (`model.predict(features).risk_prob`) and the rule TEACHER
   (`weak_label(bundle).risk`) on the **SAME** bundles, **unweighted**.
4. Carry an `obfuscated` flag per eval example; report a dedicated **obfuscated-slice
   delta table** (model − teacher) for AUROC / F1 / recall, plus the CLEAN-slice
   comparison (where the teacher is strong) so the narrative is "comparable when
   literal, far better under obfuscation" — not a rigged win.
5. **Recall-at-fixed-FPR** operating point (pick each predictor's threshold yielding
   the same FPR on benign, then compare scam recall) neutralizes the teacher's
   coarse 0..100 scale. Also report at the engine's native bands
   (`medium=40/high=65/critical=88`) by scaling `prob*100` (reuse `app.config`).
6. Report ECE/Brier pre- and post-calibration; emit per-bin reliability-curve data
   into the metrics json. Add a "teacher-agreement vs ground-truth-agreement"
   decomposition and an uncertainty-band audit (error rate inside vs outside the
   `[unc_low,unc_high]` band).
7. Write `ML_DIR/MODEL_CARD.md` (RU) + a metrics json (single source of truth). Card
   sections: назначение, данные (synth+weak, seeds, sizes), метрики (overall +
   per-dim + category, pre/post calibration), сравнение с учителем (общая + срез по
   обфускации), калибровка (T, ECE до/после), ограничения и этические риски
   (weak-supervision наследует слепые зоны лексиконов; человек в цикле через
   uncertain-флаг), версия = `cfg.version`.

---

## 14. Registry / serving firewall — `registry.py`, `inference.py`

- **`registry.py`** (= `app.ml.registry`): `save_artifact(model, metrics, cfg) ->
  path`; `load_active() -> RiskModel | None` (cached; **LAZILY** import numpy &
  model_np; return None on ImportError or missing file; **NEVER raise**);
  `current_version()`; `list_versions()`; maintains `ML_DIR/ACTIVE.json`.
- **`inference.py`**: `score_bundle(bundle, kb_similarity=0.0) -> Prediction | None`.
  If not `ml_config.enable` -> None. Lazily `load_active()`; if None -> None (engine
  falls back to rules). `featurize.extract(bundle)`, set `kb_similarity` onto the
  RawFeatures, `model.predict`. **NEVER raise** (catch -> None).

This firewall (lazy numpy import + degrade to None) lets the engine run rules-only in
lite mode and is non-negotiable. Train-side modules may import numpy at top level.

---

## 15. Active learning + drift — `active_learning.py`

`select_uncertain(predictions, k)` (pick by `uncertain` flag / distance to 0.5);
`write_review_queue(path, items)` as JSONL. PSI drift (pure numpy):
`feature_stats(X) -> stats`, `psi(base, cur) -> float`,
`check_drift(base, cur, threshold=0.2) -> report dict`. Bake the numeric
normalization constants into the meta so the PSI baseline is stable.

---

## 16. Training driver + CLI — `train.py`, `cli.py`

- **`train.train(cfg=ml_config) -> dict`**: `build_examples` -> `to_arrays` ->
  `split` -> `NpRiskModel.fit` -> `calibrate_model` -> val metrics ->
  `registry.save_artifact` -> return a metrics dict. Print concise per-epoch
  progress.
- **`cli.py`**: argparse subcommands `gen-data | train | evaluate | predict`.
  `train -> train.train()`; `evaluate -> evaluate.evaluate()`; `gen-data` writes a
  small synthetic sample; `predict --text [--hashtags]` builds a text bundle, runs
  inference, prints the `Prediction` as JSON. Print metrics readably.

---

## 17. Optional torch fusion variant — `model_torch.py` (spec only, NOT default)

True late-fusion multimodal model behind the SAME `RiskModel` Protocol and SAME
`Prediction` contract (drop-in artifact swap behind `registry.load_active`, NumPy
student remains the always-available fallback):
- TEXT branch: small token/char embedding OR a frozen multilingual MiniLM/LaBSE
  (384-d) -> 1–2 Transformer/MLP layers -> 256-d.
- NUMERIC branch: the 32 engineered features -> MLP -> 64-d.
- VISUAL branch: CLIP per-dna `visual_scores` -> projection -> 32-d.
- Concat `[256+64+32]=352` -> fusion MLP (2 layers, GELU, LayerNorm, residual) ->
  256 trunk -> identical risk/dims/category heads, SAME multi-task weighted loss +
  label-distillation. Trainable end-to-end (AdamW, cosine LR, optional focal loss for
  imbalance). **Scale path:** distill its soft outputs back into the NumPy student to
  preserve portable single-numpy inference.

---

## 18. Determinism & hard rules (global)

- Single `cfg.seed` seeds: weight init, synth, stratified split, epoch shuffle,
  dropout masks, **and the feature hash**. No time-based randomness.
- Reuse `app.config.risk_level / clamp_score` for ALL 0..100 mapping and thresholds;
  reuse `app.scoring` strictly as the distillation teacher. Never duplicate
  thresholds/lexicons.
- `inference.py` and `registry.load_active` import numpy/model_np LAZILY and degrade
  to None. Other ml modules may import numpy at top level.
- Russian for human-facing strings; concise docstrings + type hints. Do NOT modify
  spine/rule files.

---

## 19. Per-file build map (every contract module)

| file | builds | key notes |
|---|---|---|
| `featurize.py` | `INPUT_DIM`, `extract`, `vectorize`, `vectorize_batch`, `numeric_feature_names`, `top_text_features` | confusable-skeleton normalization; signed two-hash (blake2b/FNV, seeded — NOT `hash()`); namespaced char+word streams; L2-norm hash block; 32 fixed numeric slots (§5); `np.add.at` |
| `weak_labels.py` | `weak_label`, `weak_label_from_text` | teacher chain; margin in app.config threshold space; agreement down-weight |
| `synth.py` | `generate(n, seed)` | template banks + obfuscation/paraphrase; paired clean/obf twins; intensity-tiered ground-truth risk; hard-negative oversampling; lang mix; deterministic Generator |
| `dataset.py` | `build_examples`, `to_arrays`, `split` | label-correctness asserts; stratified-by-is_scam split |
| `model_np.py` | `NpRiskModel` (`fit/predict/predict_batch/save/load`, `temperature`) | shared trunk + 3 heads; manual Adam + backprop; soft-target multi-task loss; best-by-val; `.npz` + meta |
| `calibrate.py` | `fit_temperature`, `apply`, `calibrate_model` | golden-section NLL; calibrate vs GROUND TRUTH; isotonic optional; never raise |
| `explain.py` | `attribute` | input-times-grad on risk logit; numeric->dna_key table + text n-grams |
| `metrics.py` | `binary_metrics`, `ece`, `dim_metrics`, `report` | rank AUROC, PR-integral AP, correct ECE bins; RU report |
| `evaluate.py` | `evaluate` | different seed + heavier obfuscation; model vs teacher on SAME set; obfuscated-slice lift; MODEL_CARD.md + metrics json |
| `registry.py` | `save_artifact`, `load_active`, `current_version`, `list_versions` | lazy numpy import; cached; never raise; `ACTIVE.json` |
| `inference.py` | `score_bundle` | flag-gated; lazy load; set kb_similarity; never raise -> None |
| `cli.py` | `gen-data \| train \| evaluate \| predict` | argparse; JSON Prediction on predict |
| `active_learning.py` | `select_uncertain`, `write_review_queue`, `feature_stats`, `psi`, `check_drift` | pure numpy; JSONL queue; PSI drift |
| `train.py` | `train(cfg)` | build->arrays->split->fit->calibrate->save; per-epoch print |
| `model_torch.py` | optional torch fusion | spec §17; same Protocol/contract; distill back to NumPy student |
