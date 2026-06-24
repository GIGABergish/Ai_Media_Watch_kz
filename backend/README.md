# AI Media Watch — Analysis Engine (Sentinel Media AI)

A tiered, multimodal **scam-risk analysis engine** for short social-media videos.
It ingests an uploaded clip (or just a URL + captions) and emits a structured
risk report that is a **drop-in replacement for the frontend's `DemoCase`
objects** — same field names, same shape, in Russian.

> **Ethics note.** The engine produces a *risk-oriented signal*, **not a legal
> verdict**. A high Risk Score means the content matches patterns frequently
> associated with fraud — it does **not** prove that any specific person committed
> a crime. Every case is marked `status: "new"` and is intended for **human
> review** before any action. This mirrors the disclaimer on the frontend
> Settings page.

---

## Architecture: the tiered cascade

The engine is deliberately **cheap-first**. It never pays for heavy ML unless the
cheap signals justify it, and it **never downloads heavy media for URL analysis**.

```
MediaInput
  │
  ├─ (1) CHEAP LANE — always, pure stdlib, milliseconds
  │       links.extract_links     → external contacts / Telegram / promo codes
  │       text_signals.analyze_text → lexical ScamDNA matches over metadata/captions
  │       behavior.analyze_behavior → CTA / urgency / referral funnel mechanics
  │       → preliminary ScamDNA + Risk Score
  │
  ├─ (2) ESCALATE  (only when source is an upload with a real file AND
  │                 prelim score ≥ AMW_SHORTCIRCUIT_BELOW, OR the metadata text
  │                 is too thin to judge < ~40 chars)
  │       media_probe.probe        → duration / dimensions / streams (ffprobe→ffmpeg)
  │       audio.extract_audio      → 16 kHz mono WAV
  │       asr.run_asr              → Whisper / faster-whisper transcript w/ timestamps
  │       keyframes.extract_keyframes → bounded, downscaled frame sample (never full-decode)
  │       ocr.run_ocr              → Tesseract on-screen text
  │       vision.run_vision        → CLIP zero-shot (casino UI, fake payouts, …)
  │       → RE-RUN the cheap lanes so transcript + OCR text is also mined
  │
  └─ (3) SCORING
          scam_dna.compute_scam_dna   → 8 ScamDNA dimensions (saturating fold)
          KB.similarity               → boosts the "reused" dimension
          risk_score.compute_risk_score → 5 weighted components + overall 0..100
          category.classify_category  → violation category (machine + RU label)
          timeline.build_timeline     → time-anchored signal events
          evidence.build_evidence     → up to 6 evidence cards
          connections.build_connections → connection graph (Telegram / hashtags / KB)
          → CaseResult + AnalysisMeta
```

Every heavy lane is **defensive**: a missing optional dependency (or any runtime
error) sets a `Degradation` flag, records a short Russian note, and returns an
empty/cheap result — it **never raises**. `orchestrator.analyze()` itself is
wrapped so that even a catastrophic failure still returns a valid minimal
`CaseResult`.

### Engine modes: lite vs hybrid vs full

The mode is computed at runtime from which optional dependencies (and the
ffmpeg/Tesseract binaries) are actually present:

| Mode     | Meaning                                       |
|----------|-----------------------------------------------|
| `lite`   | No ML lanes available → cheap text/link/behavior analysis only (pure stdlib). Works out of the box. |
| `hybrid` | Some ML lanes available (e.g. OCR but no CLIP). |
| `full`   | ASR **and** OCR **and** vision all available.  |

Check the live mode + capabilities at `GET /api/health` or `GET /`.

### No-heavy-download philosophy (URL analysis)

`POST /api/analyze/url` **does not fetch the video**. It analyzes only the
*lightweight* signals you supply — title, description, hashtags, and optional
pre-extracted captions/transcript. This keeps URL analysis instant, side-effect
free, and legally clean. If you already have platform captions, pass them in
`transcript` and they are treated exactly like an ASR result (timeline-anchored),
skipping Whisper entirely.

---

## Risk Score formula

The overall Risk Score (0..100) is a **weighted blend of five components**, each
itself a saturating fold of the relevant ScamDNA dimensions. Weights live in
`app/config.py` (`RiskWeights`) and **must sum to 1.0**:

| Component         | Weight | Driven by ScamDNA dimensions            |
|-------------------|:------:|-----------------------------------------|
| Текст и речь (`text_speech`)    | **0.35** | `profit`, `urgency` |
| Визуальные признаки (`visual`)  | **0.25** | `gambling`, `visual` |
| Метаданные и ссылки (`metadata_links`) | **0.15** | `hashtags`, `messenger` |
| Поведенческие паттерны (`behavior`) | **0.15** | `referral`, `urgency`, `messenger` |
| Похожесть на базу (`db_similarity`) | **0.10** | `reused` (knowledge-base similarity) |

```
overall = Σ (componentᵢ × weightᵢ)
overall ×= (1 − min(0.55, negative_markers / 180))   # educational/anti-fraud damping
```

Educational and anti-fraud content (detected via `NEGATIVE_MARKERS`) is damped so
explainer clips land in the **low** band and are categorized accordingly.

Risk levels come from `risk_level()` thresholds (`app/config.py`):
`critical ≥ 88`, `high ≥ 65`, `medium ≥ 40`, else `low`. **Never hard-code these.**

The eight **ScamDNA** dimensions: `profit`, `urgency`, `gambling`, `referral`,
`messenger`, `visual`, `reused`, `hashtags`.

---

## Endpoints

All endpoints are under the `/api` prefix. Interactive docs at `/docs`.

### `GET /api/health`
Liveness + capability snapshot.
```bash
curl http://127.0.0.1:8000/api/health
```
```json
{ "status": "ok", "version": "1.0.0", "engineMode": "lite",
  "capabilities": { "ffmpeg": false, "asr": false, "ocr": false, "vision": false, "pillow": false } }
```

### `POST /api/analyze`
Multipart upload — runs the full cascade. `file` is required; `title`,
`platform`, `description`, `hashtags` (comma-separated) are optional form fields.
Rejects files larger than `AMW_MAX_UPLOAD_MB` with `413`.
```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -F "file=@clip.mp4" \
  -F "title=Занос в слотах" \
  -F "platform=Instagram" \
  -F "description=Лёгкий заработок, пиши в директ" \
  -F "hashtags=#казино,#заработок,#бонус"
```

### `POST /api/analyze/url`
Analyze by reference — **no download**. JSON body (`AnalyzeUrlRequest`):
```bash
curl -X POST http://127.0.0.1:8000/api/analyze/url \
  -H "Content-Type: application/json" \
  -d '{
        "url": "https://www.instagram.com/reel/XXXX/",
        "platform": "Instagram",
        "title": "Пассивный доход за месяц",
        "description": "Гарантированная прибыль, пиши + в директ",
        "hashtags": ["#пассивныйдоход", "#инвестиции"],
        "transcript": "Привет! Хочешь зарабатывать из дома? Пиши плюс в личку..."
      }'
```
Both analyze endpoints return `{ "case": CaseResult, "meta": AnalysisMeta }`.

### `GET /api/cases`
List recently analyzed cases (newest first).
```bash
curl http://127.0.0.1:8000/api/cases
```

### `GET /api/cases/{id}`
Fetch one stored case, or `404`.
```bash
curl http://127.0.0.1:8000/api/cases/case-1a2b3c4d
```

---

## Running

### Quick start (lite mode)
```bash
# POSIX
./run.sh
# Windows (PowerShell)
./run.ps1
```
Or manually:
```bash
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Full multimodal mode
Install the optional ML stack on top of the core requirements:
```bash
pip install -r requirements.txt -r requirements-ml.txt
```
`requirements-ml.txt` brings imageio-ffmpeg (bundled ffmpeg), Pillow, numpy,
faster-whisper, pytesseract, torch and open-clip-torch.

**OCR additionally needs the system Tesseract binary** on `PATH`
(`apt install tesseract-ocr tesseract-ocr-rus`, or the UB-Mannheim installer on
Windows). Without it the OCR lane degrades gracefully and the engine reports
`hybrid` mode.

---

## Configuration (env vars)

Every knob is an `AMW_`-prefixed environment variable — see
[`.env.example`](./.env.example) for the full annotated list. Highlights:

| Variable | Default | Purpose |
|----------|---------|---------|
| `AMW_DATA_DIR` / `AMW_UPLOAD_DIR` / `AMW_DB_PATH` | `./data*` | storage locations |
| `AMW_ENABLE_ASR` / `AMW_ENABLE_OCR` / `AMW_ENABLE_VISION` | `true` | force-skip a lane |
| `AMW_WHISPER_MODEL` / `AMW_WHISPER_DEVICE` | `base` / `cpu` | ASR backend |
| `AMW_CLIP_MODEL` / `AMW_CLIP_PRETRAINED` | `ViT-B-32` / `openai` | vision backend |
| `AMW_TESSERACT_LANG` | `rus+eng` | OCR languages |
| `AMW_KEYFRAME_INTERVAL` / `AMW_MAX_KEYFRAMES` / `AMW_KEYFRAME_MAX_DIM` | `2.0` / `24` / `720` | frame sampling budget |
| `AMW_HOST` / `AMW_PORT` / `AMW_MAX_UPLOAD_MB` | `127.0.0.1` / `8000` / `500` | server |
| `AMW_CORS_ORIGINS` | Vite dev/preview ports | CORS allow-list |

---

## Wiring the frontend

The engine emits exactly the `DemoCase` shape, so the React app can consume it
directly. Point the frontend at the engine with a Vite env var:

```bash
# frontend/.env
VITE_API_BASE=http://127.0.0.1:8000/api
```

Then in the frontend, upload via `POST ${VITE_API_BASE}/analyze`, analyze a URL
via `POST ${VITE_API_BASE}/analyze/url`, and read `response.case` (a `DemoCase`)
plus `response.meta` (engine telemetry: `engineMode`, `lanesRun`, `degraded`,
`elapsedMs`, weighted `components`, and human-readable `notes`). CORS already
allows the default Vite ports `5173` / `4173`.
