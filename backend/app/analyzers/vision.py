"""Vision lane — CLIP zero-shot scoring of sampled keyframes.

Each keyframe is matched against the ``VISION_PROMPTS`` (casino interfaces, fake
bank overlays, profit charts, cash, messenger contacts, plus benign anchors).
Non-neutral prompts that fire raise the ``gambling`` / ``visual`` / ``messenger``
ScamDNA dimensions downstream — which is why ``VisualHit.label`` is set EXACTLY
to the english ``prompt.label`` so ``scam_dna`` can look the prompt up again.

Heavy deps (``torch``, ``open_clip``/``clip``, ``PIL``) are imported lazily and
wrapped: a missing dependency degrades the lane to a no-op and never raises.
"""
from __future__ import annotations

from typing import List, Optional

from app.config import settings
from app.models import registry
from app.pipeline.contracts import Keyframe, SignalBundle, VisualHit
from app.scoring.lexicons import VISION_PROMPTS

# Probability above which a non-neutral prompt counts as a real visual signal.
_PROB_THRESHOLD = 0.18
# Keep only the strongest visual hits overall (avoid flooding the timeline).
_MAX_HITS = 6


def run_vision(bundle: SignalBundle, keyframes: List[Keyframe]) -> None:
    """Fill ``bundle.visual_hits`` with CLIP zero-shot matches per keyframe.

    Degrades to a no-op (and sets ``bundle.degradation.vision``) when vision is
    disabled, the CLIP stack is unavailable, or there are no keyframes.
    Never raises.
    """
    if (
        not settings.enable_vision
        or not registry.capabilities()["vision"]
        or not keyframes
    ):
        bundle.degradation.vision = True
        if not settings.enable_vision:
            bundle.degradation.notes.append("Зрение отключено в настройках.")
        elif not registry.capabilities()["vision"]:
            bundle.degradation.notes.append(
                "Зрение недоступно: модель CLIP не установлена."
            )
        else:
            bundle.degradation.notes.append(
                "Зрение пропущено: нет кадров для анализа."
            )
        return

    try:
        clip_ctx = registry.cached("clip", _load_clip)
        if clip_ctx is None:
            bundle.degradation.vision = True
            err = registry.load_error("clip") or "не удалось загрузить CLIP"
            bundle.degradation.notes.append(f"Зрение недоступно: {err}")
            return

        text_features = registry.cached("clip_text_features", lambda: _encode_text(clip_ctx))
        if text_features is None:
            bundle.degradation.vision = True
            bundle.degradation.notes.append(
                "Зрение недоступно: не удалось закодировать промпты."
            )
            return

        hits = _score_keyframes(keyframes, clip_ctx, text_features)
        hits.sort(key=lambda h: h.score, reverse=True)
        bundle.visual_hits.extend(hits[:_MAX_HITS])
        bundle.lanes_run.append("vision")
    except Exception as exc:  # noqa: BLE001 - any failure degrades the lane
        bundle.degradation.vision = True
        bundle.degradation.notes.append(f"Зрение прервано: {exc!r}")


# --------------------------------------------------------------------------- #
# Lazy model loading (cached once via registry).
# --------------------------------------------------------------------------- #
def _load_clip() -> Optional[dict]:
    """Build the CLIP (model, preprocess, tokenizer, device) once.

    Prefers ``open_clip``; falls back to OpenAI ``clip``. Returns None on any
    failure so the cache records the miss and the lane degrades.
    """
    import torch  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preferred backend: open_clip ------------------------------------------ #
    if registry.has_module("open_clip"):
        import open_clip  # type: ignore

        model, _, preprocess = open_clip.create_model_and_transforms(
            settings.clip_model,
            pretrained=settings.clip_pretrained,
        )
        tokenizer = open_clip.get_tokenizer(settings.clip_model)
        model = model.to(device).eval()
        return {
            "backend": "open_clip",
            "model": model,
            "preprocess": preprocess,
            "tokenizer": tokenizer,
            "device": device,
            "torch": torch,
        }

    # Fallback backend: OpenAI clip ----------------------------------------- #
    import clip  # type: ignore

    # OpenAI clip uses names like "ViT-B/32" rather than open_clip's "ViT-B-32".
    name = settings.clip_model.replace("-", "/", 1) if "/" not in settings.clip_model else settings.clip_model
    model, preprocess = clip.load(name, device=device)
    model = model.eval()
    return {
        "backend": "clip",
        "model": model,
        "preprocess": preprocess,
        "tokenizer": clip.tokenize,
        "device": device,
        "torch": torch,
    }


def _encode_text(ctx: dict) -> object:
    """Encode the VISION_PROMPTS labels into normalised text features once."""
    torch = ctx["torch"]
    model = ctx["model"]
    tokenizer = ctx["tokenizer"]
    device = ctx["device"]

    labels = [p.label for p in VISION_PROMPTS]
    tokens = tokenizer(labels)
    if hasattr(tokens, "to"):
        tokens = tokens.to(device)

    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def _score_keyframes(
    keyframes: List[Keyframe],
    ctx: dict,
    text_features: object,
) -> List[VisualHit]:
    """Softmax CLIP over the prompts for each frame; collect non-neutral hits."""
    from PIL import Image  # type: ignore

    torch = ctx["torch"]
    model = ctx["model"]
    preprocess = ctx["preprocess"]
    device = ctx["device"]

    hits: List[VisualHit] = []
    for kf in keyframes:
        image = _frame_image(kf, Image)
        if image is None:
            continue
        try:
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = 100.0 * image_features @ text_features.T
                probs = logits.softmax(dim=-1).squeeze(0).tolist()
        except Exception:  # noqa: BLE001 - skip frames the model can't handle
            continue

        for prompt, prob in zip(VISION_PROMPTS, probs):
            if prompt.is_neutral:
                continue
            if prob <= _PROB_THRESHOLD:
                continue
            hits.append(
                VisualHit(
                    time_s=kf.time_s,
                    label=prompt.label,        # EXACT english label for scam_dna lookup
                    label_ru=prompt.label_ru,
                    score=float(prob),
                )
            )
    return hits


def _frame_image(kf: Keyframe, Image) -> object:
    """Return a PIL image for the keyframe (in-memory or opened from disk)."""
    if kf.image is not None:
        return kf.image
    if kf.path:
        try:
            return Image.open(kf.path).convert("RGB")
        except Exception:  # noqa: BLE001 - skip unreadable frame
            return None
    return None
