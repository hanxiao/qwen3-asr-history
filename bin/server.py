#!/usr/bin/env python3
"""
MLX Serving - Unified ASR + TTS + Translate server.
Keeps models loaded in memory for fast inference.
Saves transcription/synthesis/translation history to ~/Documents/qwen3-asr-history/history/
"""
import base64
import gc
import os
import sys
import tempfile
import subprocess
import time
import json
import uuid
import threading
import re
import logging
import logging.handlers
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_load, generate as lm_generate
from mlx_audio.stt.utils import load_model as stt_load_model
from mlx_audio.stt.generate import generate_transcription
from mlx_audio.tts.utils import load_model as tts_load_fn
from mlx_audio.tts.generate import generate_audio
from mlx_vlm import load as vlm_load, generate as vlm_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config as vlm_load_config

# mflux site-packages path (installed in separate uv tool venv, added to sys.path at load time)
# mflux import stays deferred because it depends on sys.path setup at runtime
MFLUX_SITE_PACKAGES = os.path.expanduser("~/.local/share/uv/tools/mflux/lib/python3.12/site-packages")

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import uvicorn

import shutil


# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(os.path.expanduser("~/.openclaw/workspace/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mls.log"

_log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_file_handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
_file_handler.setFormatter(_log_formatter)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

logger = logging.getLogger("mls")
logger.setLevel(logging.INFO)
logger.addHandler(_file_handler)
logger.addHandler(_console_handler)

# Also capture uvicorn logs to file
for _uv_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
    _uv_logger = logging.getLogger(_uv_name)
    _uv_logger.addHandler(_file_handler)


# ============================================================
# ASR globals
# ============================================================
model = None
load_fn = None
generate_fn = None

AVAILABLE_MODELS = [
    "mlx-community/Qwen3-ASR-0.6B-bf16",
    "mlx-community/Qwen3-ASR-1.7B-8bit",
    "mlx-community/whisper-large-v3-turbo-asr-fp16",
    "mlx-community/whisper-large-v3-asr-8bit",
]

current_model_name = "mlx-community/Qwen3-ASR-1.7B-8bit"
server_paused = False

# ============================================================
# TTS globals
# ============================================================
tts_model = None
tts_model_name = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
tts_model_loading = False
tts_server_paused = False

AVAILABLE_TTS_MODELS = [
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
    "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-8bit",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
]

# Default TTS model path on USB drive
TTS_MODEL_PATH = "/Volumes/One Touch/ai-models/mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"

# Default voices for Qwen3-TTS
TTS_VOICES = [
    {"id": "Chelsie", "name": "Chelsie", "language": "en", "gender": "female"},
    {"id": "Ethan", "name": "Ethan", "language": "en", "gender": "male"},
    {"id": "Vivian", "name": "Vivian", "language": "zh", "gender": "female"},
]

# Supported languages
TTS_LANGUAGES = {
    "chinese": "zh",
    "english": "en",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "spanish": "es",
    "japanese": "ja",
    "korean": "ko",
    "french": "fr",
    "russian": "ru",
}

# ============================================================
# Translate globals
# ============================================================
translate_model = None
translate_tokenizer = None
translate_model_name = "translategemma-12b-it-8bit"
translate_model_dir = "/Volumes/One Touch/ai-models/mlx-community/translategemma-12b-it-8bit"
translate_model_loading = False
translate_server_paused = False

# ============================================================
# Vision globals
# ============================================================
vision_model = None
vision_processor = None
vision_config = None
vision_model_name = "jinaai/jina-vlm-mlx"
vision_model_loading = False
vision_server_paused = False
VISION_MODEL_PATH = "/Volumes/One Touch/ai-models/jinaai/jina-vlm-mlx"

# Single lock for ALL GPU inference - MLX Metal cannot handle concurrent GPU access.
# Separate per-service locks allowed ASR+TTS+Translate+Image to hit the GPU
# simultaneously, causing Metal OOM crashes.
_gpu_lock = threading.Lock()
_gpu_request_count = 0
_gpu_queue_waiting = 0  # number of threads waiting to acquire _gpu_lock
_gpu_inflight = 0  # total in-flight inference requests (from HTTP entry to response)
_GPU_CACHE_CLEAR_INTERVAL = 20


@contextmanager
def _track_request():
    """Track an in-flight inference request for the queue display."""
    global _gpu_inflight
    _gpu_inflight += 1
    try:
        yield
    finally:
        _gpu_inflight -= 1


class _gpu_queue:
    """Context manager that tracks queue depth around _gpu_lock."""
    def __enter__(self):
        global _gpu_queue_waiting
        _gpu_queue_waiting += 1
        _gpu_lock.acquire()
        _gpu_queue_waiting -= 1
        return self
    def __exit__(self, *args):
        _gpu_lock.release()


def _maybe_clear_gpu_cache():
    """Clear Metal cache periodically to prevent memory buildup.
    Must be called while holding _gpu_lock."""
    global _gpu_request_count
    _gpu_request_count += 1
    if _gpu_request_count % _GPU_CACHE_CLEAR_INTERVAL == 0:
        gc.collect()
        mx.clear_cache()
        logger.info(f"Metal cache cleared (request #{_gpu_request_count})")


# Track background file translations
translate_file_jobs: dict[str, dict] = {}

# Track background file synthesis jobs
file_synth_jobs: dict[str, dict] = {}

_JOB_MAX_AGE = 3600  # prune completed jobs older than 1h

def _prune_jobs(jobs: dict):
    """Remove completed jobs older than _JOB_MAX_AGE seconds."""
    now = time.time()
    stale = [k for k, v in jobs.items()
             if v.get("status") in ("done", "error")
             and now - v.get("_ts", 0) > _JOB_MAX_AGE]
    for k in stale:
        del jobs[k]

SUPPORTED_TRANSLATE_LANGS = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "az": "Azerbaijani",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "fa": "Persian", "fi": "Finnish", "fr": "French",
    "ga": "Irish", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian",
    "hy": "Armenian", "id": "Indonesian", "ig": "Igbo", "is": "Icelandic",
    "it": "Italian", "ja": "Japanese", "jv": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean",
    "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy",
    "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "ms": "Malay", "mt": "Maltese", "my": "Burmese",
    "ne": "Nepali", "nl": "Dutch", "no": "Norwegian", "ny": "Chichewa",
    "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
    "ru": "Russian", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "so": "Somali", "sq": "Albanian", "sr": "Serbian",
    "su": "Sundanese", "sv": "Swedish", "sw": "Swahili", "ta": "Tamil",
    "te": "Telugu", "tg": "Tajik", "th": "Thai", "tl": "Filipino",
    "tr": "Turkish", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek",
    "vi": "Vietnamese", "yo": "Yoruba", "zh": "Chinese", "zu": "Zulu",
}

HOST = "127.0.0.1"
PORT = 18321
_SERVER_START_TIME = time.time()
PROJECT_DIR = Path(__file__).parent.parent.resolve()
HISTORY_DIR = PROJECT_DIR / "history"
TTS_OUTPUT_DIR = PROJECT_DIR / "tts_output"
VISION_OUTPUT_DIR = PROJECT_DIR / "vision_output"

@asynccontextmanager
async def lifespan(app):
    """Load models in a background thread so /health and /history are available immediately."""
    def _load_and_log(fn, name):
        try:
            with _gpu_lock:
                fn()
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    loaders = [
        (load_model, "ASR"),
        (load_tts_model, "TTS"),
        (load_translate_model, "Translate"),
        (load_image_model, "Image"),
        (load_vision_model, "Vision"),
    ]
    threads = []
    for fn, name in loaders:
        t = threading.Thread(target=_load_and_log, args=(fn, name), daemon=True)
        t.start()
        threads.append(t)

    def _wait_all():
        for t in threads:
            t.join()
        logger.info("All models loaded")
    threading.Thread(target=_wait_all, daemon=True).start()
    yield

app = FastAPI(title="MLX Serving", lifespan=lifespan)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler to prevent server crashes."""
    try:
        detail = str(exc)
    except (UnicodeDecodeError, UnicodeEncodeError):
        detail = repr(exc)
    logger.error(f"Unhandled exception on {request.url.path}: {detail}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors safely, avoiding UnicodeDecodeError."""
    try:
        detail = exc.errors()
    except (UnicodeDecodeError, UnicodeEncodeError):
        detail = [{"msg": "Request validation failed (non-UTF-8 content)"}]
    logger.warning(f"Validation error on {request.url.path}: {detail}")
    return JSONResponse(status_code=422, content={"detail": detail})


# Paths that perform GPU inference - tracked for queue display
_INFERENCE_PATHS = {
    "/transcribe", "/synthesize", "/synthesize_file",
    "/translate", "/api/image/generate",
    "/v1/chat/completions", "/api/vision/analyze",
}

@app.middleware("http")
async def track_inflight_requests(request, call_next):
    global _gpu_inflight
    if request.url.path in _INFERENCE_PATHS:
        _gpu_inflight += 1
        try:
            return await call_next(request)
        finally:
            _gpu_inflight -= 1
    return await call_next(request)


# ============================================================
# Image Generation globals
# ============================================================
IMAGE_OUTPUT_DIR = PROJECT_DIR / "image_output"
MFLUX_MODEL_PATH = "/Volumes/One Touch/ai-models/mflux/z-image-turbo-8bit/"
image_model = None
image_model_loading = False

# ============================================================
# ASR helpers
# ============================================================

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in milliseconds using ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, timeout=10
        )
        return float(result.stdout.strip()) * 1000
    except Exception:
        return 0.0


def save_to_history(audio_path: str, text: str, latency_ms: float, audio_duration_ms: float, model_name: str):
    """Save transcription to history (stores reference to original audio path)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_dir = HISTORY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    audio_src = Path(audio_path)

    record = {
        "timestamp": now.isoformat(),
        "audio_path": str(audio_src.resolve()),
        "audio_file": audio_src.name,
        "text": text,
        "latency_ms": round(latency_ms, 2),
        "audio_duration_ms": round(audio_duration_ms, 2),
        "model": model_name.split("/")[-1]
    }
    jsonl_path = day_dir / "transcripts.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_tts_history(text: str, voice: str, language: str, audio_path: str,
                     audio_duration_ms: float, latency_ms: float, model_name: str):
    """Save TTS generation to history."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_dir = HISTORY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "type": "tts",
        "timestamp": now.isoformat(),
        "text": text,
        "voice": voice,
        "language": language,
        "audio_path": str(Path(audio_path).resolve()),
        "audio_file": Path(audio_path).name,
        "audio_duration_ms": round(audio_duration_ms, 2),
        "latency_ms": round(latency_ms, 2),
        "model": model_name.split("/")[-1]
    }
    jsonl_path = day_dir / "tts_history.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_translate_history(source_text: str, translated_text: str, source_lang: str,
                           target_lang: str, latency_ms: float):
    """Save translation to history."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_dir = HISTORY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "type": "translate",
        "timestamp": now.isoformat(),
        "source_text": source_text,
        "translated_text": translated_text,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "source_lang_name": SUPPORTED_TRANSLATE_LANGS.get(source_lang, source_lang),
        "target_lang_name": SUPPORTED_TRANSLATE_LANGS.get(target_lang, target_lang),
        "latency_ms": round(latency_ms, 2),
        "model": translate_model_name,
    }
    jsonl_path = day_dir / "translate_history.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Image Gen helpers
# ============================================================

def save_image_history(prompt: str, image_path: str, latency_ms: float, resolution: str,
                       seed: int | None = None, steps: int = 9):
    """Save image generation to history."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_dir = HISTORY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "type": "image",
        "timestamp": now.isoformat(),
        "prompt": prompt,
        "image_path": str(Path(image_path).resolve()),
        "image_file": Path(image_path).name,
        "resolution": resolution,
        "latency_ms": round(latency_ms, 2),
        "model": "z-image-turbo-8bit",
        "seed": seed,
        "steps": steps,
    }
    jsonl_path = day_dir / "image_history.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# ============================================================
# Vision helpers
# ============================================================

def save_vision_history(prompt: str, image_path: str, response_text: str,
                        latency_ms: float, prompt_tokens: int = 0,
                        generation_tokens: int = 0, generation_tps: float = 0.0):
    """Save vision analysis to history."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    day_dir = HISTORY_DIR / date_str
    day_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "type": "vision",
        "timestamp": now.isoformat(),
        "prompt": prompt,
        "image_path": str(Path(image_path).resolve()),
        "image_file": Path(image_path).name,
        "response_text": response_text,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "generation_tokens": generation_tokens,
        "generation_tps": round(generation_tps, 2),
        "model": vision_model_name,
    }
    jsonl_path = day_dir / "vision_history.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# Translate helpers
# ============================================================

def translate_text(text: str, source: str, target: str) -> tuple[str, float]:
    """Translate a single text. Returns (translation, elapsed_seconds).
    Thread-safe: uses _gpu_lock to serialize Metal GPU access."""
    with _gpu_queue():
        return _translate_text_impl(text, source, target)


def _translate_text_impl(text: str, source: str, target: str) -> tuple[str, float]:
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": text,
             "source_lang_code": source, "target_lang_code": target}
        ]}
    ]
    prompt = translate_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    t0 = time.time()
    result = lm_generate(
        translate_model, translate_tokenizer, prompt=prompt,
        max_tokens=1024, verbose=False,
    )
    elapsed = time.time() - t0

    clean = result.split("<end_of_turn>")[0].strip()
    _maybe_clear_gpu_cache()
    return clean, elapsed


def _translate_file_worker(src_path: Path, out_path: Path, source: str, target: str, delimiter: str):
    """Background worker for file translation."""
    job = translate_file_jobs[str(out_path)]
    content = src_path.read_text(encoding="utf-8")
    segments = content.split("\n") if delimiter == "\n" else content.split(delimiter)
    job["lines"] = len(segments)

    t0 = time.time()
    translated = []
    errors = 0
    for i, segment in enumerate(segments):
        stripped = segment.strip()
        if not stripped:
            translated.append("")
            continue
        try:
            result, elapsed = translate_text(stripped, source, target)
            translated.append(result)
            job["done"] = i + 1
            logger.info(f"Translate file [{i+1}/{len(segments)}] {len(stripped)}ch -> {len(result)}ch in {elapsed:.2f}s")
        except Exception as e:
            logger.error(f"Translate file [{i+1}/{len(segments)}] ERROR: {e}")
            translated.append(stripped)
            errors += 1
            job["done"] = i + 1

    total_elapsed = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text = delimiter.join(translated) if delimiter != "\n" else "\n".join(translated)
    out_path.write_text(out_text, encoding="utf-8")

    job.update({"status": "done", "errors": errors, "elapsed": round(total_elapsed, 2)})
    logger.info(f"Translate file done: {len(segments)} lines, {errors} errors, {total_elapsed:.1f}s -> {out_path}")


# ============================================================
# Model loading
# ============================================================

class TranscribeRequest(BaseModel):
    path: str
    language: str = "zh"


class TranscribeResponse(BaseModel):
    text: str
    latency_ms: float


class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "Chelsie"
    language: str = "en"
    speed: float = 1.0
    format: str = "ogg"  # "ogg" (default) or "wav"
    instruct: str = None  # Voice description for VoiceDesign models


class SwitchModelRequest(BaseModel):
    model: str


class SwitchTTSModelRequest(BaseModel):
    model: str


# Translate request/response models (Google Translate compatible)
class TranslateRequest(BaseModel):
    q: str | list[str]
    source: str
    target: str
    format: str = "text"


class Translation(BaseModel):
    translatedText: str
    detectedSourceLanguage: str | None = None
    model: str = "translategemma-12b-it-8bit"


class TranslateResponseData(BaseModel):
    translations: list[Translation]


class TranslateResponse(BaseModel):
    data: TranslateResponseData


class FileTranslateRequest(BaseModel):
    file: str
    source: str
    target: str
    output: str | None = None
    delimiter: str = "\n"


class ImageGenRequest(BaseModel):
    prompt: str
    resolution: str = "1024x1024"  # WxH format or shorthand like 1024x1024
    negative_prompt: str | None = None
    seed: int | None = None
    steps: int = 9


class VisionRequest(BaseModel):
    image: str  # local file path or uploaded filename
    prompt: str = "What is in this image? Describe everything you can see."
    max_tokens: int = 512


# ============================================================
# OpenAI-compatible Chat Completion models
# ============================================================

class ChatImageUrl(BaseModel):
    url: str
    detail: str = "auto"

class ChatContentPart(BaseModel):
    type: str  # "text" or "image_url"
    text: str | None = None
    image_url: ChatImageUrl | None = None

class ChatMessage(BaseModel):
    role: str
    content: str | list[ChatContentPart]

class ChatCompletionRequest(BaseModel):
    model: str = "jinaai/jina-vlm-mlx"
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False


def load_model(model_name: str = None):
    """Load ASR model."""
    global model, load_fn, generate_fn, current_model_name

    if model_name:
        current_model_name = model_name

    logger.info(f"Loading ASR model {current_model_name}...")
    start = time.time()

    load_fn = stt_load_model
    generate_fn = generate_transcription
    model = stt_load_model(current_model_name)

    elapsed = time.time() - start
    logger.info(f"ASR model loaded in {elapsed:.2f}s")
    return model


def load_tts_model(model_name: str = None):
    """Load TTS model."""
    global tts_model, tts_model_name, tts_model_loading

    if model_name:
        tts_model_name = model_name

    tts_model_loading = True
    logger.info(f"Loading TTS model {tts_model_name}...")
    start = time.time()

    try:
        # Try USB path first for all models
        usb_base = "/Volumes/One Touch/ai-models/mlx-community"
        usb_path = os.path.join(usb_base, tts_model_name.split("/")[-1])
        if os.path.exists(usb_path):
            model_path = usb_path
        else:
            model_path = tts_model_name

        tts_model = tts_load_fn(model_path=model_path)

        elapsed = time.time() - start
        logger.info(f"TTS model loaded in {elapsed:.2f}s")

        if hasattr(tts_model, 'get_supported_speakers'):
            speakers = tts_model.get_supported_speakers()
            if speakers:
                logger.info(f"Supported speakers: {speakers}")
        if hasattr(tts_model, 'get_supported_languages'):
            langs = tts_model.get_supported_languages()
            logger.info(f"Supported languages: {langs}")
    except Exception as e:
        logger.error(f"Failed to load TTS model: {e}")
        tts_model = None
    finally:
        tts_model_loading = False

    return tts_model


def load_translate_model():
    """Load translate model (TranslateGemma)."""
    global translate_model, translate_tokenizer, translate_model_loading

    translate_model_loading = True
    logger.info(f"Loading translate model from {translate_model_dir}...")
    start = time.time()

    try:
        translate_model, translate_tokenizer = mlx_load(translate_model_dir)
        # TranslateGemma uses <end_of_turn> (token 106) as stop signal, but
        # the tokenizer only has <eos> (token 1) in eos_token_ids. Without this
        # patch, mlx_lm.generate runs to max_tokens on every request.
        eot_id = translate_tokenizer.encode("<end_of_turn>", add_special_tokens=False)
        if eot_id and hasattr(translate_tokenizer, '_eos_token_ids'):
            translate_tokenizer._eos_token_ids = translate_tokenizer.eos_token_ids | set(eot_id)
        elif eot_id:
            translate_tokenizer.eos_token_ids = translate_tokenizer.eos_token_ids | set(eot_id)
        elapsed = time.time() - start
        logger.info(f"Translate model loaded in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load translate model: {e}")
        translate_model = None
        translate_tokenizer = None
    finally:
        translate_model_loading = False


def load_image_model():
    """Load Z-Image-Turbo model for persistent in-memory image generation."""
    global image_model, image_model_loading

    if not os.path.isdir(MFLUX_MODEL_PATH):
        logger.warning(f"Image model not available (USB not mounted?): {MFLUX_MODEL_PATH}")
        return

    if not os.path.isdir(MFLUX_SITE_PACKAGES):
        logger.warning(f"mflux site-packages not found: {MFLUX_SITE_PACKAGES}")
        return

    # Append mflux site-packages (not insert - avoid overriding venv's numpy)
    if MFLUX_SITE_PACKAGES not in sys.path:
        sys.path.append(MFLUX_SITE_PACKAGES)

    image_model_loading = True
    logger.info(f"Loading image model from {MFLUX_MODEL_PATH}...")
    start = time.time()

    try:
        from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo

        image_model = ZImageTurbo(
            quantize=8,
            model_path=MFLUX_MODEL_PATH,
        )
        elapsed = time.time() - start
        logger.info(f"Image model loaded in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load image model: {e}")
        image_model = None
    finally:
        image_model_loading = False


def load_vision_model():
    """Load Vision-Language model (jina-vlm-mlx)."""
    global vision_model, vision_processor, vision_config, vision_model_loading

    if not os.path.isdir(VISION_MODEL_PATH):
        logger.warning(f"Vision model not available (USB not mounted?): {VISION_MODEL_PATH}")
        return

    vision_model_loading = True
    logger.info(f"Loading vision model from {VISION_MODEL_PATH}...")
    start = time.time()

    try:
        vision_model, vision_processor = vlm_load(VISION_MODEL_PATH)
        vision_config = vlm_load_config(VISION_MODEL_PATH)
        elapsed = time.time() - start
        logger.info(f"Vision model loaded in {elapsed:.2f}s")
    except Exception as e:
        logger.error(f"Failed to load vision model: {e}")
        vision_model = None
        vision_processor = None
        vision_config = None
    finally:
        vision_model_loading = False


def convert_wav_to_ogg(wav_path: str, ogg_path: str) -> bool:
    """Convert WAV to OGG Opus using ffmpeg. Returns True on success."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "48k", ogg_path],
            capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"WAV->OGG conversion failed: {e}")
        return False


def convert_to_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV for best results."""
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    result = subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], capture_output=True)
    if result.returncode != 0:
        raise FileNotFoundError(f"ffmpeg conversion failed: {result.stderr[:200]}")
    return wav_path


# ============================================================
# Health / Status endpoints
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": current_model_name,
        "loaded": model is not None,
        "tts_model": tts_model_name,
        "tts_loaded": tts_model is not None,
        "translate_model": translate_model_name,
        "translate_loaded": translate_model is not None,
        "image_model": "z-image-turbo-8bit",
        "image_loaded": image_model is not None,
        "vision_model": vision_model_name,
        "vision_loaded": vision_model is not None,
    }


@app.get("/api/status")
async def get_status():
    """Get ASR server status including model info."""
    return {
        "model": current_model_name,
        "model_short": current_model_name.split("/")[-1],
        "loaded": model is not None,
        "paused": server_paused,
        "available_models": AVAILABLE_MODELS
    }


# ============================================================
# ASR endpoints
# ============================================================

@app.post("/api/server/pause")
async def pause_server():
    global server_paused
    server_paused = True
    return {"status": "paused"}


@app.post("/api/server/resume")
async def resume_server():
    global server_paused
    server_paused = False
    return {"status": "active"}


@app.post("/api/server/restart")
async def restart_server():
    global model
    model = None
    gc.collect()
    mx.clear_cache()
    load_model()
    return {"status": "restarted", "model": current_model_name}


@app.post("/api/model/switch")
async def switch_model(req: SwitchModelRequest):
    global model
    if req.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model not available: {req.model}")
    model = None
    load_model(req.model)
    return {"status": "switched", "model": current_model_name}


@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(req: TranscribeRequest):
    global model

    if server_paused:
        raise HTTPException(status_code=503, detail="Server is paused")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not os.path.exists(req.path):
        raise HTTPException(status_code=400, detail=f"Audio file not found: {req.path}")

    start = time.time()
    wav_path = None
    txt_file = None
    try:
        wav_path = convert_to_wav(req.path)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            output_path = f.name.replace(".txt", "")
        txt_file = output_path + ".txt"

        with _gpu_queue():
            segments = generate_fn(
                model=model,
                audio=wav_path,
                output_path=output_path,
                format="txt",
                verbose=False,
                language=req.language or "zh"
            )
            _maybe_clear_gpu_cache()

        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                text = f.read().strip()
        else:
            if segments:
                text = " ".join(s.get("text", "") for s in segments if isinstance(s, dict))
            else:
                text = str(segments) if segments else ""
    except FileNotFoundError as e:
        logger.error(f"ASR FileNotFoundError: {e}")
        raise HTTPException(status_code=500, detail=f"Audio file processing failed: {e}")
    except Exception as e:
        logger.error(f"ASR transcription error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)
        if txt_file and os.path.exists(txt_file):
            os.unlink(txt_file)

    latency_ms = (time.time() - start) * 1000
    audio_duration_ms = get_audio_duration(req.path)

    if text:
        save_to_history(req.path, text, latency_ms, audio_duration_ms, current_model_name)

    logger.info(f"ASR: {Path(req.path).name}, {len(text)}ch, {latency_ms:.0f}ms, audio={audio_duration_ms:.0f}ms")
    return TranscribeResponse(text=text, latency_ms=latency_ms)


@app.get("/api/dates")
async def get_dates():
    """Get list of dates with transcription history."""
    dates = []
    if HISTORY_DIR.exists():
        for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
            if d.is_dir() and (d / "transcripts.jsonl").exists():
                dates.append(d.name)
    return JSONResponse(dates)


@app.get("/api/transcripts/{date}")
async def get_transcripts(date: str):
    """Get transcripts for a specific date."""
    jsonl_path = HISTORY_DIR / date / "transcripts.jsonl"
    if not jsonl_path.exists():
        raise HTTPException(status_code=404, detail="No transcripts for this date")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return JSONResponse(records)


@app.get("/audio/{date}/{filename}")
async def get_audio(date: str, filename: str):
    """Serve audio file from original location or local copy."""
    jsonl_path = HISTORY_DIR / date / "transcripts.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("audio_file") == filename:
                        if "audio_path" in record:
                            original_path = Path(record["audio_path"])
                            if original_path.exists():
                                return FileResponse(original_path, media_type="audio/ogg")
                        break
    local_path = HISTORY_DIR / date / filename
    if local_path.exists():
        return FileResponse(local_path, media_type="audio/ogg")
    raise HTTPException(status_code=404, detail="Audio file not found")


# ============================================================
# TTS endpoints
# ============================================================

@app.get("/api/tts/status")
async def get_tts_status():
    """Get TTS model status."""
    speakers = []
    languages = list(TTS_LANGUAGES.keys())
    if tts_model and hasattr(tts_model, 'get_supported_speakers'):
        speakers = tts_model.get_supported_speakers()
    if tts_model and hasattr(tts_model, 'get_supported_languages'):
        languages = tts_model.get_supported_languages()

    return {
        "model": tts_model_name,
        "model_short": tts_model_name.split("/")[-1],
        "loaded": tts_model is not None,
        "loading": tts_model_loading,
        "paused": tts_server_paused,
        "available_models": AVAILABLE_TTS_MODELS,
        "speakers": speakers,
        "languages": languages
    }


@app.post("/api/tts/model/switch")
async def switch_tts_model(req: SwitchTTSModelRequest):
    """Switch TTS model."""
    global tts_model
    if req.model not in AVAILABLE_TTS_MODELS:
        raise HTTPException(status_code=400, detail=f"TTS model not available: {req.model}")
    tts_model = None
    gc.collect()
    mx.clear_cache()
    load_tts_model(req.model)
    return {"status": "switched", "model": tts_model_name}


@app.get("/api/tts/voices")
async def get_tts_voices():
    """List available voices."""
    voices = list(TTS_VOICES)
    if tts_model and hasattr(tts_model, 'get_supported_speakers'):
        model_speakers = tts_model.get_supported_speakers()
        existing_ids = {v["id"] for v in voices}
        for sp in model_speakers:
            if sp not in existing_ids:
                voices.append({"id": sp, "name": sp, "language": "auto", "gender": "unknown"})
    return JSONResponse([v for v in voices])


@app.get("/api/tts/history")
async def get_tts_history():
    """Get TTS generation history (all dates)."""
    all_records = []
    if HISTORY_DIR.exists():
        for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
            if d.is_dir():
                jsonl_path = d / "tts_history.jsonl"
                if jsonl_path.exists():
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record["date"] = d.name
                                all_records.append(record)
    all_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return JSONResponse(all_records)


@app.get("/api/tts/history/{date}")
async def get_tts_history_by_date(date: str):
    """Get TTS history for a specific date."""
    jsonl_path = HISTORY_DIR / date / "tts_history.jsonl"
    if not jsonl_path.exists():
        return JSONResponse([])

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return JSONResponse(records)


@app.get("/tts_audio/{filename}")
async def get_tts_audio(filename: str):
    """Serve TTS generated audio file."""
    audio_path = TTS_OUTPUT_DIR / filename
    if audio_path.exists():
        media_type = "audio/ogg" if filename.endswith(".ogg") else "audio/wav"
        return FileResponse(audio_path, media_type=media_type)
    raise HTTPException(status_code=404, detail="Audio file not found")


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest):
    """Synthesize speech from text."""
    global tts_model

    if tts_server_paused:
        raise HTTPException(status_code=503, detail="TTS server is paused")
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")

    start = time.time()

    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4().hex[:12]
    file_prefix = f"tts_{file_id}"
    output_file = TTS_OUTPUT_DIR / f"{file_prefix}.wav"

    try:
        lang_map = {"zh": "chinese", "en": "english", "de": "german",
                     "it": "italian", "pt": "portuguese", "es": "spanish",
                     "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}
        lang_code = lang_map.get(req.language, req.language)

        gen_kwargs = dict(
            text=req.text,
            model=tts_model,
            voice=req.voice,
            lang_code=lang_code,
            speed=req.speed,
            output_path=str(TTS_OUTPUT_DIR),
            file_prefix=file_prefix,
            audio_format="wav",
            join_audio=True,
            play=False,
            verbose=False,
            stt_model=None,
        )
        # VoiceDesign models require an instruct parameter
        instruct = req.instruct
        if not instruct and tts_model_name and "VoiceDesign" in tts_model_name:
            voice_defaults = {
                "Chelsie": "A young American female speaker with a clear and friendly voice",
                "Ethan": "A young American male speaker with a confident and natural voice",
                "Vivian": "A young Chinese female speaker with a soft and warm voice",
            }
            instruct = voice_defaults.get(req.voice, "A young Chinese male speaker with a Beijing accent")
        if instruct:
            gen_kwargs["instruct"] = instruct

        with _gpu_queue():
            generate_audio(**gen_kwargs)
            _maybe_clear_gpu_cache()

        actual_file = None
        for candidate in [
            TTS_OUTPUT_DIR / f"{file_prefix}.wav",
            TTS_OUTPUT_DIR / f"{file_prefix}_000.wav",
        ]:
            if candidate.exists():
                actual_file = candidate
                break

        if actual_file is None:
            for f in TTS_OUTPUT_DIR.iterdir():
                if f.name.startswith(file_prefix) and f.suffix == ".wav":
                    actual_file = f
                    break

        if actual_file is None:
            raise HTTPException(status_code=500, detail="Audio generation failed - no output file")

        if actual_file != output_file:
            actual_file.rename(output_file)

        final_file = output_file
        if req.format == "ogg":
            ogg_file = output_file.with_suffix(".ogg")
            if convert_wav_to_ogg(str(output_file), str(ogg_file)):
                output_file.unlink(missing_ok=True)
                final_file = ogg_file
            else:
                final_file = output_file

        latency_ms = (time.time() - start) * 1000
        audio_duration_ms = get_audio_duration(str(final_file))

        save_tts_history(
            text=req.text,
            voice=req.voice,
            language=req.language,
            audio_path=str(final_file),
            audio_duration_ms=audio_duration_ms,
            latency_ms=latency_ms,
            model_name=tts_model_name
        )

        logger.info(f"TTS: {len(req.text)}ch, voice={req.voice}, lang={req.language}, {latency_ms:.0f}ms, audio={audio_duration_ms:.0f}ms")

        return JSONResponse({
            "status": "ok",
            "audio_url": f"/tts_audio/{final_file.name}",
            "audio_file": final_file.name,
            "latency_ms": round(latency_ms, 2),
            "audio_duration_ms": round(audio_duration_ms, 2),
            "text": req.text,
            "voice": req.voice,
            "language": req.language,
            "format": "ogg" if final_file.suffix == ".ogg" else "wav",
        })

    except HTTPException:
        raise
    except Exception as e:
        latency_ms = (time.time() - start) * 1000
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


# ============================================================
# File-based TTS endpoints
# ============================================================

class SynthesizeFileRequest(BaseModel):
    file: str
    language: str = "chinese"
    voice: str = "Chelsie"
    output: str | None = None
    format: str = "ogg"


def _synthesize_file_worker(src_path: Path, out_path: Path, language: str, voice: str, fmt: str):
    """Background worker for file-based TTS synthesis."""
    job = file_synth_jobs[str(out_path)]
    t0 = time.time()

    try:
        content = src_path.read_text(encoding="utf-8")
    except Exception as e:
        job.update({"status": "error", "error": str(e)})
        return

    segments = [s.strip() for s in content.split("\n\n") if s.strip()]
    if not segments:
        segments = [s.strip() for s in content.split("\n") if s.strip()]
    if not segments:
        job.update({"status": "error", "error": "Input file is empty"})
        return

    job["segments"] = len(segments)

    lang_map = {"zh": "chinese", "en": "english", "de": "german",
                "it": "italian", "pt": "portuguese", "es": "spanish",
                "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}
    lang_code = lang_map.get(language, language)

    temp_dir = Path(tempfile.mkdtemp(prefix="tts_file_"))
    segment_files = []
    errors = 0

    # VoiceDesign models require instruct
    file_instruct = None
    if tts_model_name and "VoiceDesign" in tts_model_name:
        voice_defaults = {
            "Chelsie": "A young American female speaker with a clear and friendly voice",
            "Ethan": "A young American male speaker with a confident and natural voice",
            "Vivian": "A young Chinese female speaker with a soft and warm voice",
        }
        file_instruct = voice_defaults.get(voice, "A young Chinese male speaker with a Beijing accent")

    for i, segment in enumerate(segments):
        try:
            seg_prefix = f"seg_{i:04d}"
            seg_kwargs = dict(
                text=segment,
                model=tts_model,
                voice=voice,
                lang_code=lang_code,
                speed=1.0,
                output_path=str(temp_dir),
                file_prefix=seg_prefix,
                audio_format="wav",
                join_audio=True,
                play=False,
                verbose=False,
                stt_model=None,
            )
            if file_instruct:
                seg_kwargs["instruct"] = file_instruct
            with _gpu_queue():
                generate_audio(**seg_kwargs)
                _maybe_clear_gpu_cache()

            seg_file = None
            for candidate in [temp_dir / f"{seg_prefix}.wav", temp_dir / f"{seg_prefix}_000.wav"]:
                if candidate.exists():
                    seg_file = candidate
                    break
            if seg_file is None:
                for f in temp_dir.iterdir():
                    if f.name.startswith(seg_prefix) and f.suffix == ".wav":
                        seg_file = f
                        break

            if seg_file:
                segment_files.append(seg_file)
            else:
                errors += 1
                logger.error(f"TTS file [{i+1}/{len(segments)}] ERROR: no output file")
        except Exception as e:
            errors += 1
            logger.error(f"TTS file [{i+1}/{len(segments)}] ERROR: {e}")

        job["done"] = i + 1
        job["errors"] = errors

    if not segment_files:
        job.update({"status": "error", "error": "All segments failed"})
        return

    try:
        concat_list = temp_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        if fmt == "ogg":
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c:a", "libopus", "-b:a", "48k", str(out_path)],
                capture_output=True, text=True, timeout=300
            )
        else:
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c:a", "pcm_s16le", str(out_path)],
                capture_output=True, text=True, timeout=300
            )

        if result.returncode != 0:
            job.update({"status": "error", "error": f"ffmpeg concat failed: {result.stderr[:200]}"})
            return

    except Exception as e:
        job.update({"status": "error", "error": str(e)})
        return
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_elapsed = time.time() - t0
    audio_duration_ms = get_audio_duration(str(out_path))
    job.update({
        "status": "done",
        "errors": errors,
        "elapsed": round(total_elapsed, 2),
        "audio_duration_ms": round(audio_duration_ms, 2),
    })
    logger.info(f"TTS file done: {len(segments)} segments, {errors} errors, {total_elapsed:.1f}s -> {out_path}")


@app.post("/synthesize/file")
async def synthesize_file(req: SynthesizeFileRequest):
    """Start background file-based TTS synthesis."""
    src_path = Path(req.file).expanduser()
    if not src_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {req.file}")
    if tts_model is None:
        raise HTTPException(status_code=503, detail="TTS model not loaded")

    ext = ".ogg" if req.format == "ogg" else ".wav"
    if req.output:
        out_path = Path(req.output).expanduser()
    else:
        out_path = src_path.with_suffix(ext)

    out_key = str(out_path)
    _prune_jobs(file_synth_jobs)
    file_synth_jobs[out_key] = {
        "status": "running",
        "segments": 0,
        "done": 0,
        "errors": 0,
        "elapsed": 0,
        "audio_duration_ms": 0,
        "_ts": time.time(),
    }

    thread = threading.Thread(
        target=_synthesize_file_worker,
        args=(src_path, out_path, req.language, req.voice, req.format),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "output": out_key}


@app.get("/synthesize/file/status")
async def synthesize_file_status(output: str):
    """Check progress of a file-based TTS job."""
    if output not in file_synth_jobs:
        raise HTTPException(status_code=404, detail=f"No job found for: {output}")
    return file_synth_jobs[output]


@app.post("/api/tts/server/pause")
async def pause_tts_server():
    global tts_server_paused
    tts_server_paused = True
    return {"status": "paused"}


@app.post("/api/tts/server/resume")
async def resume_tts_server():
    global tts_server_paused
    tts_server_paused = False
    return {"status": "active"}


@app.post("/api/tts/server/restart")
async def restart_tts_server():
    global tts_model
    tts_model = None
    gc.collect()
    mx.clear_cache()
    load_tts_model()
    return {"status": "restarted", "model": tts_model_name}


# ============================================================
# Translate endpoints
# ============================================================

@app.get("/api/translate/status")
async def get_translate_status():
    """Get translate model status."""
    return {
        "model": translate_model_name,
        "model_dir": translate_model_dir,
        "loaded": translate_model is not None,
        "loading": translate_model_loading,
        "paused": translate_server_paused,
        "languages_count": len(SUPPORTED_TRANSLATE_LANGS),
    }


@app.post("/api/translate/server/pause")
async def pause_translate_server():
    global translate_server_paused
    translate_server_paused = True
    return {"status": "paused"}


@app.post("/api/translate/server/resume")
async def resume_translate_server():
    global translate_server_paused
    translate_server_paused = False
    return {"status": "active"}


@app.post("/api/translate/server/restart")
async def restart_translate_server():
    global translate_model, translate_tokenizer
    translate_model = None
    translate_tokenizer = None
    gc.collect()
    mx.clear_cache()
    load_translate_model()
    return {"status": "restarted", "model": translate_model_name}


@app.post("/translate", response_model=TranslateResponse)
def post_translate(req: TranslateRequest):
    if translate_server_paused:
        raise HTTPException(503, "Translate server is paused")
    if translate_model is None:
        raise HTTPException(503, "Translate model not loaded")

    texts = [req.q] if isinstance(req.q, str) else req.q
    if not texts:
        raise HTTPException(400, "q must not be empty")
    if req.source not in SUPPORTED_TRANSLATE_LANGS:
        raise HTTPException(400, f"Unsupported source language: {req.source}")
    if req.target not in SUPPORTED_TRANSLATE_LANGS:
        raise HTTPException(400, f"Unsupported target language: {req.target}")

    results = []
    for text in texts:
        translation, elapsed = translate_text(text, req.source, req.target)
        results.append(Translation(
            translatedText=translation,
            detectedSourceLanguage=req.source,
        ))
        # Save to history
        save_translate_history(text, translation, req.source, req.target, elapsed * 1000)
        logger.info(f"Translate: [{req.source}->{req.target}] {len(text)}ch -> {len(translation)}ch in {elapsed:.2f}s")
    return TranslateResponse(data=TranslateResponseData(translations=results))


@app.get("/translate", response_model=TranslateResponse)
def get_translate(
    q: list[str] = Query(...),
    source: str = Query(...),
    target: str = Query(...),
):
    if translate_server_paused:
        raise HTTPException(503, "Translate server is paused")
    if translate_model is None:
        raise HTTPException(503, "Translate model not loaded")
    if source not in SUPPORTED_TRANSLATE_LANGS:
        raise HTTPException(400, f"Unsupported source language: {source}")
    if target not in SUPPORTED_TRANSLATE_LANGS:
        raise HTTPException(400, f"Unsupported target language: {target}")

    results = []
    for text in q:
        translation, elapsed = translate_text(text, source, target)
        results.append(Translation(
            translatedText=translation,
            detectedSourceLanguage=source,
        ))
        save_translate_history(text, translation, source, target, elapsed * 1000)
        logger.info(f"Translate: [{source}->{target}] {len(text)}ch -> {len(translation)}ch in {elapsed:.2f}s")
    return TranslateResponse(data=TranslateResponseData(translations=results))


@app.post("/translate/file")
def translate_file_endpoint(req: FileTranslateRequest):
    if translate_model is None:
        raise HTTPException(503, "Translate model not loaded")

    src_path = Path(req.file).expanduser()
    if not src_path.exists():
        raise HTTPException(400, f"File not found: {req.file}")
    if req.source not in SUPPORTED_TRANSLATE_LANGS:
        raise HTTPException(400, f"Unsupported source language: {req.source}")
    if req.target not in SUPPORTED_TRANSLATE_LANGS:
        raise HTTPException(400, f"Unsupported target language: {req.target}")

    if req.output:
        out_path = Path(req.output).expanduser()
    else:
        out_path = src_path.with_suffix(f".{req.target}")

    out_key = str(out_path)
    _prune_jobs(translate_file_jobs)
    translate_file_jobs[out_key] = {"status": "running", "lines": 0, "done": 0, "errors": 0, "elapsed": 0, "_ts": time.time()}

    thread = threading.Thread(
        target=_translate_file_worker,
        args=(src_path, out_path, req.source, req.target, req.delimiter),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "output": out_key, "message": "Use GET /translate/file/status?output=<path> to check progress."}


@app.get("/translate/file/status")
def translate_file_status(output: str = Query(...)):
    if output not in translate_file_jobs:
        raise HTTPException(404, f"No job found for: {output}")
    return translate_file_jobs[output]


@app.get("/languages")
async def get_languages():
    return {"data": {"languages": [
        {"language": code, "name": name}
        for code, name in sorted(SUPPORTED_TRANSLATE_LANGS.items())
    ]}}


@app.get("/api/translate/history")
async def get_translate_history():
    """Get translation history (all dates)."""
    all_records = []
    if HISTORY_DIR.exists():
        for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
            if d.is_dir():
                jsonl_path = d / "translate_history.jsonl"
                if jsonl_path.exists():
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record["date"] = d.name
                                all_records.append(record)
    all_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return JSONResponse(all_records)

# ============================================================
# Image Gen endpoints
# ============================================================

@app.get("/api/image/status")
async def get_image_status():
    return {
        "model": "z-image-turbo-8bit",
        "model_path": MFLUX_MODEL_PATH,
        "loaded": image_model is not None,
        "loading": image_model_loading,
        "model_on_disk": os.path.isdir(MFLUX_MODEL_PATH),
        "available": image_model is not None,
    }

@app.post("/api/image/generate")
def generate_image(req: ImageGenRequest):
    if image_model is None:
        raise HTTPException(503, "Image model not loaded (USB not mounted or load failed)")

    start = time.time()
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse resolution to width x height
    resolution = req.resolution.strip()
    try:
        if "x" in resolution.lower():
            parts = resolution.lower().split("x")
            width, height = int(parts[0]), int(parts[1])
        else:
            width = height = 1024
    except (ValueError, IndexError):
        width = height = 1024

    file_id = uuid.uuid4().hex[:12]
    output_file = IMAGE_OUTPUT_DIR / f"img_{file_id}.png"

    try:
        logger.info(f"Image gen start: {width}x{height}, steps={req.steps}, seed={req.seed}, prompt={req.prompt[:80]}")

        with _gpu_queue():
            image = image_model.generate_image(
                seed=req.seed if req.seed is not None else int(time.time()) % (2**31),
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=req.steps,
            )
            # Image gen is the biggest memory consumer - always clear cache after
            gc.collect()
            mx.clear_cache()

        image.save(path=str(output_file))

        if not output_file.exists():
            raise HTTPException(500, "Image generation produced no output file")

        latency_ms = (time.time() - start) * 1000
        save_image_history(req.prompt, str(output_file), latency_ms,
                           f"{width}x{height}", seed=req.seed, steps=req.steps)

        logger.info(f"Image gen done: {latency_ms:.0f}ms -> {output_file.name}")

        return {
            "status": "ok",
            "image_url": f"/image_output/{output_file.name}",
            "image_file": output_file.name,
            "latency_ms": round(latency_ms, 2),
            "resolution": f"{width}x{height}",
            "seed": req.seed,
            "steps": req.steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image gen error: {e}")
        raise HTTPException(500, f"Image generation failed: {str(e)}")

@app.get("/image_output/{filename}")
async def get_image_file(filename: str):
    """Serve generated image file."""
    image_path = IMAGE_OUTPUT_DIR / filename
    if image_path.exists():
        return FileResponse(image_path)
    raise HTTPException(404, "Image not found")

@app.get("/api/image/history")
async def get_image_history():
    """Get image history."""
    all_records = []
    if HISTORY_DIR.exists():
        for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
            if d.is_dir():
                jsonl_path = d / "image_history.jsonl"
                if jsonl_path.exists():
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record["date"] = d.name
                                all_records.append(record)
    all_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return JSONResponse(all_records)


# ============================================================
# OpenAI-compatible Chat Completions API
# ============================================================

def _resolve_base64_image(data_uri: str) -> str:
    """Decode a data:image/...;base64,... URI to a temp file. Returns file path."""
    # Format: data:image/png;base64,iVBOR...
    header, _, b64data = data_uri.partition(",")
    if not b64data:
        raise ValueError("Invalid data URI: no base64 data after comma")
    # Extract extension from mime type
    ext = ".png"
    if "image/" in header:
        mime = header.split("image/")[1].split(";")[0]
        ext_map = {"png": ".png", "jpeg": ".jpg", "jpg": ".jpg", "gif": ".gif",
                    "webp": ".webp", "bmp": ".bmp", "tiff": ".tiff"}
        ext = ext_map.get(mime, f".{mime}")
    VISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dest = VISION_OUTPUT_DIR / f"vision_{uuid.uuid4().hex[:12]}{ext}"
    dest.write_bytes(base64.b64decode(b64data))
    return str(dest)


def _extract_chat_content(messages: list[ChatMessage]) -> tuple[str, str | None]:
    """Extract text prompt and image path from OpenAI-format messages.
    Returns (prompt_text, image_path_or_None)."""
    text_parts = []
    image_path = None

    for msg in messages:
        if msg.role == "system":
            # Prepend system message to prompt
            if isinstance(msg.content, str):
                text_parts.insert(0, msg.content)
            continue
        if msg.role != "user":
            continue

        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "text" and part.text:
                    text_parts.append(part.text)
                elif part.type == "image_url" and part.image_url:
                    url = part.image_url.url
                    if image_path is not None:
                        continue  # only use first image
                    if url.startswith("data:"):
                        image_path = _resolve_base64_image(url)
                    elif url.startswith("/") or url.startswith("~"):
                        # Local file path
                        expanded = os.path.expanduser(url)
                        if os.path.exists(expanded):
                            image_path = expanded
                        else:
                            raise ValueError(f"Image file not found: {url}")
                    elif url.startswith("http://") or url.startswith("https://"):
                        raise ValueError(
                            "HTTP image URLs not supported - use base64 data URI or local file path"
                        )
                    else:
                        # Try as filename in VISION_OUTPUT_DIR
                        candidate = VISION_OUTPUT_DIR / url
                        if candidate.exists():
                            image_path = str(candidate)
                        elif os.path.exists(url):
                            image_path = url
                        else:
                            raise ValueError(f"Image not found: {url}")

    prompt = "\n".join(text_parts) if text_parts else "Describe this image."
    return prompt, image_path


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint for vision model."""
    if req.stream:
        raise HTTPException(400, "Streaming not supported - set stream=false")
    if vision_server_paused:
        raise HTTPException(503, "Vision server is paused")
    if vision_model is None:
        raise HTTPException(503, "Vision model not loaded")

    try:
        prompt, image_path = _extract_chat_content(req.messages)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if image_path is None:
        raise HTTPException(400, "No image provided in messages. Include an image_url content part.")

    # Copy to vision_output if not already there
    VISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(image_path)
    if VISION_OUTPUT_DIR not in src.resolve().parents:
        dest = VISION_OUTPUT_DIR / f"vision_{uuid.uuid4().hex[:12]}{src.suffix}"
        shutil.copy2(image_path, dest)
        image_path = str(dest)

    start = time.time()
    try:
        formatted = apply_chat_template(
            vision_processor, vision_config, prompt, num_images=1
        )
        with _gpu_queue():
            result = vlm_generate(
                vision_model, vision_processor,
                prompt=formatted, image=image_path,
                max_tokens=req.max_tokens, verbose=False,
            )
            _maybe_clear_gpu_cache()

        latency_ms = (time.time() - start) * 1000
        response_text = result.text if hasattr(result, 'text') else str(result)
        prompt_tokens = getattr(result, 'prompt_tokens', 0)
        generation_tokens = getattr(result, 'generation_tokens', 0)
        generation_tps = getattr(result, 'generation_tps', 0.0)

        # Determine finish reason
        finish_reason = "stop"
        if generation_tokens >= req.max_tokens:
            finish_reason = "length"

        save_vision_history(
            prompt=prompt,
            image_path=image_path,
            response_text=response_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            generation_tps=generation_tps,
        )

        logger.info(f"Vision: {len(prompt)}ch prompt, {len(response_text)}ch response in {latency_ms:.0f}ms")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": vision_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": generation_tokens,
                "total_tokens": prompt_tokens + generation_tokens,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(500, f"Chat completion failed: {str(e)}")


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible model listing endpoint."""
    models = []
    if vision_model is not None:
        models.append({
            "id": vision_model_name,
            "object": "model",
            "created": 0,
            "owned_by": "local",
        })
    return {"object": "list", "data": models}


# ============================================================
# Vision endpoints (internal dashboard API)
# ============================================================

@app.get("/api/vision/status")
async def get_vision_status():
    return {
        "model": vision_model_name,
        "model_short": vision_model_name.split("/")[-1],
        "loaded": vision_model is not None,
        "loading": vision_model_loading,
        "paused": vision_server_paused,
    }


@app.post("/api/vision/upload")
async def upload_vision_file(file: UploadFile = File(...)):
    """Upload image file, save to VISION_OUTPUT_DIR, return path."""
    VISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex[:12]
    ext = Path(file.filename).suffix or ".png"
    dest = VISION_OUTPUT_DIR / f"vision_{file_id}{ext}"
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    return {"path": str(dest), "filename": dest.name}


@app.post("/api/vision/analyze")
def analyze_vision(req: VisionRequest):
    if vision_server_paused:
        raise HTTPException(503, "Vision server is paused")
    if vision_model is None:
        raise HTTPException(503, "Vision model not loaded")

    # Resolve image path
    image_path = req.image
    if not os.path.isabs(image_path):
        # Treat as filename in VISION_OUTPUT_DIR
        candidate = VISION_OUTPUT_DIR / image_path
        if candidate.exists():
            image_path = str(candidate)
    if not os.path.exists(image_path):
        raise HTTPException(400, f"Image not found: {req.image}")

    # Copy to vision_output if not already there, so history thumbnails work
    VISION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(image_path)
    if VISION_OUTPUT_DIR not in src.resolve().parents:
        dest = VISION_OUTPUT_DIR / f"vision_{uuid.uuid4().hex[:12]}{src.suffix}"
        shutil.copy2(image_path, dest)
        image_path = str(dest)

    start = time.time()
    try:
        formatted = apply_chat_template(
            vision_processor, vision_config, req.prompt, num_images=1
        )
        with _gpu_queue():
            result = vlm_generate(
                vision_model, vision_processor,
                prompt=formatted, image=image_path,
                max_tokens=req.max_tokens, verbose=False,
            )
            _maybe_clear_gpu_cache()

        latency_ms = (time.time() - start) * 1000
        response_text = result.text if hasattr(result, 'text') else str(result)
        prompt_tokens = getattr(result, 'prompt_tokens', 0)
        generation_tokens = getattr(result, 'generation_tokens', 0)
        generation_tps = getattr(result, 'generation_tps', 0.0)

        save_vision_history(
            prompt=req.prompt,
            image_path=image_path,
            response_text=response_text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            generation_tps=generation_tps,
        )

        logger.info(f"Vision: {len(req.prompt)}ch prompt, {len(response_text)}ch response in {latency_ms:.0f}ms")

        return {
            "status": "ok",
            "response": response_text,
            "latency_ms": round(latency_ms, 2),
            "image": req.image,
            "prompt": req.prompt,
            "prompt_tokens": prompt_tokens,
            "generation_tokens": generation_tokens,
            "generation_tps": round(generation_tps, 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vision error: {e}", exc_info=True)
        raise HTTPException(500, f"Vision analysis failed: {str(e)}")


@app.get("/vision_output/{filename}")
async def get_vision_file(filename: str):
    """Serve uploaded/processed vision images."""
    file_path = VISION_OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(404, "Image not found")


@app.get("/api/vision/history")
async def get_vision_history():
    """Get vision history (all dates)."""
    all_records = []
    if HISTORY_DIR.exists():
        for d in sorted(HISTORY_DIR.iterdir(), reverse=True):
            if d.is_dir():
                jsonl_path = d / "vision_history.jsonl"
                if jsonl_path.exists():
                    with open(jsonl_path, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record["date"] = d.name
                                all_records.append(record)
    all_records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return JSONResponse(all_records)


@app.post("/api/vision/server/pause")
async def pause_vision_server():
    global vision_server_paused
    vision_server_paused = True
    return {"status": "paused"}


@app.post("/api/vision/server/resume")
async def resume_vision_server():
    global vision_server_paused
    vision_server_paused = False
    return {"status": "active"}


@app.post("/api/vision/server/restart")
async def restart_vision_server():
    global vision_model, vision_processor, vision_config
    vision_model = None
    vision_processor = None
    vision_config = None
    gc.collect()
    mx.clear_cache()
    load_vision_model()
    return {"status": "restarted", "model": vision_model_name}


# ============================================================
# GPU stats endpoint
# ============================================================

_gpu_stats_cache: dict = {"data": None, "ts": 0.0}

def _get_gpu_stats() -> dict:
    """Get GPU utilization, memory, and disk stats. Cached for 2s to avoid ioreg overhead."""
    import re
    import shutil

    now = time.time()
    if _gpu_stats_cache["data"] is not None and now - _gpu_stats_cache["ts"] < 2.0:
        return _gpu_stats_cache["data"]

    stats = {
        "metal_active_mb": round(mx.get_active_memory() / 1024 / 1024),
        "metal_peak_mb": round(mx.get_peak_memory() / 1024 / 1024),
        "metal_cache_mb": round(mx.get_cache_memory() / 1024 / 1024),
    }

    # Device info
    try:
        info = mx.device_info()
        stats["device"] = info.get("device_name", "unknown")
        stats["total_memory_gb"] = round(info.get("memory_size", 0) / 1024**3, 1)
    except Exception:
        stats["device"] = "unknown"
        stats["total_memory_gb"] = 0

    # GPU utilization from ioreg (macOS Apple Silicon)
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"],
            capture_output=True, text=True, timeout=2,
        )
        out = result.stdout
        m = re.search(r'"Device Utilization %"=(\d+)', out)
        stats["gpu_utilization"] = int(m.group(1)) if m else 0
        m = re.search(r'"Renderer Utilization %"=(\d+)', out)
        stats["renderer_utilization"] = int(m.group(1)) if m else 0
        m = re.search(r'"In use system memory"=(\d+)', out)
        if m:
            stats["gpu_sys_memory_mb"] = round(int(m.group(1)) / 1024 / 1024)
        m = re.search(r'"Alloc system memory"=(\d+)', out)
        if m:
            stats["gpu_alloc_memory_mb"] = round(int(m.group(1)) / 1024 / 1024)
    except Exception:
        stats["gpu_utilization"] = 0
        stats["renderer_utilization"] = 0

    # Local disk usage (root volume)
    try:
        du = shutil.disk_usage("/")
        stats["disk_total_gb"] = round(du.total / 1024**3, 1)
        stats["disk_free_gb"] = round(du.free / 1024**3, 1)
        stats["disk_used_gb"] = round(du.used / 1024**3, 1)
    except Exception:
        pass

    # USB disk usage (if mounted)
    usb_path = "/Volumes/One Touch"
    if os.path.ismount(usb_path):
        try:
            du = shutil.disk_usage(usb_path)
            stats["usb_mounted"] = True
            stats["usb_total_gb"] = round(du.total / 1024**3, 1)
            stats["usb_free_gb"] = round(du.free / 1024**3, 1)
            stats["usb_used_gb"] = round(du.used / 1024**3, 1)
        except Exception:
            stats["usb_mounted"] = False
    else:
        stats["usb_mounted"] = False

    _gpu_stats_cache["data"] = stats
    _gpu_stats_cache["ts"] = time.time()
    return stats


@app.get("/api/gpu")
async def get_gpu_stats():
    stats = _get_gpu_stats()
    # Queue count is live, not cached
    stats["gpu_queue_waiting"] = _gpu_queue_waiting
    stats["gpu_lock_held"] = _gpu_lock.locked()
    stats["gpu_inflight"] = _gpu_inflight
    stats["uptime_s"] = int(time.time() - _SERVER_START_TIME)
    return stats


# ============================================================
# Log endpoints
# ============================================================

from collections import Counter
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/api/logs/histogram")
async def get_log_histogram():
    """Return log line counts per minute for the last 60 minutes."""
    now = datetime.now()
    counts = Counter()
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    # Parse timestamp: "2026-02-09 12:42:21 [INFO] ..."
                    if len(line) >= 19:
                        try:
                            ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                            diff = (now - ts).total_seconds()
                            if 0 <= diff < 3600:
                                minute = int(diff // 60)
                                counts[minute] = counts.get(minute, 0) + 1
                        except ValueError:
                            pass
        except Exception:
            pass
    # Return array of 60 values, index 0 = most recent minute
    histogram = [counts.get(i, 0) for i in range(60)]
    return {"histogram": histogram, "total": sum(histogram)}


@app.get("/api/logs/stream")
async def stream_logs():
    """Stream log file in real-time (tail -f style via SSE)."""
    async def _generate():
        # Send last 200 lines via backward seek (avoids reading entire file)
        lines = []
        if LOG_FILE.exists():
            try:
                size = LOG_FILE.stat().st_size
                # Read last 64KB at most for tail
                chunk_size = min(size, 64 * 1024)
                with open(LOG_FILE, "rb") as f:
                    f.seek(max(0, size - chunk_size))
                    raw = f.read()
                    lines = raw.decode("utf-8", errors="replace").splitlines()
                    # Drop first partial line if we seeked mid-file
                    if size > chunk_size and lines:
                        lines = lines[1:]
            except Exception:
                pass
        for line in lines[-200:]:
            yield f"data: {json.dumps(line)}\n\n"
        # Tail for new lines using binary seek (reliable byte offsets)
        last_pos = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0
        while True:
            await asyncio.sleep(0.5)
            try:
                size = LOG_FILE.stat().st_size
                if size > last_pos:
                    with open(LOG_FILE, "rb") as f:
                        f.seek(last_pos)
                        chunk = f.read()
                        last_pos = f.tell()
                    new_lines = chunk.decode("utf-8", errors="replace").splitlines()
                    for line in new_lines:
                        if line.strip():
                            yield f"data: {json.dumps(line)}\n\n"
                elif size < last_pos:
                    # File was rotated
                    last_pos = 0
            except Exception:
                pass

    return StreamingResponse(_generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ============================================================
# Server restart (re-exec process to pick up code changes)
# ============================================================

@app.post("/restart")
async def restart_process():
    """Restart the entire server process to load new code changes."""
    def _do_restart():
        time.sleep(0.5)  # let response flush
        logger.info("Server restarting via /restart endpoint")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    threading.Thread(target=_do_restart, daemon=True).start()
    return {"status": "restarting"}


# ============================================================
# HTML Dashboard (served from static/index.html)
# ============================================================

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
@app.get("/history", response_class=HTMLResponse)
async def history_page():
    """Serve the history viewer HTML page from static/index.html."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard HTML not found")
    return Response(
        content=html_path.read_text(encoding="utf-8"),
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

if __name__ == "__main__":
    logger.info(f"Starting MLX Serving on {HOST}:{PORT}")
    logger.info(f"Log file: {LOG_FILE}")
    # Limit threadpool: only 1 GPU so extra threads just waste memory waiting on _gpu_lock
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", workers=1)