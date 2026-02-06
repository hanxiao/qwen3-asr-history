#!/usr/bin/env python3
"""
MLX Serving - Unified ASR + TTS + Translate server.
Keeps models loaded in memory for fast inference.
Saves transcription/synthesis/translation history to ~/Documents/qwen3-asr-history/history/
"""
import os
import sys
import tempfile
import subprocess
import time
import json
import uuid
import threading
import re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

import shutil


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

current_model_name = "mlx-community/Qwen3-ASR-0.6B-bf16"
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

# Serialize all translate inference - MLX Metal can't handle concurrent GPU access
_translate_inference_lock = threading.Lock()
_translate_request_count = 0
_TRANSLATE_CACHE_CLEAR_INTERVAL = 50

# Track background file translations
translate_file_jobs: dict[str, dict] = {}

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
PROJECT_DIR = Path(__file__).parent.parent.resolve()
HISTORY_DIR = PROJECT_DIR / "history"
TTS_OUTPUT_DIR = PROJECT_DIR / "tts_output"

app = FastAPI(title="MLX Serving")

# Track background file synthesis jobs
file_synth_jobs: dict[str, dict] = {}


# ============================================================
# Image Generation globals
# ============================================================
IMAGE_OUTPUT_DIR = PROJECT_DIR / "image_output"
MFLUX_CLI = "/Users/hanxiao/.local/bin/mflux-generate-z-image-turbo"
MFLUX_MODEL_PATH = "/Volumes/One Touch/ai-models/mflux/z-image-turbo-8bit/"

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
    except:
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
                       seed: int | None = None, steps: int = 4):
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
# Translate helpers
# ============================================================

def translate_text(text: str, source: str, target: str) -> tuple[str, float]:
    """Translate a single text. Returns (translation, elapsed_seconds).
    Thread-safe: uses _translate_inference_lock to serialize Metal GPU access."""
    with _translate_inference_lock:
        return _translate_text_impl(text, source, target)


def _translate_text_impl(text: str, source: str, target: str) -> tuple[str, float]:
    global _translate_request_count
    import mlx.core as mx
    from mlx_lm import stream_generate

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
    result_parts = []
    for resp in stream_generate(
        translate_model, translate_tokenizer, prompt=prompt,
        max_tokens=1024,
    ):
        segment = resp.text
        if "<end_of_turn>" in segment:
            result_parts.append(segment.split("<end_of_turn>")[0])
            break
        result_parts.append(segment)
    elapsed = time.time() - t0

    clean = "".join(result_parts).strip()

    # Periodic Metal cache clear to prevent long-running buildup
    _translate_request_count += 1
    if _translate_request_count % _TRANSLATE_CACHE_CLEAR_INTERVAL == 0:
        mx.metal.clear_cache()

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
            print(f"  file [{i+1}/{len(segments)}] {len(stripped)}ch -> {len(result)}ch in {elapsed:.2f}s")
        except Exception as e:
            print(f"  file [{i+1}/{len(segments)}] ERROR: {e}")
            translated.append(stripped)
            errors += 1
            job["done"] = i + 1

    total_elapsed = time.time() - t0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text = delimiter.join(translated) if delimiter != "\n" else "\n".join(translated)
    out_path.write_text(out_text, encoding="utf-8")

    job.update({"status": "done", "errors": errors, "elapsed": round(total_elapsed, 2)})
    print(f"  File done: {len(segments)} lines, {errors} errors, {total_elapsed:.1f}s -> {out_path}")


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


def load_model(model_name: str = None):
    """Load ASR model."""
    global model, load_fn, generate_fn, current_model_name

    if model_name:
        current_model_name = model_name

    print(f"Loading ASR model {current_model_name}...")
    start = time.time()

    from mlx_audio.stt.utils import load_model
    from mlx_audio.stt.generate import generate_transcription

    load_fn = load_model
    generate_fn = generate_transcription
    model = load_model(current_model_name)

    elapsed = time.time() - start
    print(f"ASR model loaded in {elapsed:.2f}s")
    return model


def load_tts_model(model_name: str = None):
    """Load TTS model."""
    global tts_model, tts_model_name, tts_model_loading

    if model_name:
        tts_model_name = model_name

    tts_model_loading = True
    print(f"Loading TTS model {tts_model_name}...")
    start = time.time()

    try:
        from mlx_audio.tts.utils import load_model as tts_load

        # Try USB path first for all models
        usb_base = "/Volumes/One Touch/ai-models/mlx-community"
        usb_path = os.path.join(usb_base, tts_model_name.split("/")[-1])
        if os.path.exists(usb_path):
            model_path = usb_path
        else:
            model_path = tts_model_name

        tts_model = tts_load(model_path=model_path)

        elapsed = time.time() - start
        print(f"TTS model loaded in {elapsed:.2f}s")

        # Print supported speakers
        if hasattr(tts_model, 'get_supported_speakers'):
            speakers = tts_model.get_supported_speakers()
            if speakers:
                print(f"Supported speakers: {speakers}")
        if hasattr(tts_model, 'get_supported_languages'):
            langs = tts_model.get_supported_languages()
            print(f"Supported languages: {langs}")
    except Exception as e:
        print(f"Failed to load TTS model: {e}")
        tts_model = None
    finally:
        tts_model_loading = False

    return tts_model


def load_translate_model():
    """Load translate model (TranslateGemma)."""
    global translate_model, translate_tokenizer, translate_model_loading

    translate_model_loading = True
    print(f"Loading translate model from {translate_model_dir}...")
    start = time.time()

    try:
        from mlx_lm import load as mlx_load
        translate_model, translate_tokenizer = mlx_load(translate_model_dir)
        elapsed = time.time() - start
        print(f"Translate model loaded in {elapsed:.2f}s")
    except Exception as e:
        print(f"Failed to load translate model: {e}")
        translate_model = None
        translate_tokenizer = None
    finally:
        translate_model_loading = False


def convert_wav_to_ogg(wav_path: str, ogg_path: str) -> bool:
    """Convert WAV to OGG Opus using ffmpeg. Returns True on success."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "48k", ogg_path],
            capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"WAV->OGG conversion failed: {e}")
        return False


def convert_to_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV for best results."""
    wav_path = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], capture_output=True)
    return wav_path


# ============================================================
# Startup
# ============================================================

@app.on_event("startup")
async def startup_event():
    load_model()
    load_tts_model()
    load_translate_model()


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
async def transcribe(req: TranscribeRequest):
    global model

    if server_paused:
        raise HTTPException(status_code=503, detail="Server is paused")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not os.path.exists(req.path):
        raise HTTPException(status_code=400, detail=f"Audio file not found: {req.path}")

    start = time.time()
    wav_path = convert_to_wav(req.path)

    try:
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            output_path = f.name.replace(".txt", "")

        segments = generate_fn(
            model=model,
            audio=wav_path,
            output_path=output_path,
            format="txt",
            verbose=False,
            language=req.language or "zh"
        )

        txt_file = output_path + ".txt"
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                text = f.read().strip()
            os.unlink(txt_file)
        else:
            if segments:
                text = " ".join(s.get("text", "") for s in segments if isinstance(s, dict))
            else:
                text = str(segments) if segments else ""
    finally:
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    latency_ms = (time.time() - start) * 1000
    audio_duration_ms = get_audio_duration(req.path)

    if text:
        save_to_history(req.path, text, latency_ms, audio_duration_ms, current_model_name)

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
    import gc
    gc.collect()
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
async def synthesize(req: SynthesizeRequest):
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
        from mlx_audio.tts.generate import generate_audio

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
        if req.instruct:
            gen_kwargs["instruct"] = req.instruct

        generate_audio(**gen_kwargs)

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

    from mlx_audio.tts.generate import generate_audio

    lang_map = {"zh": "chinese", "en": "english", "de": "german",
                "it": "italian", "pt": "portuguese", "es": "spanish",
                "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}
    lang_code = lang_map.get(language, language)

    temp_dir = Path(tempfile.mkdtemp(prefix="tts_file_"))
    segment_files = []
    errors = 0

    for i, segment in enumerate(segments):
        try:
            seg_prefix = f"seg_{i:04d}"
            generate_audio(
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
                print(f"  file tts [{i+1}/{len(segments)}] ERROR: no output file")
        except Exception as e:
            errors += 1
            print(f"  file tts [{i+1}/{len(segments)}] ERROR: {e}")

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
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    total_elapsed = time.time() - t0
    audio_duration_ms = get_audio_duration(str(out_path))
    job.update({
        "status": "done",
        "errors": errors,
        "elapsed": round(total_elapsed, 2),
        "audio_duration_ms": round(audio_duration_ms, 2),
    })
    print(f"  File TTS done: {len(segments)} segments, {errors} errors, {total_elapsed:.1f}s -> {out_path}")


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
    file_synth_jobs[out_key] = {
        "status": "running",
        "segments": 0,
        "done": 0,
        "errors": 0,
        "elapsed": 0,
        "audio_duration_ms": 0,
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
    import gc
    gc.collect()
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
    import gc
    gc.collect()
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
        print(f"  [{req.source}->{req.target}] {len(text)}ch -> {len(translation)}ch in {elapsed:.2f}s")
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
        print(f"  [{source}->{target}] {len(text)}ch -> {len(translation)}ch in {elapsed:.2f}s")
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
    translate_file_jobs[out_key] = {"status": "running", "lines": 0, "done": 0, "errors": 0, "elapsed": 0}

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
    cli_available = shutil.which(MFLUX_CLI) is not None or os.path.isfile(MFLUX_CLI)
    model_available = os.path.isdir(MFLUX_MODEL_PATH)
    return {
        "model": "z-image-turbo-8bit",
        "cli_path": MFLUX_CLI,
        "model_path": MFLUX_MODEL_PATH,
        "cli_available": cli_available,
        "model_available": model_available,
        "available": cli_available and model_available,
    }

@app.post("/api/image/generate")
async def generate_image(req: ImageGenRequest):
    cli_available = shutil.which(MFLUX_CLI) is not None or os.path.isfile(MFLUX_CLI)
    if not cli_available:
        raise HTTPException(503, "mflux-generate-z-image-turbo CLI not found")
    if not os.path.isdir(MFLUX_MODEL_PATH):
        raise HTTPException(503, f"Model not found at {MFLUX_MODEL_PATH}")

    start = time.time()
    IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Parse resolution to width x height
    resolution = req.resolution.strip()
    try:
        if "x" in resolution.lower():
            parts = resolution.lower().split("x")
            width, height = int(parts[0]), int(parts[1])
        else:
            # Default to square
            width = height = 1024
    except (ValueError, IndexError):
        width = height = 1024

    file_id = uuid.uuid4().hex[:12]
    output_file = IMAGE_OUTPUT_DIR / f"img_{file_id}.png"

    cmd = [
        MFLUX_CLI,
        "--prompt", req.prompt,
        "--model", MFLUX_MODEL_PATH,
        "--quantize", "8",
        "--steps", str(req.steps),
        "--width", str(width),
        "--height", str(height),
        "--output", str(output_file),
    ]

    if req.seed is not None:
        cmd.extend(["--seed", str(req.seed)])

    if req.negative_prompt:
        cmd.extend(["--negative-prompt", req.negative_prompt])

    try:
        print(f"Image gen: {width}x{height}, steps={req.steps}, seed={req.seed}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"mflux stderr: {result.stderr}")
            raise HTTPException(500, f"Image generation failed: {result.stderr[:500]}")

        if not output_file.exists():
            raise HTTPException(500, "Image generation produced no output file")

        latency_ms = (time.time() - start) * 1000
        save_image_history(req.prompt, str(output_file), latency_ms,
                           f"{width}x{height}", seed=req.seed, steps=req.steps)

        return {
            "status": "ok",
            "image_url": f"/image_output/{output_file.name}",
            "image_file": output_file.name,
            "latency_ms": round(latency_ms, 2),
            "resolution": f"{width}x{height}",
            "seed": req.seed,
            "steps": req.steps,
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(504, "Image generation timed out (120s)")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image gen error: {e}")
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
# HTML Dashboard (served from static/index.html)
# ============================================================

STATIC_DIR = Path(__file__).parent / "static"


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
    print(f"Starting MLX Serving on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")