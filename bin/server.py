#!/usr/bin/env python3
"""
Persistent Qwen3 Speech server (ASR + TTS).
Keeps models loaded in memory for fast inference.
Saves transcription/synthesis history to ~/Documents/qwen3-asr-history/history/
"""
import os
import sys
import tempfile
import subprocess
import time
import json
import uuid
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

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
tts_model_name = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
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

HOST = "127.0.0.1"
PORT = 18321
PROJECT_DIR = Path(__file__).parent.parent.resolve()
HISTORY_DIR = PROJECT_DIR / "history"
TTS_OUTPUT_DIR = PROJECT_DIR / "tts_output"

app = FastAPI(title="Qwen3-Speech Server")

# Track background file synthesis jobs
file_synth_jobs: dict[str, dict] = {}  # output_path -> {status, segments, done, errors, elapsed, audio_duration_ms}


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


class SwitchModelRequest(BaseModel):
    model: str


class SwitchTTSModelRequest(BaseModel):
    model: str


def load_model(model_name: str = None):
    """Load ASR model."""
    global model, load_fn, generate_fn, current_model_name

    if model_name:
        current_model_name = model_name

    print(f"Loading ASR model {current_model_name}...")
    start = time.time()

    from mlx_audio.stt import load
    from mlx_audio.stt.generate import generate_transcription

    load_fn = load
    generate_fn = generate_transcription
    model = load(current_model_name)

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

        # Try USB path first for the default model
        model_path = tts_model_name
        if tts_model_name == "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16" and os.path.exists(TTS_MODEL_PATH):
            model_path = TTS_MODEL_PATH

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


# ============================================================
# ASR endpoints
# ============================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": current_model_name,
        "loaded": model is not None,
        "tts_model": tts_model_name,
        "tts_loaded": tts_model is not None
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
    # Add model-specific speakers if available
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
    # Sort by timestamp descending
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

    # Create output directory
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_id = uuid.uuid4().hex[:12]
    file_prefix = f"tts_{file_id}"
    output_file = TTS_OUTPUT_DIR / f"{file_prefix}.wav"

    try:
        from mlx_audio.tts.generate import generate_audio

        # Map short language codes to full names for Qwen3-TTS
        lang_map = {"zh": "chinese", "en": "english", "de": "german",
                     "it": "italian", "pt": "portuguese", "es": "spanish",
                     "ja": "japanese", "ko": "korean", "fr": "french", "ru": "russian"}
        lang_code = lang_map.get(req.language, req.language)

        generate_audio(
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

        # Find the generated file (could be file_prefix.wav or file_prefix_000.wav)
        actual_file = None
        for candidate in [
            TTS_OUTPUT_DIR / f"{file_prefix}.wav",
            TTS_OUTPUT_DIR / f"{file_prefix}_000.wav",
        ]:
            if candidate.exists():
                actual_file = candidate
                break

        if actual_file is None:
            # Check for any file with the prefix
            for f in TTS_OUTPUT_DIR.iterdir():
                if f.name.startswith(file_prefix) and f.suffix == ".wav":
                    actual_file = f
                    break

        if actual_file is None:
            raise HTTPException(status_code=500, detail="Audio generation failed - no output file")

        # Rename to standard name if needed
        if actual_file != output_file:
            actual_file.rename(output_file)

        # Convert to OGG Opus if requested (default)
        final_file = output_file
        if req.format == "ogg":
            ogg_file = output_file.with_suffix(".ogg")
            if convert_wav_to_ogg(str(output_file), str(ogg_file)):
                # Delete intermediate WAV
                output_file.unlink(missing_ok=True)
                final_file = ogg_file
            else:
                # Conversion failed, fall back to WAV
                final_file = output_file

        latency_ms = (time.time() - start) * 1000
        audio_duration_ms = get_audio_duration(str(final_file))

        # Save to history
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
    file: str                    # input text file path
    language: str = "chinese"
    voice: str = "Chelsie"
    output: str | None = None    # output file path (default: {input}.ogg)
    format: str = "ogg"          # "ogg" (default) or "wav"


def _synthesize_file_worker(src_path: Path, out_path: Path, language: str, voice: str, fmt: str):
    """Background worker for file-based TTS synthesis."""
    job = file_synth_jobs[str(out_path)]
    t0 = time.time()

    try:
        content = src_path.read_text(encoding="utf-8")
    except Exception as e:
        job.update({"status": "error", "error": str(e)})
        return

    # Split by double newlines (paragraphs) first, then by single newlines for long text
    segments = [s.strip() for s in content.split("\n\n") if s.strip()]
    if not segments:
        segments = [s.strip() for s in content.split("\n") if s.strip()]
    if not segments:
        job.update({"status": "error", "error": "Input file is empty"})
        return

    job["segments"] = len(segments)

    # Generate audio for each segment into temp WAV files
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

            # Find generated file
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

    # Concatenate all segment WAVs using ffmpeg
    try:
        concat_list = temp_dir / "concat.txt"
        with open(concat_list, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        if fmt == "ogg":
            # Concatenate and convert to OGG Opus in one step
            result = subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
                 "-c:a", "libopus", "-b:a", "48k", str(out_path)],
                capture_output=True, text=True, timeout=300
            )
        else:
            # Concatenate to WAV
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
        # Cleanup temp files
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
# HTML Dashboard
# ============================================================

HISTORY_HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>qwen3-speech</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'JetBrains Mono', monospace;
            background: #fafafa;
            color: #1a1f36;
            font-size: 14px;
        }
        .container { display: flex; height: 100vh; }
        .sidebar {
            width: 220px;
            background: #1a1f36;
            color: #fff;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        .logo {
            padding: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px dashed #3a4060;
        }
        .logo svg { width: 28px; height: 28px; }
        .logo-text { font-weight: 600; font-size: 15px; }

        /* Tab navigation in sidebar */
        .tab-nav {
            display: flex;
            border-bottom: 1px dashed #3a4060;
        }
        .tab-btn {
            flex: 1;
            padding: 10px 0;
            text-align: center;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            color: #6b7280;
            background: transparent;
            border: none;
            cursor: pointer;
            transition: all 0.15s;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .tab-btn:hover { color: #a0a8c0; }
        .tab-btn.active {
            color: #fff;
            background: rgba(226, 95, 74, 0.2);
            border-bottom: 2px solid #e25f4a;
        }

        .sidebar-content { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        /* ASR sidebar */
        .asr-sidebar { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
        .tts-sidebar { flex: 1; display: flex; flex-direction: column; overflow: hidden; display: none; }

        .dates-container { flex: 1; overflow-y: auto; padding: 10px 0; }
        .date-item {
            padding: 10px 20px;
            cursor: pointer;
            color: #a0a8c0;
            transition: all 0.15s;
        }
        .date-item:hover { color: #fff; background: rgba(255,255,255,0.05); }
        .date-item.active {
            color: #fff;
            background: #e25f4a;
        }
        .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

        /* Main panel tabs */
        .main-content { flex: 1; overflow-y: auto; padding: 30px 40px; }

        .header {
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px dashed #d0d5dd;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-controls {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header h2 {
            font-size: 13px;
            font-weight: 500;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .sort-controls {
            display: flex;
            gap: 8px;
            align-items: center;
        }
        .sort-label {
            font-size: 11px;
            color: #9ca3af;
        }
        .sort-btn {
            background: #fff;
            border: 1px dashed #d0d5dd;
            color: #6b7280;
            padding: 4px 10px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .sort-btn:hover {
            border-color: #e25f4a;
            color: #e25f4a;
        }
        .sort-btn.active {
            background: #e25f4a;
            border-color: #e25f4a;
            color: #fff;
        }
        .sort-btn svg {
            width: 10px;
            height: 10px;
        }
        .speed-controls {
            display: flex;
            gap: 4px;
            align-items: center;
            margin-left: 8px;
        }
        .speed-label {
            font-size: 11px;
            color: #9ca3af;
            margin-right: 4px;
        }
        .speed-btn {
            background: #fff;
            border: 1px dashed #d0d5dd;
            color: #6b7280;
            padding: 4px 8px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s;
            min-width: 28px;
            text-align: center;
        }
        .speed-btn:hover {
            border-color: #e25f4a;
            color: #e25f4a;
        }
        .speed-btn.active {
            background: #e25f4a;
            border-color: #e25f4a;
            color: #fff;
        }
        .transcript {
            background: #fff;
            border: 1px dashed #d0d5dd;
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: border-color 0.15s;
        }
        .transcript:hover { border-color: #e25f4a; }
        .transcript-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
        }
        .transcript-time {
            color: #6b7280;
            font-size: 12px;
        }
        .transcript-latency {
            color: #e25f4a;
            font-size: 11px;
            padding: 2px 6px;
            background: #fef2f0;
            border-radius: 3px;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        .transcript-latency svg {
            width: 12px;
            height: 12px;
        }
        .transcript-model {
            color: #6b7280;
            font-size: 10px;
            padding: 2px 6px;
            background: #f3f4f6;
            border-radius: 3px;
            margin-left: auto;
        }
        .transcript-text {
            font-size: 15px;
            line-height: 1.6;
            color: #1a1f36;
        }
        .transcript-footer {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px dashed #e5e7eb;
        }
        .transcript-file {
            color: #9ca3af;
            font-size: 11px;
            flex: 1;
        }
        .play-btn {
            background: #1a1f36;
            color: white;
            border: none;
            padding: 6px 14px;
            cursor: pointer;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            display: flex;
            align-items: center;
            gap: 6px;
            transition: background 0.15s;
        }
        .play-btn:hover { background: #e25f4a; }
        .play-btn svg { width: 12px; height: 12px; }
        .waveform-container {
            position: relative;
            height: 48px;
            margin-top: 12px;
            background: #f8f9fa;
            border-radius: 4px;
            cursor: pointer;
            overflow: hidden;
        }
        .waveform-container canvas {
            display: block;
            width: 100%;
            height: 100%;
        }
        .waveform-progress {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: rgba(226, 95, 74, 0.15);
            pointer-events: none;
            transition: width 0.1s linear;
        }
        .waveform-time {
            position: absolute;
            bottom: 4px;
            right: 8px;
            font-size: 10px;
            color: #6b7280;
            pointer-events: none;
        }
        .waveform-loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 11px;
            color: #9ca3af;
        }
        audio {
            display: none;
        }
        .empty {
            color: #9ca3af;
            text-align: center;
            padding: 60px 40px;
            border: 1px dashed #d0d5dd;
            background: #fff;
        }
        .stats {
            padding: 16px 20px;
            border-top: 1px dashed #3a4060;
            font-size: 11px;
            color: #6b7280;
        }
        .stats-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }
        .stats-value { color: #e25f4a; }

        /* Section label in sidebar */
        .section-label {
            padding: 8px 20px 4px;
            font-size: 9px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #4a5070;
            border-top: 1px dashed #3a4060;
        }

        /* Model status section */
        .model-status {
            padding: 16px 20px;
            border-top: 1px dashed #3a4060;
            font-size: 11px;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        .status-dot.paused {
            background: #fbbf24;
            animation: none;
        }
        .status-dot.loading {
            background: #60a5fa;
            animation: pulse 0.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .status-text {
            color: #a0a8c0;
            font-size: 11px;
        }
        .model-name {
            color: #fff;
            font-size: 11px;
            margin-bottom: 12px;
            word-break: break-all;
            line-height: 1.4;
        }
        .model-controls {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .ctrl-btn {
            background: rgba(255,255,255,0.1);
            border: 1px dashed #3a4060;
            color: #a0a8c0;
            padding: 4px 8px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: all 0.15s;
        }
        .ctrl-btn:hover {
            background: #e25f4a;
            border-color: #e25f4a;
            color: #fff;
        }
        .ctrl-btn.active {
            background: #e25f4a;
            border-color: #e25f4a;
            color: #fff;
        }
        .model-select {
            width: 100%;
            margin-top: 8px;
            padding: 6px 8px;
            background: rgba(255,255,255,0.05);
            border: 1px dashed #3a4060;
            color: #a0a8c0;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
        }
        .model-select:focus {
            outline: none;
            border-color: #e25f4a;
        }
        .model-select option {
            background: #1a1f36;
            color: #fff;
        }

        /* TTS panel styles */
        .tts-panel { display: none; }
        .tts-panel.active { display: block; }
        .asr-panel { display: block; }

        .tts-compose {
            background: #fff;
            border: 1px dashed #d0d5dd;
            padding: 20px;
            margin-bottom: 24px;
        }
        .tts-compose-header {
            font-size: 13px;
            font-weight: 500;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 16px;
        }
        .tts-textarea {
            width: 100%;
            min-height: 100px;
            padding: 12px;
            border: 1px dashed #d0d5dd;
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
            color: #1a1f36;
            background: #fafafa;
        }
        .tts-textarea:focus {
            outline: none;
            border-color: #e25f4a;
        }
        .tts-controls {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 12px;
        }
        .tts-select {
            padding: 8px 12px;
            border: 1px dashed #d0d5dd;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            color: #1a1f36;
            background: #fff;
            cursor: pointer;
        }
        .tts-select:focus {
            outline: none;
            border-color: #e25f4a;
        }
        .synthesize-btn {
            background: #e25f4a;
            color: #fff;
            border: none;
            padding: 8px 20px;
            font-size: 12px;
            font-family: 'JetBrains Mono', monospace;
            cursor: pointer;
            transition: background 0.15s;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .synthesize-btn:hover { background: #d04a35; }
        .synthesize-btn:disabled {
            background: #9ca3af;
            cursor: not-allowed;
        }
        .synthesize-btn svg { width: 14px; height: 14px; }

        .tts-speed-input {
            width: 60px;
            padding: 8px;
            border: 1px dashed #d0d5dd;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            text-align: center;
            color: #1a1f36;
        }
        .tts-speed-input:focus {
            outline: none;
            border-color: #e25f4a;
        }

        /* TTS history item (reuses transcript styles) */
        .tts-item {
            background: #fff;
            border: 1px dashed #d0d5dd;
            padding: 16px 20px;
            margin-bottom: 12px;
            transition: border-color 0.15s;
        }
        .tts-item:hover { border-color: #e25f4a; }

        .tts-voice-badge {
            color: #6b7280;
            font-size: 10px;
            padding: 2px 6px;
            background: #e8f4fd;
            border-radius: 3px;
        }
        .tts-lang-badge {
            color: #6b7280;
            font-size: 10px;
            padding: 2px 6px;
            background: #f0fdf4;
            border-radius: 3px;
        }

        /* TTS dates list */
        .tts-dates-container { flex: 1; overflow-y: auto; padding: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <div class="logo">
                <svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <!-- cute microphone with face -->
                    <rect x="30" y="15" width="40" height="55" rx="20" fill="#e25f4a"/>
                    <rect x="38" y="75" width="24" height="8" fill="#e25f4a"/>
                    <rect x="35" y="83" width="30" height="6" rx="2" fill="#e25f4a"/>
                    <!-- eyes -->
                    <circle cx="42" cy="38" r="5" fill="#fff"/>
                    <circle cx="58" cy="38" r="5" fill="#fff"/>
                    <circle cx="43" cy="39" r="2" fill="#1a1f36"/>
                    <circle cx="59" cy="39" r="2" fill="#1a1f36"/>
                    <!-- smile -->
                    <path d="M42 52 Q50 58 58 52" stroke="#fff" stroke-width="3" fill="none" stroke-linecap="round"/>
                    <!-- sound waves -->
                    <path d="M18 35 Q12 45 18 55" stroke="#e25f4a" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.6"/>
                    <path d="M10 30 Q2 45 10 60" stroke="#e25f4a" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.3"/>
                    <path d="M82 35 Q88 45 82 55" stroke="#e25f4a" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.6"/>
                    <path d="M90 30 Q98 45 90 60" stroke="#e25f4a" stroke-width="3" fill="none" stroke-linecap="round" opacity="0.3"/>
                </svg>
                <span class="logo-text">qwen3-speech</span>
            </div>

            <!-- Tab navigation -->
            <div class="tab-nav">
                <button class="tab-btn active" id="tab-asr-btn" onclick="switchTab('asr')">ASR</button>
                <button class="tab-btn" id="tab-tts-btn" onclick="switchTab('tts')">TTS</button>
            </div>

            <div class="sidebar-content">
                <!-- ASR sidebar content -->
                <div class="asr-sidebar" id="asr-sidebar">
                    <div class="dates-container" id="dates"></div>
                    <div class="section-label">ASR Model</div>
                    <div class="model-status" id="model-status">
                        <div class="status-indicator">
                            <div class="status-dot" id="status-dot"></div>
                            <span class="status-text" id="status-text">loading...</span>
                        </div>
                        <div class="model-name" id="model-name">-</div>
                        <div class="model-controls">
                            <button class="ctrl-btn" id="btn-pause" onclick="togglePause()">pause</button>
                            <button class="ctrl-btn" id="btn-restart" onclick="restartModel()">restart</button>
                        </div>
                        <select class="model-select" id="model-select" onchange="switchModel(this.value)">
                        </select>
                    </div>
                    <div class="stats" id="stats">
                        <div class="stats-row"><span>total</span><span class="stats-value"><span id="stat-total">-</span> / <span id="stat-total-duration">-</span></span></div>
                        <div class="stats-row"><span>today</span><span class="stats-value"><span id="stat-today">-</span> / <span id="stat-today-duration">-</span></span></div>
                    </div>
                </div>

                <!-- TTS sidebar content -->
                <div class="tts-sidebar" id="tts-sidebar">
                    <div class="tts-dates-container" id="tts-dates"></div>
                    <div class="section-label">TTS Model</div>
                    <div class="model-status" id="tts-model-status">
                        <div class="status-indicator">
                            <div class="status-dot" id="tts-status-dot"></div>
                            <span class="status-text" id="tts-status-text">loading...</span>
                        </div>
                        <div class="model-name" id="tts-model-name">-</div>
                        <div class="model-controls">
                            <button class="ctrl-btn" id="tts-btn-pause" onclick="toggleTTSPause()">pause</button>
                            <button class="ctrl-btn" id="tts-btn-restart" onclick="restartTTSModel()">restart</button>
                        </div>
                        <select class="model-select" id="tts-model-select" onchange="switchTTSModel(this.value)">
                        </select>
                    </div>
                    <div class="stats" id="tts-stats">
                        <div class="stats-row"><span>generated</span><span class="stats-value" id="tts-stat-total">-</span></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="main">
            <!-- ASR main panel -->
            <div class="main-content asr-panel" id="asr-panel">
                <div class="header">
                    <h2>Transcripts</h2>
                    <div class="header-controls">
                        <div class="sort-controls">
                            <span class="sort-label">sort:</span>
                            <button class="sort-btn active" id="sort-time" onclick="setSort('time')">
                                time
                                <svg id="sort-time-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M7 14l5-5 5 5z"/></svg>
                            </button>
                            <button class="sort-btn" id="sort-length" onclick="setSort('length')">
                                length
                                <svg id="sort-length-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M7 14l5-5 5 5z"/></svg>
                            </button>
                            <button class="sort-btn" id="sort-latency" onclick="setSort('latency')">
                                latency
                                <svg id="sort-latency-icon" viewBox="0 0 24 24" fill="currentColor"><path d="M7 14l5-5 5 5z"/></svg>
                            </button>
                        </div>
                        <div class="speed-controls">
                            <span class="speed-label">speed:</span>
                            <button class="speed-btn" data-speed="1" onclick="setSpeed(1)">1x</button>
                            <button class="speed-btn active" data-speed="2" onclick="setSpeed(2)">2x</button>
                            <button class="speed-btn" data-speed="4" onclick="setSpeed(4)">4x</button>
                        </div>
                    </div>
                </div>
                <div id="transcripts"><div class="empty">select a date to view transcripts</div></div>
            </div>

            <!-- TTS main panel -->
            <div class="main-content tts-panel" id="tts-panel">
                <div class="tts-compose">
                    <div class="tts-compose-header">Synthesize Speech</div>
                    <textarea class="tts-textarea" id="tts-text" placeholder="Enter text to synthesize..."></textarea>
                    <div class="tts-controls">
                        <select class="tts-select" id="tts-voice-select">
                            <option value="Chelsie">Chelsie (EN/F)</option>
                            <option value="Ethan">Ethan (EN/M)</option>
                            <option value="Vivian">Vivian (ZH/F)</option>
                        </select>
                        <select class="tts-select" id="tts-lang-select">
                            <option value="en">English</option>
                            <option value="zh">Chinese</option>
                            <option value="ja">Japanese</option>
                            <option value="ko">Korean</option>
                            <option value="fr">French</option>
                            <option value="de">German</option>
                            <option value="es">Spanish</option>
                            <option value="ru">Russian</option>
                            <option value="it">Italian</option>
                            <option value="pt">Portuguese</option>
                        </select>
                        <label style="font-size:11px;color:#9ca3af;">speed:</label>
                        <input type="number" class="tts-speed-input" id="tts-speed" value="1.0" min="0.5" max="2.0" step="0.1">
                        <button class="synthesize-btn" id="synthesize-btn" onclick="synthesize()">
                            <svg viewBox="0 0 24 24" fill="currentColor"><path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z"/></svg>
                            <span id="synthesize-btn-text">Synthesize</span>
                        </button>
                    </div>
                </div>

                <div class="header" style="margin-top: 0;">
                    <h2>TTS History</h2>
                </div>
                <div id="tts-history"><div class="empty">no TTS history yet</div></div>
            </div>
        </div>
    </div>
    <script>
        // ========== Shared state ==========
        let currentTab = 'asr';
        let currentAudio = null;

        // ========== ASR state ==========
        let currentDate = null;
        let lastRecordCount = 0;
        let knownDates = [];
        let currentSort = 'time';
        let sortAscending = false;
        let currentSpeed = 2;

        // ========== TTS state ==========
        let ttsCurrentDate = null;
        let ttsHistory = [];
        let ttsPaused = false;

        // ========== Tab switching ==========
        function switchTab(tab) {
            currentTab = tab;
            document.getElementById('tab-asr-btn').classList.toggle('active', tab === 'asr');
            document.getElementById('tab-tts-btn').classList.toggle('active', tab === 'tts');

            document.getElementById('asr-sidebar').style.display = tab === 'asr' ? 'flex' : 'none';
            document.getElementById('tts-sidebar').style.display = tab === 'tts' ? 'flex' : 'none';

            document.getElementById('asr-panel').style.display = tab === 'asr' ? 'block' : 'none';
            document.getElementById('asr-panel').classList.toggle('active', tab === 'asr');
            document.getElementById('tts-panel').style.display = tab === 'tts' ? 'block' : 'none';
            document.getElementById('tts-panel').classList.toggle('active', tab === 'tts');

            if (tab === 'tts') {
                loadTTSHistory();
                loadTTSStatus();
            }
        }

        // ========== ASR functions ==========

        function setSpeed(speed) {
            currentSpeed = speed;
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.toggle('active', parseInt(btn.dataset.speed) === speed);
            });
            if (currentAudio) {
                currentAudio.playbackRate = speed;
            }
        }

        async function loadDates() {
            const res = await fetch('/api/dates');
            const dates = await res.json();
            const container = document.getElementById('dates');

            const hadNewDate = dates.length > knownDates.length;
            knownDates = dates;

            if (dates.length === 0) {
                container.innerHTML = '<div class="empty" style="border:none;background:transparent;color:#6b7280;">no history</div>';
                return;
            }
            container.innerHTML = dates.map(d =>
                `<div class="date-item" data-date="${d}">${d}</div>`
            ).join('');
            container.querySelectorAll('.date-item').forEach(el => {
                el.onclick = () => loadTranscripts(el.dataset.date);
            });

            const now = new Date();
            const today = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;

            try {
                const todayRes = await fetch(`/api/transcripts/${today}`);
                if (todayRes.ok) {
                    const todayRecords = await todayRes.json();
                    document.getElementById('stat-today').textContent = todayRecords.length + ' items';
                    const todayDuration = todayRecords.reduce((sum, r) => sum + (r.audio_duration_ms || 0), 0);
                    document.getElementById('stat-today-duration').textContent = formatDurationHHMM(todayDuration);
                } else {
                    document.getElementById('stat-today').textContent = '0 items';
                    document.getElementById('stat-today-duration').textContent = '00:00';
                }
            } catch (e) {
                document.getElementById('stat-today').textContent = '0 items';
                document.getElementById('stat-today-duration').textContent = '00:00';
            }

            let totalDuration = 0;
            for (const date of dates) {
                try {
                    const res = await fetch(`/api/transcripts/${date}`);
                    if (res.ok) {
                        const records = await res.json();
                        totalDuration += records.reduce((sum, r) => sum + (r.audio_duration_ms || 0), 0);
                    }
                } catch (e) {}
            }
            document.getElementById('stat-total-duration').textContent = formatDurationHHMM(totalDuration);

            if (dates.includes(today)) {
                loadTranscripts(today);
            } else if (!currentDate || hadNewDate) {
                loadTranscripts(dates[0]);
            }

            document.getElementById('stat-total').textContent = dates.length + ' days';
        }

        async function loadTranscripts(date) {
            currentDate = date;
            document.querySelectorAll('.date-item').forEach(el => {
                el.classList.toggle('active', el.dataset.date === date);
            });
            const res = await fetch(`/api/transcripts/${date}`);
            const records = await res.json();
            const container = document.getElementById('transcripts');

            const hadNewRecords = records.length > lastRecordCount;
            lastRecordCount = records.length;

            const now = new Date(); const today = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
            if (date === today) {
                document.getElementById('stat-today').textContent = records.length + ' items';
            }

            if (records.length === 0) {
                container.innerHTML = '<div class="empty">no transcripts for this date</div>';
                return;
            }

            const sortedRecords = sortRecords(records);

            container.innerHTML = sortedRecords.map((r, i) => {
                const time = new Date(r.timestamp).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
                const latency = (r.latency_ms || r.duration_ms || 0) / 1000;
                const audioLen = (r.audio_duration_ms || 0) / 1000;
                const model = r.model || 'unknown';
                return `
                    <div class="transcript" data-audio-url="/audio/${date}/${r.audio_file}">
                        <div class="transcript-header">
                            <span class="transcript-time">${time}</span>
                            <span class="transcript-latency">
                                <svg viewBox="0 0 24 24" fill="currentColor">
                                    <circle cx="12" cy="13" r="8" fill="none" stroke="currentColor" stroke-width="2"/>
                                    <path d="M12 9v4l2.5 2.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                    <circle cx="10" cy="12" r="0.8"/><circle cx="14" cy="12" r="0.8"/>
                                    <path d="M10 15.5q2 1.5 4 0" fill="none" stroke="currentColor" stroke-width="0.8" stroke-linecap="round"/>
                                    <rect x="10" y="3" width="4" height="2" rx="0.5"/>
                                </svg>
                                ${latency.toFixed(1)}s
                            </span>
                            <span class="transcript-model">${model}</span>
                        </div>
                        <div class="transcript-text">${escapeHtml(r.text)}</div>
                        <div class="waveform-container" onclick="seekAudio(event, this)">
                            <canvas></canvas>
                            <div class="waveform-progress"></div>
                            <div class="waveform-time">0:00 / ${audioLen.toFixed(1)}s</div>
                            <div class="waveform-loading">loading waveform...</div>
                        </div>
                        <div class="transcript-footer">
                            <span class="transcript-file">${r.audio_file}</span>
                            <button class="play-btn" onclick="toggleAudio(this)">
                                <svg class="icon-play" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                                <svg class="icon-pause" viewBox="0 0 24 24" fill="currentColor" style="display:none"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                                <span class="btn-text">play</span>
                            </button>
                        </div>
                    </div>
                `;
            }).join('');

            document.querySelectorAll('.transcript').forEach(el => loadWaveform(el));

            if (hadNewRecords) {
                container.scrollTop = container.scrollHeight;
            }
        }

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const waveformCache = new Map();

        async function loadWaveform(transcriptEl) {
            const url = transcriptEl.dataset.audioUrl;
            const container = transcriptEl.querySelector('.waveform-container');
            const canvas = container.querySelector('canvas');
            const loading = container.querySelector('.waveform-loading');
            const timeDisplay = container.querySelector('.waveform-time');

            try {
                let waveformData = waveformCache.get(url);
                let duration = 0;
                if (!waveformData) {
                    const response = await fetch(url);
                    const arrayBuffer = await response.arrayBuffer();
                    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
                    waveformData = extractWaveform(audioBuffer, 100);
                    waveformData.duration = audioBuffer.duration;
                    waveformCache.set(url, waveformData);
                }
                duration = waveformData.duration || 0;
                if (duration > 0) {
                    timeDisplay.textContent = '0:00 / ' + duration.toFixed(1) + 's';
                }
                drawWaveform(canvas, waveformData, 0);
                loading.style.display = 'none';
            } catch (e) {
                loading.textContent = 'waveform unavailable';
            }
        }

        function extractWaveform(audioBuffer, samples) {
            const channelData = audioBuffer.getChannelData(0);
            const blockSize = Math.floor(channelData.length / samples);
            const waveform = [];
            for (let i = 0; i < samples; i++) {
                let sum = 0;
                for (let j = 0; j < blockSize; j++) {
                    sum += Math.abs(channelData[i * blockSize + j]);
                }
                waveform.push(sum / blockSize);
            }
            const max = Math.max(...waveform);
            return waveform.map(v => v / max);
        }

        function drawWaveform(canvas, data, progress) {
            const ctx = canvas.getContext('2d');
            const dpr = window.devicePixelRatio || 1;
            const rect = canvas.getBoundingClientRect();
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            ctx.scale(dpr, dpr);

            const width = rect.width;
            const height = rect.height;
            const barWidth = width / data.length;
            const progressIndex = Math.floor(progress * data.length);

            ctx.clearRect(0, 0, width, height);

            data.forEach((v, i) => {
                const barHeight = Math.max(2, v * (height - 8));
                const x = i * barWidth;
                const y = (height - barHeight) / 2;
                ctx.fillStyle = i < progressIndex ? '#e25f4a' : '#d0d5dd';
                ctx.fillRect(x + 1, y, barWidth - 2, barHeight);
            });
        }

        function toggleAudio(btn) {
            const transcript = btn.closest('.transcript') || btn.closest('.tts-item');
            const url = transcript.dataset.audioUrl;

            if (currentAudio && currentAudio.dataset.url === url) {
                if (currentAudio.paused) {
                    currentAudio.play();
                } else {
                    currentAudio.pause();
                }
                return;
            }

            if (currentAudio) {
                currentAudio.pause();
                resetTranscriptUI(currentAudio.transcriptEl);
            }

            const audio = new Audio(url);
            audio.dataset.url = url;
            audio.transcriptEl = transcript;
            currentAudio = audio;

            const container = transcript.querySelector('.waveform-container');
            const timeDisplay = container.querySelector('.waveform-time');
            const progressBar = container.querySelector('.waveform-progress');
            const canvas = container.querySelector('canvas');
            const playIcon = btn.querySelector('.icon-play');
            const pauseIcon = btn.querySelector('.icon-pause');
            const btnText = btn.querySelector('.btn-text');

            audio.ontimeupdate = () => {
                const progress = audio.currentTime / audio.duration;
                progressBar.style.width = (progress * 100) + '%';
                const current = formatTime(audio.currentTime);
                const total = formatTime(audio.duration);
                timeDisplay.textContent = current + ' / ' + total;
                const waveformData = waveformCache.get(url);
                if (waveformData) drawWaveform(canvas, waveformData, progress);
            };

            audio.onplay = () => {
                playIcon.style.display = 'none';
                pauseIcon.style.display = 'block';
                btnText.textContent = 'pause';
            };

            audio.onpause = () => {
                playIcon.style.display = 'block';
                pauseIcon.style.display = 'none';
                btnText.textContent = 'play';
            };

            audio.onended = () => {
                resetTranscriptUI(transcript);
                currentAudio = null;
            };

            audio.playbackRate = currentSpeed;
            audio.play();
        }

        function seekAudio(event, container) {
            const transcript = container.closest('.transcript') || container.closest('.tts-item');
            const url = transcript.dataset.audioUrl;
            const rect = container.getBoundingClientRect();
            const progress = (event.clientX - rect.left) / rect.width;

            if (currentAudio && currentAudio.dataset.url === url) {
                currentAudio.currentTime = progress * currentAudio.duration;
            } else {
                const btn = transcript.querySelector('.play-btn');
                toggleAudio(btn);
                currentAudio.oncanplay = () => {
                    currentAudio.currentTime = progress * currentAudio.duration;
                    currentAudio.oncanplay = null;
                };
            }
        }

        function resetTranscriptUI(transcript) {
            const container = transcript.querySelector('.waveform-container');
            const progressBar = container.querySelector('.waveform-progress');
            const canvas = container.querySelector('canvas');
            const btn = transcript.querySelector('.play-btn');
            const playIcon = btn.querySelector('.icon-play');
            const pauseIcon = btn.querySelector('.icon-pause');
            const btnText = btn.querySelector('.btn-text');

            progressBar.style.width = '0%';
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
            btnText.textContent = 'play';
            const url = transcript.dataset.audioUrl;
            const waveformData = waveformCache.get(url);
            if (waveformData) drawWaveform(canvas, waveformData, 0);
        }

        function formatTime(seconds) {
            if (isNaN(seconds)) return '0:00';
            const m = Math.floor(seconds / 60);
            const s = Math.floor(seconds % 60);
            return m + ':' + s.toString().padStart(2, '0');
        }

        function formatDurationHHMM(ms) {
            const totalSeconds = Math.floor(ms / 1000);
            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            return hours.toString().padStart(2, '0') + ':' + minutes.toString().padStart(2, '0');
        }

        function sortRecords(records) {
            const sorted = [...records];
            if (currentSort === 'time') {
                sorted.sort((a, b) => {
                    const diff = new Date(a.timestamp) - new Date(b.timestamp);
                    return sortAscending ? diff : -diff;
                });
            } else if (currentSort === 'length') {
                sorted.sort((a, b) => {
                    const diff = (a.audio_duration_ms || 0) - (b.audio_duration_ms || 0);
                    return sortAscending ? diff : -diff;
                });
            } else if (currentSort === 'latency') {
                sorted.sort((a, b) => {
                    const diff = (a.latency_ms || a.duration_ms || 0) - (b.latency_ms || b.duration_ms || 0);
                    return sortAscending ? diff : -diff;
                });
            }
            return sorted;
        }

        function setSort(type) {
            if (currentSort === type) {
                sortAscending = !sortAscending;
            } else {
                currentSort = type;
                sortAscending = false;
            }

            document.getElementById('sort-time').classList.toggle('active', type === 'time');
            document.getElementById('sort-length').classList.toggle('active', type === 'length');
            document.getElementById('sort-latency').classList.toggle('active', type === 'latency');

            const upArrow = '<path d="M7 14l5-5 5 5z"/>';
            const downArrow = '<path d="M7 10l5 5 5-5z"/>';

            document.getElementById('sort-time-icon').innerHTML =
                (currentSort === 'time' && sortAscending) ? upArrow : downArrow;
            document.getElementById('sort-length-icon').innerHTML =
                (currentSort === 'length' && sortAscending) ? upArrow : downArrow;
            document.getElementById('sort-latency-icon').innerHTML =
                (currentSort === 'latency' && sortAscending) ? upArrow : downArrow;

            if (currentDate) {
                loadTranscripts(currentDate);
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function autoRefresh() {
            if (currentAudio && !currentAudio.paused) return;
            if (currentTab !== 'asr') return;

            try {
                const res = await fetch('/api/dates');
                if (!res.ok) return;
                const dates = await res.json();

                if (JSON.stringify(dates) !== JSON.stringify(knownDates)) {
                    knownDates = dates;
                    const container = document.getElementById('dates');
                    container.innerHTML = dates.map(d =>
                        `<div class="date-item${d === currentDate ? ' active' : ''}" data-date="${d}">${d}</div>`
                    ).join('');
                    container.querySelectorAll('.date-item').forEach(el => {
                        el.onclick = () => loadTranscripts(el.dataset.date);
                    });
                    document.getElementById('stat-total').textContent = dates.length + ' days';
                }

                if (currentDate) {
                    const res2 = await fetch(`/api/transcripts/${currentDate}`);
                    if (res2.ok) {
                        const records = await res2.json();
                        if (records.length > lastRecordCount) {
                            lastRecordCount = records.length;
                            await loadTranscripts(currentDate);
                        }
                    }
                }

                const now = new Date(); const today = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
                const todayRes = await fetch(`/api/transcripts/${today}`);
                if (todayRes.ok) {
                    const todayRecords = await todayRes.json();
                    document.getElementById('stat-today').textContent = todayRecords.length + ' items';
                    const todayDuration = todayRecords.reduce((sum, r) => sum + (r.audio_duration_ms || 0), 0);
                    document.getElementById('stat-today-duration').textContent = formatDurationHHMM(todayDuration);
                }
            } catch (e) {}
        }

        // ========== ASR Model status ==========

        let serverPaused = false;

        async function loadStatus() {
            try {
                const res = await fetch('/api/status');
                const status = await res.json();

                const dot = document.getElementById('status-dot');
                const text = document.getElementById('status-text');
                const modelName = document.getElementById('model-name');
                const pauseBtn = document.getElementById('btn-pause');
                const select = document.getElementById('model-select');

                serverPaused = status.paused;

                if (!status.loaded) {
                    dot.className = 'status-dot loading';
                    text.textContent = 'loading model...';
                } else if (status.paused) {
                    dot.className = 'status-dot paused';
                    text.textContent = 'paused';
                    pauseBtn.textContent = 'resume';
                    pauseBtn.classList.add('active');
                } else {
                    dot.className = 'status-dot';
                    text.textContent = 'active';
                    pauseBtn.textContent = 'pause';
                    pauseBtn.classList.remove('active');
                }

                modelName.textContent = status.model_short;

                if (select.options.length === 0) {
                    status.available_models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.textContent = m.split('/').pop();
                        if (m === status.model) opt.selected = true;
                        select.appendChild(opt);
                    });
                }
            } catch (e) {
                document.getElementById('status-dot').className = 'status-dot paused';
                document.getElementById('status-text').textContent = 'offline';
            }
        }

        async function togglePause() {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');

            dot.className = 'status-dot loading';
            text.textContent = 'updating...';

            try {
                const endpoint = serverPaused ? '/api/server/resume' : '/api/server/pause';
                await fetch(endpoint, { method: 'POST' });
                await loadStatus();
            } catch (e) {
                text.textContent = 'error';
            }
        }

        async function restartModel() {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');
            const btn = document.getElementById('btn-restart');

            dot.className = 'status-dot loading';
            text.textContent = 'restarting...';
            btn.classList.add('active');

            try {
                await fetch('/api/server/restart', { method: 'POST' });
                await loadStatus();
            } catch (e) {
                text.textContent = 'error';
            }
            btn.classList.remove('active');
        }

        async function switchModel(modelName) {
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');

            dot.className = 'status-dot loading';
            text.textContent = 'switching model...';

            try {
                await fetch('/api/model/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelName })
                });
                await loadStatus();
            } catch (e) {
                text.textContent = 'error';
            }
        }

        // ========== TTS functions ==========

        async function loadTTSStatus() {
            try {
                const res = await fetch('/api/tts/status');
                const status = await res.json();

                const dot = document.getElementById('tts-status-dot');
                const text = document.getElementById('tts-status-text');
                const modelName = document.getElementById('tts-model-name');
                const pauseBtn = document.getElementById('tts-btn-pause');
                const select = document.getElementById('tts-model-select');

                ttsPaused = status.paused;

                if (status.loading) {
                    dot.className = 'status-dot loading';
                    text.textContent = 'loading model...';
                } else if (!status.loaded) {
                    dot.className = 'status-dot paused';
                    text.textContent = 'not loaded';
                } else if (status.paused) {
                    dot.className = 'status-dot paused';
                    text.textContent = 'paused';
                    pauseBtn.textContent = 'resume';
                    pauseBtn.classList.add('active');
                } else {
                    dot.className = 'status-dot';
                    text.textContent = 'active';
                    pauseBtn.textContent = 'pause';
                    pauseBtn.classList.remove('active');
                }

                modelName.textContent = status.model_short;

                if (select.options.length === 0) {
                    status.available_models.forEach(m => {
                        const opt = document.createElement('option');
                        opt.value = m;
                        opt.textContent = m.split('/').pop();
                        if (m === status.model) opt.selected = true;
                        select.appendChild(opt);
                    });
                }
            } catch (e) {
                document.getElementById('tts-status-dot').className = 'status-dot paused';
                document.getElementById('tts-status-text').textContent = 'offline';
            }
        }

        async function toggleTTSPause() {
            const dot = document.getElementById('tts-status-dot');
            const text = document.getElementById('tts-status-text');

            dot.className = 'status-dot loading';
            text.textContent = 'updating...';

            try {
                const endpoint = ttsPaused ? '/api/tts/server/resume' : '/api/tts/server/pause';
                await fetch(endpoint, { method: 'POST' });
                await loadTTSStatus();
            } catch (e) {
                text.textContent = 'error';
            }
        }

        async function restartTTSModel() {
            const dot = document.getElementById('tts-status-dot');
            const text = document.getElementById('tts-status-text');
            const btn = document.getElementById('tts-btn-restart');

            dot.className = 'status-dot loading';
            text.textContent = 'restarting...';
            btn.classList.add('active');

            try {
                await fetch('/api/tts/server/restart', { method: 'POST' });
                await loadTTSStatus();
            } catch (e) {
                text.textContent = 'error';
            }
            btn.classList.remove('active');
        }

        async function switchTTSModel(modelName) {
            const dot = document.getElementById('tts-status-dot');
            const text = document.getElementById('tts-status-text');

            dot.className = 'status-dot loading';
            text.textContent = 'switching model...';

            try {
                await fetch('/api/tts/model/switch', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model: modelName })
                });
                await loadTTSStatus();
            } catch (e) {
                text.textContent = 'error';
            }
        }

        async function synthesize() {
            const textEl = document.getElementById('tts-text');
            const text = textEl.value.trim();
            if (!text) return;

            const voice = document.getElementById('tts-voice-select').value;
            const language = document.getElementById('tts-lang-select').value;
            const speed = parseFloat(document.getElementById('tts-speed').value) || 1.0;
            const btn = document.getElementById('synthesize-btn');
            const btnText = document.getElementById('synthesize-btn-text');

            btn.disabled = true;
            btnText.textContent = 'Generating...';

            try {
                const res = await fetch('/synthesize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text, voice, language, speed })
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert('Error: ' + (err.detail || 'Unknown error'));
                    return;
                }

                const result = await res.json();

                // Reload TTS history
                await loadTTSHistory();

                // Auto-play the new audio
                setTimeout(() => {
                    const firstItem = document.querySelector('.tts-item');
                    if (firstItem) {
                        const playBtn = firstItem.querySelector('.play-btn');
                        if (playBtn) toggleAudio(playBtn);
                    }
                }, 200);

            } catch (e) {
                alert('Network error: ' + e.message);
            } finally {
                btn.disabled = false;
                btnText.textContent = 'Synthesize';
            }
        }

        async function loadTTSHistory() {
            try {
                const res = await fetch('/api/tts/history');
                const records = await res.json();
                ttsHistory = records;

                const container = document.getElementById('tts-history');

                if (records.length === 0) {
                    container.innerHTML = '<div class="empty">no TTS history yet</div>';
                    document.getElementById('tts-stat-total').textContent = '0 items';
                    return;
                }

                document.getElementById('tts-stat-total').textContent = records.length + ' items';

                // Build dates sidebar
                const dates = [...new Set(records.map(r => r.date || r.timestamp.split('T')[0]))];
                const datesContainer = document.getElementById('tts-dates');
                datesContainer.innerHTML = dates.map(d =>
                    `<div class="date-item" data-date="${d}">${d}</div>`
                ).join('');

                container.innerHTML = records.map((r, i) => {
                    const time = new Date(r.timestamp).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
                    const date = r.date || r.timestamp.split('T')[0];
                    const latency = (r.latency_ms || 0) / 1000;
                    const audioLen = (r.audio_duration_ms || 0) / 1000;
                    const model = r.model || 'unknown';
                    const audioUrl = `/tts_audio/${r.audio_file}`;
                    return `
                        <div class="tts-item" data-audio-url="${audioUrl}">
                            <div class="transcript-header">
                                <span class="transcript-time">${date} ${time}</span>
                                <span class="transcript-latency">
                                    <svg viewBox="0 0 24 24" fill="currentColor">
                                        <circle cx="12" cy="13" r="8" fill="none" stroke="currentColor" stroke-width="2"/>
                                        <path d="M12 9v4l2.5 2.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                                    </svg>
                                    ${latency.toFixed(1)}s
                                </span>
                                <span class="tts-voice-badge">${escapeHtml(r.voice || '-')}</span>
                                <span class="tts-lang-badge">${escapeHtml(r.language || '-')}</span>
                                <span class="transcript-model">${model}</span>
                            </div>
                            <div class="transcript-text">${escapeHtml(r.text)}</div>
                            <div class="waveform-container" onclick="seekAudio(event, this)">
                                <canvas></canvas>
                                <div class="waveform-progress"></div>
                                <div class="waveform-time">0:00 / ${audioLen.toFixed(1)}s</div>
                                <div class="waveform-loading">loading waveform...</div>
                            </div>
                            <div class="transcript-footer">
                                <span class="transcript-file">${r.audio_file}</span>
                                <button class="play-btn" onclick="toggleAudio(this)">
                                    <svg class="icon-play" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
                                    <svg class="icon-pause" viewBox="0 0 24 24" fill="currentColor" style="display:none"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                                    <span class="btn-text">play</span>
                                </button>
                            </div>
                        </div>
                    `;
                }).join('');

                // Load waveforms for TTS items
                document.querySelectorAll('.tts-item').forEach(el => loadWaveform(el));

            } catch (e) {
                console.error('Failed to load TTS history:', e);
            }
        }

        // ========== Init ==========
        loadDates();
        loadStatus();
        loadTTSStatus();
        setInterval(autoRefresh, 3000);
        setInterval(loadStatus, 5000);
        setInterval(() => { if (currentTab === 'tts') loadTTSStatus(); }, 5000);
    </script>
</body>
</html>
"""

@app.get("/history", response_class=HTMLResponse)
async def history_page():
    """Serve the history viewer HTML page."""
    from fastapi.responses import Response
    return Response(
        content=HISTORY_HTML,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

if __name__ == "__main__":
    print(f"Starting Qwen3-Speech server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
