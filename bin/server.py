#!/usr/bin/env python3
"""
Persistent Qwen3-ASR transcription server.
Keeps model loaded in memory for fast inference.
Saves transcription history to ~/Documents/qwen3-asr-history/history/
"""
import os
import sys
import tempfile
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Lazy imports for mlx_audio (heavy)
model = None
load_fn = None
generate_fn = None

# Available models (all use mlx-audio generate_transcription API)
AVAILABLE_MODELS = [
    "mlx-community/Qwen3-ASR-0.6B-bf16",      # 0.6B - fastest, smallest
    "mlx-community/Qwen3-ASR-1.7B-8bit",      # 1.7B - better quality
    "mlx-community/whisper-large-v3-turbo-asr-fp16",  # whisper turbo
    "mlx-community/whisper-large-v3-asr-8bit",        # whisper v3 full
]

current_model_name = "mlx-community/Qwen3-ASR-0.6B-bf16"
server_paused = False

HOST = "127.0.0.1"
PORT = 18321
PROJECT_DIR = Path(__file__).parent.parent.resolve()
HISTORY_DIR = PROJECT_DIR / "history"

app = FastAPI(title="Qwen3-ASR Server")

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

    # Store reference to original audio path (no copying)
    audio_src = Path(audio_path)

    # Append to transcripts.jsonl
    record = {
        "timestamp": now.isoformat(),
        "audio_path": str(audio_src.resolve()),  # full path to original file
        "audio_file": audio_src.name,  # just filename for display
        "text": text,
        "latency_ms": round(latency_ms, 2),  # processing time
        "audio_duration_ms": round(audio_duration_ms, 2),  # actual audio length
        "model": model_name.split("/")[-1]  # short model name
    }
    jsonl_path = day_dir / "transcripts.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

class TranscribeRequest(BaseModel):
    path: str
    language: str = "zh"  # None = auto-detect

class TranscribeResponse(BaseModel):
    text: str
    latency_ms: float

def load_model(model_name: str = None):
    """Load model once at startup."""
    global model, load_fn, generate_fn, current_model_name

    if model_name:
        current_model_name = model_name

    print(f"Loading model {current_model_name}...")
    start = time.time()

    from mlx_audio.stt import load
    from mlx_audio.stt.generate import generate_transcription

    load_fn = load
    generate_fn = generate_transcription
    model = load(current_model_name)

    elapsed = time.time() - start
    print(f"Model loaded in {elapsed:.2f}s")
    return model

def convert_to_wav(audio_path: str) -> str:
    """Convert audio to 16kHz mono WAV for best results."""
    wav_path = tempfile.mktemp(suffix=".wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-ar", "16000", "-ac", "1", wav_path
    ], capture_output=True)
    return wav_path

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/health")
async def health():
    return {"status": "ok", "model": current_model_name, "loaded": model is not None}

@app.get("/api/status")
async def get_status():
    """Get server status including model info."""
    return {
        "model": current_model_name,
        "model_short": current_model_name.split("/")[-1],
        "loaded": model is not None,
        "paused": server_paused,
        "available_models": AVAILABLE_MODELS
    }

@app.post("/api/server/pause")
async def pause_server():
    """Pause the server (stop accepting transcriptions)."""
    global server_paused
    server_paused = True
    return {"status": "paused"}

@app.post("/api/server/resume")
async def resume_server():
    """Resume the server."""
    global server_paused
    server_paused = False
    return {"status": "active"}

@app.post("/api/server/restart")
async def restart_server():
    """Reload the current model."""
    global model
    model = None
    load_model()
    return {"status": "restarted", "model": current_model_name}

class SwitchModelRequest(BaseModel):
    model: str

@app.post("/api/model/switch")
async def switch_model(req: SwitchModelRequest):
    """Switch to a different model."""
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

    # Convert to WAV
    wav_path = convert_to_wav(req.path)

    try:
        # Use a temp file for output
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            output_path = f.name.replace(".txt", "")

        # Generate transcription using the loaded model
        segments = generate_fn(
            model=model,
            audio=wav_path,
            output_path=output_path,
            format="txt",
            verbose=False,
            language=req.language or "zh"
        )

        # Read the output
        txt_file = output_path + ".txt"
        if os.path.exists(txt_file):
            with open(txt_file, "r") as f:
                text = f.read().strip()
            os.unlink(txt_file)
        else:
            # Try to extract from segments directly
            if segments:
                text = " ".join(s.get("text", "") for s in segments if isinstance(s, dict))
            else:
                text = str(segments) if segments else ""

    finally:
        # Cleanup
        if os.path.exists(wav_path):
            os.unlink(wav_path)

    latency_ms = (time.time() - start) * 1000
    audio_duration_ms = get_audio_duration(req.path)

    # Save to history
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
    # Try to find the original path from transcripts
    jsonl_path = HISTORY_DIR / date / "transcripts.jsonl"
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("audio_file") == filename:
                        # New format: use original path
                        if "audio_path" in record:
                            original_path = Path(record["audio_path"])
                            if original_path.exists():
                                return FileResponse(original_path, media_type="audio/ogg")
                        break
    # Fallback: serve from local copy (old format)
    local_path = HISTORY_DIR / date / filename
    if local_path.exists():
        return FileResponse(local_path, media_type="audio/ogg")
    raise HTTPException(status_code=404, detail="Audio file not found")

HISTORY_HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>qwen3-asr</title>
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
        .main { flex: 1; padding: 30px 40px; overflow-y: auto; }
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
            background: #1a1f36;
            border-color: #1a1f36;
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
                <span class="logo-text">qwen3-asr</span>
            </div>
            <div class="dates-container" id="dates"></div>
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
                <div class="stats-row"><span>total</span><span class="stats-value" id="stat-total">-</span></div>
                <div class="stats-row"><span>today</span><span class="stats-value" id="stat-today">-</span></div>
            </div>
        </div>
        <div class="main">
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
    </div>
    <script>
        let currentDate = null;
        let currentAudio = null;
        let lastRecordCount = 0;
        let knownDates = [];
        let currentSort = 'time';
        let sortAscending = false; // false = descending (newest/longest first)
        let currentSpeed = 2; // default 2x

        function setSpeed(speed) {
            currentSpeed = speed;
            document.querySelectorAll('.speed-btn').forEach(btn => {
                btn.classList.toggle('active', parseInt(btn.dataset.speed) === speed);
            });
            // Apply to currently playing audio if any
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

            // Use local date, not UTC
            const now = new Date();
            const today = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;

            // Always fetch today's count independently
            try {
                const todayRes = await fetch(`/api/transcripts/${today}`);
                if (todayRes.ok) {
                    const todayRecords = await todayRes.json();
                    document.getElementById('stat-today').textContent = todayRecords.length + ' items';
                } else {
                    document.getElementById('stat-today').textContent = '0 items';
                }
            } catch (e) {
                document.getElementById('stat-today').textContent = '0 items';
            }

            // Load transcripts for display
            if (dates.includes(today)) {
                loadTranscripts(today);
            } else if (!currentDate || hadNewDate) {
                loadTranscripts(dates[0]);
            }

            // Update stats
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

            // Update today stat
            const now = new Date(); const today = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
            if (date === today) {
                document.getElementById('stat-today').textContent = records.length + ' items';
            }

            if (records.length === 0) {
                container.innerHTML = '<div class="empty">no transcripts for this date</div>';
                return;
            }

            // Sort records
            const sortedRecords = sortRecords(records);

            container.innerHTML = sortedRecords.map((r, i) => {
                const time = new Date(r.timestamp).toLocaleTimeString('zh-CN', {hour: '2-digit', minute: '2-digit', second: '2-digit'});
                const latency = (r.latency_ms || r.duration_ms || 0) / 1000;  // fallback for old records
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

            // Load waveforms for all transcripts
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
                    waveformData.duration = audioBuffer.duration;  // store duration
                    waveformCache.set(url, waveformData);
                }
                duration = waveformData.duration || 0;
                // Update time display with actual audio duration
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
            const transcript = btn.closest('.transcript');
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
            const transcript = container.closest('.transcript');
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
                // Toggle direction
                sortAscending = !sortAscending;
            } else {
                currentSort = type;
                sortAscending = false; // Default to descending
            }

            // Update button states
            document.getElementById('sort-time').classList.toggle('active', type === 'time');
            document.getElementById('sort-length').classList.toggle('active', type === 'length');
            document.getElementById('sort-latency').classList.toggle('active', type === 'latency');

            // Update icons (up arrow = ascending, down arrow = descending)
            const upArrow = '<path d="M7 14l5-5 5 5z"/>';
            const downArrow = '<path d="M7 10l5 5 5-5z"/>';

            document.getElementById('sort-time-icon').innerHTML =
                (currentSort === 'time' && sortAscending) ? upArrow : downArrow;
            document.getElementById('sort-length-icon').innerHTML =
                (currentSort === 'length' && sortAscending) ? upArrow : downArrow;
            document.getElementById('sort-latency-icon').innerHTML =
                (currentSort === 'latency' && sortAscending) ? upArrow : downArrow;

            // Reload with new sort
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
            // Skip refresh while audio is playing
            if (currentAudio && !currentAudio.paused) return;

            try {
                const res = await fetch('/api/dates');
                if (!res.ok) return;
                const dates = await res.json();

                // Only update dates sidebar if changed
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

                // Check for new transcripts
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

                // Always update today stat
                const now = new Date(); const today = `${now.getFullYear()}-${String(now.getMonth()+1).padStart(2,'0')}-${String(now.getDate()).padStart(2,'0')}`;
                const todayRes = await fetch(`/api/transcripts/${today}`);
                if (todayRes.ok) {
                    const todayRecords = await todayRes.json();
                    document.getElementById('stat-today').textContent = todayRecords.length + ' items';
                }
            } catch (e) {
                // Network error - server may be restarting, ignore
            }
        }

        // Model status and control functions
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

                // Populate model select
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
            const btn = document.getElementById('btn-pause');
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

        loadDates();
        loadStatus();
        setInterval(autoRefresh, 3000);
        setInterval(loadStatus, 5000);
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
    print(f"Starting Qwen3-ASR server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
