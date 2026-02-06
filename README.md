# <img src="logo.svg" width="30" height="30" align="top"> MLS

**MLX Local Serving** provides unified local serving for ASR, TTS, and Translation on Apple Silicon.

![screenshot](screenshot.png)

## Purpose

This project empowers **OpenClaw** with high-quality, local, and privacy-first AI capabilities. It replaces cloud APIs and allows the AI agent to:

1. **Hear** using Qwen3 ASR
2. **Speak** using Qwen3 TTS with custom voice instructions
3. **Translate** using TranslateGemma 12B without leaking data

## OpenClaw Integration

Add the following to your `openclaw.json` or `TOOLS.md` to enable MLS support.

### 1. Hearing
Configure the `media` tool to use the CLI wrapper. This wrapper tries the MLS server first and falls back to a cold start if needed.

```json
{
  "tools": {
    "media": {
      "audio": {
        "enabled": true,
        "models": [
          {
            "type": "cli",
            "command": "~/Documents/mls/bin/qwen3-asr",
            "args": ["{{MediaPath}}"],
            "timeoutSeconds": 60
          }
        ]
      }
    }
  }
}
```

### 2. Speaking and Translating
Since OpenClaw uses `curl` for these tasks, you can register them in your `TOOLS.md` or skills directory.

**TTS Skill Pattern**
```bash
curl -X POST http://127.0.0.1:18321/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello",
    "instruct": "A young female speaker with a cheerful tone"
  }' > output.json
```

**Translate Skill Pattern**
```bash
curl -X POST http://127.0.0.1:18321/translate \
  -H "Content-Type: application/json" \
  -d '{"q": "Text", "source": "en", "target": "zh"}'
```

## Features

* **ASR** runs Qwen3 ASR models for fast and accurate speech-to-text.
* **TTS** uses Qwen3 TTS VoiceDesign 1.7B, which supports natural speech generation from instructions.
* **Translate** is powered by TranslateGemma 12B and handles high-quality document translation across 55 languages.
* **Dashboard** is a unified web UI running on port 18321 with an accordion sidebar and mini-calendar.

## Quick Start

**Install**
```bash
git clone https://github.com/hanxiao/mls ~/Documents/mls
cd ~/Documents/mls
uv sync
```

**Run Server**
```bash
uv run bin/server.py
# Server starts on http://127.0.0.1:18321
```

## API

### ASR
```bash
curl -X POST http://127.0.0.1:18321/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/audio.ogg"}'
```

### TTS
```bash
curl -X POST http://127.0.0.1:18321/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "instruct": "Professional narrator"}' > output.json
```

### Translate
```bash
curl -X POST http://127.0.0.1:18321/translate \
  -H "Content-Type: application/json" \
  -d '{"q": "Hello world", "source": "en", "target": "zh"}'
```

## Models

All models run locally on Metal GPU via MLX.

* **ASR** uses `mlx-community/Qwen3-ASR-0.6B-bf16` by default.
* **TTS** uses `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` by default.
* **Translate** uses `mlx-community/translategemma-12b-it-8bit`.

## Requirements

* macOS 14+ with Apple Silicon
* Python 3.12+
* `uv` package manager
