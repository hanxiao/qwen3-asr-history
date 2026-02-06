# <img src="logo.svg" width="30" height="30" align="top"> MLS

**MLX Local Serving** provides unified local serving for ASR, TTS, Translation, and Image Generation on Apple Silicon. All four models run on Metal GPU via MLX and stay resident in memory.

![screenshot](screenshot.png)

## Quick Start

```bash
git clone https://github.com/hanxiao/mls ~/Documents/mls
cd ~/Documents/mls
uv sync
uv run bin/server.py
# http://127.0.0.1:18321
```

Dashboard at `http://127.0.0.1:18321/history`

## Models

| Service | Model | Size |
|---------|-------|------|
| ASR | Qwen3-ASR-0.6B-bf16 | 1.2 GB |
| TTS | Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 | 3.6 GB |
| Translate | TranslateGemma-12B-8bit | 12 GB |
| Image | Z-Image-Turbo-8bit (mflux) | 10 GB |

## API

### Health Check
```bash
curl http://127.0.0.1:18321/health
```

### ASR (Speech-to-Text)
```bash
curl -X POST http://127.0.0.1:18321/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/audio.ogg", "language": "zh"}'
# -> {"text": "...", "latency_ms": 1234}
```

### TTS (Text-to-Speech)
```bash
curl -X POST http://127.0.0.1:18321/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "language": "english",
    "format": "ogg",
    "instruct": "A young male speaker with a calm tone"
  }'
# -> {"audio_url": "/tts_audio/tts_xxx.ogg", "latency_ms": 3400, "audio_duration_ms": 6000}
```

### Translate
```bash
curl -X POST http://127.0.0.1:18321/translate \
  -H "Content-Type: application/json" \
  -d '{"q": "Hello world", "source": "en", "target": "zh"}'
# -> {"data": {"translations": [{"translatedText": "..."}]}}
```

### Image Generation
```bash
curl -X POST http://127.0.0.1:18321/api/image/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a pixel art cat", "steps": 9}'
# -> {"image_url": "/image_output/img_xxx.png", "latency_ms": 28000}
```

## OpenClaw Integration

The best way to use MLS with OpenClaw is via the included skill. Copy `skills/SKILL.md` to your OpenClaw workspace:

```bash
cp skills/SKILL.md ~/.openclaw/workspace/skills/local-model/SKILL.md
```

This gives the agent full API reference for all four services.

For automatic voice message transcription, add the ASR wrapper to `openclaw.json`:

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

## Requirements

- macOS 14+ with Apple Silicon
- Python 3.12+
- `uv` package manager
- `ffmpeg` and `ffprobe` (for audio conversion)
