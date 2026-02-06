# MLS (MLX Local Serving)

Unified local serving for ASR, TTS, and Translation on Apple Silicon via MLX.

## Features

- **ASR**: Qwen2.5-ASR (0.6B/1.7B) - Fast, accurate speech-to-text
- **TTS**: Qwen2.5-TTS (1.7B VoiceDesign) - Natural speech with instruct support (e.g., accents)
- **Translate**: TranslateGemma 12B - High-quality document translation (55+ languages)
- **Dashboard**: Unified web UI on port 18321 with accordion sidebar, mini-calendar, and 3 service tabs

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

### ASR (Speech-to-Text)
```bash
curl -X POST http://127.0.0.1:18321/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/audio.ogg"}'
```

### TTS (Text-to-Speech)
```bash
curl -X POST http://127.0.0.1:18321/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好，世界",
    "instruct": "A young Chinese male speaker with a Beijing accent"
  }' > output.json
```

### Translate
```bash
curl -X POST http://127.0.0.1:18321/translate \
  -H "Content-Type: application/json" \
  -d '{"q": "Hello world", "source": "en", "target": "zh"}'
```

## Models

All models run locally on Metal GPU via MLX.

- **ASR**: `mlx-community/Qwen2.5-ASR-0.6B-bf16` (Default)
- **TTS**: `mlx-community/Qwen2.5-TTS-12Hz-1.7B-VoiceDesign-bf16` (Default, supports prompts)
- **Translate**: `mlx-community/translategemma-12b-it-8bit` (Requires ~12GB VRAM)

## Requirements

- macOS 14+ with Apple Silicon (M1/M2/M3)
- Python 3.12+
- `uv` package manager
