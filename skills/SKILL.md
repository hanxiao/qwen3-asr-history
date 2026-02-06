# Local Model (MLS - MLX Local Serving)

Four on-device ML models running on Mac Studio M3 Ultra via `http://127.0.0.1:18321`.
All models stay resident in GPU memory after server startup.

Project: `~/Documents/mls/`
Dashboard: `http://127.0.0.1:18321/history`

---

## 1. ASR (Speech-to-Text)

**Model**: Qwen3-ASR-0.6B-bf16 (HuggingFace cache)

```
POST /transcribe
Content-Type: application/json

{
  "path": "/absolute/path/to/audio.ogg",   # required, local file path
  "language": "zh"                          # optional, default "zh"
}

Response:
{
  "text": "transcribed text",
  "latency_ms": 1234.56
}
```

Notes:
- Input: any audio format (auto-converts to WAV internally)
- `path` must be an absolute local file path, not a URL
- OpenClaw voice messages already route through this via `qwen3-asr` wrapper script

---

## 2. TTS (Text-to-Speech)

**Model**: Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 (USB: `/Volumes/One Touch/ai-models/mlx-community/`)

```
POST /synthesize
Content-Type: application/json

{
  "text": "content to speak",              # required
  "language": "chinese",                   # optional, default "en"
  "voice": "Chelsie",                      # optional, default "Chelsie"
  "speed": 1.0,                            # optional, default 1.0
  "format": "ogg",                         # optional, "ogg" (default) or "wav"
  "instruct": "A young Chinese male speaker with a Beijing accent"  # optional, for VoiceDesign model
}

Response:
{
  "status": "ok",
  "audio_url": "/tts_audio/tts_abc123.ogg",
  "audio_file": "tts_abc123.ogg",
  "latency_ms": 3400.00,
  "audio_duration_ms": 6000.00,
  "text": "content to speak",
  "voice": "Chelsie",
  "language": "chinese",
  "format": "ogg"
}
```

Download audio: `curl http://127.0.0.1:18321/tts_audio/<filename> -o output.ogg`

Language codes: `chinese`, `english`, `german`, `italian`, `portuguese`, `spanish`, `japanese`, `korean`, `french`, `russian`

File-based TTS (long text):
```
POST /synthesize/file
{
  "file": "/path/to/text.txt",
  "language": "chinese",
  "voice": "Chelsie",
  "format": "ogg"
}

# Poll progress:
GET /synthesize/file/status?output=<output_path>
```

Notes:
- VoiceDesign model: use `instruct` to control voice characteristics
- Default instruct for Han's voice messages: `"A young Chinese male speaker with a Beijing accent"`
- Performance: ~3.4s generation for 6s audio (1.8x realtime)

---

## 3. Translate

**Model**: TranslateGemma-12B-8bit (USB: `/Volumes/One Touch/ai-models/mlx-community/translategemma-12b-it-8bit/`)

```
POST /translate
Content-Type: application/json

{
  "q": "text to translate",               # required, string or list of strings
  "source": "en",                          # required, ISO 639-1 code
  "target": "zh"                           # required, ISO 639-1 code
}

Response:
{
  "data": {
    "translations": [
      {
        "translatedText": "translated result",
        "detectedSourceLanguage": null,
        "model": "translategemma-12b-it-8bit"
      }
    ]
  }
}
```

GET also works: `GET /translate?q=hello&source=en&target=zh`

Batch: pass `"q": ["text1", "text2"]` for multiple texts.

File-based translation:
```
POST /translate/file
{
  "file": "/path/to/input.txt",
  "source": "en",
  "target": "zh",
  "delimiter": "\n"
}

# Poll progress:
GET /translate/file/status?output=<output_path>
```

Available languages: `GET /languages`

Notes:
- 55 languages supported
- ~0.5-1s per sentence
- Always prefer this over LLM translation

---

## 4. Image Generation

**Model**: Z-Image-Turbo-8bit (USB: `/Volumes/One Touch/ai-models/mflux/z-image-turbo-8bit/`)

```
POST /api/image/generate
Content-Type: application/json

{
  "prompt": "description of image",        # required
  "resolution": "1024x1024",               # optional, default "1024x1024"
  "negative_prompt": "things to avoid",    # optional
  "seed": 42,                              # optional, for reproducibility
  "steps": 9                               # optional, default 9, high quality 20
}

Response:
{
  "status": "ok",
  "image_url": "/image_output/img_abc123.png",
  "image_file": "img_abc123.png",
  "latency_ms": 28000.00,
  "resolution": "1024x1024",
  "seed": 42,
  "steps": 9
}
```

Download image: `curl http://127.0.0.1:18321/image_output/<filename> -o output.png`

Resolutions: `1024x1024`, `768x1024`, `1024x768`, `512x512`, etc.

Notes:
- Default 9 steps (~28s), high quality 20 steps (~60s)
- Model stays in GPU memory, no cold start penalty
- Output is PNG

---

## Health Check

```
GET /health

{
  "status": "ok",
  "model": "mlx-community/Qwen3-ASR-0.6B-bf16",
  "loaded": true,
  "tts_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
  "tts_loaded": true,
  "translate_model": "translategemma-12b-it-8bit",
  "translate_loaded": true,
  "image_model": "z-image-turbo-8bit",
  "image_loaded": true
}
```

## Server Management

```
POST /api/server/pause          # Pause ASR
POST /api/server/resume         # Resume ASR
POST /api/server/restart        # Restart ASR model
POST /api/tts/server/pause      # Pause TTS
POST /api/tts/server/resume     # Resume TTS
POST /api/tts/server/restart    # Restart TTS model
POST /api/translate/server/pause
POST /api/translate/server/resume
POST /api/translate/server/restart
```
