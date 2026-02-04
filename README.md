# Qwen3-ASR History

Local speech-to-text server using Qwen3-ASR on Apple Silicon via MLX. Model stays loaded in memory for fast inference.

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- ffmpeg

## Install

```bash
git clone <repo> ~/Documents/qwen3-asr-history
cd ~/Documents/qwen3-asr-history
./setup.sh
```

Server starts automatically at `http://127.0.0.1:18321`.

## OpenClaw Setup

In OpenClaw settings, set transcription to use local server:

```
Transcription URL: http://127.0.0.1:18321/transcribe
```

The server reads audio directly from OpenClaw's media folder (`~/.openclaw/media/inbound/`).

## Usage

CLI:
```bash
qwen3-asr recording.ogg        # Chinese (default)
qwen3-asr recording.ogg en     # English
```

API:
```bash
curl -X POST http://127.0.0.1:18321/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/audio.ogg", "language": "zh"}'
```

History UI: http://127.0.0.1:18321/history

## Service

```bash
launchctl list | grep qwen3-asr                              # status
launchctl unload ~/Library/LaunchAgents/ai.openclaw.qwen3-asr.plist  # stop
launchctl load ~/Library/LaunchAgents/ai.openclaw.qwen3-asr.plist    # start
tail -f ~/Documents/qwen3-asr-history/logs/server.log        # logs
```

## Models

Available models (switch via UI):
- `mlx-community/Qwen3-ASR-0.6B-bf16` - fastest, default
- `mlx-community/Qwen3-ASR-1.7B-8bit` - better quality
- `mlx-community/whisper-large-v3-turbo-asr-fp16` - whisper turbo
- `mlx-community/whisper-large-v3-asr-8bit` - whisper v3
