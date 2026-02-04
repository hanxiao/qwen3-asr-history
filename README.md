# Qwen3-ASR History

Local speech-to-text server for OpenClaw using Qwen3-ASR on Apple Silicon via MLX.

![screenshot](https://github.com/hanxiao/qwen3-asr-history/blob/main/screenshot.png?raw=true)

## Quick Start

**Install** - clone and run setup, server starts automatically via launchd.
```bash
git clone https://github.com/hanxiao/qwen3-asr-history ~/Documents/qwen3-asr-history
cd ~/Documents/qwen3-asr-history && ./setup.sh
```

**CLI** - transcribe audio files (language auto-detected).
```bash
bin/qwen3-asr recording.ogg
```

**API** - POST audio path for transcription.
```bash
curl -X POST http://127.0.0.1:18321/transcribe \
  -H "Content-Type: application/json" \
  -d '{"path": "/path/to/audio.ogg"}'
```

**UI** - browse history at http://127.0.0.1:18321/history

**Service** - manage the background server.
```bash
launchctl list | grep qwen3-asr       # status
launchctl stop ai.openclaw.qwen3-asr  # stop
launchctl start ai.openclaw.qwen3-asr # start
tail -f logs/server.log               # logs
```

## OpenClaw Setup

Add to `~/.openclaw/openclaw.json`:

```json
{
  "tools": {
    "media": {
      "audio": {
        "enabled": true,
        "models": [
          {
            "type": "cli",
            "command": "~/Documents/qwen3-asr-history/bin/qwen3-asr",
            "args": ["{{MediaPath}}"],
            "timeoutSeconds": 60
          }
        ]
      }
    }
  }
}
```

## Models

Switch via UI dropdown:
- `mlx-community/Qwen3-ASR-0.6B-bf16` - fastest, default
- `mlx-community/Qwen3-ASR-1.7B-8bit` - better quality
- `mlx-community/whisper-large-v3-turbo-asr-fp16` - whisper turbo
- `mlx-community/whisper-large-v3-asr-8bit` - whisper v3

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- ffmpeg
