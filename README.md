# Qwen3-ASR History

Local speech-to-text server using Qwen3-ASR on Apple Silicon via MLX. Model stays loaded in memory for fast inference (~120-200ms hot).

## Requirements

- macOS with Apple Silicon
- Python 3.12+
- ffmpeg

## Install

```bash
git clone https://github.com/hanxiao/qwen3-asr-history ~/Documents/qwen3-asr-history
cd ~/Documents/qwen3-asr-history
./setup.sh
```

Server starts automatically at `http://127.0.0.1:18321`.

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

The CLI wrapper calls the local server (fast) with fallback to cold start if server is down.

## Usage

CLI:
```bash
bin/qwen3-asr recording.ogg        # Chinese (default)
bin/qwen3-asr recording.ogg en     # English
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

Available models (switch via environment or config):
- `mlx-community/Qwen3-ASR-0.6B-bf16` - fastest, default
- `mlx-community/Qwen3-ASR-1.7B-8bit` - better quality

## Performance

- Cold start: ~3s (Python + model loading)
- Hot (server running): ~120-200ms
