# Rosely Voice + Ditto — Vast Serverless worker

## What this repo is

A Vast custom PyWorker for one RTX 3090 24 GB: speech-to-text, text-to-speech, Ditto talking-head video replies and live WebRTC calls. At boot it downloads every checkpoint directly from Hugging Face at pinned revisions and verifies each file against `models.sha256` / `ditto_models.sha256` (`fetch_models.py`); Ditto source comes from GitHub at a pinned commit.

### Model servers and endpoints

- Qwen3-ASR-0.6B on loopback `8001` (`/v1/stt` JSON via gateway)
- Chatterbox Turbo on loopback `8002` (`/v1/tts` JSON via gateway)
- Ditto (TensorRT) on loopback `8005` (video reply clips, motion-clip registration)
- Live call agent on loopback `8006` (optional, `LIVE_CALL_ENABLED=1`)
- Authenticated FastAPI gateway on `8000` (not exposed to the Internet by this template)
- Local JSON adapter on loopback `8004` (Vast PyWorker forwards JSON to here)
- Vast PyWorker listens on its platform port (typically `3000`) and exposes `/v1/tts`, `/v1/stt`, `/v1/avatar/generate`, `/v1/ditto/warm`, `/v1/live/join`.

`/v1/avatar/generate` takes a still portrait (`image_base64`, png/jpg/webp) and **one** of `audio_base64` or `input` text, plus `emotion` (the chat reply's voice_style) and `gesture` (`none`, `nod`, `shake`, `tilt`, `wink`). It returns `{video_base64, content_type, size_bytes, engine: "ditto", ditto_source}`. For text it generates speech with Chatterbox. Ditto animates the whole portrait (head motion, blinks, expression); `acting.py` maps emotion and gesture to Ditto controls. Measured on an RTX 3090: about 4–9 s per reply including TTS. One GPU job runs at a time.

## Vast template configuration

- Image: `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` (official Vast CUDA Ubuntu 22.04 variant; validate chosen host driver and image availability)
- Minimum GPU: RTX 3090 24 GB. Allow 32 GB RAM and 8 vCPU if possible.
- Disk: `100` GB (checkpoints, three isolated Python environments, Ditto's motion-clip registration cache of up to 20 GB).
- Container port mapping: `-p 3000:3000`. **Do not** map the gateway 8000 or internal model ports publicly.
- `SERVERLESS=true`
- `PYWORKER_REPO=https://github.com/ai-chad/rosely-vast-voice-avatar`
- `PYWORKER_REF=main`
- `PROVISIONING_SCRIPT=https://raw.githubusercontent.com/ai-chad/rosely-vast-voice-avatar/main/provision_serverless.sh`
- `HUGGINGFACE_TOKEN`: a **read-only** Hugging Face token. All model repos are public; the token avoids anonymous rate limits. Put it only in a **private Vast template**, never git. (`HF_TOKEN` is accepted as a fallback.) No S3 or AWS credentials are needed.
- `FETCH_BUILTIN_VOICES=1`

Use a purpose-specific CUDA/base image rather than ComfyUI; this does not run ComfyUI, and **do not set `BACKEND=comfyui-json`**.

### What happens on boot

1. Vast base image executes `PROVISIONING_SCRIPT` as root after Supervisor starts.
2. It clones `PYWORKER_REPO` and invokes `provision_vast.sh`, which downloads the ASR and TTS checkpoints from Hugging Face (pinned revisions, SHA256-verified, resumable) into isolated Python 3.10 environments, then `provision_ditto.sh` (Ditto source, TensorRT checkpoints, the `rosely-venvs/ditto` environment with a private cuDNN 8, the LiveKit client). A worker that already has a complete bundle skips the download.
3. It installs `vastai-sdk` and registers a Supervisor program `rosely-serverless`.
4. Supervisor runs `serverless_start.sh`, which starts the model services and the gateway, waits until they are healthy and starts `worker.py`.
5. PyWorker benchmarks the TTS route (short input) and announces readiness via Vast.

### Motion clips

`/v1/avatar/generate` also accepts `motion_sha256` plus either `motion_url` (HTTPS) or `motion_base64` for a short pre-generated clip of the character. Ditto keeps the clip's body, hand and head motion and drives only the mouth and expression. The gateway verifies the digest and keeps clips in `DITTO_CLIP_DIR` (default `$AVATAR_MEDIA_ROOT/ditto-clips`), so later turns can send just `motion_sha256`; a missing or invalid clip falls back to the still portrait (`ditto_source` in the response says which). Registering a 10 s clip takes ~6 s and ~2.3 GB, so registrations are cached in RAM (`DITTO_SOURCE_CACHE_RAM`, ~2.3 GB each) and on disk (`DITTO_SOURCE_CACHE_DIR`, `DITTO_SOURCE_CACHE_DISK_GB` default 20; ~1.3 GB per clip as float16, output byte-identical). `/v1/ditto/warm` registers a character's clips ahead of its replies.

### Live calls (WebRTC)

`provision_ditto.sh` adds the LiveKit client and voice activity detection to the Ditto environment and sets `LIVE_CALL_ENABLED=1`. `start_services.sh` then starts `live_agent.py` on loopback `8006` (streaming Ditto, ~2.6 GB more VRAM). If it fails to start, turn-based voice/video keep working and the gateway's health ignores it.

The backend starts a call with `/v1/live/join` (room URL, a short-lived LiveKit token, the caller's identity, the character's portrait URL, voice and motion clips, and a per-call secret for turn callbacks). The worker joins the room as the character. It detects the end of each caller turn (webrtcvad), transcribes it with the local ASR and posts it to the backend's `turn_url`; the backend runs the character chat and billing and returns the reply. The worker then synthesizes it sentence by sentence and streams Ditto video at 20 fps with the matching audio, from the motion clip the reply's emotion calls for (loaded from the registration cache) or the still portrait. Between replies the still portrait shows, and speaking over the character interrupts it. Only one call runs per worker at a time (`409 live_call_busy`). The worker never holds LiveKit API credentials.

Measured on an RTX 3090: Ditto online renders ~22–24 fps (every fifth frame is skipped before warp/decode for 20 fps output), and lips land within about one frame of the audio. From reply text to the first spoken frame takes ~2.5–3.5 s, mostly Chatterbox synthesizing the first sentence.

### Vast endpoint

**`vastai create instance` only creates a GPU instance / candidate worker.** You must also create or attach it through Vast's Serverless **endpoint / worker group** configuration (UI or SDK), and configure a suitable warm minimum. Start with one worker, then scale after measuring latency, VRAM and response limits.

## Client smoke tests

```bash
pip install vastai-sdk
export VAST_API_KEY='YOUR_VAST_KEY'
python test_serverless.py --endpoint YOUR_ENDPOINT_NAME
python test_serverless.py --endpoint YOUR_ENDPOINT_NAME --image character.png --out demo.mp4
```

For the video test, start with a short utterance. Clip audio is limited by `MAX_AVATAR_AUDIO_SECONDS=12`. Returned MP4s travel as base64, so keep clips short.

## Diagnostics on a worker

```bash
supervisorctl status rosely-serverless
nvidia-smi
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8004/health
tail -n 100 /workspace/rosely-logs/serverless-adapter.log
tail -n 100 /workspace/rosely-logs/ditto.log
tail -n 100 /workspace/rosely-logs/live.log
```

## Architectural limitations

- One GPU serves ASR, TTS, Ditto clips and one live call; the gateway serializes GPU-heavy clip requests, and a live call holds the streaming renderer.
- Custom voice references saved in the older Vast instance are not included; only the publicly hosted built-in voice references are downloaded by the provisioner.
- All model/image/voice likeness and checkpoint licenses, consent and age verification remain your responsibility.
