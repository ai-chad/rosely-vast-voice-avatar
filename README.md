# Rosely Voice + MuseTalk 1.5 — Vast Serverless POC

## What this repo is

A Vast custom PyWorker for queued clip inference on one RTX 3090 24 GB. At boot it downloads every checkpoint directly from Hugging Face at pinned revisions and verifies each file against `models.sha256` (`fetch_models.py`); MuseTalk source comes from GitHub at a pinned commit. It is **not** a persistent WebRTC video-call service and is **not** tested on a rented Vast RTX 3090 yet. The earlier `README_PERSISTENT.md` documents standalone/interactive execution.

### Model servers and endpoints

- Qwen3-ASR-0.6B on loopback `8001` (`/v1/stt` JSON via gateway)
- Chatterbox Turbo on loopback `8002` (`/v1/tts` JSON via gateway)
- MuseTalk 1.5 on loopback `8003` (clip preparation/render)
- Authenticated FastAPI gateway on `8000` (not exposed to the Internet by this template)
- Local JSON adapter on loopback `8004` (Vast PyWorker forwards JSON to here)
- Vast PyWorker listens on its platform port (typically `3000`) and exposes `/v1/tts`, `/v1/stt`, `/v1/avatar/generate`.

`/v1/avatar/generate` **requires** `image_base64` and **one** of `audio_base64` or `input` text. It returns `{video_base64, content_type, size_bytes, avatar_id}`. For user text, it generates speech via the existing Chatterbox model. A prepared avatar is cached only on the responding worker's local disk; the route always accepts source image so no cross-worker state is assumed. This POC holds one GPU job at a time; MuseTalk is invoked via an upstream subprocess and reloads weights for each clip.

## Create a new public repo

Upload the extracted contents at the root of the repo (not the parent folder), so `provision_serverless.sh`, `worker.py`, `requirements.txt` are at the repository root. Do **not** upload model checkpoints or real credentials. For a private GitHub repository, this `git clone` startup needs GitHub authentication configured separately.

## Vast template configuration

- Image: `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310` (official Vast CUDA Ubuntu 22.04 variant; validate chosen host driver and image availability)
- Minimum GPU: RTX 3090 24 GB. Allow 32 GB RAM and 8 vCPU if possible.
- Disk: `100` GB is enough (about 12 GB of checkpoints plus three isolated Python environments and build caches; 32 GB used in practice).
- Container port mapping: `-p 3000:3000`. **Do not** map the gateway 8000 or internal model ports publicly.
- `SERVERLESS=true`
- `PYWORKER_REPO=https://github.com/ai-chad/rosely-vast-voice-avatar`
- `PYWORKER_REF=main`
- `PROVISIONING_SCRIPT=https://raw.githubusercontent.com/ai-chad/rosely-vast-voice-avatar/main/provision_serverless.sh`
- `HUGGINGFACE_TOKEN`: a **read-only** Hugging Face token. All model repos are public; the token avoids anonymous rate limits. Put it only in a **private Vast template**, never git. (`HF_TOKEN` is accepted as a fallback.) No S3 or AWS credentials are needed.
- `MUSE_BATCH_SIZE=4`, `FETCH_BUILTIN_VOICES=1`
- Optional `ENABLE_DITTO=1`: also installs the Ditto talking-head engine (see below).

Use a purpose-specific CUDA/base image rather than ComfyUI; this does not run ComfyUI, and **do not set `BACKEND=comfyui-json`**. Also do not use the `-e AWS_ZIT_IMAGE_*` vars, which these scripts do not read.

### What happens on boot

1. Vast base image executes `PROVISIONING_SCRIPT` as root after Supervisor starts.
2. It clones `PYWORKER_REPO` and invokes `provision_vast.sh`, which fetches MuseTalk source from GitHub, downloads the 30 checkpoint files from Hugging Face (pinned revisions, SHA256-verified, resumable), and installs 3 isolated Python 3.10 model environments. A worker that already has a complete bundle skips the download.
3. It installs `vastai-sdk` and registers a Supervisor program `rosely-serverless`.
4. Supervisor runs `serverless_start.sh`, starts 4 FastAPI processes, waits until they are healthy and starts `worker.py`.
5. PyWorker benchmarks the TTS route (short input) and announces readiness via Vast.

The first provisioning can be lengthy because it installs CUDA, PyTorch and MMLab dependencies; the ~12 GB Hugging Face download itself took about 3 minutes on a Vast RTX 3090 host. A prebuilt derivative Docker image is preferable once inference has been validated.

### Optional Ditto engine (dev)

With `ENABLE_DITTO=1`, provisioning runs `provision_ditto.sh`: Ditto source at a pinned commit, its TensorRT checkpoints from Hugging Face (pinned revision, verified against `ditto_models.sha256`), and an isolated `rosely-venvs/ditto` environment (`requirements-ditto.txt`, TensorRT 8.6.1 with a private cuDNN 8). `start_services.sh` then starts `ditto_server.py` on loopback `8005` (~2.6 GB VRAM, resident).

`/v1/avatar/generate` accepts `engine: "ditto"` with `emotion` (the chat reply's voice_style) and `gesture` (`none`, `nod`, `shake`, `tilt`, `wink`). Ditto animates the whole portrait (head motion, blinks, expression) instead of only the mouth; `acting.py` maps emotion and gesture to Ditto controls. Without Ditto, or if it fails, the same request renders with MuseTalk, and the response's `engine` field says which one ran. Measured on an RTX 3090: about 4–9 s per reply including TTS, against roughly 80 s for MuseTalk's prepare + render.

Motion clips (optional): `/v1/avatar/generate` with `engine: "ditto"` also accepts `motion_sha256` plus either `motion_url` (HTTPS) or `motion_base64` for a short pre-generated clip of the character. Ditto keeps the clip's body, hand and head motion and drives only the mouth and expression. The gateway verifies the digest and keeps clips in `DITTO_CLIP_DIR` (default `$AVATAR_MEDIA_ROOT/ditto-clips`), so later turns can send just `motion_sha256`; a missing or invalid clip falls back to the still portrait (`ditto_source` in the response says which). Registering a 10 s clip takes ~6 s and ~2.3 GB, so registrations are cached in RAM (`DITTO_SOURCE_CACHE_RAM`, default 3 clips, ~2.3 GB each) and on disk (`DITTO_SOURCE_CACHE_DIR`, `DITTO_SOURCE_CACHE_DISK_GB` default 20; ~1.3 GB per clip as float16, output byte-identical). Measured on an RTX 3090: first use ~11 s render, RAM hit ~4.5 s, disk hit ~7.3 s for a ~3.6 s reply.

### Vast endpoint

**`vastai create instance` only creates a GPU instance / candidate worker.** You must also create or attach it through Vast's Serverless **endpoint / worker group** configuration (UI or SDK), and configure a suitable warm minimum. For an inexpensive POC, start with one worker, then scale after measuring latency, VRAM and response limits.

## Client smoke tests

```bash
pip install vastai-sdk
export VAST_API_KEY='YOUR_VAST_KEY'
python test_serverless.py --endpoint YOUR_ENDPOINT_NAME
python test_serverless.py --endpoint YOUR_ENDPOINT_NAME --image character.png --out demo.mp4
```

If you use the full video test, start with a short utterance. Clip audio is limited by `MAX_AVATAR_AUDIO_SECONDS=12` and MuseTalk rendering may take much longer than the audio duration. Since returned MP4 is base64, output size limits at the Vast API must be validated before sending large clips. Production should write outputs to S3 and return presigned URLs instead.

## Diagnostics on a worker

```bash
supervisorctl status rosely-serverless
nvidia-smi
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8004/health
tail -n 100 /workspace/rosely-logs/serverless-adapter.log
tail -n 100 /workspace/rosely-logs/avatar.log
```

## Architectural limitations

- Serverless is suitable for on-demand **clips**, not continuous 2-way video calls / WebRTC connections.
- The current implementation restarts MuseTalk inference on every prepare/render (not a hot resident model).
- The ASR/TTS services use the same GPU as MuseTalk. Admission control serializes GPU-intensive gateway requests but VRAM may still be exhausted with all model processes loaded.
- Custom voice references saved in the older Vast instance are not included in the combined checkpoint archive; only the publicly hosted built-in voice references are downloaded by the provisioner.
- Actual GPU inference is not yet validated; MMLab/PyTorch installation may require fixes for this exact image.
- All model/image/voice likeness and checkpoint licenses, consent and age verification remain your responsibility.
