# Rosely Voice + MuseTalk 1.5 — Vast Serverless POC

## What this repo is

A Vast custom PyWorker for queued clip inference on one RTX 3090 24 GB. It installs the exact combined checkpoint archive uploaded to `s3://rosely-infrastructure/serverless/voice-avatar/rosely-voice-musetalk-v1.tar.zst`. It is **not** a persistent WebRTC video-call service and is **not** tested on a rented Vast RTX 3090 yet. The earlier `README_PERSISTENT.md` documents standalone/interactive execution.

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
- Disk: `150` GB recommended for downloading/extracting the 9.9 GiB model bundle plus three isolated Python environments and build caches.
- Container port mapping: `-p 3000:3000`. **Do not** map the gateway 8000 or internal model ports publicly.
- `SERVERLESS=true`
- `PYWORKER_REPO=https://github.com/ai-chad/rosely-vast-voice-avatar`
- `PYWORKER_REF=main`
- `PROVISIONING_SCRIPT=https://raw.githubusercontent.com/ai-chad/rosely-vast-voice-avatar/main/provision_serverless.sh`
- `MODEL_S3_URI=s3://rosely-infrastructure/serverless/voice-avatar/rosely-voice-musetalk-v1.tar.zst`
- `ROSELY_AVATAR_S3_ACCESS_KEY_ID`, `ROSELY_AVATAR_S3_SECRET_ACCESS_KEY`: restricted `s3:GetObject` credentials for **two objects** (archive and `.sha256`). Put them only in a **private Vast template**, never git or a publicly viewable environment. They are deliberately custom names to avoid Vast-cached standard AWS variables; provisioning maps them to standard AWS CLI credentials only in the child process.
- `ROSELY_AVATAR_S3_REGION=us-east-1`
- Optional: `MODEL_ARCHIVE_SHA256` and `MODEL_ARCHIVE_SIZE`, actual values only; remote `.sha256` is downloaded and validated even without optional values.
- `MUSE_BATCH_SIZE=4`, `FETCH_BUILTIN_VOICES=1`

Use a purpose-specific CUDA/base image rather than ComfyUI; this does not run ComfyUI, and **do not set `BACKEND=comfyui-json`**. Also do not use the `-e AWS_ZIT_IMAGE_*` vars, which these scripts do not read.

### What happens on boot

1. Vast base image executes `PROVISIONING_SCRIPT` as root after Supervisor starts.
2. It clones `PYWORKER_REPO`, invokes `provision_vast.sh`, downloads the bundle and its checksum, validates both archive and inner weights, and installs 3 isolated Python 3.10 model environments.
3. It installs `vastai-sdk` and registers a Supervisor program `rosely-serverless`.
4. Supervisor runs `serverless_start.sh`, starts 4 FastAPI processes, waits until they are healthy and starts `worker.py`.
5. PyWorker benchmarks the TTS route (short input) and announces readiness via Vast.

The first provisioning can be lengthy because it downloads 9.9 GiB and installs CUDA, PyTorch and MMLab dependencies. A prebuilt derivative Docker image is preferable once inference has been validated.

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
