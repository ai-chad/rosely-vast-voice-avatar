# Rosely Voice + MuseTalk 1.5 — Vast RTX 3090 POC

**New, separate deployment. Does not modify the working RTX 5060 / `rosely-vast-voice` endpoint.**

This repository deploys three existing/open-source model stacks, downloaded at provisioning time directly from **Hugging Face** (pinned revisions, SHA256-verified), plus an authenticated API gateway:

| Service | Model | Loopback port | External gateway route |
|---|---|---:|---|
| ASR | Qwen3-ASR-0.6B | 8001 | `POST /v1/stt` — JSON/base64 |
| TTS | Chatterbox Turbo 350M | 8002 | `POST /v1/tts` — JSON/base64 |
| Avatar | MuseTalk 1.5 | 8003 | `POST /v1/avatar/prepare`, `/v1/avatar/render` — multipart |
| Gateway | FastAPI | **8000** | Single authenticated public entrypoint |

> **This is a working *clip-generation POC design*, not a finished live video-call system.** MuseTalk's upstream `realtime_inference.py` preprocesses avatars and renders audio clips; it is not a LiveKit/WebRTC server. This version runs upstream CLI in a subprocess per prepare/render operation (thus reloads weights per request) and returns MP4 files. After proving quality and GPU capacity, refactor MuseTalk into a resident frame streamer and integrate self-hosted LiveKit. No end-to-end inference has been run on an RTX 3090 in this package-building environment.

## Repository structure

```text
rosely-vast-voice-avatar/
├── provision_vast.sh         # Hugging Face model download, isolated Python environments
├── fetch_models.py           # pinned HF repos/revisions; verifies every file against models.sha256
├── models.sha256             # expected SHA256 of the 30 runtime checkpoint files
├── start_services.sh         # 8001 ASR, 8002 TTS, 8003 avatar, 8000 gateway
├── asr_server.py             # existing Qwen3-ASR request contract
├── tts_server.py             # existing Chatterbox Turbo endpoints + voice references
├── voice_gateway.py          # /v1/stt and /v1/tts unchanged; adds avatar routes
├── avatar_server.py          # MuseTalk 1.5 prep and MP4 render endpoints
├── test_poc.py               # tests one character end-to-end from your laptop
├── tests/                   # local no-GPU API checks
├── requirements-dev.txt
└── .env.example
```

**Do not commit checkpoints or tokens to GitHub.** `fetch_models.py` downloads them from these Hugging Face repos (revisions pinned in the script): `Qwen/Qwen3-ASR-0.6B`, `ResembleAI/chatterbox-turbo`, `TMElyralab/MuseTalk` (v1.5 UNet), `stabilityai/sd-vae-ft-mse`, `openai/whisper-tiny`, `yzd-v/DWPose`, `ByteDance/LatentSync`, and `ManyOtherFunctions/face-parse-bisent` (byte-identical mirror of the face-parsing weights upstream fetches from Google Drive). MuseTalk source is fetched from GitHub at commit `0a89dec`.

The resulting bundle layout:

```text
rosely-voice-musetalk-v1/
├── models/
│   ├── Qwen3-ASR-0.6B/
│   └── chatterbox-turbo/
├── source/MuseTalk/
│   ├── models/musetalkV15/
│   ├── models/sd-vae/
│   ├── models/whisper/
│   ├── models/dwpose/
│   ├── models/face-parse-bisent/
│   ├── models/syncnet/
│   ├── scripts/
│   ├── musetalk/
│   └── requirements.txt
└── manifest/SHA256SUMS
```

## 1. Vast instance configuration

Choose **Vast On-demand/Reserved Instance (persistent)** rather than Serverless, with:

- **NVIDIA RTX 3090 24 GB**, one GPU. Prefer a host with driver support for CUDA 12.4 (NVIDIA driver **550+**), Ubuntu 22.04 CUDA/PyTorch-compatible image.
- **At least 80 GB disk**, ideally 100+ GB, 32 GB system RAM and 8 CPU cores for comfortable installation, preprocessing and clip testing.
- Expose **container TCP port `8000`** in Vast. The externally mapped *host* port may differ; check Vast's port mapping page. Internal ports 8001–8003 bind only to 127.0.0.1.
- Allow outbound HTTPS to Hugging Face, GitHub, PyPI/PyTorch/MMLab, Astral Python runtime downloads, and optionally the old public GitHub repo for the 28 style voice samples.

For a persistent instance, **do not use `PYWORKER_REPO` / Vast PyWorker** as the deployment mechanism. Those are for Vast Serverless queued JSON jobs; this prototype uses a regular instance's port mapping. Streaming calls need a permanent connection in the next phase.

### Hugging Face token

Put a **read-only** `HUGGINGFACE_TOKEN` in Vast **private environment settings**, not in Git. The repos are public, so the token only avoids anonymous rate limits; `HF_TOKEN` is accepted as a fallback.

## 2. Create a new GitHub repository

Unzip this package and push it to your own newly created repository; models are downloaded from Hugging Face at provisioning time:

```bash
unzip rosely-vast-voice-avatar-serverless-repo-updated.zip
cd rosely-vast-voice-avatar
git init
git add .
git commit -m "Initial Rosely voice and avatar RTX 3090 POC"
git branch -M main
git remote add origin https://github.com/ai-chad/rosely-vast-voice-avatar.git
git push -u origin main
```

## 3. Provision new Vast RTX 3090

SSH to the **new** Vast instance (never execute these on the old voice GPU). Use the instance's mapped SSH port or Vast web terminal.

```bash
cd /workspace
git clone https://github.com/ai-chad/rosely-vast-voice-avatar.git rosely-vast-voice-avatar
cd /workspace/rosely-vast-voice-avatar

nvidia-smi
# HUGGINGFACE_TOKEN should already be available as an environment variable.
sudo -E bash provision_vast.sh
sudo -E bash start_services.sh
```

Provisioner behavior:

1. Fetches MuseTalk source from GitHub (pinned commit) into `/workspace/rosely-voice-musetalk-v1/source/MuseTalk`.
2. Downloads the 30 checkpoint files (~12 GB) from Hugging Face at pinned revisions and verifies each against `models.sha256`; reruns skip files that already match.
3. These are the same bytes as the original S3 bundle (checksums identical), so behavior is unchanged.
4. Creates isolated Python 3.10 environments in `/workspace/rosely-venvs/{asr,tts,muse}` to keep model dependencies compatible.
5. Installs the current voice dependencies, then **upstream MuseTalk 1.5** dependencies: Torch 2.0.1/CUDA11.8 and MMLab. This step needs network access and may require troubleshooting CUDA/MMLab wheels on your chosen Vast OS image.
6. By default, downloads the same **28 `en_female_*` / `en_male_*` style voice samples** from the old public voice GitHub repository, preserving Chatterbox voice IDs. Set `FETCH_BUILTIN_VOICES=0` if you will supply these WAVs yourself.
7. Generates `/workspace/rosely-voice-avatar.env` containing new private internal/API gateway keys. Existing keys are preserved if you rerun the provisioner.

**Do not copy voice sample WAVs from arbitrary sources; ensure you have rights for each reference voice.** User-uploaded custom voice references on the old 5060 are **not** part of the model download. Copy them separately only when authorized, if you need the same personal/custom IDs.

If provisioning fails during MuseTalk dependency installation, fix the failure before starting services. Checkpoints do not include preinstalled CUDA libraries or a working inference environment.

## 4. Check services and API key

```bash
# On the new Vast instance
curl -s http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8002/health
curl -s http://127.0.0.1:8003/health
curl -s http://127.0.0.1:8000/health

# Display the new gateway key locally; do NOT paste into a public repository/chat.
grep '^GATEWAY_API_KEY=' /workspace/rosely-voice-avatar.env

# Logs and diagnostics
nvidia-smi
tail -n 80 /workspace/rosely-logs/avatar.log
tail -n 80 /workspace/rosely-logs/tts.log
tail -n 80 /workspace/rosely-logs/asr.log
```

Ports `8001`, `8002` and `8003` are **loopback only**. Only the gateway listens on `0.0.0.0:8000`, and it requires `X-API-Key` on inference routes. `GET /health` intentionally does not require authentication and only reports readiness; place a firewall/IP allowlist or TLS reverse proxy in front of the mapped Vast endpoint before external deployment.

## 5. Generate a complete test clip

From your **laptop or EC2** with `requests` installed, using the Vast mapped URL (not the loopback port on your laptop):

```bash
pip install requests
python test_poc.py \
  --url http://VAST_HOST:MAPPED_PORT \
  --api-key 'YOUR_NEW_GATEWAY_API_KEY' \
  --avatar-file ./character.png \
  --avatar-id demo_adult_01 \
  --voice en_female_happy \
  --text 'Hi! How was your day?' \
  --out ./rosely_demo.mp4
```

The client performs: `/health` → `/v1/avatar/prepare` (only once per ID) → `/v1/tts` → `/v1/avatar/render`, and saves an MP4. **Only test with characters you have a right to depict, and with adult characters.**

### Direct endpoints

All POSTs to the gateway require `X-API-Key`:

| Endpoint | Data | Response |
|---|---|---|
| `POST /v1/tts` | JSON `{ "input":"Hello", "voice":"en_female_happy" }` | JSON `audio_base64`, `content_type`, `sample_rate`, `size_bytes` |
| `POST /v1/stt` | JSON `{ "audio_base64":"...", "filename":"audio.wav" }` | JSON `text`, `language` |
| `POST /v1/avatar/prepare` | `multipart/form-data`, `avatar_id`, `file` (PNG/JPEG/WEBP/MP4/MOV/WEBM) | JSON `{ "ok":true, "avatar_id":"...", "prepared":true }` |
| `POST /v1/avatar/render` | `multipart/form-data`, `avatar_id`, `audio` (WAV or other ffmpeg-decodable audio) | MP4 bytes |

POC limits: 25 MiB source media, 10 MiB source audio, 12-second audio clip, two-second normalized idle video, one avatar job at a time. Set `MAX_AVATAR_AUDIO_SECONDS` etc. in `avatar_server.py`'s environment if you need different limits. The default render batch size is `4` to conserve RTX 3090 VRAM.

### Design and performance notes

- `/v1/tts` still returns **whole-utterance WAV**, not streaming PCM. `/v1/stt` still transcribes completed audio.
- `prepare` preprocesses/caches an avatar. Use a source image with a clear, frontal face. The image can be converted to a two-second static video; MuseTalk cannot create independent full-body gestures.
- This POC **restarts the MuseTalk model for every render**. The first response will be slow. For production, maintain the inference model in GPU memory, render frames continuously and publish time-stamped audio/video through self-hosted LiveKit/WebRTC.
- **Sharing one 24 GB GPU does not guarantee all models fit or can infer concurrently.** Monitor peak VRAM with `nvidia-smi` while invoking TTS and MuseTalk. Reduce Muse batch size to 1–2 if needed; benchmark separate inference workers if you hit OOM or miss frame deadlines.
- **Do not run more than one avatar inference job at a time.** If the GPU isn't fast enough, the next phase can split voice and avatar workers again. This POC uses an in-process lock rather than a global distributed scheduler.
- The MuseTalk model and supporting tools have their own licenses, which must be reviewed before commercial deployment. Self-hosting doesn't confer consent for real people's identities or voices.

## 6. Troubleshooting

**Hugging Face 401/429** — Check `HUGGINGFACE_TOKEN` is set in the Vast template and is a valid read token. Rerun `provision_vast.sh`; files already verified are skipped.

**SHA256 mismatch** — `fetch_models.py` deletes the bad file and stops. Rerun to download it again; a persistent mismatch means the pinned revision or `models.sha256` was edited inconsistently.

**MuseTalk dependency/MMLab installation failed** — Choose Ubuntu 22.04 with modern NVIDIA driver; this project intentionally uses upstream Python3.10, Torch2.0.1+cu118 and MMLab versions, separate from Torch2.6+cu124 voice environments. Read pip/mim error output. Some GPUs/images may need compatible prebuilt `mmcv` wheels. GPU-dependent install/inference was not exercised when this ZIP was authored.

**CUDA OOM** — Run `nvidia-smi`, reduce `MUSE_BATCH_SIZE=1`, retry after restart. For the first video-only test, stop ASR/TTS temporarily to isolate VRAM demand, then test all three together.

**Avatar already prepared, HTTP 409** — Use the same `avatar_id` with `/render`; preparation is only required once. To update character content, use a new ID for the POC.

**Wrong external port** — Vast maps internal TCP 8000 to a platform-assigned public port. Use its actual public mapping and restrict access.

**GPU media not fluid** — `/render` produces an MP4 clip, not real-time RTP frames. Measuring upstream offline FPS does not prove that live video calling works.

## 7. Next phase (not bundled)

A resident MuseTalk inference engine, streaming STT/TTS, interruptible frame queue, LiveKit agent/session and proper per-minute billing. These require additional engineering after the first 3090 quality/VRAM benchmark.
