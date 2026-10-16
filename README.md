# Rosely Voice + MuseTalk 1.5 — Vast RTX 3090 POC

**New, separate deployment. Does not modify the working RTX 5060 / `rosely-vast-voice` endpoint.**

This repository deploys three existing/open-source model stacks from the **single model archive already uploaded to S3**, plus an authenticated API gateway:

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
├── provision_vast.sh         # private S3 download, digest verification, isolated Python environments
├── start_services.sh         # 8001 ASR, 8002 TTS, 8003 avatar, 8000 gateway
├── asr_server.py             # existing Qwen3-ASR request contract
├── tts_server.py             # existing Chatterbox Turbo endpoints + voice references
├── voice_gateway.py          # /v1/stt and /v1/tts unchanged; adds avatar routes
├── avatar_server.py          # MuseTalk 1.5 prep and MP4 render endpoints
├── test_poc.py               # tests one character end-to-end from your laptop
├── tests/                   # local no-GPU API checks
├── requirements-dev.txt
├── .env.example
└── docs/S3_READ_ONLY_POLICY.json
```

**Do not commit the 9.9 GiB S3 checkpoint archive or credentials to GitHub.** It is retrieved at provisioning time from:

```text
s3://rosely-infrastructure/serverless/voice-avatar/rosely-voice-musetalk-v1.tar.zst
s3://rosely-infrastructure/serverless/voice-avatar/rosely-voice-musetalk-v1.tar.zst.sha256
```

The S3 bundle contains:

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
- Allow outbound HTTPS to S3, PyPI/PyTorch/MMLab, Astral Python runtime downloads, and optionally the old public GitHub repo for the 28 style voice samples.

For a persistent instance, **do not use `PYWORKER_REPO` / Vast PyWorker** as the deployment mechanism. Those are for Vast Serverless queued JSON jobs; this prototype uses a regular instance's port mapping. Streaming calls need a permanent connection in the next phase.

### Give the new instance S3 read-only access

Option A — *recommended for repeatable boots:* Create a dedicated access key restricted to `GetObject` on the two archive/checksum keys. See `docs/S3_READ_ONLY_POLICY.json`. Put `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in Vast **private environment settings**, not in Git. With a fresh machine terminal, check:

```bash
aws sts get-caller-identity
```

Option B — pre-signed temporary HTTPS URLs. On your EC2 with AWS access, generate presigned GET URLs for each S3 object using `aws s3 presign ... --expires-in 21600`, then set `MODEL_ARCHIVE_URL` and `MODEL_SHA_URL` in the Vast instance environment. These URLs expire; the archive only needs to be downloaded during initial provisioning. Treat URLs as credentials.

## 2. Create a new GitHub repository

Unzip this package and push it to your own newly created repository; models remain in S3:

```bash
unzip rosely-vast-voice-avatar-repo.zip
cd rosely-vast-voice-avatar
git init
git add .
git commit -m "Initial Rosely voice and avatar RTX 3090 POC"
git branch -M main
git remote add origin https://github.com/YOUR_ORG/YOUR_NEW_REPO.git
git push -u origin main
```

## 3. Provision new Vast RTX 3090

SSH to the **new** Vast instance (never execute these on the old voice GPU). Use the instance's mapped SSH port or Vast web terminal.

```bash
cd /workspace
git clone https://github.com/YOUR_ORG/YOUR_NEW_REPO.git rosely-vast-voice-avatar
cd /workspace/rosely-vast-voice-avatar

nvidia-smi
# Your restricted S3 credentials must already be available as environment variables.
sudo -E bash provision_vast.sh
sudo -E bash start_services.sh
```

Provisioner behavior:

1. Downloads the archived **9.9 GiB** model bundle and its checksum from the exact S3 path.
2. Verifies `sha256sum` before extracting to `/workspace/rosely-voice-musetalk-v1` and verifies per-model file checksums from the manifest.
3. Reuses the original Qwen and Chatterbox checkpoints; no Hugging Face model download takes place.
4. Creates isolated Python 3.10 environments in `/workspace/rosely-venvs/{asr,tts,muse}` to keep model dependencies compatible.
5. Installs the current voice dependencies, then **upstream MuseTalk 1.5** dependencies: Torch 2.0.1/CUDA11.8 and MMLab. This step needs network access and may require troubleshooting CUDA/MMLab wheels on your chosen Vast OS image.
6. By default, downloads the same **28 `en_female_*` / `en_male_*` style voice samples** from the old public voice GitHub repository, preserving Chatterbox voice IDs. Set `FETCH_BUILTIN_VOICES=0` if you will supply these WAVs yourself.
7. Generates `/workspace/rosely-voice-avatar.env` containing new private internal/API gateway keys. Existing keys are preserved if you rerun the provisioner.

**Do not copy voice sample WAVs from arbitrary sources; ensure you have rights for each reference voice.** User-uploaded custom voice references on the old 5060 are **not** included in the S3 model archive. Copy them separately only when authorized, if you need the same personal/custom IDs.

If provisioning fails during MuseTalk dependency installation, fix the failure before starting services. A checkpoint archive does not include preinstalled CUDA libraries or a working inference environment.

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

**S3 AccessDenied** — Outside AWS, Vast has no EC2 instance IAM role; use scoped S3 credentials or presigned URLs. Confirm `aws sts get-caller-identity`. Do not grant `s3:*` or put credentials in Git.

**S3 checksum failed** — Do not extract. Recheck S3 object key and the separately uploaded SHA file; delete the partial local archive, redownload and verify again.

**MuseTalk dependency/MMLab installation failed** — Choose Ubuntu 22.04 with modern NVIDIA driver; this project intentionally uses upstream Python3.10, Torch2.0.1+cu118 and MMLab versions, separate from Torch2.6+cu124 voice environments. Read pip/mim error output. Some GPUs/images may need compatible prebuilt `mmcv` wheels. GPU-dependent install/inference was not exercised when this ZIP was authored.

**CUDA OOM** — Run `nvidia-smi`, reduce `MUSE_BATCH_SIZE=1`, retry after restart. For the first video-only test, stop ASR/TTS temporarily to isolate VRAM demand, then test all three together.

**Avatar already prepared, HTTP 409** — Use the same `avatar_id` with `/render`; preparation is only required once. To update character content, use a new ID for the POC.

**Wrong external port** — Vast maps internal TCP 8000 to a platform-assigned public port. Use its actual public mapping and restrict access.

**GPU media not fluid** — `/render` produces an MP4 clip, not real-time RTP frames. Measuring upstream offline FPS does not prove that live video calling works.

## 7. Next phase (not bundled)

A resident MuseTalk inference engine, streaming STT/TTS, interruptible frame queue, LiveKit agent/session and proper per-minute billing. These require additional engineering after the first 3090 quality/VRAM benchmark.
