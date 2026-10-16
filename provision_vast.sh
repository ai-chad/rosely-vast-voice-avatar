#!/usr/bin/env bash
# Fresh-install on a persistent Vast RTX 3090 instance. NEVER touches old RTX 5060.
set -Eeuo pipefail
log(){ printf '\n[%s] %s\n' "$(date -Is)" "$*"; }
die(){ printf '[FATAL] %s\n' "$*" >&2; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
BUNDLE_NAME="rosely-voice-musetalk-v1"
BUNDLE_ROOT="${BUNDLE_ROOT:-${WORKSPACE}/${BUNDLE_NAME}}"
ARCHIVE_DIR="${WORKSPACE}/rosely-bundle-download"
CACHE_ARCHIVE="${ARCHIVE_DIR}/${BUNDLE_NAME}.tar.zst"
CACHE_SHA="${CACHE_ARCHIVE}.sha256"
S3_PREFIX="${S3_PREFIX:-s3://rosely-infrastructure/serverless/voice-avatar}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION

[[ "$(id -u)" == 0 ]] || die 'Run sudo -E bash provision_vast.sh (root required for apt packages).'
command -v nvidia-smi >/dev/null || die 'NVIDIA GPU is not visible.'
log 'Detected NVIDIA GPU'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
command -v apt-get >/dev/null || die 'Ubuntu/Debian instance required.'
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y --no-install-recommends \
  awscli ca-certificates curl ffmpeg git openssl python3 python3-venv python3-pip \
  libsndfile1 libsox-fmt-all sox zstd libgl1 libglib2.0-0 libgomp1

mkdir -p "$WORKSPACE" "$ARCHIVE_DIR"
ESSENTIAL=(
  "$BUNDLE_ROOT/models/Qwen3-ASR-0.6B/model.safetensors"
  "$BUNDLE_ROOT/models/chatterbox-turbo/t3_turbo_v1.safetensors"
  "$BUNDLE_ROOT/models/chatterbox-turbo/s3gen_meanflow.safetensors"
  "$BUNDLE_ROOT/source/MuseTalk/models/musetalkV15/unet.pth"
  "$BUNDLE_ROOT/source/MuseTalk/models/musetalkV15/musetalk.json"
  "$BUNDLE_ROOT/source/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin"
  "$BUNDLE_ROOT/source/MuseTalk/models/whisper/pytorch_model.bin"
  "$BUNDLE_ROOT/source/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth"
  "$BUNDLE_ROOT/source/MuseTalk/models/syncnet/latentsync_syncnet.pt"
  "$BUNDLE_ROOT/source/MuseTalk/models/face-parse-bisent/79999_iter.pth"
)
models_complete(){
  local f
  for f in "${ESSENTIAL[@]}"; do [[ -s "$f" ]] || return 1; done
  [[ -s "$BUNDLE_ROOT/manifest/SHA256SUMS" ]]
}

if models_complete; then
  log 'Model files already extracted; reusing local bundle.'
else
  if [[ ! -s "$CACHE_ARCHIVE" ]]; then
    if [[ -n "${MODEL_ARCHIVE_URL:-}" ]]; then
      [[ -n "${MODEL_SHA_URL:-}" ]] || die 'MODEL_SHA_URL required with MODEL_ARCHIVE_URL.'
      log 'Downloading private archive from presigned HTTPS URL'
      curl -fL --retry 5 --retry-delay 5 "$MODEL_ARCHIVE_URL" -o "$CACHE_ARCHIVE"
      curl -fL --retry 5 --retry-delay 5 "$MODEL_SHA_URL" -o "$CACHE_SHA"
    else
      log "Downloading $S3_PREFIX/${BUNDLE_NAME}.tar.zst (requires read-only S3 IAM credentials)"
      aws s3 cp "$S3_PREFIX/${BUNDLE_NAME}.tar.zst" "$CACHE_ARCHIVE" --region "$AWS_DEFAULT_REGION"
      aws s3 cp "$S3_PREFIX/${BUNDLE_NAME}.tar.zst.sha256" "$CACHE_SHA" --region "$AWS_DEFAULT_REGION"
    fi
  elif [[ ! -s "$CACHE_SHA" ]]; then
    if [[ -n "${MODEL_SHA_URL:-}" ]]; then
      curl -fL --retry 5 "$MODEL_SHA_URL" -o "$CACHE_SHA"
    else
      aws s3 cp "$S3_PREFIX/${BUNDLE_NAME}.tar.zst.sha256" "$CACHE_SHA" --region "$AWS_DEFAULT_REGION"
    fi
  fi
  log 'Verifying downloaded archive SHA256 before extracting'
  (cd "$ARCHIVE_DIR" && sha256sum -c "$(basename "$CACHE_SHA")") || die 'Archive digest mismatch.'
  log 'Extracting prebuilt checkpoints and MuseTalk source (no Hugging Face downloads)'
  tar --zstd -xf "$CACHE_ARCHIVE" -C "$WORKSPACE" --no-same-owner
  models_complete || die 'Extracted archive missing expected checkpoints or manifest.'
  log 'Verifying all inner checkpoint/file digests'
  (cd "$BUNDLE_ROOT" && sha256sum -c manifest/SHA256SUMS >/dev/null) || die 'Inner bundle integrity validation failed.'
  log 'Deleting downloaded 9.9 GiB compressed file after verified extraction'
  rm -f "$CACHE_ARCHIVE" "$CACHE_SHA"
fi

log 'Ensuring uv Python bootstrap exists'
BOOT_VENV="${WORKSPACE}/rosely-bootstrap-venv"
if [[ ! -x "$BOOT_VENV/bin/uv" ]]; then
  python3 -m venv "$BOOT_VENV"
  "$BOOT_VENV/bin/python" -m pip install --upgrade pip uv
fi
UV="$BOOT_VENV/bin/uv"
"$UV" python install 3.10
ENVS="${WORKSPACE}/rosely-venvs"
mkdir -p "$ENVS"
create_env(){
  local name="$1"
  if [[ ! -x "$ENVS/$name/bin/python" ]]; then
    "$UV" venv --python 3.10 --seed "$ENVS/$name"
  fi
}
for env in asr tts muse; do create_env "$env"; done

install_asr(){
  local py="$ENVS/asr/bin/python"
  log 'Installing Qwen3-ASR packages in isolated Python 3.10 environment'
  "$py" -m pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  "$py" -m pip install --no-cache-dir 'qwen-asr==0.0.6' 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' \
    'python-multipart>=0.0.18' 'httpx>=0.28,<1'
}
install_tts(){
  local py="$ENVS/tts/bin/python"
  log 'Installing Chatterbox Turbo packages in isolated Python 3.10 environment'
  "$py" -m pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
  "$py" -m pip install --no-cache-dir 'chatterbox-tts==0.1.7' 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' \
    'python-multipart>=0.0.18' 'httpx>=0.28,<1'
  "$py" -c 'import perth; assert perth.PerthImplicitWatermarker is not None'
}
install_muse(){
  local py="$ENVS/muse/bin/python"
  log 'Installing upstream MuseTalk Python 3.10 / CUDA 11.8 dependencies'
  "$py" -m pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  "$py" -m pip install --no-cache-dir -r "$BUNDLE_ROOT/source/MuseTalk/requirements.txt"
  "$py" -m pip install --no-cache-dir openmim 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.30,<1' 'python-multipart>=0.0.18'
  "$ENVS/muse/bin/mim" install mmengine
  "$ENVS/muse/bin/mim" install 'mmcv==2.0.1'
  "$ENVS/muse/bin/mim" install 'mmdet==3.1.0'
  "$ENVS/muse/bin/mim" install 'mmpose==1.1.0'
}

if [[ ! -e "$ENVS/asr/.rosely_installed" ]]; then install_asr && touch "$ENVS/asr/.rosely_installed"; fi
if [[ ! -e "$ENVS/tts/.rosely_installed" ]]; then install_tts && touch "$ENVS/tts/.rosely_installed"; fi
if [[ ! -e "$ENVS/muse/.rosely_installed" ]]; then install_muse && touch "$ENVS/muse/.rosely_installed"; fi

log 'Checking service source code syntax'
for entry in asr_server.py tts_server.py voice_gateway.py avatar_server.py; do
  "$ENVS/asr/bin/python" -m py_compile "$APP_DIR/$entry"
done

log 'Writing persistent secure service configuration (existing keys are preserved)'
ENV_FILE="${WORKSPACE}/rosely-voice-avatar.env"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi
VOICE_API_KEY="${VOICE_API_KEY:-$(openssl rand -hex 32)}"
GATEWAY_API_KEY="${GATEWAY_API_KEY:-$(openssl rand -hex 32)}"
cat > "$ENV_FILE" <<EOF
VOICE_API_KEY=${VOICE_API_KEY}
GATEWAY_API_KEY=${GATEWAY_API_KEY}
BUNDLE_ROOT=${BUNDLE_ROOT}
ASR_MODEL_PATH=${BUNDLE_ROOT}/models/Qwen3-ASR-0.6B
TTS_MODEL_PATH=${BUNDLE_ROOT}/models/chatterbox-turbo
VOICES_DIR=${WORKSPACE}/rosely-voices
AVATAR_MEDIA_ROOT=${WORKSPACE}/rosely-avatar-data
MUSE_PYTHON=${ENVS}/muse/bin/python
ASR_PORT=8001
TTS_PORT=8002
AVATAR_PORT=8003
GATEWAY_PORT=8000
MUSE_BATCH_SIZE=${MUSE_BATCH_SIZE:-4}
EOF
chmod 600 "$ENV_FILE"
mkdir -p "$WORKSPACE/rosely-voices"

if [[ "${FETCH_BUILTIN_VOICES:-1}" == 1 ]]; then
  log 'Preserving original 28 Rosely English voice reference IDs (download from current public GitHub repo)'
  REFS="https://raw.githubusercontent.com/robert2398/rosely-vast-voice/main/voice_sample/english"
  for gender in female male; do
    for style in original anger calm confused enthused excited frustrated happy neutral sad shout surprised tired worried; do
      id="en_${gender}_${style}"
      destination="${WORKSPACE}/rosely-voices/${id}.wav"
      [[ -s "$destination" ]] && continue
      tmp="$(mktemp --suffix=.flac)"
      curl -fLsS --retry 4 "${REFS}/${gender}/${style}.flac" -o "$tmp" || die "Could not download voice reference: $id"
      ffmpeg -nostdin -hide_banner -loglevel error -y -i "$tmp" -ac 1 -ar 24000 "$destination"
      rm -f "$tmp"
    done
  done
fi

log 'Provisioned. Start via: sudo -E bash start_services.sh'
log "Configuration at $ENV_FILE. Do not commit this file or AWS secrets."
