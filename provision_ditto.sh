#!/usr/bin/env bash
# Ditto talking-head engine (video replies and live calls).
# Called by provision_vast.sh after the base install; safe to re-run.
set -Eeuo pipefail
log(){ printf '\n[%s] %s\n' "$(date -Is)" "$*"; }
die(){ printf '[FATAL] %s\n' "$*" >&2; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
BUNDLE_ROOT="${BUNDLE_ROOT:-${WORKSPACE}/rosely-voice-v1}"
ENVS="${WORKSPACE}/rosely-venvs"
UV="${WORKSPACE}/rosely-bootstrap-venv/bin/uv"
BOOT_PY="${WORKSPACE}/rosely-bootstrap-venv/bin/python"
DITTO_COMMIT="c3e47eee2e626500017a0556b470d6d4182f85e8"
DITTO_ROOT="${BUNDLE_ROOT}/source/ditto-talkinghead"
VENV="${ENVS}/ditto"
CUDNN8_DIR="${VENV}/cudnn8"
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"

if [[ ! -s "$DITTO_ROOT/stream_pipeline_offline.py" ]]; then
  log "Fetching Ditto source ${DITTO_COMMIT:0:7} from GitHub"
  mkdir -p "$DITTO_ROOT"
  curl -fsSL --retry 5 --retry-delay 3 \
    "https://codeload.github.com/antgroup/ditto-talkinghead/tar.gz/${DITTO_COMMIT}" |
    tar xz -C "$DITTO_ROOT" --strip-components=1
fi

log 'Downloading Ditto TensorRT checkpoints from Hugging Face (pinned, SHA256-verified)'
"$BOOT_PY" "$APP_DIR/fetch_models.py" "$BUNDLE_ROOT" --ditto

if [[ ! -e "$VENV/.rosely_installed" ]]; then
  log 'Installing Ditto Python 3.10 environment (torch 2.5.1/cu121 + TensorRT 8.6.1)'
  [[ -x "$VENV/bin/python" ]] || "$UV" venv --python 3.10 --seed "$VENV"
  "$VENV/bin/python" -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
  # Locked from the environment validated on the RTX 3090. TensorRT 8.6.1 wheels live on NVIDIA's index.
  "$VENV/bin/python" -m pip install --no-cache-dir -r "$APP_DIR/requirements-ditto.txt" \
    --extra-index-url https://pypi.nvidia.com
  # The tensorrt==8.6.1 meta-package cannot build in isolation; it only re-exports tensorrt_bindings.
  site="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  mkdir -p "$site/tensorrt"
  cat > "$site/tensorrt/__init__.py" <<'EOF'
# Stand-in for the tensorrt==8.6.1 meta-package: expose tensorrt_bindings as tensorrt.
import sys
import tensorrt_bindings as _trt
sys.modules[__name__] = _trt
EOF
  # TensorRT 8.6.1 needs cuDNN 8; torch keeps cuDNN 9. Keep 8 private to the Ditto service.
  tmp="$(mktemp -d)"
  "$VENV/bin/python" -m pip download -q --no-deps --only-binary=:all: nvidia-cudnn-cu12==8.9.7.29 -d "$tmp"
  rm -rf "$CUDNN8_DIR" && mkdir -p "$CUDNN8_DIR"
  "$VENV/bin/python" -m zipfile -e "$tmp"/nvidia_cudnn_cu12-8.9.7.29-*.whl "$CUDNN8_DIR"
  rm -rf "$tmp"
  LD_LIBRARY_PATH="$CUDNN8_DIR/nvidia/cudnn/lib" "$VENV/bin/python" -c \
    'import torch, tensorrt; assert torch.cuda.is_available(); print("Ditto env ok: torch", torch.__version__, "tensorrt", tensorrt.__version__)'
  touch "$VENV/.rosely_installed"
fi
# Live-call packages came later; add them to environments provisioned before that.
if ! "$VENV/bin/python" -c 'import livekit.rtc, webrtcvad' 2>/dev/null; then
  log 'Adding live call packages (LiveKit client, voice activity detection) to the Ditto environment'
  # shellcheck disable=SC2046
  "$VENV/bin/python" -m pip install --no-cache-dir     $(grep -E '^(aiofiles|livekit|protobuf|types-protobuf|webrtcvad-wheels)==' "$APP_DIR/requirements-ditto.txt")
fi

log 'Enabling Ditto in the service configuration'
ENV_FILE="${WORKSPACE}/rosely-voice-avatar.env"
[[ -f "$ENV_FILE" ]] || die "Missing $ENV_FILE; run provision_vast.sh first."
sed -i '/^DITTO_/d; /^LIVE_CALL_ENABLED=/d' "$ENV_FILE"
cat >> "$ENV_FILE" <<EOF
LIVE_CALL_ENABLED=${LIVE_CALL_ENABLED:-1}
DITTO_PORT=8005
DITTO_ROOT=${DITTO_ROOT}
DITTO_MODELS=${BUNDLE_ROOT}/models/ditto
DITTO_PYTHON_ENV=${VENV}
DITTO_CUDNN8_DIR=${CUDNN8_DIR}/nvidia/cudnn/lib
EOF
log 'Ditto provisioned. It starts with the other services (start_services.sh).'
