#!/usr/bin/env bash
set -Eeuo pipefail
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="${WORKSPACE}/rosely-voice-avatar.env"
[[ -f "$ENV_FILE" ]] || { echo "Run provision_vast.sh first" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
export VOICE_API_KEY GATEWAY_API_KEY BUNDLE_ROOT ASR_MODEL_PATH TTS_MODEL_PATH VOICES_DIR \
  AVATAR_MEDIA_ROOT MUSE_PYTHON ASR_PORT TTS_PORT AVATAR_PORT GATEWAY_PORT MUSE_BATCH_SIZE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_MODULE_LOADING=LAZY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${WORKSPACE}/rosely-hf-cache"
LOGS="${WORKSPACE}/rosely-logs"
RUN="${WORKSPACE}/rosely-run"
mkdir -p "$LOGS" "$RUN" "$AVATAR_MEDIA_ROOT" "$HF_HOME"
cd "$APP_DIR"
stop_pid(){
  local pidfile="$1" pid
  [[ -s "$pidfile" ]] || return 0
  pid="$(cat "$pidfile")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    # Refuse to kill an unrelated process if an old PID was recycled.
    if tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q 'uvicorn'; then kill "$pid" || true; fi
  fi
  rm -f "$pidfile"
}
wait_health(){
  local url="$1" logfile="$2" name="$3"
  for _ in $(seq 1 120); do
    if curl -fsS "$url" | grep -q '"ok":true'; then
      echo "[Rosely] $name ready: $url"
      return 0
    fi
    sleep 2
  done
  echo "[ERROR] $name failed; tail of $logfile:" >&2
  tail -n 120 "$logfile" >&2 || true
  exit 1
}
for name in gateway avatar tts asr; do stop_pid "$RUN/$name.pid"; done
sleep 2
start_api(){
  local name="$1" envname="$2" module="$3" bind="$4" port="$5"
  local uvicorn="${WORKSPACE}/rosely-venvs/${envname}/bin/uvicorn"
  echo "[Rosely] Starting $name on $bind:$port"
  nohup "$uvicorn" "${module}:app" --host "$bind" --port "$port" --workers 1 \
    >"$LOGS/$name.log" 2>&1 &
  echo "$!" >"$RUN/$name.pid"
  wait_health "http://127.0.0.1:${port}/health" "$LOGS/$name.log" "$name"
}
start_api asr asr asr_server 127.0.0.1 "$ASR_PORT"
start_api tts tts tts_server 127.0.0.1 "$TTS_PORT"
start_api avatar muse avatar_server 127.0.0.1 "$AVATAR_PORT"
start_api gateway asr voice_gateway 0.0.0.0 "$GATEWAY_PORT"
echo '[Rosely] All three model services and authenticated gateway are online.'
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv || true
