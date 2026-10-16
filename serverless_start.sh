#!/usr/bin/env bash
set -Eeuo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${WORKSPACE}/rosely-vast-voice-avatar"
ENV_FILE="${WORKSPACE}/rosely-voice-avatar.env"
source "$ENV_FILE"
export VOICE_API_KEY GATEWAY_API_KEY BUNDLE_ROOT ASR_MODEL_PATH TTS_MODEL_PATH VOICES_DIR
export AVATAR_MEDIA_ROOT MUSE_PYTHON ASR_PORT TTS_PORT AVATAR_PORT GATEWAY_PORT MUSE_BATCH_SIZE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SERVERLESS_ADAPTER_PORT="${SERVERLESS_ADAPTER_PORT:-8004}"
mkdir -p "$WORKSPACE/rosely-logs" "$WORKSPACE/rosely-run"
# Existing launcher starts each service on internal ports and waits until healthy.
bash "$APP_DIR/start_services.sh"
cd "$APP_DIR"
adapter_pidfile="$WORKSPACE/rosely-run/serverless-adapter.pid"
if [[ -f "$adapter_pidfile" ]]; then
  old_pid="$(cat "$adapter_pidfile" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    if tr '\0' ' ' <"/proc/$old_pid/cmdline" | grep -q 'serverless_adapter'; then
      kill "$old_pid" || true
      sleep 2
    fi
  fi
fi
"$WORKSPACE/rosely-venvs/asr/bin/uvicorn" serverless_adapter:app \
  --host 127.0.0.1 --port "$SERVERLESS_ADAPTER_PORT" --workers 1 \
  > "$WORKSPACE/rosely-logs/serverless-adapter.log" 2>&1 &
adapter_pid=$!
printf '%s\n' "$adapter_pid" > "$adapter_pidfile"
trap 'kill "$adapter_pid" 2>/dev/null || true; rm -f "$adapter_pidfile"' EXIT
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${SERVERLESS_ADAPTER_PORT}/health" >/dev/null 2>&1; then break; fi
  sleep 2
done
curl -fsS "http://127.0.0.1:${SERVERLESS_ADAPTER_PORT}/health" >/dev/null
"$WORKSPACE/rosely-venvs/asr/bin/python" "$APP_DIR/worker.py"
