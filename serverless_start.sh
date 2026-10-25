#!/usr/bin/env bash
set -Eeuo pipefail
WORKSPACE="${WORKSPACE:-/workspace}"
APP_DIR="${WORKSPACE}/rosely-vast-voice-avatar"
ENV_FILE="${WORKSPACE}/rosely-voice-avatar.env"
source "$ENV_FILE"
export VOICE_API_KEY GATEWAY_API_KEY BUNDLE_ROOT ASR_MODEL_PATH TTS_MODEL_PATH VOICES_DIR
export AVATAR_MEDIA_ROOT ASR_PORT TTS_PORT GATEWAY_PORT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SERVERLESS_ADAPTER_PORT="${SERVERLESS_ADAPTER_PORT:-8004}"

# Vast official pyworker bootstrap supplies these defaults. Our custom worker
# launches the SDK directly, so it must supply them itself.
export WORKER_PORT="${WORKER_PORT:-3000}"
export REPORT_ADDR="${REPORT_ADDR:-https://run.vast.ai}"

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
if [[ "${SERVERLESS:-false}" == "true" ]]; then
  # The metrics SDK reads VAST_TCP_PORT_${WORKER_PORT}, PUBLIC_IPADDR,
  # CONTAINER_ID, REPORT_ADDR. Never invent the actual mapped public port.
  for required in PUBLIC_IPADDR CONTAINER_ID; do
    if [[ -z "${!required:-}" ]]; then
      echo "[Rosely ERROR] Required Vast worker variable ${required} is not available" >&2
      exit 1
    fi
  done
  mapped_port_var="VAST_TCP_PORT_${WORKER_PORT}"
  if [[ -z "${!mapped_port_var:-}" ]]; then
    echo "[Rosely ERROR] ${mapped_port_var} missing. Check that port ${WORKER_PORT} is mapped in the Vast template." >&2
    exit 1
  fi
  echo "[Rosely] Starting Vast PyWorker on internal port ${WORKER_PORT} (public mapping exists)"
  "$WORKSPACE/rosely-venvs/asr/bin/python" "$APP_DIR/worker.py"
else
  echo '[Rosely] Standalone debug mode: APIs running; PyWorker intentionally disabled'
  wait "$adapter_pid"
fi
