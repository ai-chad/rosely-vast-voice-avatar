#!/usr/bin/env bash
# Vast base-image PROVISIONING_SCRIPT entrypoint. Do not run on existing RTX 5060.
set -Eeuo pipefail
export WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${PYWORKER_REPO:?Set PYWORKER_REPO to your new public GitHub repository}"
REF="${PYWORKER_REF:-main}"
APP_DIR="$WORKSPACE/rosely-vast-voice-avatar"
mkdir -p "$WORKSPACE"
if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --depth 1 --branch "$REF" "$REPO" "$APP_DIR"
fi
cd "$APP_DIR"
# Model archive, voice checkpoints and MuseTalk environments are installed by the
# existing provisioner. This fetches from S3 and checks the archive SHA256.
bash provision_vast.sh
"$WORKSPACE/rosely-venvs/asr/bin/python" -m pip install --no-cache-dir -r requirements.txt
chmod +x serverless_start.sh
# Supervisor is included in Vast base-image; run one manager for server+PyWorker.
cat > /etc/supervisor/conf.d/rosely-serverless.conf <<EOF
[program:rosely-serverless]
environment=PROC_NAME="%(program_name)s"
command=/bin/bash ${APP_DIR}/serverless_start.sh
directory=${APP_DIR}
autostart=true
autorestart=true
startsecs=10
startretries=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=30
stdout_logfile=/dev/stdout
stdout_logfile_maxbytes=0
redirect_stderr=true
EOF
supervisorctl reread
supervisorctl update
printf '\n[Rosely] Serverless provisioning installed. Check supervisor: supervisorctl status rosely-serverless\n'
