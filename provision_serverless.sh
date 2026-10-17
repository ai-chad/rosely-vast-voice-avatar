#!/usr/bin/env bash
# Vast base-image PROVISIONING_SCRIPT entrypoint (RTX 3090). Independent of old RTX 5060.
set -Eeuo pipefail
export WORKSPACE="${WORKSPACE:-/workspace}"
REPO="${PYWORKER_REPO:-https://github.com/ai-chad/rosely-vast-voice-avatar}"
REF="${PYWORKER_REF:-main}"
APP_DIR="$WORKSPACE/rosely-vast-voice-avatar"
log(){ printf '[Rosely boot] %s\n' "$*"; }
die(){ printf '[Rosely boot ERROR] %s\n' "$*" >&2; exit 1; }

# Vast launches supervisord before the custom provisioning script. Its
# Docker ENTRYPOINT mode DOES NOT automatically install an SSH daemon.
# This is intentionally before git clone, S3 download, and pip installs so
# that an interactive shell is available throughout long provisioning.
configure_debug_ssh(){
  [[ "${ROSELY_ENABLE_SSH:-false}" == "true" ]] || { log 'SSH bootstrap disabled (ROSELY_ENABLE_SSH != true)'; return 0; }
  [[ "$(id -u)" == 0 ]] || die 'SSH bootstrap requires root; Vast provisioning should run as root.'
  log 'Preparing key-only SSH on container port 22 before any model download'

  # Vast populates /root/.ssh/authorized_keys for SSH-enabled instances.
  # Optionally fall back to the public key Vast supplies as SSH_PUBLIC_KEY.
  install -d -m 700 /root/.ssh
  touch /root/.ssh/authorized_keys
  if [[ ! -s /root/.ssh/authorized_keys && -n "${SSH_PUBLIC_KEY:-}" ]]; then
    printf '%s\n' "$SSH_PUBLIC_KEY" >> /root/.ssh/authorized_keys
  fi
  chmod 600 /root/.ssh/authorized_keys
  chown root:root /root/.ssh /root/.ssh/authorized_keys
  [[ -s /root/.ssh/authorized_keys ]] || die 'No root authorized SSH key. Add your public key in Vast > Account > SSH Keys before launching. No passwords are permitted.'
  # ssh-keygen may not exist until openssh-server is installed.
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends openssh-server
  ssh-keygen -lf /root/.ssh/authorized_keys >/dev/null 2>&1 || die 'Root authorized_keys does not contain a valid SSH public key.'
  install -d -m 755 /run/sshd
  ssh-keygen -A >/dev/null

  # All restrictions are explicit in the sshd invocation, rather than relying
  # on the contents/order of Ubuntu /etc/ssh/sshd_config.d files.
  local -a ssh_opts=(
    -o PermitRootLogin=prohibit-password
    -o PasswordAuthentication=no
    -o KbdInteractiveAuthentication=no
    -o PubkeyAuthentication=yes
    -o AuthorizedKeysFile=.ssh/authorized_keys
    -o AllowTcpForwarding=yes
    -o PermitEmptyPasswords=no
  )
  /usr/sbin/sshd -t -f /etc/ssh/sshd_config "${ssh_opts[@]}" || die 'sshd configuration test failed'

  cat > /etc/supervisor/conf.d/rosely-debug-sshd.conf <<'CONF'
[program:rosely-debug-sshd]
command=/usr/sbin/sshd -D -e -f /etc/ssh/sshd_config -o PermitRootLogin=prohibit-password -o PasswordAuthentication=no -o KbdInteractiveAuthentication=no -o PubkeyAuthentication=yes -o AuthorizedKeysFile=.ssh/authorized_keys -o AllowTcpForwarding=yes -o PermitEmptyPasswords=no
user=root
autostart=true
autorestart=true
startsecs=2
startretries=10
stopsignal=TERM
stopwaitsecs=10
stdout_logfile=/workspace/rosely-logs/rosely-serverless-supervisor.log
stdout_logfile_maxbytes=20MB
stdout_logfile_backups=3
redirect_stderr=true
CONF
  supervisorctl reread
  supervisorctl update
  local i
  for i in $(seq 1 20); do
    if supervisorctl status rosely-debug-sshd 2>/dev/null | grep -q 'RUNNING'; then
      log 'SSH supervisor service RUNNING; use the NEW Vast Connect > Terminal Connections host and public port (22 is internal).'
      return 0
    fi
    sleep 1
  done
  supervisorctl status rosely-debug-sshd || true
  die 'sshd did not reach RUNNING state. Check Vast instance logs before provisioning.'
}

mkdir -p "$WORKSPACE"
configure_debug_ssh

# Clone current code only after SSH is reachable so pip/S3 failures are
# debuggable without having to destroy/recreate the GPU instance.
if [[ ! -d "$APP_DIR/.git" ]]; then
  git clone --depth 1 --branch "$REF" "$REPO" "$APP_DIR"
else
  log 'Existing cloned repository found; using its checked-out files.'
fi
cd "$APP_DIR"
bash provision_vast.sh
"$WORKSPACE/rosely-venvs/asr/bin/python" -m pip install --no-cache-dir -r requirements.txt
chmod +x serverless_start.sh
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
log 'Model provisioning and service configuration complete.'
supervisorctl status rosely-serverless || true
if [[ "${ROSELY_ENABLE_SSH:-false}" == true ]]; then supervisorctl status rosely-debug-sshd || true; fi
