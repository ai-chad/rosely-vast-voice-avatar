# Rosely RTX 3090 – permanent SSH access and model provisioning patch

## Files to replace (repository root)

- `provision_serverless.sh` **required** – starts key-only `sshd` via Supervisor BEFORE cloning/downloading S3 models, regardless of Vast's `SERVERLESS` status.
- `provision_vast.sh` **required** – same AWS CLI v2, SHA256, chumpy fixes as current GitHub code, with a setuptools 80.10.2 pin before the Perth TTS import check.
- `serverless_start.sh` **recommended** – only launches `worker.py` when `SERVERLESS=true`; otherwise runs API adapter persistently for standalone debugging.
- `VAST_TEMPLATE_COMMAND.sh` **optional** – corrected example template command, 100 GB disk. Don't put keys in git.

## Required Vast template configuration

Select **Docker ENTRYPOINT**. Leave args and on-start command EMPTY.

Image: `vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310`
Disk: `100` GB (watch space)
Expose exactly `-p 3000:3000 -p 22:22` (8080 and 1111 are not required for SSH)
Keep your S3/GitHub values and custom `ROSELY_AVATAR_S3_*` keys.

Add/modify these env vars for a **managed Serverless workgroup debug worker**:

```
SERVERLESS=true
ROSELY_ENABLE_SSH=true
SUPERVISOR_SKIP_PYWORKER=true
```

`SUPERVISOR_SKIP_PYWORKER=true` prevents Vast's built-in generic PyWorker from racing with this repository's custom `worker.py`.
`ROSELY_ENABLE_SSH=true` installs SSH even if Vast forces `SERVERLESS=true` (which otherwise disables the portal/Jupyter).

For a **standalone instance not part of Vast Serverless**, set `SERVERLESS=false`; the custom `serverless_start.sh` then leaves the APIs active without running the Vast worker.

### SSH public key

Add your Windows *public* key to **Vast Account > SSH Keys**, BEFORE launching the worker. The Vast base-image entrypoint normally injects it into `/root/.ssh/authorized_keys`. The patch validates it before any 9.9 GiB model download. It optionally accepts the Vast-provided `SSH_PUBLIC_KEY` env var if authorized_keys is empty. It NEVER enables passwords.

Do **not** put your private SSH key in any template. Do not add S3 secrets to GitHub.

### After you push the files

Create a NEW instance/worker from the updated template. Old workers still have the previously cloned scripts and the Docker port mappings from their original creation; pushing GitHub doesn't alter them.

In Vast's **Terminal Connections** popup, copy the CURRENT host and mapped SSH port. It changes between workers. In Windows PowerShell:

```powershell
Test-NetConnection <HOST_FROM_VAST> -Port <PUBLIC_PORT_FROM_VAST>
ssh -p <PUBLIC_PORT_FROM_VAST> root@<HOST_FROM_VAST>
```

Use `/root/.ssh/authorized_keys` with your Vast-registered PUBLIC key. If SSH key auth fails, choose the correct private key (`ssh -i "$env:USERPROFILE\.ssh\id_ed25519" ...`).

### Once connected

```bash
supervisorctl status rosely-debug-sshd
supervisorctl status rosely-serverless
ss -ltnp | grep ':22'
nvidia-smi
df -h /workspace
tail -n 80 /workspace/rosely-logs/tts.log
```

For optional JupyterLab in managed Serverless mode, Vast's built-in Jupyter is deliberately disabled; do not assume port 8080 works. Once SSH works, you may start Jupyter manually bound to localhost and forward its port through SSH. The custom SSH patch does NOT automatically expose Jupyter to the public internet.

### Boundaries / tests

Bash syntax has been checked for each modified script. The install/boot behavior is grounded in Vast's published base-image source but HAS NOT been verified on your rented host. Network policy and GPU service readiness still require live tests. This patch should eliminate the known `Connection refused` case caused by *no sshd process*, provided Vast maps internal TCP 22 to an externally reachable port and the account's public key is injected.
