#!/usr/bin/env bash
# RTX 3090 Serverless template with OPT-IN SSH debug access.
# Always select Docker ENTRYPOINT (NOT SSH/Jupyter runtime) and leave entrypoint args blank.
# Usage: export OFFER_ID=...; export ROSELY_AVATAR_S3_ACCESS_KEY_ID=...;
#        export ROSELY_AVATAR_S3_SECRET_ACCESS_KEY=...; bash VAST_TEMPLATE_COMMAND.sh
# CAUTION: Vast CLI receives S3 secrets as instance environment variables.
set -Eeuo pipefail
: "${OFFER_ID:?Set OFFER_ID to your GPU offer ID}"
: "${ROSELY_AVATAR_S3_ACCESS_KEY_ID:?S3 key missing}"
: "${ROSELY_AVATAR_S3_SECRET_ACCESS_KEY:?S3 secret missing}"
GITHUB_REPO=https://github.com/ai-chad/rosely-vast-voice-avatar
RAW_REPO=https://raw.githubusercontent.com/ai-chad/rosely-vast-voice-avatar/main
STS_OPTION=''
if [[ -n "${ROSELY_AVATAR_S3_SESSION_TOKEN:-}" ]]; then
  STS_OPTION=" -e ROSELY_AVATAR_S3_SESSION_TOKEN=${ROSELY_AVATAR_S3_SESSION_TOKEN}"
fi
vastai create instance "$OFFER_ID" \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env "-p 3000:3000 -p 22:22 -e SERVERLESS=true -e ROSELY_ENABLE_SSH=true -e SUPERVISOR_SKIP_PYWORKER=true -e PYWORKER_REPO=${GITHUB_REPO} -e PYWORKER_REF=main -e PROVISIONING_SCRIPT=${RAW_REPO}/provision_serverless.sh -e MODEL_S3_URI=s3://rosely-infrastructure/serverless/voice-avatar/rosely-voice-musetalk-v1.tar.zst -e ROSELY_AVATAR_S3_ACCESS_KEY_ID=${ROSELY_AVATAR_S3_ACCESS_KEY_ID} -e ROSELY_AVATAR_S3_SECRET_ACCESS_KEY=${ROSELY_AVATAR_S3_SECRET_ACCESS_KEY} -e ROSELY_AVATAR_S3_REGION=${ROSELY_AVATAR_S3_REGION:-us-east-1}${STS_OPTION} -e MUSE_BATCH_SIZE=4 -e FETCH_BUILTIN_VOICES=1 -e GATEWAY_PORT=8000 -e SERVERLESS_ADAPTER_PORT=8004" \
  --disk 100
