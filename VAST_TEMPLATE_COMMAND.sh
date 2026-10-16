#!/usr/bin/env bash
# Requires Vast CLI authenticated: vastai set api-key <key>
# This creates an instance for testing. Attach the new template to a Vast Serverless endpoint/worker group separately.
set -Eeuo pipefail
: "${OFFER_ID:?Export OFFER_ID (your RTX 3090 offer ID)}"
GITHUB_REPO="${GITHUB_REPO:-https://github.com/ai-chad/rosely-vast-voice-avatar}"
: "${ROSELY_AVATAR_S3_ACCESS_KEY_ID:?Provide dedicated read-only key as an environment variable}"
: "${ROSELY_AVATAR_S3_SECRET_ACCESS_KEY:?Provide dedicated read-only secret as an environment variable}"
REPO_RAW="https://raw.githubusercontent.com/${GITHUB_REPO#https://github.com/}/main"
# Optional temporary STS session token, only when the access key requires one.
SESSION_TOKEN_OPT=""
if [[ -n "${ROSELY_AVATAR_S3_SESSION_TOKEN:-}" ]]; then
  SESSION_TOKEN_OPT=" -e ROSELY_AVATAR_S3_SESSION_TOKEN=${ROSELY_AVATAR_S3_SESSION_TOKEN}"
fi
vastai create instance "$OFFER_ID" \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env "-p 3000:3000 -e SERVERLESS=true -e PYWORKER_REPO=${GITHUB_REPO} -e PYWORKER_REF=main -e PROVISIONING_SCRIPT=${REPO_RAW}/provision_serverless.sh -e MODEL_S3_URI=s3://rosely-infrastructure/serverless/voice-avatar/rosely-voice-musetalk-v1.tar.zst -e ROSELY_AVATAR_S3_ACCESS_KEY_ID=${ROSELY_AVATAR_S3_ACCESS_KEY_ID} -e ROSELY_AVATAR_S3_SECRET_ACCESS_KEY=${ROSELY_AVATAR_S3_SECRET_ACCESS_KEY} -e ROSELY_AVATAR_S3_REGION=${ROSELY_AVATAR_S3_REGION:-us-east-1}${SESSION_TOKEN_OPT} -e FETCH_BUILTIN_VOICES=1 -e MUSE_BATCH_SIZE=4 -e GATEWAY_PORT=8000 -e SERVERLESS_ADAPTER_PORT=8004" \
  --disk 150
