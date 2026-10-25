#!/usr/bin/env bash
# RTX 3090 Serverless template with OPT-IN SSH debug access.
# Always select Docker ENTRYPOINT (NOT SSH/Jupyter runtime) and leave entrypoint args blank.
# Usage: export OFFER_ID=...; export HUGGINGFACE_TOKEN=hf_...; bash VAST_TEMPLATE_COMMAND.sh
# Models download from Hugging Face at boot (fetch_models.py). A read-only token
# avoids anonymous rate limits; Vast receives it as an instance environment variable.
set -Eeuo pipefail
: "${OFFER_ID:?Set OFFER_ID to your GPU offer ID}"
: "${HUGGINGFACE_TOKEN:?Set HUGGINGFACE_TOKEN to a read-only Hugging Face token}"
GITHUB_REPO=https://github.com/ai-chad/rosely-vast-voice-avatar
RAW_REPO=https://raw.githubusercontent.com/ai-chad/rosely-vast-voice-avatar/main
vastai create instance "$OFFER_ID" \
  --image vastai/base-image:cuda-12.4.1-cudnn-devel-ubuntu22.04-py310 \
  --env "-p 3000:3000 -p 22:22 -e SERVERLESS=true -e ROSELY_ENABLE_SSH=true -e SUPERVISOR_SKIP_PYWORKER=true -e PYWORKER_REPO=${GITHUB_REPO} -e PYWORKER_REF=main -e PROVISIONING_SCRIPT=${RAW_REPO}/provision_serverless.sh -e HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN} -e FETCH_BUILTIN_VOICES=1 -e GATEWAY_PORT=8000 -e SERVERLESS_ADAPTER_PORT=8004" \
  --disk 100
