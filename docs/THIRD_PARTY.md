# Attribution and upstream notes

This repository's voice service code closely follows the user's existing `robert2398/rosely-vast-voice` code, with the new local model paths and authenticated gateway. Checkpoints are downloaded from Hugging Face at pinned revisions (`fetch_models.py`).

Upstream resources to review and respect:

- https://github.com/antgroup/ditto-talkinghead (video model)
- https://huggingface.co/digital-avatar/ditto-talkinghead
- https://github.com/livekit/python-sdks (live calls)
- https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- https://huggingface.co/ResembleAI/chatterbox-turbo
- https://github.com/vast-ai/pyworker (NOT used in this regular-instance deployment)

Always review checkpoint and third-party dependency licenses separately from their code repositories. The application is not a sexual-content compliance or age-verification system; upstream restrictions and relevant laws still apply.
