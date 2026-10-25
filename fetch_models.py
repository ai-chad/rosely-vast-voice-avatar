"""Download every runtime checkpoint straight from Hugging Face into BUNDLE_ROOT.

Revisions are pinned and every file is checked against models.sha256, so a fresh
worker gets exactly the bytes the services were validated with. Files that
already match are skipped, so re-running after a partial download resumes.

Usage: python fetch_models.py BUNDLE_ROOT [--ditto]
  --ditto fetches the Ditto TensorRT set instead (ditto_models.sha256).
Token: HUGGINGFACE_TOKEN (or HF_TOKEN). All repos are public; the token avoids
anonymous rate limits. It is never printed.
"""
import hashlib
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

# repo, pinned revision, local directory (relative to BUNDLE_ROOT), files in the repo
SOURCES = [
    ("Qwen/Qwen3-ASR-0.6B", "5eb144179a02acc5e5ba31e748d22b0cf3e303b0", "models/Qwen3-ASR-0.6B", [
        "chat_template.json", "config.json", "generation_config.json", "merges.txt", "model.safetensors",
        "preprocessor_config.json", "tokenizer_config.json", "vocab.json"]),
    ("ResembleAI/chatterbox-turbo", "749d1c1a46eb10492095d68fbcf55691ccf137cd", "models/chatterbox-turbo", [
        "added_tokens.json", "conds.pt", "merges.txt", "s3gen.safetensors", "s3gen_meanflow.safetensors",
        "special_tokens_map.json", "t3_turbo_v1.safetensors", "t3_turbo_v1.yaml", "tokenizer_config.json",
        "ve.safetensors", "vocab.json"]),
]
# Ditto talking-head engine (TensorRT engines built for Ampere and newer).
DITTO_ENGINES = ["appearance_extractor_fp16", "blaze_face_fp16", "decoder_fp16", "face_mesh_fp16", "hubert_fp32",
                 "insightface_det_fp16", "landmark106_fp16", "landmark203_fp16", "lmdm_v0.4_hubert_fp32",
                 "motion_extractor_fp32", "stitch_network_fp16", "warp_network_fp16"]
DITTO_SOURCES = [
    ("digital-avatar/ditto-talkinghead", "e4a2f60328ee7c32af585ac4b3cce299e4c8e254", "models/ditto", [
        "ditto_cfg/v0.4_hubert_cfg_trt.pkl", "ditto_cfg/v0.4_hubert_cfg_trt_online.pkl",
        *(f"ditto_trt_Ampere_Plus/{e}.engine" for e in DITTO_ENGINES)]),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(bundle_root, ditto=False):
    sources = DITTO_SOURCES if ditto else SOURCES
    root = Path(bundle_root)
    manifest = Path(__file__).with_name("ditto_models.sha256" if ditto else "models.sha256")
    expected = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            expected[rel] = digest
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None
    print(f"[models] Hugging Face token {'provided' if token else 'not set (anonymous)'}", flush=True)

    listed = set()
    for repo, revision, local_dir, files in sources:
        for name in files:
            rel = f"{local_dir}/{name}"
            listed.add(rel)
            target = root / rel
            if target.is_file() and sha256(target) == expected[rel]:
                print(f"[models] ok (cached) {rel}", flush=True)
                continue
            print(f"[models] downloading {repo}@{revision[:8]} {name}", flush=True)
            downloaded = hf_hub_download(repo, name, revision=revision, token=token, local_dir=root / local_dir)
            if Path(downloaded).resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(downloaded, target)
            actual = sha256(target)
            if actual != expected[rel]:
                target.unlink()
                sys.exit(f"[models] SHA256 mismatch for {rel}: got {actual}, expected {expected[rel]}")
            print(f"[models] verified {rel}", flush=True)

    missing = set(expected) - listed
    if missing:
        sys.exit(f"[models] models.sha256 lists files with no source: {sorted(missing)}")
    # hf_hub_download keeps per-directory download metadata; it is not needed at runtime.
    for _, _, local_dir, _ in sources:
        shutil.rmtree(root / local_dir / ".cache", ignore_errors=True)
    (root / "manifest").mkdir(exist_ok=True)
    shutil.copyfile(manifest, root / "manifest" / ("DITTO_SHA256SUMS" if ditto else "SHA256SUMS"))
    print(f"[models] all {len(expected)} files verified in {root}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--ditto"]
    if len(args) != 1:
        sys.exit("usage: python fetch_models.py BUNDLE_ROOT [--ditto]")
    main(args[0], ditto="--ditto" in sys.argv[1:])
