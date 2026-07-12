#!/usr/bin/env python3
"""
Downloads and converts a supported GPT-2-architecture model to the ggml
format gpt-2-server loads, into models/<name>/ggml-model.bin.

Usage:
    python scripts/get_model.py --list
    python scripts/get_model.py lamini-774m

Requires the packages in requirements.txt for the "lamini-*" models
(huggingface_hub, transformers, torch, numpy). The "gpt2-*-base" models are
pre-converted downloads and need no Python ML dependencies.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "gpt2-117m-base": {
        "description": "Original GPT-2 117M. Not instruction-tuned, fastest, least reliable for chat.",
        "chat_template": "raw",
        "license": "MIT (original OpenAI GPT-2 weights)",
        "kind": "prebuilt-ggml",
        "size_arg": "117M",
    },
    "gpt2-345m-base": {
        "description": "Original GPT-2 345M. Not instruction-tuned.",
        "chat_template": "raw",
        "license": "MIT (original OpenAI GPT-2 weights)",
        "kind": "prebuilt-ggml",
        "size_arg": "345M",
    },
    "lamini-124m": {
        "description": "GPT-2 124M fine-tuned on 2.58M instructions. Small and fast; the default.",
        "chat_template": "alpaca",
        "license": "CC-BY-NC-4.0 (non-commercial)",
        "kind": "hf-convert",
        "hf_repo": "MBZUAI/LaMini-GPT-124M",
    },
    "lamini-774m": {
        "description": (
            "GPT-2 Large 774M fine-tuned on 2.58M instructions. The LaMini-LM "
            "paper's own pick for best overall size/performance trade-off. "
            "Slower and needs more RAM than lamini-124m (roughly 1.6GB for "
            "weights plus ~360MB per concurrent request slot at CTX_SIZE=1024)."
        ),
        "chat_template": "alpaca",
        "license": "CC-BY-NC-4.0 (non-commercial)",
        "kind": "hf-convert",
        "hf_repo": "MBZUAI/LaMini-GPT-774M",
    },
}


def list_models():
    print("Available models:\n")
    for name, info in MODELS.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    chat_template={info['chat_template']}  license={info['license']}")
        print()


def get_prebuilt_ggml(name, info):
    script = REPO_ROOT / "examples" / "gpt-2" / "download-ggml-model.sh"
    dest = REPO_ROOT / "models" / f"gpt-2-{info['size_arg']}"
    subprocess.run(["bash", str(script), info["size_arg"]], cwd=REPO_ROOT, check=True)
    return dest / "ggml-model.bin"


def get_hf_convert(name, info):
    dest_dir = REPO_ROOT / "models" / name
    model_bin = dest_dir / "ggml-model.bin"

    if model_bin.exists():
        print(f"{model_bin} already exists, skipping download/convert.")
        return model_bin

    from huggingface_hub import snapshot_download

    print(f"Downloading {info['hf_repo']} ...")
    snapshot_download(
        repo_id=info["hf_repo"],
        local_dir=str(dest_dir),
        allow_patterns=["*.json", "*.txt", "*.bin"],
    )

    print("Converting to ggml format ...")
    convert_script = REPO_ROOT / "examples" / "gpt-2" / "convert-h5-to-ggml.py"
    subprocess.run([sys.executable, str(convert_script), str(dest_dir)], check=True)

    return model_bin


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="model name, see --list")
    parser.add_argument("--list", action="store_true", help="list available models")
    args = parser.parse_args()

    if args.list or not args.name:
        list_models()
        if not args.name:
            sys.exit(0 if args.list else 1)

    if args.name not in MODELS:
        print(f"Unknown model: {args.name}", file=sys.stderr)
        list_models()
        sys.exit(1)

    info = MODELS[args.name]

    if info["kind"] == "prebuilt-ggml":
        model_path = get_prebuilt_ggml(args.name, info)
    else:
        model_path = get_hf_convert(args.name, info)

    rel_path = model_path.relative_to(REPO_ROOT)
    print("\nDone. To use this model, set:\n")
    print(f"  MODEL_PATH={rel_path.as_posix()}")
    print(f"  CHAT_TEMPLATE={info['chat_template']}")
    print("\nFor docker compose: put those two lines in your .env file, then")
    print("  docker compose up -d")
    print("\nFor a local build:")
    print(f"  ./build/bin/gpt-2-server -m {rel_path.as_posix()} --chat-template {info['chat_template']}")


if __name__ == "__main__":
    main()
