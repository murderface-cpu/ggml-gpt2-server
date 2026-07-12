#!/usr/bin/env python3
"""
Downloads a supported GGUF model into models/<name>/model.gguf, for
llama.cpp's server to load.

Usage:
    python scripts/get_model.py --list
    python scripts/get_model.py qwen2.5-1.5b

Requires the huggingface_hub package (pip install huggingface_hub).

llama.cpp can also pull a model directly with no separate download step,
via `-hf <repo>:<quant>` instead of `-m <path>` (see the server's --help).
This script exists for the explicit download-then-mount workflow used by
docker-compose.yml, which avoids depending on llama.cpp's HF-cache
directory persisting correctly across container restarts.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = {
    "qwen2.5-1.5b": {
        "description": (
            "Qwen2.5-1.5B-Instruct. Apache 2.0, 32K native context, RoPE + GQA. "
            "The default: strong quality for its size, commercially usable."
        ),
        "hf_repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "hf_filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "license": "Apache-2.0",
    },
    "qwen2.5-0.5b": {
        "description": (
            "Qwen2.5-0.5B-Instruct. Same family/license/context as qwen2.5-1.5b, "
            "smaller and faster; a lighter option for a small VPS."
        ),
        "hf_repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "hf_filename": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "license": "Apache-2.0",
    },
}


def list_models():
    print("Available models:\n")
    for name, info in MODELS.items():
        print(f"  {name}")
        print(f"    {info['description']}")
        print(f"    license={info['license']}")
        print()


def get_model(name, info):
    dest_dir = REPO_ROOT / "models" / name
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_path = dest_dir / "model.gguf"

    if model_path.exists():
        print(f"{model_path} already exists, skipping download.")
        return model_path

    from huggingface_hub import hf_hub_download

    print(f"Downloading {info['hf_repo']}/{info['hf_filename']} ...")
    downloaded = hf_hub_download(repo_id=info["hf_repo"], filename=info["hf_filename"])

    # hf_hub_download caches into ~/.cache/huggingface; link/copy it to our
    # own predictable path so docker-compose's MODEL_PATH stays simple
    import shutil
    shutil.copyfile(downloaded, model_path)

    return model_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", help="model name, see --list")
    parser.add_argument("--list", action="store_true", help="list available models")
    args = parser.parse_args()

    if args.list or not args.name:
        list_models()
        sys.exit(0 if args.list else 1)

    if args.name not in MODELS:
        print(f"Unknown model: {args.name}", file=sys.stderr)
        list_models()
        sys.exit(1)

    info = MODELS[args.name]
    model_path = get_model(args.name, info)

    rel_path = model_path.relative_to(REPO_ROOT)
    print("\nDone. To use this model, set in .env:\n")
    print(f"  MODEL_PATH=/models/{args.name}/model.gguf")
    print("\nthen:")
    print("  docker compose up -d")
    print(f"\n(local file: {rel_path.as_posix()})")


if __name__ == "__main__":
    main()
