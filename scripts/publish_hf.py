#!/usr/bin/env python3
"""Publish ced.cpp GGUF quantizations + model card to a HuggingFace model repo.

Idempotent: creates the repo if missing and uploads each file (overwriting).

Usage:
    python scripts/publish_hf.py --repo mudler/ced-gguf \
        --files models/ced-*.gguf \
        --card scripts/hf_model_card.md
"""
import argparse
import os
import sys

from huggingface_hub import HfApi, create_repo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF model repo id, e.g. mudler/ced-base-gguf")
    ap.add_argument("--files", nargs="+", required=True, help="GGUF files to upload")
    ap.add_argument("--card", default="scripts/hf_model_card.md", help="model card -> README.md")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    print(f"== ensuring repo {args.repo}", file=sys.stderr)
    create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    if args.card and os.path.exists(args.card):
        print(f"== uploading model card {args.card} -> README.md", file=sys.stderr)
        api.upload_file(
            path_or_fileobj=args.card,
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Update model card",
        )

    for f in args.files:
        if not os.path.exists(f):
            print(f"!! missing file, skipping: {f}", file=sys.stderr)
            continue
        name = os.path.basename(f)
        print(f"== uploading {f} -> {name}", file=sys.stderr)
        api.upload_file(
            path_or_fileobj=f,
            path_in_repo=name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Add {name}",
        )

    print(f"== done: https://huggingface.co/{args.repo}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
