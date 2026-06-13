from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Sequence

from .client import resolve_api_key
from .probes import OpenAIProbeService, format_timestamp

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(description="Standalone OpenAI probes")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "vector-stores",
        help="List vector stores and attached files",
    )
    subparsers.add_parser(
        "files",
        help="List uploaded OpenAI files",
    )

    vision = subparsers.add_parser(
        "vision-probe",
        help="Upload a file and probe vision until it succeeds",
    )
    vision.add_argument("--file-path", required=True)
    vision.add_argument("--model", default="gpt-5-mini")
    vision.add_argument(
        "--prompt",
        default=(
            "Analyze the attached image and return a concise "
            "procurement-focused summary. Keep it brief, visible-only, "
            "and do not guess any hidden details."
        ),
    )
    vision.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 means retry forever",
    )

    attach = subparsers.add_parser(
        "attach-probe",
        help="Create a temporary vector store and attach an uploaded file",
    )
    attach.add_argument("--file-path", required=True)
    attach.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 means retry forever",
    )

    return parser


async def _run_async(args: argparse.Namespace) -> int:
    service = OpenAIProbeService(api_key=resolve_api_key())

    if args.command == "vector-stores":
        vector_stores = await service.list_vector_stores()
        print(f"Vector store count: {len(vector_stores)}")
        for store in vector_stores:
            print(
                f"Vector store: id={store.vector_store_id} "
                f"name={store.name!r} created_at={format_timestamp(store.created_at)} "
                f"attached_file_count={store.attached_file_count}",
            )
        if vector_stores:
            print("Summary:")
            for store in vector_stores:
                print(f"- vector_store_name={store.name!r}")
                print(f"  - vector_store_id={store.vector_store_id}")
                print(f"  - total_files={store.attached_file_count}")
        return 0

    if args.command == "files":
        files = await service.list_files()
        print(f"Total files: {files.total_files}")
        return 0

    if args.command == "vision-probe":
        attempts = None if args.max_attempts == 0 else args.max_attempts
        result = await service.probe_vision(
            args.file_path,
            model=args.model,
            prompt=args.prompt,
            attempts=attempts,
        )
        print(f"Upload seconds: {result.upload_seconds:.3f}")
        print(f"Processing seconds: {result.processing_seconds:.3f}")
        print(f"Visibility seconds: {result.visibility_seconds:.3f}")
        print(f"Response seconds: {result.response_seconds:.3f}")
        print(f"Total seconds: {result.total_seconds:.3f}")
        print(f"File ID: {result.file_id}")
        print("Response text:")
        print(result.response_text)
        return 0

    if args.command == "attach-probe":
        attempts = None if args.max_attempts == 0 else args.max_attempts
        result = await service.probe_vector_store_attach(
            args.file_path,
            attempts=attempts,
        )
        print(f"Upload seconds: {result.upload_seconds:.3f}")
        print(f"Attach seconds: {result.attach_seconds:.3f}")
        print(f"Total seconds: {result.total_seconds:.3f}")
        print(f"File ID: {result.file_id}")
        print(f"Vector store ID: {result.vector_store_id}")
        print(f"Vector store file ID: {result.vector_store_file_id}")
        return 0

    raise ValueError(f"Unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    logger.info("Starting OpenAI probe CLI command")
    parser = build_parser()
    args = parser.parse_args(argv)
    result = asyncio.run(_run_async(args))
    logger.info(
        "Finished OpenAI probe CLI command command=%s exit_code=%s",
        args.command,
        result,
    )
    return result
