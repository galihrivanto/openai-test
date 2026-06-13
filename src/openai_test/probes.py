from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from .client import build_client
from .retry import retry_until_success

logger = logging.getLogger(__name__)


def fingerprint_key(api_key: str) -> str:
    """Create a short stable fingerprint for logging."""

    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def format_timestamp(timestamp: int | None) -> str:
    """Format a Unix timestamp for human-readable output."""

    if timestamp is None:
        return "-"
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))


@dataclass(slots=True)
class VectorStoreInfo:
    vector_store_id: str
    name: str | None
    created_at: int | None
    attached_file_count: int


@dataclass(slots=True)
class FileInventoryResult:
    total_files: int


@dataclass(slots=True)
class VisionProbeResult:
    upload_seconds: float
    processing_seconds: float
    visibility_seconds: float
    response_seconds: float
    total_seconds: float
    file_id: str
    response_text: str


@dataclass(slots=True)
class VectorStoreAttachProbeResult:
    upload_seconds: float
    attach_seconds: float
    total_seconds: float
    file_id: str
    vector_store_id: str
    vector_store_file_id: str


class OpenAIProbeService:
    """Standalone OpenAI probing service."""

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key
        self._client = build_client(api_key=api_key)
        key = api_key or ""
        self._api_key_fingerprint = fingerprint_key(key)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def list_vector_stores(self) -> list[VectorStoreInfo]:
        stores: list[VectorStoreInfo] = []
        logger.info(
            "Listing vector stores api_key_fingerprint=%s",
            self._api_key_fingerprint,
        )
        async for vector_store in self._client.vector_stores.list(limit=100):
            logger.info(
                "Reading vector store api_key_fingerprint=%s vector_store_id=%s",
                self._api_key_fingerprint,
                vector_store.id,
            )
            attached_file_count = 0
            async for _attachment in self._client.vector_stores.files.list(
                vector_store_id=vector_store.id,
                limit=100,
            ):
                attached_file_count += 1
            stores.append(
                VectorStoreInfo(
                    vector_store_id=vector_store.id,
                    name=getattr(vector_store, "name", None),
                    created_at=getattr(vector_store, "created_at", None),
                    attached_file_count=attached_file_count,
                ),
            )
        logger.info(
            "Completed vector store listing api_key_fingerprint=%s vector_store_count=%s",
            self._api_key_fingerprint,
            len(stores),
        )
        return stores

    async def list_files(self) -> FileInventoryResult:
        total_files = 0
        logger.info(
            "Listing OpenAI files api_key_fingerprint=%s",
            self._api_key_fingerprint,
        )
        async for file_obj in self._client.files.list(limit=100):
            total_files += 1
        logger.info(
            "Completed file listing api_key_fingerprint=%s file_count=%s",
            self._api_key_fingerprint,
            total_files,
        )
        return FileInventoryResult(
            total_files=total_files,
        )

    async def probe_vision(
        self,
        file_path: str,
        *,
        model: str,
        prompt: str,
        attempts: int | None,
        initial_delay_seconds: float = 1.0,
    ) -> VisionProbeResult:
        start = time.perf_counter()
        file_id = ""
        with Path(file_path).open("rb") as file:
            logger.info(
                "Uploading probe file for vision api_key_fingerprint=%s file_path=%s",
                self._api_key_fingerprint,
                file_path,
            )
            upload_started = time.perf_counter()
            uploaded_file = await self._client.files.create(
                file=file,
                purpose="assistants",
            )
            upload_finished = time.perf_counter()
            logger.info(
                "Waiting for uploaded file processing api_key_fingerprint=%s file_id=%s",
                self._api_key_fingerprint,
                uploaded_file.id,
            )
            await self._client.files.wait_for_processing(
                uploaded_file.id,
                max_wait_seconds=60,
            )
            processed_finished = time.perf_counter()
            logger.info("Waiting for file visibility file_id=%s", uploaded_file.id)
            ready_file = await self._wait_for_file_visibility(uploaded_file.id)
            ready_finished = time.perf_counter()
            file_id = ready_file.id

            logger.info("Calling vision model file_id=%s model=%s", file_id, model)
            response = await retry_until_success(
                lambda: self._client.responses.create(
                    model=model,
                    reasoning={"effort": "minimal", "summary": "auto"},
                    instructions=(
                        "You are a vision assistant for procurement support. "
                        "Describe only what is directly visible in the image. "
                        "Do not guess missing details."
                    ),
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "detail": "auto",
                                    "file_id": file_id,
                                },
                            ],
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                ),
                attempts=attempts,
                initial_delay_seconds=initial_delay_seconds,
            )
            response_finished = time.perf_counter()

        response_text = getattr(response, "output_text", "") or str(response)
        try:
            return VisionProbeResult(
                upload_seconds=upload_finished - upload_started,
                processing_seconds=processed_finished - upload_finished,
                visibility_seconds=ready_finished - processed_finished,
                response_seconds=response_finished - ready_finished,
                total_seconds=response_finished - start,
                file_id=file_id,
                response_text=response_text,
            )
        finally:
            if file_id:
                with contextlib.suppress(Exception):
                    await self._client.files.delete(file_id=file_id)

    async def probe_vector_store_attach(
        self,
        file_path: str,
        *,
        attempts: int | None,
        initial_delay_seconds: float = 1.0,
    ) -> VectorStoreAttachProbeResult:
        start = time.perf_counter()
        file_id = ""
        vector_store_id = ""
        vector_store_file_id = ""
        logger.info("Creating temporary vector store")
        vector_store = await self._client.vector_stores.create(
            name=f"probe-{Path(file_path).stem}-{int(time.time())}",
        )
        vector_store_id = vector_store.id
        try:
            with Path(file_path).open("rb") as file:
                logger.info(
                    "Uploading probe file for attach api_key_fingerprint=%s file_path=%s vector_store_id=%s",
                    self._api_key_fingerprint,
                    file_path,
                    vector_store_id,
                )
                upload_started = time.perf_counter()
                uploaded_file = await self._client.files.create(
                    file=file,
                    purpose="assistants",
                )
                upload_finished = time.perf_counter()
                logger.info(
                    "Waiting for uploaded file processing file_id=%s",
                    uploaded_file.id,
                )
                await self._client.files.wait_for_processing(
                    uploaded_file.id,
                    max_wait_seconds=60,
                )
                logger.info(
                    "Waiting for file visibility file_id=%s",
                    uploaded_file.id,
                )
                ready_file = await self._wait_for_file_visibility(uploaded_file.id)
                file_id = ready_file.id

            logger.info(
                "Attaching file to temporary vector store file_id=%s vector_store_id=%s",
                file_id,
                vector_store_id,
            )
            attach_started = time.perf_counter()
            vector_store_file = await retry_until_success(
                lambda: self._client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=file_id,
                ),
                attempts=attempts,
                initial_delay_seconds=initial_delay_seconds,
            )
            attach_finished = time.perf_counter()
            vector_store_file_id = vector_store_file.id

            return VectorStoreAttachProbeResult(
                upload_seconds=upload_finished - upload_started,
                attach_seconds=attach_finished - attach_started,
                total_seconds=attach_finished - start,
                file_id=file_id,
                vector_store_id=vector_store_id,
                vector_store_file_id=vector_store_file_id,
            )
        finally:
            if vector_store_file_id:
                logger.info(
                    "Cleaning up attached vector-store file file_id=%s vector_store_id=%s",
                    file_id,
                    vector_store_id,
                )
                with contextlib.suppress(Exception):
                    await self._client.vector_stores.files.delete(
                        vector_store_id=vector_store_id,
                        file_id=file_id,
                    )
            if file_id:
                logger.info("Cleaning up uploaded file file_id=%s", file_id)
                with contextlib.suppress(Exception):
                    await self._client.files.delete(file_id=file_id)
            if vector_store_id:
                logger.info("Cleaning up temporary vector store vector_store_id=%s", vector_store_id)
                with contextlib.suppress(Exception):
                    await self._client.vector_stores.delete(
                        vector_store_id=vector_store_id,
                    )

    async def probe_existing_vector_store_attach(
        self,
        file_path: str,
        *,
        vector_store_id: str,
        attempts: int | None,
        initial_delay_seconds: float = 1.0,
    ) -> VectorStoreAttachProbeResult:
        start = time.perf_counter()
        file_id = ""
        vector_store_file_id = ""
        logger.info(
            "Using existing vector store vector_store_id=%s",
            vector_store_id,
        )
        try:
            with Path(file_path).open("rb") as file:
                logger.info(
                    "Uploading probe file for existing-store attach api_key_fingerprint=%s file_path=%s vector_store_id=%s",
                    self._api_key_fingerprint,
                    file_path,
                    vector_store_id,
                )
                upload_started = time.perf_counter()
                uploaded_file = await self._client.files.create(
                    file=file,
                    purpose="assistants",
                )
                upload_finished = time.perf_counter()
                logger.info(
                    "Waiting for uploaded file processing file_id=%s",
                    uploaded_file.id,
                )
                await self._client.files.wait_for_processing(
                    uploaded_file.id,
                    max_wait_seconds=60,
                )
                logger.info("Waiting for file visibility file_id=%s", uploaded_file.id)
                ready_file = await self._wait_for_file_visibility(uploaded_file.id)
                file_id = ready_file.id

            logger.info(
                "Attaching file to existing vector store file_id=%s vector_store_id=%s",
                file_id,
                vector_store_id,
            )
            attach_started = time.perf_counter()
            vector_store_file = await retry_until_success(
                lambda: self._client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=file_id,
                ),
                attempts=attempts,
                initial_delay_seconds=initial_delay_seconds,
            )
            attach_finished = time.perf_counter()
            vector_store_file_id = vector_store_file.id

            return VectorStoreAttachProbeResult(
                upload_seconds=upload_finished - upload_started,
                attach_seconds=attach_finished - attach_started,
                total_seconds=attach_finished - start,
                file_id=file_id,
                vector_store_id=vector_store_id,
                vector_store_file_id=vector_store_file_id,
            )
        finally:
            if vector_store_file_id:
                logger.info(
                    "Cleaning up attached vector-store file file_id=%s vector_store_id=%s",
                    file_id,
                    vector_store_id,
                )
                with contextlib.suppress(Exception):
                    await self._client.vector_stores.files.delete(
                        vector_store_id=vector_store_id,
                        file_id=file_id,
                    )
            if file_id:
                logger.info("Cleaning up uploaded file file_id=%s", file_id)
                with contextlib.suppress(Exception):
                    await self._client.files.delete(file_id=file_id)

    async def _wait_for_file_visibility(self, file_id: str) -> FileObject:
        logger.info("Retrying until file is visible file_id=%s", file_id)
        return await retry_until_success(
            lambda: self._client.files.retrieve(file_id=file_id),
            attempts=8,
            initial_delay_seconds=1.0,
        )
