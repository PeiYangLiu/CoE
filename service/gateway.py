from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from service.app import (
    AskRequest,
    REPO_ROOT,
    _check_content_length,
    _cleanup_old_sessions,
    _create_session_from_uploads,
    _session_path,
    settings,
    slide as serve_slide,
)


LOGGER = logging.getLogger("coe_slidevqa_gateway")
logging.basicConfig(
    level=os.environ.get("COE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class GatewaySettings:
    backend_urls = [
        url.strip().rstrip("/")
        for url in os.environ.get(
            "COE_BACKEND_URLS",
            "http://127.0.0.1:7861,http://127.0.0.1:7862,http://127.0.0.1:7863,http://127.0.0.1:7864",
        ).split(",")
        if url.strip()
    ]
    queue_max_size = int(os.environ.get("COE_QUEUE_MAX_SIZE", "32"))
    max_queue_wait_seconds = float(os.environ.get("COE_MAX_QUEUE_WAIT_SECONDS", "180"))
    request_timeout_seconds = float(os.environ.get("COE_REQUEST_TIMEOUT_SECONDS", "900"))
    backend_health_interval = float(os.environ.get("COE_BACKEND_HEALTH_INTERVAL", "3"))
    cache_max_entries = int(os.environ.get("COE_CACHE_MAX_ENTRIES", "256"))
    cache_ttl_seconds = float(os.environ.get("COE_CACHE_TTL_SECONDS", "3600"))


gateway_settings = GatewaySettings()
gateway = FastAPI(title="CoE SlideVQA Gateway", version="0.2.0")
gateway.mount("/static", StaticFiles(directory=REPO_ROOT / "service" / "static"), name="static")


@dataclass
class BackendState:
    index: int
    url: str
    ready: bool = False
    in_flight: int = 0
    processed: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_error: str = ""
    last_health_at: float = 0.0


@dataclass
class GatewayJob:
    request_id: str
    payload: dict
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)
    cache_key: Optional[str] = None


class TTLCache:
    def __init__(self, max_entries: int, ttl_seconds: float):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[dict]:
        now = time.time()
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        ts, value = item
        if now - ts > self.ttl_seconds:
            self._items.pop(key, None)
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        cached = json.loads(json.dumps(value, ensure_ascii=False))
        cached["cache_hit"] = True
        cached["queue_wait_seconds"] = 0.0
        return cached

    def set(self, key: str, value: dict) -> None:
        if not value.get("parse_ok", False):
            return
        self._items[key] = (time.time(), json.loads(json.dumps(value, ensure_ascii=False)))
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._items)


backends = [BackendState(index=i, url=url) for i, url in enumerate(gateway_settings.backend_urls)]
response_cache = TTLCache(gateway_settings.cache_max_entries, gateway_settings.cache_ttl_seconds)
upload_semaphore = asyncio.Semaphore(settings.upload_concurrency)
queue: asyncio.Queue[GatewayJob]
client: httpx.AsyncClient
workers: List[asyncio.Task] = []
health_task: Optional[asyncio.Task] = None
total_accepted = 0
total_rejected = 0


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not settings.api_token:
        return
    expected = f"Bearer {settings.api_token}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _request_payload(req: AskRequest) -> dict:
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()


def _norm_question(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _cache_key(req: AskRequest) -> str:
    top_k = min(max(1, req.top_k or settings.default_top_k), settings.max_top_k)
    slide_indices = sorted(set(int(i) for i in (req.slide_indices or [])))
    payload = {
        "session_id": req.session_id,
        "question": _norm_question(req.question),
        "top_k": top_k,
        "slide_indices": slide_indices,
        "max_new_tokens": req.max_new_tokens or settings.max_new_tokens,
        "temperature": float(req.temperature or 0.0),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _ready_backends() -> List[BackendState]:
    return [backend for backend in backends if backend.ready]


async def _probe_backend(backend: BackendState) -> None:
    try:
        resp = await client.get(f"{backend.url}/api/health", timeout=2.0)
        data = resp.json() if resp.status_code == 200 else {}
        backend.ready = resp.status_code == 200 and bool(data.get("model_loaded"))
        backend.last_error = "" if backend.ready else "loading"
        backend.last_health_at = time.time()
        if backend.ready:
            backend.consecutive_failures = 0
    except Exception as exc:
        backend.ready = False
        backend.last_error = str(exc)[:200]
        backend.last_health_at = time.time()


async def _health_loop() -> None:
    while True:
        await asyncio.gather(*(_probe_backend(backend) for backend in backends))
        await asyncio.sleep(gateway_settings.backend_health_interval)


def _set_future_exception(job: GatewayJob, exc: HTTPException) -> None:
    if not job.future.done():
        job.future.set_exception(exc)


async def _backend_worker(backend: BackendState) -> None:
    global total_rejected
    while True:
        if not backend.ready:
            await asyncio.sleep(0.5)
            continue
        job = await queue.get()
        try:
            if job.future.cancelled():
                continue
            queue_wait = time.time() - job.enqueued_at
            if queue_wait > gateway_settings.max_queue_wait_seconds:
                _set_future_exception(
                    job,
                    HTTPException(status_code=503, detail="Request waited too long in the inference queue"),
                )
                continue

            backend.in_flight += 1
            timeout = httpx.Timeout(
                connect=5.0,
                read=gateway_settings.request_timeout_seconds,
                write=30.0,
                pool=5.0,
            )
            try:
                headers = {"X-CoE-Request-ID": job.request_id}
                if settings.api_token:
                    headers["Authorization"] = f"Bearer {settings.api_token}"
                resp = await client.post(
                    f"{backend.url}/api/ask",
                    json=job.payload,
                    headers=headers,
                    timeout=timeout,
                )
                if resp.status_code >= 500:
                    backend.consecutive_failures += 1
                    backend.failures += 1
                    backend.last_error = resp.text[:200]
                    if backend.consecutive_failures >= 2:
                        backend.ready = False
                if resp.status_code != 200:
                    detail: Any
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text
                    _set_future_exception(job, HTTPException(status_code=resp.status_code, detail=detail))
                    continue

                backend.consecutive_failures = 0
                backend.processed += 1
                result = resp.json()
                result.update({
                    "cache_hit": False,
                    "request_id": job.request_id,
                    "backend": {"index": backend.index, "url": backend.url},
                    "queue_wait_seconds": queue_wait,
                })
                if job.cache_key:
                    response_cache.set(job.cache_key, result)
                if not job.future.done():
                    job.future.set_result(result)
            except Exception as exc:
                backend.consecutive_failures += 1
                backend.failures += 1
                backend.last_error = str(exc)[:200]
                if backend.consecutive_failures >= 2:
                    backend.ready = False
                _set_future_exception(job, HTTPException(status_code=502, detail=f"Backend failed: {backend.last_error}"))
        finally:
            backend.in_flight = max(0, backend.in_flight - 1)
            queue.task_done()


async def _await_future(request: Request, future: asyncio.Future) -> dict:
    deadline = time.time() + gateway_settings.request_timeout_seconds + gateway_settings.max_queue_wait_seconds
    while True:
        if future.done():
            return future.result()
        if await request.is_disconnected():
            future.cancel()
            raise HTTPException(status_code=499, detail="Client disconnected")
        if time.time() > deadline:
            future.cancel()
            raise HTTPException(status_code=504, detail="Inference request timed out")
        await asyncio.sleep(0.25)


async def _submit_ask(request: Request, req: AskRequest) -> dict:
    global total_accepted, total_rejected
    _session_path(req.session_id)
    cache_key = _cache_key(req)
    if float(req.temperature or 0.0) == 0.0:
        cached = response_cache.get(cache_key)
        if cached is not None:
            return cached

    if queue.full():
        total_rejected += 1
        raise HTTPException(
            status_code=429,
            detail="Inference queue is full; retry later",
            headers={"Retry-After": "5"},
        )

    future: asyncio.Future = asyncio.get_running_loop().create_future()
    job = GatewayJob(
        request_id=uuid.uuid4().hex,
        payload=_request_payload(req),
        future=future,
        cache_key=cache_key if float(req.temperature or 0.0) == 0.0 else None,
    )
    try:
        queue.put_nowait(job)
        total_accepted += 1
    except asyncio.QueueFull:
        total_rejected += 1
        raise HTTPException(
            status_code=429,
            detail="Inference queue is full; retry later",
            headers={"Retry-After": "5"},
        )
    return await _await_future(request, future)


@gateway.on_event("startup")
async def startup() -> None:
    global queue, client, workers, health_task
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_sessions()
    queue = asyncio.Queue(maxsize=gateway_settings.queue_max_size)
    client = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=max(16, len(backends) * 4), max_keepalive_connections=max(8, len(backends) * 2))
    )
    health_task = asyncio.create_task(_health_loop())
    workers = [asyncio.create_task(_backend_worker(backend)) for backend in backends]
    LOGGER.info("Gateway started with backends: %s", ", ".join(backend.url for backend in backends))
    if not settings.api_token:
        LOGGER.warning("COE_API_TOKEN is not set; gateway endpoints are unauthenticated")


@gateway.on_event("shutdown")
async def shutdown() -> None:
    if health_task:
        health_task.cancel()
    for task in workers:
        task.cancel()
    await client.aclose()


@gateway.get("/")
def index() -> FileResponse:
    return FileResponse(REPO_ROOT / "service" / "static" / "index.html")


@gateway.get("/api/health")
def health() -> dict:
    ready = _ready_backends()
    return {
        "ok": bool(ready),
        "ready": bool(ready),
        "model_loaded": bool(ready),
        "model_name": settings.public_model_name,
        "ready_backends": len(ready),
        "total_backends": len(backends),
        "queue_size": queue.qsize() if "queue" in globals() else 0,
        "queue_max_size": gateway_settings.queue_max_size,
        "top_k_default": settings.default_top_k,
        "top_k_max": settings.max_top_k,
        "eval_resolution": settings.eval_resolution,
        "image_max_pixels": settings.image_max_pixels,
    }


@gateway.get("/api/status")
def status(_: None = Depends(require_auth)) -> dict:
    return {
        "queue": {
            "size": queue.qsize(),
            "max_size": gateway_settings.queue_max_size,
            "accepted": total_accepted,
            "rejected": total_rejected,
        },
        "cache": {
            "size": response_cache.size,
            "max_entries": response_cache.max_entries,
            "hits": response_cache.hits,
            "misses": response_cache.misses,
            "ttl_seconds": response_cache.ttl_seconds,
        },
        "backends": [backend.__dict__ for backend in backends],
    }


@gateway.get("/metrics")
def metrics() -> PlainTextResponse:
    lines = [
        f"coe_queue_size {queue.qsize()}",
        f"coe_queue_max_size {gateway_settings.queue_max_size}",
        f"coe_requests_accepted_total {total_accepted}",
        f"coe_requests_rejected_total {total_rejected}",
        f"coe_cache_entries {response_cache.size}",
        f"coe_cache_hits_total {response_cache.hits}",
        f"coe_cache_misses_total {response_cache.misses}",
    ]
    for backend in backends:
        labels = f'backend="{backend.index}"'
        lines.extend([
            f"coe_backend_ready{{{labels}}} {1 if backend.ready else 0}",
            f"coe_backend_in_flight{{{labels}}} {backend.in_flight}",
            f"coe_backend_processed_total{{{labels}}} {backend.processed}",
            f"coe_backend_failures_total{{{labels}}} {backend.failures}",
        ])
    return PlainTextResponse("\n".join(lines) + "\n")


@gateway.post("/api/upload")
async def upload(request: Request, files: List[UploadFile] = File(...), _: None = Depends(require_auth)) -> dict:
    _check_content_length(request)
    async with upload_semaphore:
        manifest = await _create_session_from_uploads(files)
    return {
        "session_id": manifest["session_id"],
        "slides": manifest["slides"],
        "slide_count": len(manifest["slides"]),
    }


@gateway.post("/api/ask")
async def ask(request: Request, req: AskRequest, _: None = Depends(require_auth)) -> dict:
    return await _submit_ask(request, req)


@gateway.post("/api/answer")
async def answer(
    request: Request,
    question: str = Form(...),
    top_k: int = Form(default=settings.default_top_k),
    files: List[UploadFile] = File(...),
    _: None = Depends(require_auth),
) -> dict:
    if queue.full():
        raise HTTPException(
            status_code=429,
            detail="Inference queue is full; retry later",
            headers={"Retry-After": "5"},
        )
    _check_content_length(request)
    async with upload_semaphore:
        manifest = await _create_session_from_uploads(files)
    req = AskRequest(session_id=manifest["session_id"], question=question, top_k=top_k)
    return await _submit_ask(request, req)


@gateway.get("/api/slides/{session_id}/{filename}")
def slide(session_id: str, filename: str) -> FileResponse:
    return serve_slide(session_id, filename)
