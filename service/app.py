from __future__ import annotations

import asyncio
import hmac
import io
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from PIL import Image

from data.dataset import (
    MULTI_HOP_USER_INSTRUCTION,
    SYSTEM_PROMPT,
    resize_image_and_bboxes,
)
from models.coe_model import parse_coe_output


REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("coe_slidevqa_service")
logging.basicConfig(
    level=os.environ.get("COE_LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


class Settings:
    runtime_dir = Path(os.environ.get("COE_RUNTIME_DIR", REPO_ROOT / "service" / "runtime"))
    sessions_dir = runtime_dir / "sessions"
    max_file_mb = int(os.environ.get("COE_MAX_FILE_MB", "150"))
    max_session_mb = int(os.environ.get("COE_MAX_SESSION_MB", "300"))
    upload_concurrency = int(os.environ.get("COE_UPLOAD_CONCURRENCY", "2"))
    max_pages = int(os.environ.get("COE_MAX_PAGES", "120"))
    pdf_dpi = int(os.environ.get("COE_PDF_DPI", "150"))
    session_ttl_seconds = int(os.environ.get("COE_SESSION_TTL_SECONDS", str(24 * 3600)))
    default_top_k = int(os.environ.get("COE_TOP_K", "20"))
    max_top_k = int(os.environ.get("COE_MAX_TOP_K", "20"))
    eval_resolution = int(os.environ.get("COE_EVAL_RESOLUTION", "1024"))
    image_max_pixels = int(os.environ.get("COE_IMAGE_MAX_PIXELS", str(1024 * 1024)))
    max_new_tokens = int(os.environ.get("COE_MAX_NEW_TOKENS", "1024"))
    torch_dtype = os.environ.get("COE_TORCH_DTYPE", "bfloat16")
    api_token = os.environ.get("COE_API_TOKEN", "")
    public_model_name = os.environ.get("COE_PUBLIC_MODEL_NAME", "CoE-SlideVQA-8B")
    preload_model = os.environ.get("COE_PRELOAD_MODEL", "0") == "1"
    use_flash_attention = os.environ.get("COE_USE_FLASH_ATTENTION", "1") != "0"


settings = Settings()
_upload_semaphore = asyncio.Semaphore(settings.upload_concurrency)


def _first_existing_model() -> str:
    explicit = os.environ.get("COE_MODEL_PATH")
    if explicit:
        return explicit
    return "PeiyangLiu/CoE-SlideVQA-8B"


def _first_existing_processor(model_path: str) -> str:
    explicit = os.environ.get("COE_PROCESSOR_PATH")
    if explicit:
        return explicit
    model_dir = Path(model_path)
    if model_dir.exists() and (model_dir / "tokenizer_config.json").exists():
        return model_path
    return "PeiyangLiu/CoE-SlideVQA-8B"


def _validate_local_model(model_path: str) -> None:
    path = Path(model_path)
    if not path.exists():
        return
    has_config = (path / "config.json").exists()
    has_weights = (
        (path / "model.safetensors.index.json").exists()
        or any(path.glob("model-*.safetensors"))
        or (path / "pytorch_model.bin").exists()
    )
    if not has_config or not has_weights:
        raise RuntimeError(
            f"Invalid model checkpoint at {path}: expected config.json and HF-format model weights."
        )


def _parse_generation(raw_output: str) -> dict:
    parsed = parse_coe_output(raw_output)
    if parsed is not None and not (
        str(parsed.get("answer", "")).lstrip().startswith("{")
        and not parsed.get("evidence_chain")
    ):
        parsed["parse_ok"] = True
        return parsed
    return {
        "answer": "",
        "evidence_chain": [],
        "parse_ok": False,
        "parse_error": "Model output was not a complete CoE JSON object.",
    }


MODEL_PATH = _first_existing_model()
PROCESSOR_PATH = _first_existing_processor(MODEL_PATH)


class AskRequest(BaseModel):
    session_id: str
    question: str = Field(min_length=1)
    top_k: Optional[int] = None
    slide_indices: Optional[List[int]] = Field(
        default=None,
        description="Optional 1-based slide indices to send to the model.",
    )
    max_new_tokens: Optional[int] = None
    temperature: float = 0.0


class ModelRuntime:
    def __init__(self, model_path: str, processor_path: str):
        self.model_path = model_path
        self.processor_path = processor_path
        self.model = None
        self.processor = None
        self.device = None
        self.loaded_at: Optional[float] = None
        self._load_lock = asyncio.Lock()
        self._infer_lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    async def ensure_loaded(self) -> None:
        if self.loaded:
            return
        async with self._load_lock:
            if self.loaded:
                return
            await run_in_threadpool(self._load)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the SlideVQA 8B inference service.")

        _validate_local_model(self.model_path)
        dtype = getattr(torch, settings.torch_dtype, torch.bfloat16)
        kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if settings.use_flash_attention:
            kwargs["attn_implementation"] = "flash_attention_2"

        LOGGER.info("Loading SlideVQA CoE model from %s", self.model_path)
        try:
            model = AutoModelForImageTextToText.from_pretrained(self.model_path, **kwargs)
        except Exception:
            if "attn_implementation" not in kwargs:
                raise
            LOGGER.exception("Model load with flash_attention_2 failed; retrying without it")
            kwargs.pop("attn_implementation", None)
            model = AutoModelForImageTextToText.from_pretrained(self.model_path, **kwargs)

        model.to("cuda")
        model.eval()
        if hasattr(model, "config"):
            model.config.use_cache = True

        LOGGER.info("Loading processor from %s", self.processor_path)
        processor = AutoProcessor.from_pretrained(self.processor_path, trust_remote_code=True)
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "left"

        self.model = model
        self.processor = processor
        self.device = "cuda"
        self.loaded_at = time.time()
        LOGGER.info("SlideVQA CoE model ready")

    async def generate(
        self,
        question: str,
        image_paths: List[Path],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[dict, List[dict]]:
        await self.ensure_loaded()
        async with self._infer_lock:
            return await run_in_threadpool(
                self._generate_sync,
                question,
                image_paths,
                max_new_tokens,
                temperature,
            )

    def _generate_sync(
        self,
        question: str,
        image_paths: List[Path],
        max_new_tokens: int,
        temperature: float,
    ) -> Tuple[dict, List[dict]]:
        import torch

        images = []
        image_meta = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            source_size = img.size
            resized, _, _ = resize_image_and_bboxes(
                img,
                [],
                target_long_side=settings.eval_resolution,
                max_pixels=settings.image_max_pixels,
            )
            images.append(resized)
            image_meta.append({
                "source_width": source_size[0],
                "source_height": source_size[1],
                "model_width": resized.size[0],
                "model_height": resized.size[1],
            })

        content = []
        for i, img in enumerate(images):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[img_{i}]"})
        content.append({
            "type": "text",
            "text": f"\nQuestion: {question}\n\n{MULTI_HOP_USER_INSTRUCTION}",
        })
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[prompt],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        tokenizer = self.processor.tokenizer
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "num_beams": 1,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        raw_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        parsed = _parse_generation(raw_output)
        parsed["raw_output"] = raw_output
        return parsed, image_meta


runtime = ModelRuntime(MODEL_PATH, PROCESSOR_PATH)
app = FastAPI(title="CoE SlideVQA Inference Service", version="0.1.0")
app.mount("/static", StaticFiles(directory=REPO_ROOT / "service" / "static"), name="static")


def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not settings.api_token:
        return
    expected = f"Bearer {settings.api_token}"
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def _session_path(session_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    path = settings.sessions_dir / session_id
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    return path


def _manifest_path(session_id: str) -> Path:
    return _session_path(session_id) / "manifest.json"


def _read_manifest(session_id: str) -> dict:
    with _manifest_path(session_id).open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(session_dir: Path, manifest: dict) -> None:
    tmp = session_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(session_dir / "manifest.json")


def _cleanup_old_sessions() -> None:
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - settings.session_ttl_seconds
    for path in settings.sessions_dir.iterdir():
        if path.is_dir() and path.stat().st_mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)


def _slide_url(session_id: str, filename: str) -> str:
    return f"/api/slides/{session_id}/{filename}"


def _add_slide(
    slides_dir: Path,
    slides: List[dict],
    image: Image.Image,
    *,
    text: str = "",
    source: str = "",
) -> None:
    index = len(slides) + 1
    filename = f"slide_{index:04d}.png"
    path = slides_dir / filename
    image = image.convert("RGB")
    width, height = image.size
    if width * height > 40_000_000:
        raise HTTPException(status_code=400, detail=f"Slide {index} is too large to process safely")
    image.save(path)
    slides.append({
        "index": index,
        "filename": filename,
        "url": "",
        "width": width,
        "height": height,
        "text": text or "",
        "source": source,
    })


def _render_pdf_bytes(data: bytes, slides_dir: Path, slides: List[dict], source: str) -> None:
    try:
        import fitz
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PyMuPDF is required for PDF rendering") from exc

    with fitz.open(stream=data, filetype="pdf") as doc:
        if len(doc) > settings.max_pages:
            raise HTTPException(status_code=400, detail=f"PDF has {len(doc)} pages; max is {settings.max_pages}")
        scale = settings.pdf_dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        for page_idx, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            text = page.get_text("text") or ""
            _add_slide(slides_dir, slides, image, text=text, source=f"{source}:page-{page_idx}")


def _extract_pptx_texts(data: bytes) -> Dict[int, str]:
    texts: Dict[int, List[str]] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                match = re.fullmatch(r"ppt/slides/slide(\d+)\.xml", name)
                if not match:
                    continue
                slide_idx = int(match.group(1))
                xml = zf.read(name).decode("utf-8", errors="ignore")
                parts = re.findall(r"<a:t>(.*?)</a:t>", xml)
                texts[slide_idx] = [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]
    except Exception:
        return {}
    return {idx: "\n".join(parts) for idx, parts in texts.items()}


def _render_ppt_bytes(data: bytes, suffix: str, slides_dir: Path, slides: List[dict], source: str) -> None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise HTTPException(
            status_code=400,
            detail="PPT/PPTX upload requires LibreOffice/soffice on the server. Export to PDF or upload slide images instead.",
        )
    pptx_texts = _extract_pptx_texts(data) if suffix == ".pptx" else {}
    with tempfile.TemporaryDirectory(prefix="coe_ppt_") as tmp:
        tmp_path = Path(tmp)
        input_path = tmp_path / f"input{suffix}"
        input_path.write_bytes(data)
        profile = tmp_path / "lo_profile"
        profile.mkdir()
        cmd = [
            soffice,
            "--headless",
            "--nologo",
            "--nofirststartwizard",
            "--norestore",
            f"-env:UserInstallation={profile.resolve().as_uri()}",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_path),
            str(input_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180, check=False)
        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f"LibreOffice failed to convert PPT/PPTX: {result.stderr.strip() or result.stdout.strip()}",
            )
        pdfs = sorted(tmp_path.glob("*.pdf"))
        if not pdfs:
            raise HTTPException(status_code=400, detail="LibreOffice did not produce a PDF")
        before = len(slides)
        _render_pdf_bytes(pdfs[0].read_bytes(), slides_dir, slides, source)
        for i, slide in enumerate(slides[before:], start=1):
            if i in pptx_texts and not slide.get("text"):
                slide["text"] = pptx_texts[i]


def _render_image_bytes(data: bytes, slides_dir: Path, slides: List[dict], source: str) -> None:
    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {source}") from exc
    _add_slide(slides_dir, slides, image, source=source)


async def _create_session_from_uploads(files: List[UploadFile]) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    settings.sessions_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex
    session_dir = settings.sessions_dir / session_id
    slides_dir = session_dir / "slides"
    slides_dir.mkdir(parents=True)

    total_bytes = 0
    slides: List[dict] = []
    try:
        for upload in files:
            chunks: List[bytes] = []
            file_bytes = 0
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                file_bytes += len(chunk)
                total_bytes += len(chunk)
                if file_bytes > settings.max_file_mb * 1024 * 1024:
                    raise HTTPException(status_code=400, detail=f"{upload.filename} exceeds {settings.max_file_mb} MB")
                if total_bytes > settings.max_session_mb * 1024 * 1024:
                    raise HTTPException(status_code=400, detail=f"Session upload exceeds {settings.max_session_mb} MB")
                chunks.append(chunk)
            data = b"".join(chunks)
            if not data:
                raise HTTPException(status_code=400, detail=f"{upload.filename or 'upload'} is empty")
            if file_bytes > settings.max_file_mb * 1024 * 1024:
                raise HTTPException(status_code=400, detail=f"{upload.filename} exceeds {settings.max_file_mb} MB")
            suffix = Path(upload.filename or "").suffix.lower()
            source = Path(upload.filename or "upload").name
            if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                await run_in_threadpool(_render_image_bytes, data, slides_dir, slides, source)
            elif suffix == ".pdf":
                await run_in_threadpool(_render_pdf_bytes, data, slides_dir, slides, source)
            elif suffix in {".ppt", ".pptx"}:
                await run_in_threadpool(_render_ppt_bytes, data, suffix, slides_dir, slides, source)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix or 'unknown'}")

        if not slides:
            raise HTTPException(status_code=400, detail="No slides were extracted from the upload")
        for slide in slides:
            slide["url"] = _slide_url(session_id, slide["filename"])
        manifest = {
            "session_id": session_id,
            "created_at": time.time(),
            "model_path": MODEL_PATH,
            "slides": slides,
        }
        _write_manifest(session_dir, manifest)
        return manifest
    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise


_TOKEN_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")


def _tokens(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text or "") if len(tok.strip()) > 1]


def _select_candidates(manifest: dict, question: str, top_k: int, slide_indices: Optional[List[int]]) -> Tuple[List[dict], str]:
    slides = manifest["slides"]
    if slide_indices:
        by_index = {int(s["index"]): s for s in slides}
        selected = []
        for idx in slide_indices:
            if idx not in by_index:
                raise HTTPException(status_code=400, detail=f"Slide index out of range: {idx}")
            selected.append(by_index[idx])
        return selected[:top_k], "explicit"

    if len(slides) <= top_k:
        return slides, "all_slides"

    q = set(_tokens(question))
    scored = []
    for slide in slides:
        slide_tokens = _tokens(slide.get("text", ""))
        overlap = len(q & set(slide_tokens))
        phrase_bonus = sum(1 for tok in q if tok in (slide.get("text", "").lower()))
        scored.append((overlap * 10 + phrase_bonus, slide["index"], slide))
    if max(score for score, _, _ in scored) <= 0:
        return slides[:top_k], "first_slides_no_text_match"
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [slide for _, _, slide in scored[:top_k]], "lexical_text"


def _candidate_paths(session_id: str, candidates: List[dict]) -> List[Path]:
    session_dir = _session_path(session_id)
    paths = []
    for cand in candidates:
        path = (session_dir / "slides" / cand["filename"]).resolve()
        if not str(path).startswith(str((session_dir / "slides").resolve())) or not path.exists():
            raise HTTPException(status_code=404, detail="Slide image missing")
        paths.append(path)
    return paths


@app.on_event("startup")
async def startup() -> None:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("COE_DISABLE_CLEANUP") != "1":
        _cleanup_old_sessions()
    if settings.preload_model:
        await runtime.ensure_loaded()
    if not settings.api_token:
        LOGGER.warning("COE_API_TOKEN is not set; upload and inference endpoints are unauthenticated")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(REPO_ROOT / "service" / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": runtime.loaded,
        "model_name": settings.public_model_name,
        "top_k_default": settings.default_top_k,
        "top_k_max": settings.max_top_k,
        "eval_resolution": settings.eval_resolution,
        "image_max_pixels": settings.image_max_pixels,
    }


@app.get("/api/status")
def status() -> dict:
    return {
        "model_loaded": runtime.loaded,
        "loaded_at": runtime.loaded_at,
        "sessions": len(list(settings.sessions_dir.glob("*"))) if settings.sessions_dir.exists() else 0,
    }


def _check_content_length(request: Request) -> None:
    value = request.headers.get("content-length")
    if not value:
        return
    try:
        n_bytes = int(value)
    except ValueError:
        return
    if n_bytes > settings.max_session_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Upload exceeds {settings.max_session_mb} MB")


@app.post("/api/upload")
async def upload(request: Request, files: List[UploadFile] = File(...), _: None = Depends(require_auth)) -> dict:
    _check_content_length(request)
    async with _upload_semaphore:
        manifest = await _create_session_from_uploads(files)
    return {
        "session_id": manifest["session_id"],
        "slides": manifest["slides"],
        "slide_count": len(manifest["slides"]),
    }


@app.post("/api/ask")
async def ask(req: AskRequest, _: None = Depends(require_auth)) -> dict:
    manifest = _read_manifest(req.session_id)
    top_k = min(max(1, req.top_k or settings.default_top_k), settings.max_top_k)
    candidates, selection_reason = _select_candidates(manifest, req.question, top_k, req.slide_indices)
    image_paths = _candidate_paths(req.session_id, candidates)
    result, image_meta = await runtime.generate(
        req.question,
        image_paths,
        max_new_tokens=req.max_new_tokens or settings.max_new_tokens,
        temperature=req.temperature,
    )
    response_candidates = []
    for i, cand in enumerate(candidates):
        enriched = dict(cand)
        enriched.update(image_meta[i])
        enriched["image_id"] = f"img_{i}"
        response_candidates.append(enriched)
    return {
        "session_id": req.session_id,
        "question": req.question,
        "selection_reason": selection_reason,
        "candidates": response_candidates,
        "answer": result.get("answer", ""),
        "evidence_chain": result.get("evidence_chain", []),
        "parse_ok": result.get("parse_ok", False),
        "parse_error": result.get("parse_error", ""),
        "raw_output": result.get("raw_output", ""),
        "model_name": settings.public_model_name,
    }


@app.post("/api/answer")
async def answer(
    request: Request,
    question: str = Form(...),
    top_k: int = Form(default=settings.default_top_k),
    files: List[UploadFile] = File(...),
    _: None = Depends(require_auth),
) -> dict:
    _check_content_length(request)
    async with _upload_semaphore:
        manifest = await _create_session_from_uploads(files)
    req = AskRequest(session_id=manifest["session_id"], question=question, top_k=top_k)
    return await ask(req)


@app.get("/api/slides/{session_id}/{filename}")
def slide(session_id: str, filename: str) -> FileResponse:
    if not re.fullmatch(r"slide_\d{4}\.png", filename):
        raise HTTPException(status_code=404, detail="Slide not found")
    session_dir = _session_path(session_id)
    path = session_dir / "slides" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Slide not found")
    return FileResponse(path)
