from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool

from . import __version__
from .config import ROOT, settings
from .engine import StandaloneH3Engine
from .inventory import resolve_asset, scan_inventory
from .runner import JobRunner
from .postprocess import capabilities as postprocess_capabilities
from .schemas import (
    EngineLoadRequest, EngineLoraRequest, GenerationRequest, InterpolateRequest, UpscaleRequest, VideoSource,
)
from .store import JobStore
from .uploads import validate_image
from .video import inspect_video, read_embedded_metadata, sha256_file
from .thumbnails import create_thumbnail
from .seedvr2 import validate_request_bounds as validate_seedvr2_request_bounds
from .generation_options import public_options


settings.ensure_dirs()
store = JobStore(settings.state)
runner = JobRunner(settings, store)


@asynccontextmanager
async def lifespan(_: FastAPI):
    runner.start()
    yield
    runner.stop()


app = FastAPI(title="H3 Studio", version=__version__, lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])


@app.middleware("http")
async def local_origin_only(request: Request, call_next):
    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        origin = request.headers.get("origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != request.headers.get("host", "").lower():
                return JSONResponse({"detail": "Cross-origin mutation is not allowed"}, status_code=403)
    return await call_next(request)
app.mount("/assets", StaticFiles(directory=ROOT / "static"), name="assets")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/status")
def status():
    engine = StandaloneH3Engine.diagnostics()
    engine["runtime"] = runner.control_status()
    return {
        "version": __version__,
        "engine": engine,
        "config": settings.public_dict(),
        "vc_attention": {
            "available": False,
            "reason": "公開論文と同等と確認できる独立カーネルがないため無効です。",
        },
    }


@app.get("/api/postprocess/capabilities")
def get_postprocess_capabilities():
    return postprocess_capabilities()


@app.get("/api/generation/options")
def generation_options():
    return public_options()


@app.get("/api/inventory")
def inventory():
    return scan_inventory(settings.model_root, settings.lora_root, settings.latent_upscaler_root)


def _validate_loras(loras) -> None:
    for item in loras:
        if item.enabled:
            resolve_asset(settings.lora_root, "lora", item.path)


@app.post("/api/engine/load", status_code=202)
def load_engine(request: EngineLoadRequest):
    try:
        resolve_asset(settings.model_root, "repository", request.model)
        _validate_loras(request.loras)
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc)) from exc
    return runner.queue_model_load(request.model_dump())


@app.post("/api/engine/loras", status_code=202)
def configure_engine_loras(request: EngineLoraRequest):
    try:
        _validate_loras(request.loras)
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc)) from exc
    return runner.queue_loras([item.model_dump() for item in request.loras])


@app.post("/api/uploads")
async def upload_image(file: UploadFile = File(...)):
    allowed = {"image/png": (".png", "PNG"), "image/jpeg": (".jpg", "JPEG"), "image/webp": (".webp", "WEBP")}
    specification = allowed.get((file.content_type or "").lower())
    if not specification:
        raise HTTPException(415, "PNG, JPEG, WebP のみ使用できます")
    suffix, expected_format = specification
    target = settings.uploads / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    try:
        with target.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 32 * 1024 * 1024:
                    raise HTTPException(413, "画像は32MB以下にしてください")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(422, "画像ファイルが空です")
        validate_image(target, expected_format)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(422, f"画像を読み込めません: {exc}") from exc
    finally:
        await file.close()
    return {"id": target.name, "name": file.filename, "size": size}


@app.post("/api/video-uploads")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename or "video.mp4"
    if Path(filename).suffix.lower() != ".mp4" or (file.content_type or "").lower() not in {"", "video/mp4", "application/mp4", "application/octet-stream"}:
        raise HTTPException(415, "MP4動画のみ使用できます")
    target = settings.uploads / f"video-{uuid.uuid4().hex}.mp4"
    details_path = target.with_suffix(".upload.json")
    size = 0
    try:
        with target.open("wb") as destination:
            while chunk := await file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > 16 * 1024**3:
                    raise HTTPException(413, "動画は16GB以下にしてください")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(422, "動画ファイルが空です")
        info = await run_in_threadpool(inspect_video, target)
        digest = await run_in_threadpool(sha256_file, target)
        upload_record = {"id": target.name, "name": filename, "size": size, "sha256": digest, **info}
        details_path.write_text(json.dumps(upload_record, ensure_ascii=False, indent=2), encoding="utf-8")
    except HTTPException:
        target.unlink(missing_ok=True); details_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        target.unlink(missing_ok=True); details_path.unlink(missing_ok=True)
        raise HTTPException(422, f"MP4を読み込めません: {exc}") from exc
    finally:
        await file.close()
    return {**upload_record, "preview_url": f"/api/video-uploads/{target.name}"}


@app.get("/api/video-uploads/{upload_id}")
def uploaded_video(upload_id: str):
    target = (settings.uploads / upload_id).resolve()
    if not target.is_relative_to(settings.uploads.resolve()) or not target.is_file() or not target.name.startswith("video-") or target.suffix.lower() != ".mp4":
        raise HTTPException(404, "アップロード動画が見つかりません")
    return FileResponse(target, media_type="video/mp4")


def _resolve_video_source(source: VideoSource) -> dict[str, object]:
    if source.job_id:
        try:
            source_job = store.get(source.job_id)
        except KeyError as exc:
            raise HTTPException(422, "元の生成ジョブが見つかりません") from exc
        relative = source_job.get("output_path")
        if not relative:
            raise HTTPException(422, "元のジョブには利用可能な動画がありません")
        target = (settings.outputs / relative).resolve()
        if not target.is_relative_to(settings.outputs.resolve()) or not target.is_file():
            raise HTTPException(422, "元の出力動画が見つかりません")
        try:
            info = inspect_video(target)
        except Exception as exc:
            raise HTTPException(422, f"元のMP4を読み込めません: {exc}") from exc
        previous_source = source_job.get("resolved", {}).get("source") or {}
        previous_lineage = previous_source.get("lineage") if isinstance(previous_source, dict) else []
        origin_resolved = {key: value for key, value in source_job["resolved"].items() if key != "source"}
        current_snapshot = {
            "job_id": source.job_id, "kind": source_job.get("kind", "generate"),
            "request": source_job["request"], "resolved": origin_resolved,
            "created_at": source_job.get("created_at"),
        }
        lineage = [*(previous_lineage if isinstance(previous_lineage, list) else []), current_snapshot][-8:]
        origin = previous_source.get("origin") if isinstance(previous_source, dict) else None
        if not isinstance(origin, dict):
            origin = current_snapshot
        return {
            "type": "output", "job_id": source.job_id, "job_kind": source_job.get("kind", "generate"),
            "path": str(target.relative_to(settings.outputs.resolve()).as_posix()), "name": target.name,
            "sha256": sha256_file(target), "timing_mode": "normalized_cfr_at_average_rate",
            "origin": origin, "lineage": lineage, **info,
        }
    target = (settings.uploads / str(source.upload_id)).resolve()
    if not target.is_relative_to(settings.uploads.resolve()) or not target.is_file() or not target.name.startswith("video-"):
        raise HTTPException(422, "アップロード動画が見つかりません")
    record_path = target.with_suffix(".upload.json")
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    try:
        info = inspect_video(target)
    except Exception as exc:
        raise HTTPException(422, f"アップロードMP4を読み込めません: {exc}") from exc
    digest = record.get("sha256") or sha256_file(target)
    embedded = read_embedded_metadata(target)
    imported_origin = {
        "kind": "import", "name": record.get("name", target.name), "sha256": digest,
        **({"embedded_metadata": embedded} if embedded is not None else {}),
    }
    return {
        "type": "upload", "upload_id": target.name, "path": target.name,
        "name": record.get("name", target.name), "sha256": digest,
        "timing_mode": "normalized_cfr_at_average_rate",
        "origin": imported_origin, "lineage": [{**imported_origin, **info}],
        **info,
    }


def _validate_postprocess_geometry(width: int, height: int) -> None:
    if width > 8192 or height > 8192 or width * height > 40_000_000:
        raise HTTPException(422, f"出力サイズ {width}x{height} は上限（各辺8192、4000万画素）を超えます")


def _method_available(operation: str, method: str, factor: int) -> bool:
    section = postprocess_capabilities()[operation]
    return any(item["id"] == method and item["available"] and factor in item["factors"] for item in section["methods"])


@app.post("/api/jobs/upscale", status_code=201)
def create_upscale_job(request: UpscaleRequest):
    if not _method_available("upscale", request.method, request.scale):
        raise HTTPException(409, f"アップスケール方式は現在利用できません: {request.method}")
    source = _resolve_video_source(request.source)
    if request.method != "lanczos" and not source["is_cfr"]:
        raise HTTPException(422, "AIアップスケールは固定フレームレートのMP4に対応しています。LanczosはCFRへ正規化できます")
    _validate_postprocess_geometry(int(source["width"]) * request.scale, int(source["height"]) * request.scale)
    if request.method == "seedvr2_3b":
        try:
            validate_seedvr2_request_bounds({"scale": request.scale, "source": source})
        except RuntimeError as exc:
            raise HTTPException(422, str(exc)) from exc
    resolved = {
        **request.model_dump(), "source": source, "kind": "upscale",
        "width": int(source["width"]) * request.scale,
        "height": int(source["height"]) * request.scale,
        "fps": source["fps"], "duration_seconds": source["duration_seconds"],
        "frame_count": source["frame_count"], "has_audio": source["has_audio"],
    }
    job_id = str(uuid.uuid4())
    job = store.create(job_id, request.model_dump(), resolved, kind="upscale")
    runner.wake()
    return job


@app.post("/api/jobs/interpolate", status_code=201)
def create_interpolate_job(request: InterpolateRequest):
    if not _method_available("interpolate", request.method, request.factor):
        raise HTTPException(409, f"フレーム補間方式は現在利用できません: {request.method}")
    source = _resolve_video_source(request.source)
    if not source["is_cfr"]:
        raise HTTPException(422, "フレーム補間は固定フレームレートのMP4に対応しています")
    if float(source["fps"]) * request.factor > 240:
        raise HTTPException(422, "補間後のフレームレートは240fps以下にしてください")
    _validate_postprocess_geometry(int(source["width"]), int(source["height"]))
    resolved = {
        **request.model_dump(), "source": source, "kind": "interpolate",
        "width": source["width"], "height": source["height"],
        "fps": float(source["fps"]) * request.factor,
        "fps_numerator": int(source["fps_numerator"]) * request.factor,
        "fps_denominator": source["fps_denominator"],
        "duration_seconds": source["duration_seconds"],
        "frame_count": int(source["frame_count"]) * request.factor, "has_audio": source["has_audio"],
    }
    job_id = str(uuid.uuid4())
    job = store.create(job_id, request.model_dump(), resolved, kind="interpolate")
    runner.wake()
    return job


@app.post("/api/jobs", status_code=201)
def create_job(request: GenerationRequest):
    try:
        resolve_asset(settings.model_root, "repository", request.model)
        for lora in request.loras:
            if lora.enabled:
                resolve_asset(settings.lora_root, "lora", lora.path)
        for upload_id in (request.first_frame, request.last_frame):
            if upload_id:
                target = (settings.uploads / upload_id).resolve()
                if not target.is_relative_to(settings.uploads.resolve()) or not target.is_file():
                    raise FileNotFoundError(f"Unknown upload: {upload_id}")
        if request.latent_refine.enabled:
            from .latent_upscale import validated_weight_path
            validated_weight_path(settings.latent_upscaler_root, str(request.latent_refine.model))
    except FileNotFoundError as exc:
        raise HTTPException(422, str(exc)) from exc
    job_id = str(uuid.uuid4())
    job = store.create(job_id, request.model_dump(), request.resolved())
    runner.wake()
    return job


@app.get("/api/jobs")
def list_jobs(limit: int = 100):
    return [_public_job(job) for job in store.list(max(1, min(limit, 500)))]


def _public_job(job: dict[str, object]) -> dict[str, object]:
    result = dict(job)
    if result.get("output_path"):
        result["thumbnail_url"] = f"/api/jobs/{result['id']}/thumbnail"
    return result


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    try:
        return _public_job(store.get(job_id))
    except KeyError as exc:
        raise HTTPException(404, "生成履歴が見つかりません") from exc


@app.get("/api/jobs/{job_id}/thumbnail")
def job_thumbnail(job_id: str):
    try:
        job = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "生成履歴が見つかりません") from exc
    relative = job.get("output_path")
    if not relative:
        raise HTTPException(404, "サムネイル対象の動画がありません")
    video = (settings.outputs / relative).resolve()
    if not video.is_relative_to(settings.outputs.resolve()) or not video.is_file() or video.suffix.lower() != ".mp4":
        raise HTTPException(404, "出力MP4が見つかりません")
    try:
        thumbnail = create_thumbnail(video, settings.thumbnails, job_id)
    except Exception as exc:
        raise HTTPException(422, f"サムネイルを作成できません: {exc}") from exc
    return FileResponse(
        thumbnail, media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=0, must-revalidate"},
    )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return runner.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, "生成履歴が見つかりません") from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.post("/api/jobs/{job_id}/replay", status_code=201)
def replay_job(job_id: str):
    try:
        original = store.get(job_id)
    except KeyError as exc:
        raise HTTPException(404, "生成履歴が見つかりません") from exc
    replay = dict(original["request"])
    kind = original.get("kind", "generate")
    if kind == "upscale":
        return create_upscale_job(UpscaleRequest.model_validate(replay))
    if kind == "interpolate":
        return create_interpolate_job(InterpolateRequest.model_validate(replay))
    replay["seed"] = original["resolved"]["seed"]
    return create_job(GenerationRequest.model_validate(replay))


@app.post("/api/jobs/{job_id}/retry-finalize")
def retry_finalize(job_id: str):
    try:
        return runner.retry_finalize(job_id)
    except KeyError as exc:
        raise HTTPException(404, "生成履歴が見つかりません") from exc
    except (RuntimeError, OSError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/outputs/{relative_path:path}")
def output_file(relative_path: str):
    target = (settings.outputs / relative_path).resolve()
    if not target.is_relative_to(settings.outputs.resolve()) or not target.is_file():
        raise HTTPException(404, "出力が見つかりません")
    media_type, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=media_type, filename=target.name if relative_path.endswith(".json") else None)


@app.post("/api/outputs/open")
def open_outputs():
    if os.name != "nt":
        raise HTTPException(501, "この環境ではフォルダを開けません")
    subprocess.Popen(["explorer.exe", str(settings.outputs)], shell=False)
    return {"opened": str(settings.outputs)}
