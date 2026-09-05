"""单工作线程后台流水线与可恢复任务状态。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .pipeline import PipelineEvent, VideoResult, run_pipeline
    from .task_store import job_file, jobs_dir, read_json, utc_now, write_json
except ImportError:
    from pipeline import PipelineEvent, VideoResult, run_pipeline
    from task_store import job_file, jobs_dir, read_json, utc_now, write_json


_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cv-toolbox")
_LOCK = threading.RLock()
_JOBS: dict[str, "BackgroundJob"] = {}


@contextmanager
def idle_outputs(outputs_root: Path):
    """删除与任务提交互斥，避免处理中删除工件或后台状态文件。"""
    with _LOCK:
        status = active_status(outputs_root)
        if status and status.get("status") in {"queued", "running", "cancelling"}:
            raise ValueError("批次正在运行，请完成或取消后再删除本地文件")
        yield


class BackgroundJob:
    def __init__(self, batch_id: str, status_path: Path):
        self.batch_id = batch_id
        self.status_path = status_path
        self.cancel_event = threading.Event()
        self.future: Future | None = None


def _serialize_result(result: VideoResult) -> dict[str, Any]:
    payload = asdict(result)
    for key in ("output_video", "frames_dir", "annotated_dir"):
        value = payload.get(key)
        payload[key] = str(value) if value else None
    return payload


def _write_status(path: Path, **changes: Any) -> dict[str, Any]:
    with _LOCK:
        current = read_json(path) or {"schema_version": 1}
        current.update(changes)
        current["updated_at"] = utc_now()
        write_json(path, current)
    return current


def _run_job(job: BackgroundJob, kwargs: dict[str, Any]) -> None:
    _write_status(job.status_path, status="running", started_at=utc_now())

    last_progress = 0.0
    last_stage = None

    def progress(event: PipelineEvent) -> None:
        nonlocal last_progress, last_stage
        now = time.monotonic()
        stage_key = (event.video_stem, event.stage)
        if (
            stage_key == last_stage
            and now - last_progress < 0.5
            and event.stage not in {"done", "error", "cancelled"}
        ):
            return
        last_progress, last_stage = now, stage_key
        _write_status(
            job.status_path,
            status="cancelling" if job.cancel_event.is_set() else "running",
            current_task=event.video_stem,
            stage=event.stage,
            current=event.current,
            total=event.total,
            message=event.message,
        )

    try:
        results = run_pipeline(
            **kwargs,
            progress_cb=progress,
            cancel_cb=job.cancel_event.is_set,
        )
        cancelled = job.cancel_event.is_set() or any(r.status == "cancelled" for r in results)
        failed = any(r.status == "failed" for r in results)
        status = "cancelled" if cancelled else ("failed" if failed else "completed")
        _write_status(
            job.status_path,
            status=status,
            stage=status,
            finished_at=utc_now(),
            results=[_serialize_result(result) for result in results],
            message={
                "completed": "批次处理完成",
                "cancelled": "批次已取消，部分工件已保留",
                "failed": "批次处理结束，存在失败任务",
            }[status],
        )
    except Exception as exc:
        _write_status(
            job.status_path,
            status="failed",
            stage="error",
            finished_at=utc_now(),
            error=str(exc),
            message=f"后台任务失败: {exc}",
        )


def submit_pipeline(
    *,
    batch_id: str,
    outputs_root: Path,
    kwargs: dict[str, Any],
    stems: list[str],
) -> dict[str, Any]:
    """提交一个批次；任意批次运行时拒绝新任务。"""
    with _LOCK:
        active = active_status(outputs_root, reconcile=False)
        if active and active.get("status") in {"queued", "running", "cancelling"}:
            raise RuntimeError("已有批次正在运行，请等待完成或先取消")
        for key, previous in list(_JOBS.items()):
            if previous.future and previous.future.done():
                del _JOBS[key]
        path = job_file(outputs_root, batch_id)
        status = {
            "schema_version": 1,
            "batch_id": batch_id,
            "mode": kwargs.get("mode", "未知"),
            "status": "queued",
            "stage": "queued",
            "created_at": utc_now(),
            "stems": stems,
            "current": 0,
            "total": 0,
            "message": "等待后台工作线程",
            "results": [],
        }
        write_json(path, status)
        job = BackgroundJob(batch_id, path)
        _JOBS[batch_id] = job
        job.future = _EXECUTOR.submit(_run_job, job, kwargs)
        return status


def cancel_batch(outputs_root: Path, batch_id: str) -> bool:
    with _LOCK:
        job = _JOBS.get(batch_id)
        if not job or not job.future or job.future.done():
            return False
        job.cancel_event.set()
        _write_status(
            job.status_path,
            status="cancelling",
            message="正在取消：当前帧或图片完成后停止",
        )
        return True


def active_status(outputs_root: Path, *, reconcile: bool = True) -> dict[str, Any] | None:
    directory = jobs_dir(outputs_root)
    if not directory.exists():
        return None
    records = []
    for path in directory.glob("*.json"):
        record = read_json(path)
        if record:
            records.append((path, record))
    if not records:
        return None
    path, latest = max(records, key=lambda item: item[0].stat().st_mtime_ns)
    if reconcile and latest.get("status") in {"queued", "running", "cancelling"}:
        batch_id = str(latest.get("batch_id", ""))
        job = _JOBS.get(batch_id)
        if not job or not job.future or job.future.done():
            latest = _write_status(
                path,
                status="interrupted",
                stage="interrupted",
                finished_at=utc_now(),
                message="服务已重启，原后台任务已中断",
            )
    return latest


def results_from_status(status: dict[str, Any] | None) -> list[VideoResult]:
    results: list[VideoResult] = []
    status_data = status or {}
    for item in status_data.get("results", []):
        results.append(
            VideoResult(
                stem=str(item.get("stem", "task")),
                started_at=item.get("started_at") or status_data.get("started_at"),
                mode=str(
                    status_data.get("mode")
                    if item.get("mode") in {None, "", "未知"}
                    else item.get("mode")
                ),
                output_video=Path(item["output_video"]) if item.get("output_video") else None,
                frames_dir=Path(item["frames_dir"]) if item.get("frames_dir") else None,
                annotated_dir=Path(item["annotated_dir"]) if item.get("annotated_dir") else None,
                error=item.get("error"),
                status=str(item.get("status", "completed")),
                stats=item.get("stats"),
            )
        )
    return results
