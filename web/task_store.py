"""任务元数据、推理统计与工件修订号的持久化。"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

META_DIR_NAME = "_meta"
TASK_FILE_NAME = "task.json"
STATS_FILE_NAME = "inference_stats.json"
STATS_CSV_NAME = "inference_images.csv"
JOBS_DIR_NAME = "_jobs"
REVISION_FILE_NAME = "artifact_revision.txt"
SCHEMA_VERSION = 1

_WRITE_LOCK = threading.RLock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with _WRITE_LOCK:
        _atomic_write_text(
            Path(path), json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        )


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def task_meta_dir(task_root: Path) -> Path:
    return Path(task_root) / META_DIR_NAME


def task_file(task_root: Path) -> Path:
    return task_meta_dir(task_root) / TASK_FILE_NAME


def read_task(task_root: Path) -> dict[str, Any] | None:
    return read_json(task_file(task_root))


def update_task(task_root: Path, **changes: Any) -> dict[str, Any]:
    path = task_file(task_root)
    with _WRITE_LOCK:
        current = read_json(path) or {"schema_version": SCHEMA_VERSION}
        current.setdefault("created_at", utc_now())
        current.update(changes)
        current["updated_at"] = utc_now()
        write_json(path, current)
    return current


def write_inference_stats(task_root: Path, stats: dict[str, Any]) -> tuple[Path, Path]:
    meta_dir = task_meta_dir(task_root)
    json_path = meta_dir / STATS_FILE_NAME
    csv_path = meta_dir / STATS_CSV_NAME
    write_json(json_path, stats)
    rows = stats.get("images", []) if isinstance(stats, dict) else []
    with _WRITE_LOCK:
        meta_dir.mkdir(parents=True, exist_ok=True)
        temp = csv_path.with_suffix(".csv.tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "file_name",
                    "status",
                    "detections",
                    "class_counts",
                    "max_confidence",
                    "avg_confidence",
                    "error",
                ],
            )
            writer.writeheader()
            for row in rows:
                item = dict(row)
                item["class_counts"] = json.dumps(
                    item.get("class_counts", {}), ensure_ascii=False, sort_keys=True
                )
                writer.writerow({key: item.get(key, "") for key in writer.fieldnames})
        temp.replace(csv_path)
    return json_path, csv_path


def jobs_dir(outputs_root: Path) -> Path:
    return Path(outputs_root) / JOBS_DIR_NAME


def job_file(outputs_root: Path, batch_id: str) -> Path:
    return jobs_dir(outputs_root) / f"{batch_id}.json"


def artifact_revision(outputs_root: Path) -> int:
    path = jobs_dir(outputs_root) / REVISION_FILE_NAME
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def bump_artifact_revision(outputs_root: Path) -> int:
    with _WRITE_LOCK:
        revision = artifact_revision(outputs_root) + 1
        _atomic_write_text(jobs_dir(outputs_root) / REVISION_FILE_NAME, str(revision))
    return revision
