"""任务元数据、推理统计与工件修订号的持久化。"""

from __future__ import annotations

import csv
import json
import threading
import time
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


class InferenceJournal:
    """逐图追加写入，内存仅保留汇总；异常退出时也保存已处理部分。"""

    def __init__(self, task_root: Path):
        self.directory = task_meta_dir(task_root)
        self.paths = (self.directory / STATS_FILE_NAME, self.directory / STATS_CSV_NAME)
        self.rows_path = self.directory / "inference_images.jsonl"
        self.count = 0
        self.failed = 0
        self.matched = 0
        self.class_counts = {}
        self.actual_device = None
        self.last_flush = 0.0

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = self.rows_path.open("w", encoding="utf-8")
        try:
            self.csv = self.paths[1].open("w", encoding="utf-8-sig", newline="")
        except BaseException:
            self.rows.close()
            raise
        self.writer = csv.DictWriter(
            self.csv,
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
        self.writer.writeheader()
        return self

    def __len__(self):
        return self.count

    def append(self, row):
        self.rows.write(json.dumps(row, ensure_ascii=False) + "\n")
        csv_row = dict(row)
        csv_row["class_counts"] = json.dumps(row.get("class_counts", {}), ensure_ascii=False)
        self.writer.writerow(csv_row)
        self.count += 1
        self.failed += row["status"] == "failed"
        self.matched += row.get("detections", 0) > 0
        for key, value in row.get("class_counts", {}).items():
            self.class_counts[key] = self.class_counts.get(key, 0) + value
        if time.monotonic() - self.last_flush >= 0.5:
            self.rows.flush()
            self.csv.flush()
            self.last_flush = time.monotonic()

    def __exit__(self, exc_type, exc, tb):
        try:
            self.rows.close()
        finally:
            self.csv.close()
        summary = {
            "total_images": self.count,
            "processed_images": self.count - self.failed,
            "failed_images": self.failed,
            "matched_images": self.matched,
            "class_counts": self.class_counts,
            "actual_device": self.actual_device,
            "status": "cancelled"
            if isinstance(exc, InterruptedError)
            else ("failed" if exc else "completed"),
            "partial": exc is not None,
        }
        temp = self.paths[0].with_suffix(".json.tmp")
        try:
            with temp.open("w", encoding="utf-8") as output:
                output.write(json.dumps(summary, ensure_ascii=False)[:-1] + ', "images": [')
                with self.rows_path.open(encoding="utf-8") as rows:
                    for index, line in enumerate(rows):
                        output.write(("," if index else "") + line.strip())
                output.write("]}")
            temp.replace(self.paths[0])
            self.rows_path.unlink(missing_ok=True)
        finally:
            temp.unlink(missing_ok=True)
        return False


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
