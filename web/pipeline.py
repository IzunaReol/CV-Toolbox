"""编排层：依次调三个脚本完成 抽帧 → 推理 → 合成视频。

加载方式：所有脚本的 import 都走 importlib.util.spec_from_file_location，
原因：脚本在 scripts/ 这个独立目录里（不在 web/ 包内），普通 import 语法找不到。

注意：永远不要在这里执行脚本的 main()，直接调用纯函数
（extract_frames / infer / create_video_from_images），否则会触发它们的
input() 交互式阻塞逻辑。
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional
from uuid import uuid4

import cv2

# 让 helpers / 三个被 importlib 加载的脚本在任何调用方式下都能找到
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .helpers import read_video_meta, safe_stem
    from .media import VerifiedVideoWriter, list_images, positive_fps
    from .task_store import bump_artifact_revision, update_task, utc_now, write_inference_stats
except ImportError:  # 当作顶层模块运行（streamlit run web/app.py）时回落
    from helpers import read_video_meta, safe_stem
    from media import VerifiedVideoWriter, list_images, positive_fps
    from task_store import bump_artifact_revision, update_task, utc_now, write_inference_stats


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


# ---------- 模块加载 ----------


def _load_module(alias: str, file_path: Path):
    """通过文件路径加载 Python 模块。"""
    spec = importlib.util.spec_from_file_location(alias, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module  # 让模块内部互引可用
    spec.loader.exec_module(module)
    return module


_extract_mod = _load_module("video_extract", SCRIPTS_DIR / "1_frame_extract.py")
_infer_mod = _load_module("model_infer", SCRIPTS_DIR / "2_model_infer.py")
_video_mod = _load_module("img_to_video", SCRIPTS_DIR / "3_images_to_video.py")

extract_frames = _extract_mod.extract_frames
infer = _infer_mod.infer
create_video_from_images = _video_mod.create_video_from_images
parse_color = _infer_mod.parse_color
parse_label_map = _infer_mod.parse_label_map
parse_classes = _infer_mod.parse_classes


# ---------- 模型元信息 ----------


def get_model_class_names(model_path: Path) -> dict[int, str]:
    """加载一次 YOLO 模型，读取其内置的类别名 {id: name}。

    调用方应自行按 (path, mtime) 缓存，避免每次 UI rerun 都重新加载权重。
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = getattr(model, "names", {}) or {}
    # `names` 的 key 可能是 int 或 str，统一转为 int
    return {int(k): str(v) for k, v in names.items()}


# ---------- 事件类型 ----------

Stage = Literal[
    "queued",
    "start",
    "extract",
    "infer",
    "encode",
    "done",
    "cancelled",
    "interrupted",
    "error",
]


@dataclass
class PipelineEvent:
    video_stem: str
    stage: Stage
    current: int = 0
    total: int = 0
    message: str = ""


@dataclass
class VideoResult:
    stem: str
    started_at: Optional[str] = None
    mode: str = "未知"
    output_video: Optional[Path] = None
    frames_dir: Optional[Path] = None
    annotated_dir: Optional[Path] = None
    error: Optional[str] = None
    status: str = "queued"
    stats: Optional[dict] = None


# ---------- 主入口 ----------


def run_pipeline(
    *,
    video_paths: list[Path],
    model_path: Optional[Path],
    frame_interval: int,
    conf: float,
    iou: float,
    device: str,
    box_color: tuple,
    label_map: Optional[dict],
    fps: Optional[float],
    selected_classes: Optional[list[int]] = None,
    mode: Literal["full", "extract", "infer", "encode"] = "full",
    frames_dir: Optional[Path] = None,
    annotated_dir: Optional[Path] = None,
    uploads_root: Path = UPLOADS_DIR,
    outputs_root: Path = OUTPUTS_DIR,
    progress_cb: Optional[Callable[[PipelineEvent], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    task_context: Optional[dict] = None,
    streaming: bool = False,
) -> list[VideoResult]:
    """对每个视频跑指定 stage。

    mode:
        full    - 抽帧 → 推理 → 合成（默认，需要 video_paths）
        extract - 仅抽帧（需要 video_paths + frame_interval）
        infer   - 仅推理（需要 model_path + frames_dir；frames_dir 默认
                  outputs/<stem>/frames/）
        encode  - 仅合成（需要 annotated_dir；fps 若为 None 则尝试从 frames_dir
                  推断或回退到 30）

    selected_classes 为 None 时推理全部模型类别；非空列表仅推理指定类别。

    进度通过 progress_cb 推送；单视频失败不会中断其他视频。
    """
    uploads_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)

    if frame_interval < 1 or not isinstance(frame_interval, int):
        raise ValueError("抽帧间隔必须是正整数")
    if mode not in {"full", "extract", "infer", "encode"}:
        raise ValueError("未知处理模式")
    if fps is not None:
        positive_fps(fps)
    if streaming and mode != "full":
        raise ValueError("流式处理仅支持全流程模式")
    # 模式前置校验
    if mode in ("full", "extract") and not video_paths:
        raise ValueError(f"{mode} 模式至少需要一个视频路径")
    if mode in ("full", "infer") and (model_path is None or not Path(model_path).exists()):
        raise ValueError(f"{mode} 模式需要有效的 model_path")
    if mode in ("full", "infer") and selected_classes is not None and not selected_classes:
        raise ValueError("至少需要选择一个推理类别")

    results: list[VideoResult] = []

    # encode 模式没有 video_paths 时按 frames_dir/annotated_dir 推断 stem
    iter_targets: list[tuple[str, Optional[Path]]]
    if mode == "encode":
        if annotated_dir:
            # v1.0.2 新约定: outputs/<stem>/_uploaded/*.jpg → parent.name = stem
            # 旧约定: outputs/<stem>/annotated/images/*.jpg → parent.parent.name = stem
            # 兜底: 直接 _annotated 目录名 (split)
            if annotated_dir.parent.name in ("annotated", "images"):
                # 旧约定: outputs/<stem>/annotated/images/...
                if annotated_dir.parent.name == "annotated":
                    stems = [safe_stem(annotated_dir.parent.parent.name)]
                else:
                    # outputs/<stem>/annotated/images → parent.parent.name = stem
                    stems = [safe_stem(annotated_dir.parent.parent.name)]
            else:
                # v1.0.2: outputs/<stem>/_uploaded/... → parent.name = stem
                stems = [safe_stem(annotated_dir.parent.name)]
        elif frames_dir:
            stems = [safe_stem(frames_dir.parent.name)]
        else:
            stems = ["output"]
    elif mode == "infer":
        # frames_dir 在 v1.0.2 走 outputs/<stem>/_uploaded/，parent.name = stem
        stems = [safe_stem(frames_dir.parent.name)] if frames_dir else ["infer"]
    else:
        # full/extract: stem 取自 uploads/<stem>/video.mp4 的父目录名
        stems = [safe_stem(p.parent.name) for p in video_paths]

    # 循环入口：encode/infer 用单一虚拟 raw_video；full/extract 用真实视频
    if mode == "encode":
        iter_targets = list(zip(stems, [None] * len(stems)))
    elif mode == "infer":
        iter_targets = list(zip(stems, [None] * len(stems)))
    else:
        iter_targets = [(safe_stem(p.parent.name), p) for p in video_paths]

    runtime = None
    for stem, raw_video in iter_targets:
        original_stem = stem
        existing = outputs_root / stem
        if (
            (existing / "_meta" / "task.json").exists()
            or (existing / "frames").exists()
            or (existing / "annotated").exists()
            or (existing / f"{stem}.mp4").exists()
        ):
            stem = f"{stem}_{uuid4().hex[:12]}"
        res = VideoResult(stem=stem, started_at=utc_now(), mode=mode, status="running")
        task_root = outputs_root / stem
        task_root.mkdir(parents=True, exist_ok=True)
        last_persisted = 0.0

        def emit(stage: Stage, current: int = 0, total: int = 0, message: str = ""):
            nonlocal last_persisted
            if progress_cb:
                progress_cb(PipelineEvent(stem, stage, current, total, message))
            now = time.monotonic()
            if (
                stage in ("start", "done", "cancelled", "interrupted", "error")
                or now - last_persisted >= 0.5
            ):
                update_task(
                    task_root,
                    task_id=stem,
                    batch_id=(task_context or {}).get("batch_id"),
                    mode=mode,
                    status={
                        "queued": "queued",
                        "start": "running",
                        "extract": "running",
                        "infer": "running",
                        "encode": "running",
                        "done": "completed",
                        "error": "failed",
                        "cancelled": "cancelled",
                        "interrupted": "interrupted",
                    }.get(stage, stage),
                    stage=stage,
                    current=current,
                    total=total,
                    message=message,
                    config=(task_context or {}).get("config", {}),
                    inputs=(task_context or {}).get("inputs", {}).get(original_stem, []),
                    started_at=res.started_at,
                    **(
                        {"finished_at": utc_now()}
                        if stage in {"done", "error", "cancelled", "interrupted"}
                        else {}
                    ),
                )
                last_persisted = now

        try:
            if cancel_cb and cancel_cb():
                raise InterruptedError("任务已取消")
            emit("start", 0, 0, f"准备处理 {stem}（mode={mode}）")

            # 默认 frames_dir / annotated_dir 推断
            cur_frames_dir = frames_dir or (outputs_root / stem / "frames")
            cur_annotated_root = (
                annotated_dir.parent if annotated_dir else outputs_root / stem / "annotated"
            )
            cur_annotated_images = annotated_dir if annotated_dir else cur_annotated_root / "images"

            effective_fps = None
            if mode in ("full", "encode"):
                original_fps = read_video_meta(raw_video)[0] if raw_video else 30.0
                effective_fps = resolve_output_fps(original_fps, fps)
                update_task(task_root, effective_fps=effective_fps, streaming=streaming)

            if mode in ("full", "extract") and not streaming:
                res.frames_dir = cur_frames_dir
                extract_frames(
                    str(raw_video),
                    str(cur_frames_dir),
                    frame_interval,
                    name_prefix=stem,
                    progress_cb=lambda current, total: emit(
                        "extract", current, total, f"抽帧 {current}/{total}"
                    ),
                    cancel_cb=cancel_cb,
                )
                bump_artifact_revision(outputs_root)

            if mode in ("full", "infer"):
                if cancel_cb and cancel_cb():
                    raise InterruptedError("任务已取消")
                emit("infer", 0, 1, "加载模型并开始推理")
                if runtime is None:
                    runtime = _infer_mod.load_model(str(model_path), device)
                infer_kwargs = dict(
                    model_path=str(model_path),
                    input_dir=str(cur_frames_dir),
                    output_dir=str(cur_annotated_root),
                    conf_thres=conf,
                    iou_thres=iou,
                    device=device,
                    box_color=box_color,
                    label_map=label_map,
                    classes=selected_classes,
                    runtime=runtime,
                    cancel_cb=cancel_cb,
                    progress_cb=lambda current, total: emit(
                        "infer", current, total, f"推理 {current}/{total}"
                    ),
                )
                if streaming:
                    out_video_path = task_root / f"{stem}.mp4"
                    _, total_frames = read_video_meta(raw_video)
                    total_images = (total_frames + frame_interval - 1) // frame_interval
                    with closing(iter_video_frames(raw_video, frame_interval, cancel_cb)) as source:
                        with VerifiedVideoWriter(out_video_path, effective_fps) as writer:
                            inference_stats = infer(
                                **infer_kwargs,
                                image_source=source,
                                total_images=total_images,
                                frame_cb=writer.write,
                                save_images=False,
                            )
                            if cancel_cb and cancel_cb():
                                raise InterruptedError("任务已取消")
                    res.output_video = out_video_path
                else:
                    if not list_images(cur_frames_dir):
                        raise RuntimeError(f"没有可推理的图片: {cur_frames_dir}")
                    res.annotated_dir = cur_annotated_images
                    inference_stats = infer(**infer_kwargs)
                res.stats = inference_stats
                stats_json, stats_csv = write_inference_stats(task_root, inference_stats)
                update_task(
                    task_root,
                    actual_device=inference_stats.get("actual_device"),
                    statistics={
                        key: inference_stats.get(key)
                        for key in (
                            "total_images",
                            "processed_images",
                            "failed_images",
                            "matched_images",
                            "class_counts",
                        )
                    },
                    statistics_files=[str(stats_json), str(stats_csv)],
                )
                if mode == "full" and inference_stats.get("failed_images", 0):
                    raise RuntimeError("部分帧无法读取，停止合成以避免视频静默丢帧")
                bump_artifact_revision(outputs_root)

            if mode in ("full", "encode") and not streaming:
                if cancel_cb and cancel_cb():
                    raise InterruptedError("任务已取消")
                out_video_path = task_root / f"{stem}.mp4"
                n_frames = len(list_images(cur_annotated_images))
                emit("encode", 0, max(n_frames, 1), f"合成视频（{effective_fps:g} fps）")
                ok = create_video_from_images(
                    str(cur_annotated_images),
                    str(out_video_path),
                    fps=effective_fps,
                    progress_cb=lambda current, total: emit(
                        "encode", current, total, f"合成 {current}/{total}"
                    ),
                    cancel_cb=cancel_cb,
                )
                if not ok:
                    raise RuntimeError("视频合成失败，请检查输入图片")
                res.output_video = out_video_path

            res.status = "completed"
            emit("done", 1, 1, f"完成 -> {res.output_video or res.annotated_dir or res.frames_dir}")
            bump_artifact_revision(outputs_root)
        except InterruptedError as exc:
            res.status = "cancelled"
            res.error = str(exc)
            emit("cancelled", 0, 0, f"{stem} 已取消，已生成文件予以保留")
            bump_artifact_revision(outputs_root)
            results.append(res)
            break
        except Exception as exc:  # 单视频失败不中断
            res.status = "failed"
            res.error = str(exc)
            emit("error", 0, 0, f"{stem} 失败: {exc}")
            bump_artifact_revision(outputs_root)
        results.append(res)

    return results


_MODEL_CACHE_LOCK = threading.Lock()


def cache_uploaded_model(
    uploaded_bytes: bytes, original_name: str, cache_root: Path = UPLOADS_DIR / "models"
) -> Path:
    """按完整 SHA-256 存储不可变模型；原子发布避免读取到半写入权重。"""
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(uploaded_bytes).hexdigest()
    target = cache_root / f"{safe_stem(Path(original_name).stem)}_{digest}.pt"
    with _MODEL_CACHE_LOCK:
        if not target.exists():
            temp = target.with_suffix(f".{uuid4().hex}.tmp")
            try:
                temp.write_bytes(uploaded_bytes)
                temp.replace(target)
            finally:
                temp.unlink(missing_ok=True)
    return target


def save_uploaded_video(
    uploaded_bytes: bytes, original_name: str, uploads_root: Path = UPLOADS_DIR
) -> Path:
    """每次上传分配独立 ID，隔离同名、同大小视频及历史输出。"""
    stem = f"{safe_stem(Path(original_name).stem)}_{uuid4().hex[:12]}"
    target_dir = uploads_root / stem
    target_dir.mkdir(parents=True, exist_ok=False)
    target = target_dir / "video.mp4"
    target.write_bytes(uploaded_bytes)
    return target


def resolve_output_fps(original_fps, fps=None):
    """自定义值优先，否则使用源视频帧率，不根据抽帧间隔换算。"""
    return positive_fps(fps if fps is not None else original_fps)


def iter_video_frames(video_path, interval, cancel_cb=None):
    """流式解码，每次仅保留当前采样帧；生成器关闭时释放视频。"""
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")
        index = 0
        while True:
            if cancel_cb and cancel_cb():
                raise InterruptedError("任务已取消")
            ok, frame = cap.read()
            if not ok:
                break
            if index % interval == 0:
                yield Path(f"frame_{index:06d}.jpg"), frame
            index += 1
    finally:
        cap.release()
