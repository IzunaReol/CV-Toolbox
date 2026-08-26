"""编排层：依次调三个脚本完成 抽帧 → 推理 → 合成视频。

加载方式：所有脚本的 import 都走 importlib.util.spec_from_file_location，
原因：脚本在 scripts/ 这个独立目录里（不在 web/ 包内），普通 import 语法找不到。

注意：永远不要在这里执行脚本的 main()，直接调用纯函数
（extract_frames / infer / create_video_from_images），否则会触发它们的
input() 交互式阻塞逻辑。
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

# 让 helpers / 三个被 importlib 加载的脚本在任何调用方式下都能找到
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .helpers import read_video_meta, safe_stem
except ImportError:  # 当作顶层模块运行（streamlit run web/app.py）时回落
    from helpers import read_video_meta, safe_stem


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

Stage = Literal["start", "extract", "infer", "encode", "done", "error"]


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
    output_video: Optional[Path] = None
    frames_dir: Optional[Path] = None
    annotated_dir: Optional[Path] = None
    error: Optional[str] = None


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
    fps: Optional[int],
    selected_classes: Optional[list[int]] = None,
    mode: Literal["full", "extract", "infer", "encode"] = "full",
    frames_dir: Optional[Path] = None,
    annotated_dir: Optional[Path] = None,
    uploads_root: Path = UPLOADS_DIR,
    outputs_root: Path = OUTPUTS_DIR,
    progress_cb: Optional[Callable[[PipelineEvent], None]] = None,
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

    # 模式前置校验
    if mode in ("full", "extract") and not video_paths:
        raise ValueError(f"{mode} 模式至少需要一个视频路径")
    if mode in ("full", "infer") and (model_path is None or not Path(model_path).exists()):
        raise ValueError(f"{mode} 模式需要有效的 model_path")
    if mode in ("full", "infer") and selected_classes is not None and not selected_classes:
        raise ValueError("至少需要选择一个推理类别")

    results: list[VideoResult] = []

    # encode 模式没有 video_paths 时按 frames_dir/annotated_dir 推断 stem
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

    for stem, raw_video in iter_targets:
        res = VideoResult(stem=stem)

        def emit(stage: Stage, current: int = 0, total: int = 0, message: str = ""):
            if progress_cb:
                progress_cb(PipelineEvent(stem, stage, current, total, message))

        try:
            emit("start", 0, 0, f"准备处理 {stem}（mode={mode}）")

            # 默认 frames_dir / annotated_dir 推断
            cur_frames_dir = frames_dir or (outputs_root / stem / "frames")
            cur_annotated_root = (annotated_dir.parent if annotated_dir
                                   else outputs_root / stem / "annotated")
            cur_annotated_images = (annotated_dir
                                    if annotated_dir
                                    else cur_annotated_root / "images")

            # ---- Stage: extract ----
            if mode in ("full", "extract"):
                cur_frames_dir.mkdir(parents=True, exist_ok=True)
                res.frames_dir = cur_frames_dir
                _, total_frames = read_video_meta(raw_video)
                total_to_save = max(total_frames // max(frame_interval, 1), 1)
                emit("extract", 0, total_to_save,
                     f"开始抽帧（间隔 {frame_interval} 帧）")

                extract_frames(
                    video_path=str(raw_video),
                    output_folder=str(cur_frames_dir),
                    frame_interval=frame_interval,
                    name_prefix=stem,
                )
                emit("extract", total_to_save, total_to_save, "抽帧完成")

            # ---- Stage: infer ----
            if mode in ("full", "infer"):
                cur_annotated_root.mkdir(parents=True, exist_ok=True)
                cur_annotated_images.mkdir(parents=True, exist_ok=True)
                res.annotated_dir = cur_annotated_images

                if not cur_frames_dir.exists() or not any(cur_frames_dir.glob("*.jpg")):
                    raise RuntimeError(f"frames 目录为空或不存在: {cur_frames_dir}")
                n_frames = sum(1 for _ in cur_frames_dir.glob("*.jpg"))
                emit("infer", 0, max(n_frames, 1), "加载模型并开始推理")

                infer(
                    model_path=str(model_path),
                    input_dir=str(cur_frames_dir),
                    output_dir=str(cur_annotated_root),
                    conf_thres=conf,
                    iou_thres=iou,
                    device=device,
                    box_color=box_color,
                    label_map=label_map,
                    classes=selected_classes,
                )
                emit("infer", n_frames, max(n_frames, 1), "推理完成")

            # ---- Stage: encode ----
            if mode in ("full", "encode"):
                out_video_path = outputs_root / stem / f"{stem}.mp4"
                res.output_video = out_video_path

                if fps is None:
                    # 优先从 annotated 推断；若没有 source video 则回退 30
                    if raw_video is not None:
                        orig_fps, _ = read_video_meta(raw_video)
                        effective_fps = max(int(round(orig_fps)), 1)
                    elif cur_frames_dir.exists():
                        effective_fps = 30
                    else:
                        effective_fps = 30
                else:
                    effective_fps = max(int(fps), 1)

                n_frames = sum(1 for _ in cur_annotated_images.glob("*.jpg")) \
                    if cur_annotated_images.exists() else 0
                emit("encode", 0, max(n_frames, 1),
                     f"合成视频（{effective_fps} fps）")

                ok = create_video_from_images(
                    image_dir=str(cur_annotated_images),
                    output_video=str(out_video_path),
                    fps=effective_fps,
                )
                if not ok:
                    raise RuntimeError(
                        "视频合成失败，请检查标注目录是否包含有效图片")

            emit("done", 1, 1, f"完成 -> {res.output_video or res.annotated_dir or res.frames_dir}")
        except Exception as exc:  # 单视频失败不中断
            res.error = str(exc)
            emit("error", 0, 0, f"{stem} 失败: {exc}")
        results.append(res)

    return results


def cache_uploaded_model(uploaded_bytes: bytes, original_name: str,
                         cache_root: Path = UPLOADS_DIR / "models") -> Path:
    """按 sha1 前 12 位把上传的模型缓存到 uploads/models/，返回落盘路径。"""
    import hashlib
    cache_root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(uploaded_bytes).hexdigest()[:12]
    safe_name = safe_stem(Path(original_name).stem)
    target = cache_root / f"{safe_name}_{digest}.pt"
    if not target.exists():
        target.write_bytes(uploaded_bytes)
    return target


def save_uploaded_video(uploaded_bytes: bytes, original_name: str,
                        uploads_root: Path = UPLOADS_DIR) -> Path:
    """把上传的视频复制到 uploads/<stem>/video.mp4，返回落盘路径。

    v1.0.1: 内部文件名从 v.mp4 改为 video.mp4，避免与字母 v 混淆。
    """
    stem = safe_stem(Path(original_name).stem)
    target_dir = uploads_root / stem
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "video.mp4"
    if not target.exists() or target.stat().st_size != len(uploaded_bytes):
        target.write_bytes(uploaded_bytes)
    return target
