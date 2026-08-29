"""工具函数：路径清洗、FPS/帧数读取、ZIP 打包、文本框解析。"""

from __future__ import annotations

import hashlib
import io
import re
import threading
import zipfile
from pathlib import Path

import cv2
import streamlit as st

_DOWNLOAD_CACHE_LOCK = threading.Lock()
_DOWNLOAD_CACHE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "_downloads"


# ---------- 文本框解析 ----------


def parse_positive_int_text(value: str, default: int) -> tuple[int, bool]:
    """把文本框输入解析为正整数。

    返回 (值, 是否回退到默认值)。空字符串、非法字符、<=0 一律回退。
    """
    if value is None:
        return default, True
    raw = str(value).strip()
    if not raw:
        return default, True
    try:
        parsed = int(raw)
        if parsed < 1:
            return default, True
        return parsed, False
    except (ValueError, TypeError):
        return default, True


def parse_positive_float_text(value: str, default: float) -> tuple[float, bool]:
    """把文本框输入解析为 (0, 1] 范围内的浮点数。

    接受 0.25 / 0,25 / .25 等写法；空或非法一律回退。
    返回 (值, 是否回退)。
    """
    if value is None:
        return default, True
    raw = str(value).strip().replace(",", ".")
    if not raw:
        return default, True
    try:
        parsed = float(raw)
        if parsed <= 0 or parsed > 1:
            return default, True
        return parsed, False
    except (ValueError, TypeError):
        return default, True


# ---------- 颜色名映射 ----------

COLOR_NAME_ZH_TO_EN: dict[str, str] = {
    "红色": "red",
    "绿色": "green",
    "蓝色": "blue",
    "黄色": "yellow",
    "青色": "cyan",
    "品红": "magenta",
    "白色": "white",
    "黑色": "black",
    "橙色": "orange",
    "紫色": "purple",
}


# ---------- 类别表 → label_map ----------


def label_table_to_dict(
    rows: list[tuple[int, str, str]],
    unified: str,
) -> dict[str, str]:
    """把"每个类别一个输入框"渲染成 infer() 期望的 {str(cid): str(name)}。

    rows: [(class_id, current_input_value, model_default_name), ...]
    unified: 非空时整体覆盖所有类；为空时按每行 current_input_value 处理。

    行为：
      - unified 非空：返回 {str(cid): unified}，覆盖模型原名。
      - 否则仅保留与模型默认不同的行，减少无意义键。
    """
    unified_clean = unified.strip()
    out: dict[str, str] = {}
    for cid, current, default in rows:
        if unified_clean:
            out[str(cid)] = unified_clean
            continue
        name = (current or "").strip()
        if not name or name == default:
            continue
        out[str(cid)] = name
    return out


# ---------- 路径清洗 ----------


def safe_stem(name: str) -> str:
    """把字符串清洗成 Windows 安全的目录名/文件名前缀。

    替换 Windows 路径非法字符与首尾空白；空字符串回退为 'output'。
    """
    if not name:
        return "output"
    cleaned = re.sub(r'[\\/:*?"<>|]', "_", name).strip().strip(".")
    return cleaned or "output"


# ---------- 视频元信息 ----------


def read_video_meta(video_path: Path) -> tuple[float, int]:
    """读取视频的 fps 与总帧数。

    OpenCV 对 VFR 视频可能返回 0，此时回落 30 fps。
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if fps <= 0:
        fps = 30.0
    return fps, max(total, 0)


# ---------- ZIP 打包 ----------


def build_results_zip(outputs_root: Path) -> bytes:
    """把所有 outputs/<stem>/<stem>.mp4 打包成单一 ZIP 字节流。

    ZIP 内部保留 <stem>/<stem>.mp4 目录结构，方便解压后辨识。
    没有 mp4 的子目录会被跳过。

    v1.0.2: 此函数被 build_session_zip 替代，仅保留作为遗留兼容。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if not outputs_root.exists():
            return buf.getvalue()
        for video_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
            mp4 = video_dir / f"{video_dir.name}.mp4"
            if mp4.exists() and mp4.is_file():
                zf.write(mp4, arcname=f"{video_dir.name}/{mp4.name}")
    return buf.getvalue()


def _zip_directory_glob(
    zf: zipfile.ZipFile, src_dir: Path | None, arc_prefix: str, pattern: str = "*.jpg"
) -> None:
    """把 src_dir 下符合 pattern 的文件按 arc_prefix/<name> 写入 zf。"""
    if not src_dir or not src_dir.exists():
        return
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            zf.write(f, arcname=f"{arc_prefix}/{f.name}")


@st.cache_data(show_spinner=False)
def build_session_zip(results: dict) -> bytes:
    """按当前会话的 VideoResult 字典打包 ZIP；只含 r.error is None 的项。

    各模式产物路径：
      - full:        只含 r.output_video（<stem>.mp4），不含中间帧/标注目录
      - encode:      r.output_video（<stem>.mp4）
      - extract:     r.frames_dir/*.jpg → <stem>/frames/<name>.jpg
      - infer:       r.annotated_dir/*.jpg → <stem>/annotated/images/<name>.jpg

    v1.0.2 新增：只打包当前会话在 st.session_state["results"] 里的 stem，
    不会把 outputs/ 下历史轮次的 mp4 一起打包。
    v1.0.3 变更：全流程模式（同时有 output_video + frames_dir + annotated_dir）
    只打包最终视频，与 UI 展示对齐。
    v1.0.6 变更：加 `@st.cache_data` 装饰，结果 zip 按 results 字典 key 缓存，
    同一会话内多次下载同一份 ZIP 不再重算。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in sorted(results.keys()):
            r = results[stem]
            # 跳过失败 / None 项；鸭子类型访问字段
            if getattr(r, "error", None):
                continue
            if getattr(r, "output_video", None) and r.output_video.exists():
                zf.write(r.output_video, arcname=f"{stem}/{r.output_video.name}")
            # 全流程模式：只打包最终视频，不打包中间帧/标注目录
            is_full = bool(
                getattr(r, "output_video", None)
                and getattr(r, "frames_dir", None)
                and getattr(r, "annotated_dir", None)
            )
            if is_full:
                continue
            _zip_directory_glob(zf, getattr(r, "frames_dir", None), arc_prefix=f"{stem}/frames")
            _zip_directory_glob(
                zf, getattr(r, "annotated_dir", None), arc_prefix=f"{stem}/annotated/images"
            )
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def build_frames_zip(frames_dir: Path) -> bytes:
    """单 stem 的 frames ZIP（仅抽帧结果区单条下载用）。

    v1.0.6 加 `@st.cache_data`：同一目录多次调用直接命中缓存，避免每次 rerun
    重新遍历 + 压缩（节省 CPU + 避免触发 `_ProactorBasePipeTransport` IO）。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if frames_dir.exists():
            for img in sorted(frames_dir.glob("*.jpg")):
                if img.is_file():
                    zf.write(img, arcname=img.name)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def build_infer_zip(annotated_dir: Path) -> bytes:
    """单 stem 的推理结果 ZIP（仅推理结果区单条下载用）。

    v1.0.6 加 `@st.cache_data`（见 build_frames_zip 说明）。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if annotated_dir.exists():
            for img in sorted(annotated_dir.glob("*.jpg")):
                if img.is_file():
                    zf.write(img, arcname=img.name)
    return buf.getvalue()


# ---------- 延迟下载（Streamlit 1.62+）----------


def _download_cache_path(kind: str, entries: list[tuple[Path, str]]) -> Path:
    """按文件路径、大小和修改时间生成稳定的 ZIP 缓存路径。"""
    digest = hashlib.sha256()
    for file_path, arcname in entries:
        stat = file_path.stat()
        digest.update(str(file_path.resolve()).encode("utf-8"))
        digest.update(arcname.encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
    _DOWNLOAD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _DOWNLOAD_CACHE_DIR / f"{safe_stem(kind)}_{digest.hexdigest()[:16]}.zip"


def _build_zip_file(kind: str, entries: list[tuple[Path, str]], progress_cb=None) -> Path:
    """把下载 ZIP 原子化落盘；相同输入后续直接复用。"""
    entries = [(p, arcname) for p, arcname in entries if p.is_file()]
    target = _download_cache_path(kind, entries)
    if target.exists():
        return target

    with _DOWNLOAD_CACHE_LOCK:
        if target.exists():
            return target
        temp = target.with_suffix(".tmp")
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for index, (file_path, arcname) in enumerate(entries, 1):
                    zf.write(file_path, arcname=arcname)
                    if progress_cb:
                        progress_cb(index, len(entries))
            temp.replace(target)
        finally:
            if temp.exists():
                temp.unlink(missing_ok=True)
    return target


def deferred_file_bytes(path: Path, allowed_root: Path | None = None):
    """返回零参数回调，让 Streamlit 在用户点击下载后再读取文件。"""
    resolved = Path(path)
    root = Path(allowed_root).resolve() if allowed_root else None

    def load() -> bytes:
        if resolved.is_symlink():
            raise ValueError(f"不允许下载符号链接: {resolved.name}")
        candidate = resolved.resolve(strict=True)
        if root is not None and not candidate.is_relative_to(root):
            raise ValueError(f"下载路径超出允许范围: {resolved.name}")
        return candidate.read_bytes()

    return load


def deferred_frames_zip(frames_dir: Path):
    """点击后才生成/读取单个抽帧 ZIP，并复用磁盘缓存。"""
    source = Path(frames_dir)

    def load() -> bytes:
        files = sorted(p for p in source.glob("*.jpg") if p.is_file())
        entries = [(p, p.name) for p in files]
        return _build_zip_file(f"{source.parent.name}_frames", entries).read_bytes()

    return load


def deferred_infer_zip(annotated_dir: Path):
    """点击后才生成/读取单个推理 ZIP，并复用磁盘缓存。"""
    source = Path(annotated_dir)

    def load() -> bytes:
        files = sorted(p for p in source.glob("*.jpg") if p.is_file())
        entries = [(p, p.name) for p in files]
        return _build_zip_file(f"{source.parent.parent.name}_infer", entries).read_bytes()

    return load


def deferred_session_zip(results: dict):
    """点击后才生成当前会话 ZIP；失败任务和全流程中间帧会被跳过。"""
    snapshot = dict(results)

    def load() -> bytes:
        entries: list[tuple[Path, str]] = []
        for stem in sorted(snapshot):
            result = snapshot[stem]
            if getattr(result, "error", None):
                continue
            output_video = getattr(result, "output_video", None)
            if output_video and output_video.is_file():
                entries.append((output_video, f"{stem}/{output_video.name}"))
            is_full = bool(
                output_video
                and getattr(result, "frames_dir", None)
                and getattr(result, "annotated_dir", None)
            )
            if is_full:
                continue
            frames_dir = getattr(result, "frames_dir", None)
            if frames_dir and frames_dir.exists():
                entries.extend(
                    (p, f"{stem}/frames/{p.name}")
                    for p in sorted(frames_dir.glob("*.jpg"))
                    if p.is_file()
                )
            annotated_dir = getattr(result, "annotated_dir", None)
            if annotated_dir and annotated_dir.exists():
                entries.extend(
                    (p, f"{stem}/annotated/images/{p.name}")
                    for p in sorted(annotated_dir.glob("*.jpg"))
                    if p.is_file()
                )
        return _build_zip_file("cv_session", entries).read_bytes()

    return load


def deferred_files_zip(files: list[Path], archive_root: Path, kind: str = "artifacts"):
    """点击后打包指定文件，ZIP 内保留相对 archive_root 的路径。"""
    selected = [Path(path) for path in files]
    root = Path(archive_root).resolve()

    def load() -> bytes:
        entries: list[tuple[Path, str]] = []
        for file_path in selected:
            if file_path.is_symlink():
                raise ValueError(f"不允许下载符号链接: {file_path}")
            resolved = file_path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise ValueError(f"下载路径超出允许范围: {file_path}")
            entries.append((resolved, resolved.relative_to(root).as_posix()))
        return _build_zip_file(kind, entries).read_bytes()

    return load


def prepare_files_zip(
    files: list[Path],
    archive_root: Path,
    kind: str = "artifacts",
    progress_cb=None,
) -> Path:
    """校验并准备 ZIP 文件，供需要可见进度的 UI 使用。"""
    root = Path(archive_root).resolve(strict=True)
    entries: list[tuple[Path, str]] = []
    for file_path in files:
        candidate = Path(file_path)
        if candidate.is_symlink():
            raise ValueError(f"不允许下载符号链接: {candidate.name}")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"下载路径超出允许范围: {candidate.name}")
        entries.append((resolved, resolved.relative_to(root).as_posix()))
    return _build_zip_file(kind, entries, progress_cb=progress_cb)


# ---------- 上传图片落盘 ----------

# 与 pipeline.OUTPUTS_DIR 同源；这里不再 import pipeline 避免循环依赖
DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parent.parent / "outputs"


# ---------- v1.0.6: 大文件 IO 缓存（解决 download_button 置灰/IO 抖动）----------


@st.cache_data(show_spinner=False)
def read_file_bytes_cached(path_str: str, mtime_ns: int) -> bytes:
    """按 (path, mtime_ns) 缓存文件读取。

    应用场景：结果区「下载视频」按钮每次 rerun 都要重读 mp4 → 大 IO 触发
    `_ProactorBasePipeTransport._call_connection_lost` 错误日志，且 Streamlit
    会重新分配 download_token，肉眼看到「按钮置灰 → 消失 → 再出现」。

    加上 cache_data 后，同一会话同一文件多次读取只发生一次 IO，后续 rerun
    直接拿缓存。文件被修改（mtime 变化）后自动重新读取。

    注意：必须在调用前显式传 mtime_ns 作为 cache key，否则 cache 只按 path 缓存
    会一直返回旧内容。"""
    return Path(path_str).read_bytes()


def save_uploaded_images(files, stem: str, outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
    """把 Streamlit 上传的图片批量写入 outputs/<stem>/_uploaded/。

    - 按文件名排序后落盘（仅合成模式依赖此顺序）
    - 文件名经 safe_stem 清洗，避免 Windows 非法字符
    - 写入前清空目标目录旧文件，保证本次上传为唯一来源
    - 返回写入目录的路径
    """
    if not files:
        raise ValueError("files 为空，无法落盘")
    target = outputs_root / stem / "_uploaded"
    target.mkdir(parents=True, exist_ok=True)
    for old in target.iterdir():
        if old.is_file():
            try:
                old.unlink()
            except OSError:
                pass
    for f in sorted(files, key=lambda x: x.name):
        p = Path(f.name)
        clean_stem = safe_stem(p.stem) or "image"
        clean_name = clean_stem + p.suffix.lower()
        # 避免重名覆盖：若同名追加 _2, _3...
        final = target / clean_name
        counter = 2
        while final.exists():
            final = target / f"{clean_stem}_{counter}{p.suffix.lower()}"
            counter += 1
        data = f.read() if hasattr(f, "read") else f
        final.write_bytes(data)
        try:
            f.seek(0)
        except Exception:
            pass
    return target
