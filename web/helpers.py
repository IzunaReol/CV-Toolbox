"""工具函数：路径清洗、FPS/帧数读取、ZIP 打包、文本框解析。"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import cv2


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
    raw = str(value).strip().replace(',', '.')
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
    cleaned = re.sub(r'[\\/:*?"<>|]', '_', name).strip().strip('.')
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


def _zip_directory_glob(zf: zipfile.ZipFile, src_dir: Path,
                        arc_prefix: str, pattern: str = "*.jpg") -> None:
    """把 src_dir 下符合 pattern 的文件按 arc_prefix/<name> 写入 zf。"""
    if not src_dir or not src_dir.exists():
        return
    for f in sorted(src_dir.glob(pattern)):
        if f.is_file():
            zf.write(f, arcname=f"{arc_prefix}/{f.name}")


def build_session_zip(results: dict) -> bytes:
    """按当前会话的 VideoResult 字典打包 ZIP；只含 r.error is None 的项。

    各模式产物路径：
      - full / encode: r.output_video（<stem>.mp4）
      - extract:      r.frames_dir/*.jpg → <stem>/frames/<name>.jpg
      - infer:        r.annotated_dir/*.jpg → <stem>/annotated/images/<name>.jpg

    v1.0.2 新增：只打包当前会话在 st.session_state["results"] 里的 stem，
    不会把 outputs/ 下历史轮次的 mp4 一起打包。
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for stem in sorted(results.keys()):
            r = results[stem]
            # 跳过失败 / None 项；鸭子类型访问字段
            if getattr(r, "error", None):
                continue
            if getattr(r, "output_video", None) and r.output_video.exists():
                zf.write(r.output_video,
                         arcname=f"{stem}/{r.output_video.name}")
            _zip_directory_glob(zf, getattr(r, "frames_dir", None),
                                arc_prefix=f"{stem}/frames")
            _zip_directory_glob(zf, getattr(r, "annotated_dir", None),
                                arc_prefix=f"{stem}/annotated/images")
    return buf.getvalue()


def build_frames_zip(frames_dir: Path) -> bytes:
    """单 stem 的 frames ZIP（仅抽帧结果区单条下载用）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if frames_dir.exists():
            for img in sorted(frames_dir.glob("*.jpg")):
                if img.is_file():
                    zf.write(img, arcname=img.name)
    return buf.getvalue()


def build_infer_zip(annotated_dir: Path) -> bytes:
    """单 stem 的推理结果 ZIP（仅推理结果区单条下载用）。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if annotated_dir.exists():
            for img in sorted(annotated_dir.glob("*.jpg")):
                if img.is_file():
                    zf.write(img, arcname=img.name)
    return buf.getvalue()


# ---------- 上传图片落盘 ----------

# 与 pipeline.OUTPUTS_DIR 同源；这里不再 import pipeline 避免循环依赖
DEFAULT_OUTPUTS_ROOT = Path(__file__).resolve().parent.parent / "outputs"


def save_uploaded_images(files, stem: str,
                         outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> Path:
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
