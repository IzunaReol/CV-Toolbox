"""CLI 与 Web 共用的图片格式和视频写入校验。"""

from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

import cv2

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})


def list_images(directory: Path) -> list[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def positive_fps(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("帧率必须是有限的正数")
    return value


class VerifiedVideoWriter:
    """先写临时 MP4，校验首尾帧后才发布成品；失败不覆盖已有视频。"""

    def __init__(self, output: Path, fps: float):
        self.output = Path(output)
        self.fps = positive_fps(fps)
        self.temp = self.output.with_name(f".{self.output.stem}_{uuid4().hex}.partial.mp4")
        self.writer = None
        self.size = None
        self.written = 0

    def __enter__(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        return self

    def write(self, frame):
        size = (frame.shape[1], frame.shape[0])
        if self.writer is None:
            self.size = size
            self.writer = cv2.VideoWriter(
                str(self.temp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size
            )
            if not self.writer.isOpened():
                raise RuntimeError("视频编码器无法打开，请检查输出路径及编码器")
        if size != self.size:
            raise ValueError(f"图片尺寸不一致：期望 {self.size}，实际 {size}")
        self.writer.write(frame)
        self.written += 1

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.writer is not None:
                self.writer.release()
            if exc_type is not None:
                return False
            if not self.written or not self.temp.is_file() or not self.temp.stat().st_size:
                raise RuntimeError("未生成有效视频文件")
            cap = cv2.VideoCapture(str(self.temp))
            try:
                if not cap.isOpened():
                    raise RuntimeError("生成的视频无法打开")
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, frame = cap.read()
                if (
                    not ok
                    or frame is None
                    or count != self.written
                    or (frame.shape[1], frame.shape[0]) != self.size
                ):
                    raise RuntimeError("生成的视频帧数或首帧校验失败")
                if self.written > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.written - 1)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError("生成的视频尾帧校验失败")
            finally:
                cap.release()
            self.temp.replace(self.output)
        finally:
            self.temp.unlink(missing_ok=True)
        return False
