# python 3_images_to_video.py --input <path/to/annotated/images> --output <path/to/output.mp4> --fps 30
# 直接运行也可，按提示依次输入图片目录、输出文件名、帧率即可

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from web.media import VerifiedVideoWriter, list_images, positive_fps


def imread_unicode(path) -> np.ndarray:
    """以 numpy.fromfile + cv2.imdecode 读取图片，绕过 OpenCV 对中文路径的支持问题。

    cv2.imread 在 Windows 上对含中文的路径会返回 None（OpenCV 内部用 ANSI 解析路径），
    这种方式先把文件以二进制读入内存，再交给 cv2 解码，可正确处理任意 Unicode 路径。
    """
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def prompt_existing_path(prompt_text: str) -> Path:
    """交互式提示用户输入一个已存在的路径，无效则反复询问"""
    while True:
        raw = input(prompt_text).strip().strip('"').strip("'")
        if not raw:
            print("路径不能为空，请重新输入")
            continue
        p = Path(raw)
        if not p.exists():
            print(f"路径不存在: {p}，请重新输入")
            continue
        return p


def prompt_output_path(prompt_text: str, default: Path) -> Path:
    """交互式提示用户输入输出路径，回车使用默认值"""
    raw = input(f"{prompt_text} (回车使用默认 {default}): ").strip().strip('"').strip("'")
    if not raw:
        return default
    return Path(raw)


def parse_positive_int(value: str, default: int) -> int:
    """解析正整数，无效时回退到默认值"""
    try:
        parsed = int(value)
        if parsed < 1:
            print(f"输入值 {value} 小于1，使用默认值 {default}")
            return default
        return parsed
    except (ValueError, TypeError):
        print(f"输入值 {value} 不是有效整数，使用默认值 {default}")
        return default


def create_video_from_images(
    image_dir: str, output_video: str, fps: float = 30, progress_cb=None, cancel_cb=None
):
    """按文件名排序合成；损坏或尺寸不一致的图片使任务失败，避免静默丢帧。"""
    if cancel_cb and cancel_cb():
        raise InterruptedError("任务已取消")
    positive_fps(fps)
    image_files = list_images(Path(image_dir))
    if not image_files:
        return False
    with VerifiedVideoWriter(Path(output_video), fps) as writer:
        for idx, img_file in enumerate(image_files, 1):
            if cancel_cb and cancel_cb():
                raise InterruptedError("任务已取消")
            frame = imread_unicode(img_file)
            if frame is None:
                raise RuntimeError(f"无法读取图片: {img_file.name}")
            writer.write(frame)
            if progress_cb:
                progress_cb(idx, len(image_files))
        if cancel_cb and cancel_cb():
            raise InterruptedError("任务已取消")
    return True


def main():
    parser = argparse.ArgumentParser(description="将图片目录合成为 MP4 视频（支持交互式输入）")
    parser.add_argument("--input", "-i", type=str, default=None, help="输入图片目录路径")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="输出视频文件路径（默认: <输入目录名>.mp4）"
    )
    parser.add_argument("--fps", type=float, default=30, help="视频帧率 (默认 30)")
    args = parser.parse_args()

    # 输入图片目录：命令行优先，否则交互输入
    if args.input:
        input_dir = Path(args.input)
        if not input_dir.is_dir():
            print(f"输入目录不存在: {input_dir}")
            return
    else:
        input_dir = prompt_existing_path("请输入图片目录路径: ")

    # 默认输出路径：<输入目录名>.mp4，放在同级目录
    default_output = input_dir.parent / f"{input_dir.name}.mp4"

    # 输出路径：命令行优先，否则交互输入（可回车）
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = prompt_output_path("请输入输出视频文件路径", default=default_output)

    # 帧率：命令行优先，否则交互输入（可回车使用默认）
    fps = args.fps
    if len(sys.argv) <= 1:
        fps_raw = input(f"请输入视频帧率 (默认 {fps}): ").strip()
        if fps_raw:
            fps = positive_fps(fps_raw)

    if not create_video_from_images(str(input_dir), str(output_path), fps=fps):
        raise RuntimeError("没有可合成的图片")


if __name__ == "__main__":
    main()
