# python 3.图片转视频.py --input "D:\CV\images" --output "D:\CV\output.mp4" --fps 30
# 直接运行也可，按提示依次输入图片目录、输出文件名、帧率即可

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp')

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

def create_video_from_images(image_dir: str, output_video: str, fps: int = 30):
    """将目录下的图片按文件名排序后合成为 MP4 视频"""
    image_path = Path(image_dir)
    if not image_path.is_dir():
        print(f"输入不是有效目录: {image_dir}")
        return False

    image_files = [f for f in image_path.iterdir() if f.suffix.lower() in IMG_EXTS and f.is_file()]
    image_files.sort()
    if not image_files:
        print(f"错误：目录 {image_dir} 中没有找到任何图片文件")
        return False

    # 读取第一张图片以获取尺寸（使用 unicode 安全读取）
    first_frame = imread_unicode(image_files[0])
    if first_frame is None:
        print(f"错误：无法读取第一张图片 {image_files[0].name}")
        return False

    height, width = first_frame.shape[:2]
    size = (width, height)

    # 确保输出目录存在
    out_path = Path(output_video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)

    print(f"图片数量: {len(image_files)}")
    print(f"视频尺寸: {width}x{height}    帧率: {fps} fps")
    print(f"输出路径: {out_path}")
    print("开始合成...")

    written = 0
    for idx, img_file in enumerate(image_files, 1):
        frame = imread_unicode(img_file)
        if frame is None:
            print(f"  警告：无法读取 {img_file.name}，已跳过")
            continue
        if frame.shape[1] != width or frame.shape[0] != height:
            print(f"  警告：{img_file.name} 尺寸 {frame.shape[1]}x{frame.shape[0]} 与首帧不一致，已跳过")
            continue
        writer.write(frame)
        written += 1
        if idx <= 5 or idx % 20 == 0 or idx == len(image_files):
            print(f"  已处理 {idx}/{len(image_files)}: {img_file.name}")

    writer.release()
    if written == 0:
        print("未能写入任何帧，请检查输入图片")
        return False

    print("=" * 50)
    print(f"视频合成完成！共写入 {written} 帧")
    print(f"保存路径: {out_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="将图片目录合成为 MP4 视频（支持交互式输入）")
    parser.add_argument("--input", "-i", type=str, default=None, help="输入图片目录路径")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出视频文件路径（默认: <输入目录名>.mp4）")
    parser.add_argument("--fps", type=int, default=30, help="视频帧率 (默认 30)")
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
            fps = parse_positive_int(fps_raw, fps)

    create_video_from_images(str(input_dir), str(output_path), fps=fps)

if __name__ == "__main__":
    main()