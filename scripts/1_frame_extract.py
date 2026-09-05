import argparse
import os

import cv2


def imwrite_unicode(output_path, image) -> None:
    """在 Windows 中文路径下可靠保存 OpenCV 图片，并校验实际落盘。"""
    path = os.fspath(output_path)
    extension = os.path.splitext(path)[1] or ".jpg"
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"图片编码失败: {path}")
    try:
        encoded.tofile(path)
    except OSError as exc:
        raise RuntimeError(f"图片保存失败: {path}: {exc}") from exc
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"图片未成功写入磁盘: {path}")


def extract_frames(
    video_path, output_folder, frame_interval=1, name_prefix="", progress_cb=None, cancel_cb=None
):
    """
    从视频中抽取帧并保存为图片

    参数:
        video_path (str): 视频文件路径
        output_folder (str): 保存图片的文件夹
        frame_interval (int): 每隔多少帧抽取一帧，默认为1(每帧都抽取)
        name_prefix (str): 文件名前缀（空字符串则保持原命名 frame_NNNNNN.jpg），
                           设置后输出形如 <prefix>_frame_NNNNNN.jpg
    """
    if not isinstance(frame_interval, int) or frame_interval < 1:
        raise ValueError("抽帧间隔必须是正整数")
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    prefix = f"{name_prefix}_frame_" if name_prefix else "frame_"
    if any(name.startswith(prefix) and name.endswith(".jpg") for name in os.listdir(output_folder)):
        raise FileExistsError("输出目录已有抽帧图片，请使用新的输出目录，避免混入历史帧")

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频文件: {video_path}")

    # 获取视频基本信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {fps:.2f} FPS, 总帧数: {total_frames}")

    frame_count = 0
    saved_count = 0

    try:
        while True:
            if cancel_cb and cancel_cb():
                raise InterruptedError("任务已取消")
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔frame_interval帧保存一次
            if frame_count % frame_interval == 0:
                # 生成文件名（带可选前缀，避免多视频合并到同一目录时冲突）
                if name_prefix:
                    fname = f"{name_prefix}_frame_{frame_count:06d}.jpg"
                else:
                    fname = f"frame_{frame_count:06d}.jpg"
                output_path = os.path.join(output_folder, fname)

                # 保存帧为图片
                imwrite_unicode(output_path, frame)
                saved_count += 1
                if saved_count <= 5 or saved_count % 100 == 0:
                    print(f"已保存 {saved_count} 帧: {output_path}")
                if progress_cb:
                    progress_cb(
                        saved_count, max((total_frames + frame_interval - 1) // frame_interval, 1)
                    )

            frame_count += 1

    finally:
        cap.release()
    if saved_count == 0:
        raise RuntimeError("视频没有可读取的帧")
    print(f"抽帧完成! 共保存了 {saved_count} 张图片到 {output_folder}")
    return {"total_frames": total_frames, "saved_frames": saved_count, "fps": fps}


def main():
    parser = argparse.ArgumentParser(description="按间隔抽取视频帧")
    parser.add_argument("--video", required=True, help="源视频路径")
    parser.add_argument("--output", required=True, help="输出图片目录")
    parser.add_argument("--interval", type=int, default=1, help="抽帧间隔（正整数）")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval 必须大于 0")
    extract_frames(args.video, args.output, args.interval)


if __name__ == "__main__":
    main()
