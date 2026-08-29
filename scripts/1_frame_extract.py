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
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

    while True:
        if cancel_cb and cancel_cb():
            cap.release()
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
            print(f"已保存: {output_path}")
            if progress_cb:
                progress_cb(
                    saved_count, max((total_frames + frame_interval - 1) // frame_interval, 1)
                )

        frame_count += 1

    cap.release()
    print(f"抽帧完成! 共保存了 {saved_count} 张图片到 {output_folder}")
    return {"total_frames": total_frames, "saved_frames": saved_count, "fps": fps}


if __name__ == "__main__":
    # 使用示例
    video_file = r"C:\wt\1.mp4"  # 替换为你的视频文件路径
    output_dir = r"C:\wt\1"  # 输出文件夹

    # 参数说明:
    # frame_interval=1 表示抽取每一帧
    extract_frames(video_file, output_dir, frame_interval=5)
