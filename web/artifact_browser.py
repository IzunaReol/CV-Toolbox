"""outputs 任务工件浏览、预览、下载与安全删除。"""
from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import os
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

try:
    from .helpers import deferred_file_bytes, deferred_files_zip
except ImportError:
    from helpers import deferred_file_bytes, deferred_files_zip


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PAGE_SIZE = 24
DIRECT_DOWNLOAD_MAX_FILES = 5
DIRECT_DOWNLOAD_MAX_BYTES = 20 * 1024 * 1024


def _is_within(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except (AttributeError, ValueError):
        return root == path or root in path.parents


def _directories_with_files(root: Path) -> list[Path]:
    """列出直接包含普通文件的目录，不跟随符号链接目录。"""
    resolved_root = Path(root).resolve(strict=True)
    directories: list[Path] = []
    for current, child_dirs, file_names in os.walk(resolved_root, followlinks=False):
        current_path = Path(current)
        child_dirs[:] = [
            name for name in child_dirs
            if not (current_path / name).is_symlink()
        ]
        if any(
            (current_path / name).is_file() and not (current_path / name).is_symlink()
            for name in file_names
        ):
            directories.append(current_path.resolve(strict=True))
    return directories


def list_task_roots(outputs_root: Path) -> list[Path]:
    """列出包含工件的顶层任务目录，隐藏内部目录和空任务。"""
    root = Path(outputs_root).resolve()
    if not root.exists():
        return []
    return sorted(
        (
            path for path in root.iterdir()
            if (
                path.is_dir()
                and not path.is_symlink()
                and not path.name.startswith("_")
                and _directories_with_files(path)
            )
        ),
        key=lambda path: path.name.casefold(),
    )


def resolve_task_root(outputs_root: Path, task_name: str) -> Path:
    """解析一个非内部任务目录，并拒绝越界或符号链接。"""
    root = Path(outputs_root).resolve()
    raw_candidate = root / task_name
    if raw_candidate.is_symlink():
        raise ValueError("不允许访问符号链接任务目录")
    candidate = raw_candidate.resolve(strict=True)
    if (
        not _is_within(candidate, root)
        or candidate.parent != root
        or candidate.name.startswith("_")
        or not candidate.is_dir()
    ):
        raise ValueError("无效的任务目录")
    return candidate


def list_directories(task_root: Path) -> list[Path]:
    """列出直接包含文件的任务目录；任务内的 _uploaded 允许展示。"""
    root = Path(task_root).resolve(strict=True)
    directories = _directories_with_files(root)
    return sorted(directories, key=lambda path: path.relative_to(root).as_posix())


def resolve_directory(task_root: Path, relative_path: str) -> Path:
    """解析任务内目录并校验其完整父链不包含符号链接。"""
    root = Path(task_root).resolve(strict=True)
    candidate = (root / relative_path).resolve(strict=True)
    if not _is_within(candidate, root) or not candidate.is_dir():
        raise ValueError("目录超出当前任务范围")
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("不允许访问符号链接目录")
    return candidate


def list_files(directory: Path, task_root: Path) -> list[Path]:
    """列出当前目录文件，不递归并跳过符号链接。"""
    current = resolve_directory(task_root, str(Path(directory).relative_to(task_root)))
    return sorted(
        (path for path in current.iterdir() if path.is_file() and not path.is_symlink()),
        key=lambda path: path.name.casefold(),
    )


def validate_selected_files(paths: list[Path], task_root: Path) -> list[Path]:
    """校验选择结果都是真实存在的任务内普通文件。"""
    root = Path(task_root).resolve(strict=True)
    validated: list[Path] = []
    for path in paths:
        candidate = Path(path)
        if candidate.is_symlink():
            raise ValueError(f"不允许操作符号链接: {candidate.name}")
        resolved = candidate.resolve(strict=True)
        if not _is_within(resolved, root) or not resolved.is_file():
            raise ValueError(f"文件超出当前任务范围: {candidate.name}")
        validated.append(resolved)
    return validated


def should_bundle(files: list[Path]) -> bool:
    """超过 5 个文件或总大小超过 20 MiB 时使用 ZIP。"""
    return len(files) > DIRECT_DOWNLOAD_MAX_FILES or sum(p.stat().st_size for p in files) > DIRECT_DOWNLOAD_MAX_BYTES


def delete_files(paths: list[Path], task_root: Path) -> int:
    """永久删除校验通过的文件，返回删除数量。"""
    files = validate_selected_files(paths, task_root)
    for path in files:
        path.unlink()
    return len(files)


def _format_size(size: int) -> str:
    if size >= 1024 * 1024:
        return f"{size / 1024 / 1024:.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


def _selection_key(task: str, directory: str, refresh_token: int) -> str:
    raw = f"{task}\0{directory}\0{refresh_token}".encode("utf-8")
    return f"artifact_selection_{hashlib.sha1(raw).hexdigest()[:12]}"


def _sync_page_state(widget_key: str, state_key: str) -> None:
    """在页码控件触发 rerun 前保存稳定页码。"""
    st.session_state[state_key] = int(st.session_state[widget_key])


def toggle_selection(selected: list[str], file_name: str) -> list[str]:
    """切换单个文件的选择状态，并保留原选择顺序。"""
    result = list(selected)
    if file_name in result:
        result.remove(file_name)
    else:
        result.append(file_name)
    return result


@st.cache_data(show_spinner=False)
def _thumbnail_data_uri(path_str: str, mtime_ns: int, size: int) -> str:
    """生成适合图片按钮使用的 JPEG 缩略图 data URI。"""
    del mtime_ns, size  # 仅作为文件变化时的缓存键
    with Image.open(path_str) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((480, 300))
        canvas = Image.new("RGB", (480, 300), "#111827")
        offset = ((480 - image.width) // 2, (300 - image.height) // 2)
        canvas.paste(image, offset)
        buffer = io.BytesIO()
        canvas.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_artifact_browser(outputs_root: Path) -> None:
    """渲染第五种运行模式：任务工件浏览器。"""
    st.subheader("📁 文件浏览")
    st.caption("浏览 outputs 中的任务工件；空目录及内部模型、源视频和下载缓存不会显示。")

    refresh_col, info_col = st.columns([1, 4])
    if refresh_col.button("🔄 刷新", key="artifact_refresh", use_container_width=True):
        st.session_state["artifact_refresh_token"] = st.session_state.get("artifact_refresh_token", 0) + 1
        st.session_state.pop("artifact_delete_pending", None)
        st.rerun()
    info_col.caption("文件变化后点击刷新即可重新扫描")

    tasks = list_task_roots(outputs_root)
    if not tasks:
        st.info("outputs 中暂无可浏览的任务工件。")
        return
    task_names = [path.name for path in tasks]
    task_key = "artifact_task"
    if st.session_state.get(task_key) not in task_names:
        st.session_state[task_key] = task_names[0]
    task_name = st.selectbox("任务", task_names, key=task_key)
    try:
        task_root = resolve_task_root(outputs_root, task_name)
        directories = list_directories(task_root)
    except (OSError, ValueError) as exc:
        st.error(f"无法打开任务目录: {exc}")
        return

    if not directories:
        st.info("当前任务暂无可浏览的文件，请刷新后重试。")
        return
    relative_dirs = ["." if path == task_root else path.relative_to(task_root).as_posix() for path in directories]
    directory_key = f"artifact_dir_{task_name}"
    if st.session_state.get(directory_key) not in relative_dirs:
        st.session_state[directory_key] = relative_dirs[0]
    relative_dir = st.selectbox("目录", relative_dirs, key=directory_key)
    try:
        current_dir = resolve_directory(task_root, relative_dir)
        files = list_files(current_dir, task_root)
    except (OSError, ValueError) as exc:
        st.error(f"无法读取目录: {exc}")
        return

    if not files:
        st.info("当前目录没有文件。")
        return

    refresh_token = st.session_state.get("artifact_refresh_token", 0)
    file_by_name = {path.name: path for path in files}
    selection_key = _selection_key(task_name, relative_dir, refresh_token)
    st.session_state.setdefault(selection_key, [])

    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    other_files = [path for path in files if path.suffix.lower() not in IMAGE_EXTENSIONS]
    if images:
        st.markdown("#### 图片预览")
        preview_key = f"artifact_preview_{hashlib.sha1(f'{task_name}/{relative_dir}'.encode()).hexdigest()[:12]}"
        preview_name = st.session_state.get(preview_key)
        image_names = [path.name for path in images]
        if preview_name not in image_names:
            preview_name = None
        if preview_name:
            preview_index = image_names.index(preview_name)
            with st.container(border=True):
                st.image(str(images[preview_index]), caption=preview_name, width="stretch")
                previous_col, position_col, next_col = st.columns([1, 2, 1])
                if previous_col.button(
                    "← 上一张", key=f"artifact_previous_{preview_key}",
                    disabled=preview_index == 0, use_container_width=True,
                ):
                    st.session_state[preview_key] = image_names[preview_index - 1]
                    st.rerun()
                position_col.markdown(
                    f"<div style='text-align:center;padding-top:.5rem'>"
                    f"{preview_index + 1} / {len(images)}</div>",
                    unsafe_allow_html=True,
                )
                if next_col.button(
                    "下一张 →", key=f"artifact_next_{preview_key}",
                    disabled=preview_index == len(images) - 1, use_container_width=True,
                ):
                    st.session_state[preview_key] = image_names[preview_index + 1]
                    st.rerun()

        page_count = max((len(images) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        context_hash = hashlib.sha1(
            f"{task_name}/{relative_dir}/{refresh_token}".encode()
        ).hexdigest()[:12]
        page_state_key = f"artifact_page_value_{context_hash}"
        page_widget_key = f"artifact_page_widget_{context_hash}"
        st.session_state.setdefault(page_state_key, 1)
        page = min(max(int(st.session_state[page_state_key]), 1), page_count)
        st.session_state[page_state_key] = page
        start = (int(page) - 1) * PAGE_SIZE
        page_images = images[start:start + PAGE_SIZE]
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] button:has(img) {
                min-height: 215px;
                padding: .35rem;
                display: flex;
                flex-direction: column;
                gap: .35rem;
            }
            div[data-testid="stButton"] button img {
                width: 100% !important;
                height: 165px !important;
                max-height: 165px !important;
                object-fit: cover;
                border-radius: .35rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        for row_start in range(0, len(page_images), 4):
            columns = st.columns(4)
            for column, image_path in zip(columns, page_images[row_start:row_start + 4]):
                with column:
                    stat = image_path.stat()
                    thumbnail = _thumbnail_data_uri(
                        str(image_path), stat.st_mtime_ns, stat.st_size
                    )
                    is_selected = image_path.name in st.session_state[selection_key]
                    marker = "✅ " if is_selected else ""
                    if st.button(
                        f"![preview]({thumbnail})\n\n{marker}{image_path.name}",
                        key=f"artifact_image_{hashlib.sha1(str(image_path).encode()).hexdigest()[:16]}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        st.session_state[selection_key] = toggle_selection(
                            st.session_state[selection_key], image_path.name
                        )
                        st.session_state[preview_key] = image_path.name
                        st.rerun()
        st.caption(f"第 {int(page)}/{page_count} 页，共 {len(images)} 张图片")

        selection_col, page_col = st.columns([3, 1])
        with selection_col:
            selected_names = st.multiselect(
                "选择文件",
                list(file_by_name),
                format_func=lambda name: f"{name}（{_format_size(file_by_name[name].stat().st_size)}）",
                key=selection_key,
            )
        with page_col:
            st.session_state.setdefault(page_widget_key, page)
            st.selectbox(
                "页码",
                options=list(range(1, page_count + 1)),
                key=page_widget_key,
                format_func=lambda value: f"第 {value} 页",
                on_change=_sync_page_state,
                args=(page_widget_key, page_state_key),
            )
    else:
        selected_names = st.multiselect(
            "选择文件",
            list(file_by_name),
            format_func=lambda name: f"{name}（{_format_size(file_by_name[name].stat().st_size)}）",
            key=selection_key,
        )

    selected = [file_by_name[name] for name in selected_names]

    if other_files:
        st.markdown("#### 其他工件")
        st.dataframe(
            [{"文件名": path.name, "类型": path.suffix.lower() or "文件", "大小": _format_size(path.stat().st_size)} for path in other_files],
            use_container_width=True,
            hide_index=True,
        )

    if not selected:
        return
    st.divider()
    total_size = sum(path.stat().st_size for path in selected)
    st.caption(f"已选择 {len(selected)} 个文件，共 {_format_size(total_size)}")

    if should_bundle(selected):
        st.download_button(
            "📦 下载所选文件 (ZIP)",
            data=deferred_files_zip(selected, task_root, f"{task_name}_artifacts"),
            file_name=f"{task_name}_artifacts.zip",
            mime="application/zip",
            key=f"artifact_zip_{task_name}_{relative_dir}",
            on_click="ignore",
            use_container_width=True,
        )
    else:
        st.markdown("#### 下载所选文件")
        for index, path in enumerate(selected):
            st.download_button(
                f"下载 {path.name}",
                data=deferred_file_bytes(path),
                file_name=path.name,
                mime=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                key=f"artifact_download_{task_name}_{relative_dir}_{index}_{path.name}",
                on_click="ignore",
                use_container_width=True,
            )

    if st.button("🗑 删除所选文件", key="artifact_delete", type="secondary"):
        st.session_state["artifact_delete_pending"] = [str(path) for path in selected]
    pending = st.session_state.get("artifact_delete_pending")
    if pending:
        st.warning(f"确认永久删除 {len(pending)} 个文件？此操作无法撤销。")
        yes_col, no_col = st.columns(2)
        if yes_col.button("确认删除", key="artifact_delete_confirm", type="primary", use_container_width=True):
            try:
                count = delete_files([Path(path) for path in pending], task_root)
            except (OSError, ValueError) as exc:
                st.error(f"删除失败: {exc}")
            else:
                st.session_state.pop("artifact_delete_pending", None)
                st.session_state["artifact_refresh_token"] = refresh_token + 1
                st.toast(f"已删除 {count} 个文件", icon="🗑️")
                st.rerun()
        if no_col.button("取消", key="artifact_delete_cancel", use_container_width=True):
            st.session_state.pop("artifact_delete_pending", None)
            st.rerun()
