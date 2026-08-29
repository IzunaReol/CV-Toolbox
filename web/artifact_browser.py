"""outputs 任务工件浏览、预览、下载与安全删除。"""

from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import os
from html import escape
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

try:
    from .helpers import deferred_file_bytes, prepare_files_zip
    from .task_store import (
        META_DIR_NAME,
        artifact_revision,
        bump_artifact_revision,
        read_task,
        task_meta_dir,
    )
except ImportError:
    from helpers import deferred_file_bytes, prepare_files_zip
    from task_store import (
        META_DIR_NAME,
        artifact_revision,
        bump_artifact_revision,
        read_task,
        task_meta_dir,
    )


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PAGE_SIZE = 24
DIRECT_DOWNLOAD_MAX_FILES = 5
DIRECT_DOWNLOAD_MAX_BYTES = 20 * 1024 * 1024
MODE_LABELS = {
    "full": "全流程",
    "extract": "仅抽帧",
    "infer": "仅推理",
    "encode": "仅合成",
}


def _centered_table(rows: list[dict]) -> None:
    if not rows:
        return
    headers = list(rows[0])
    head = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{escape(str(row.get(header, '')))}</td>" for header in headers)
        + "</tr>"
        for row in rows
    )
    st.markdown(
        "<style>.cv-centered-table{width:100%;border-collapse:collapse;margin:.5rem 0;}"
        ".cv-centered-table th,.cv-centered-table td{text-align:center!important;"
        "padding:.45rem;border-bottom:1px solid rgba(128,128,128,.25);}</style>"
        f"<table class='cv-centered-table'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>",
        unsafe_allow_html=True,
    )


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
            name
            for name in child_dirs
            if name != META_DIR_NAME and not (current_path / name).is_symlink()
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
            path
            for path in root.iterdir()
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
        if part == META_DIR_NAME:
            raise ValueError("不允许访问内部任务目录")
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
    return (
        len(files) > DIRECT_DOWNLOAD_MAX_FILES
        or sum(p.stat().st_size for p in files) > DIRECT_DOWNLOAD_MAX_BYTES
    )


def _has_public_files(task_root: Path) -> bool:
    return bool(_directories_with_files(task_root)) if Path(task_root).exists() else False


def cleanup_empty_directories(task_root: Path) -> bool:
    """清理空子目录；无公开工件时连同内部元数据和任务根一起移除。"""
    root = Path(task_root).resolve(strict=True)
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        if directory.name == META_DIR_NAME:
            continue
        try:
            directory.rmdir()
        except OSError:
            pass
    if _has_public_files(root):
        return False
    meta = task_meta_dir(root)
    if meta.exists() and not meta.is_symlink():
        for path in sorted(meta.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if path.is_file() and not path.is_symlink():
                path.unlink(missing_ok=True)
            elif path.is_dir() and not path.is_symlink():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            meta.rmdir()
        except OSError:
            pass
    try:
        root.rmdir()
        return True
    except OSError:
        return False


def delete_files(paths: list[Path], task_root: Path, progress_cb=None) -> int:
    """永久删除校验通过的文件，返回删除数量。"""
    files = validate_selected_files(paths, task_root)
    for index, path in enumerate(files, 1):
        path.unlink()
        if progress_cb:
            progress_cb(index, len(files), path.name)
    cleanup_empty_directories(task_root)
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


def next_preview_after_delete(image_names: list[str], preview_index: int) -> str | None:
    """返回删除当前图片后应预览的原列表下一张；末张删除后关闭预览。"""
    next_index = preview_index + 1
    return image_names[next_index] if next_index < len(image_names) else None


@st.cache_data(show_spinner=False)
def _thumbnail_data_uri(path_str: str, mtime_ns: int, size: int) -> str:
    """生成适合图片按钮使用的 JPEG 缩略图 data URI。"""
    del mtime_ns, size  # 仅作为文件变化时的缓存键
    try:
        with Image.open(path_str) as image:
            converted = ImageOps.exif_transpose(image).convert("RGB")
            converted.thumbnail((480, 300))
            canvas = Image.new("RGB", (480, 300), "#111827")
            offset = ((480 - converted.width) // 2, (300 - converted.height) // 2)
            canvas.paste(converted, offset)
            buffer = io.BytesIO()
            canvas.save(buffer, format="JPEG", quality=82, optimize=True)
    except (OSError, ValueError):
        return ""
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _task_summary(task_root: Path, directories: list[Path]) -> dict:
    files = [file for directory in directories for file in list_files(directory, task_root)]
    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    metadata = read_task(task_root)
    return {
        "legacy": metadata is None,
        "status": (metadata or {}).get("status", "历史任务"),
        "mode": (metadata or {}).get("mode", "未知"),
        "updated_at": (metadata or {}).get("updated_at"),
        "file_count": len(files),
        "image_count": len(images),
        "total_size": sum(path.stat().st_size for path in files),
        "statistics": (metadata or {}).get("statistics", {}),
        "config": (metadata or {}).get("config", {}),
    }


@st.cache_data(show_spinner=False)
def scan_artifact_snapshot(outputs_root_str: str, refresh_token: int, revision: int) -> list[dict]:
    """按刷新令牌和工件修订号缓存完整浏览快照。"""
    del refresh_token, revision
    outputs_root = Path(outputs_root_str)
    snapshot = []
    for task_root in list_task_roots(outputs_root):
        directories = list_directories(task_root)
        snapshot.append(
            {
                "name": task_root.name,
                "directories": [
                    "." if path == task_root.resolve() else path.relative_to(task_root).as_posix()
                    for path in directories
                ],
                "summary": _task_summary(task_root, directories),
            }
        )
    return snapshot


def render_artifact_browser(outputs_root: Path) -> None:
    """渲染第五种运行模式：任务工件浏览器。"""
    st.subheader("📁 文件浏览")
    st.caption("浏览 outputs 中的任务工件；空目录及内部模型、源视频和下载缓存不会显示。")

    refresh_col, info_col = st.columns([1, 4])
    if refresh_col.button("🔄 刷新", key="artifact_refresh", use_container_width=True):
        st.session_state["artifact_refresh_token"] = (
            st.session_state.get("artifact_refresh_token", 0) + 1
        )
        st.session_state.pop("artifact_delete_pending", None)
        st.rerun()
    info_col.caption("文件变化后点击刷新即可重新扫描")

    refresh_token = st.session_state.get("artifact_refresh_token", 0)
    revision = artifact_revision(outputs_root)
    snapshot = scan_artifact_snapshot(str(Path(outputs_root).resolve()), refresh_token, revision)
    if not snapshot:
        st.info("outputs 中暂无可浏览的任务工件。")
        return
    task_names = [item["name"] for item in snapshot]
    task_key = "artifact_task"
    if st.session_state.get(task_key) not in task_names:
        st.session_state[task_key] = task_names[0]
    task_name = st.selectbox("任务", task_names, key=task_key)
    task_snapshot = next(item for item in snapshot if item["name"] == task_name)
    try:
        task_root = resolve_task_root(outputs_root, task_name)
    except (OSError, ValueError) as exc:
        st.error(f"无法打开任务目录: {exc}")
        return

    relative_dirs = list(task_snapshot["directories"])
    if not relative_dirs:
        st.info("当前任务暂无可浏览的文件，请刷新后重试。")
        return
    summary = task_snapshot["summary"]
    status_labels = {
        "completed": "已完成",
        "running": "处理中",
        "cancelled": "已取消",
        "failed": "失败",
        "interrupted": "已中断",
        "历史任务": "历史任务",
    }
    with st.container(border=True):
        st.markdown(f"**任务摘要 · {status_labels.get(summary['status'], summary['status'])}**")
        cols = st.columns(4)
        raw_mode = str(summary.get("mode", "未知"))
        cols[0].metric("模式", MODE_LABELS.get(raw_mode, raw_mode))
        cols[1].metric("文件", summary.get("file_count", 0))
        cols[2].metric("图片", summary.get("image_count", 0))
        cols[3].metric("大小", _format_size(summary.get("total_size", 0)))
        stats = summary.get("statistics") or {}
        if stats:
            st.caption(
                f"推理：共 {stats.get('total_images', 0)} 张，"
                f"命中 {stats.get('matched_images', 0)} 张，"
                f"读取失败 {stats.get('failed_images', 0)} 张"
            )
            if stats.get("class_counts"):
                rows = [
                    {
                        "序号": index,
                        "类别": str(raw_name).split(":", 1)[-1],
                        "数量": count,
                    }
                    for index, (raw_name, count) in enumerate(stats["class_counts"].items(), 1)
                ]
                _centered_table(rows)
            stats_json = task_meta_dir(task_root) / "inference_stats.json"
            stats_csv = task_meta_dir(task_root) / "inference_images.csv"
            stats_downloads = st.columns(2)
            if stats_json.is_file():
                stats_downloads[0].download_button(
                    "下载统计 JSON",
                    data=deferred_file_bytes(stats_json, task_root),
                    file_name=f"{task_name}_inference_stats.json",
                    mime="application/json",
                    key=f"artifact_stats_json_{task_name}",
                    on_click="ignore",
                    use_container_width=True,
                )
            if stats_csv.is_file():
                stats_downloads[1].download_button(
                    "下载逐图统计 CSV",
                    data=deferred_file_bytes(stats_csv, task_root),
                    file_name=f"{task_name}_inference_images.csv",
                    mime="text/csv",
                    key=f"artifact_stats_csv_{task_name}",
                    on_click="ignore",
                    use_container_width=True,
                )
        if summary.get("legacy"):
            st.caption("历史任务：未找到 v2.1.0 任务配置，摘要由现有工件生成。")
        elif summary.get("config"):
            with st.expander("查看任务配置"):
                st.json(summary["config"], expanded=True)
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

    file_by_name = {path.name: path for path in files}
    selection_key = _selection_key(task_name, relative_dir, refresh_token)
    st.session_state.setdefault(selection_key, [])
    valid_selection = [name for name in st.session_state[selection_key] if name in file_by_name]
    if valid_selection != st.session_state[selection_key]:
        st.session_state[selection_key] = valid_selection

    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    other_files = [path for path in files if path.suffix.lower() not in IMAGE_EXTENSIONS]
    if images:
        st.markdown("#### 图片预览")
        preview_key = f"artifact_preview_{hashlib.sha1(f'{task_name}/{relative_dir}'.encode()).hexdigest()[:12]}"
        preview_name = st.session_state.get(preview_key)
        image_names = [path.name for path in images]
        page_count = max((len(images) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        context_hash = hashlib.sha1(
            f"{task_name}/{relative_dir}/{refresh_token}".encode()
        ).hexdigest()[:12]
        page_state_key = f"artifact_page_value_{context_hash}"
        page_widget_key = f"artifact_page_widget_{context_hash}"
        st.session_state.setdefault(page_state_key, 1)
        page = min(max(int(st.session_state[page_state_key]), 1), page_count)
        st.session_state[page_state_key] = page
        if preview_name not in image_names:
            preview_name = None
        if preview_name:
            preview_index = image_names.index(preview_name)
            preview_path = images[preview_index]
            with st.container(border=True):
                try:
                    with Image.open(preview_path) as opened:
                        width, height = opened.size
                        opened.verify()
                    st.image(str(preview_path), caption=preview_name, width="stretch")
                    st.caption(
                        f"{width} × {height}　{_format_size(preview_path.stat().st_size)}　"
                        f"第 {preview_index + 1}/{len(images)} 张"
                    )
                except (OSError, ValueError):
                    st.warning(f"无法预览 {preview_name}，文件可能损坏或尚未写完。")
                close_col, select_col, download_col, delete_col = st.columns(4)
                if close_col.button(
                    "关闭预览", key=f"artifact_close_{preview_key}", use_container_width=True
                ):
                    st.session_state.pop(preview_key, None)
                    st.rerun()
                selected_now = preview_name in st.session_state[selection_key]
                if select_col.button(
                    "取消选择" if selected_now else "选择当前图片",
                    key=f"artifact_select_preview_{preview_key}",
                    use_container_width=True,
                ):
                    st.session_state[selection_key] = toggle_selection(
                        st.session_state[selection_key], preview_name
                    )
                    st.rerun()
                download_col.download_button(
                    "下载当前图片",
                    data=deferred_file_bytes(preview_path, task_root),
                    file_name=preview_name,
                    mime=mimetypes.guess_type(preview_name)[0] or "application/octet-stream",
                    key=f"artifact_preview_download_{preview_key}_{preview_name}",
                    on_click="ignore",
                    use_container_width=True,
                )
                preview_delete_key = f"artifact_preview_delete_{preview_key}"
                if delete_col.button(
                    "删除该图片",
                    key=f"artifact_preview_delete_button_{preview_key}_{preview_name}",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state[preview_delete_key] = preview_name
                    st.rerun()
                if st.session_state.get(preview_delete_key) == preview_name:
                    st.warning(f"确认永久删除图片「{preview_name}」吗？此操作无法撤销。")
                    confirm_col, cancel_col = st.columns(2)
                    if confirm_col.button(
                        "确认删除",
                        key=f"artifact_preview_delete_confirm_{preview_key}_{preview_name}",
                        type="primary",
                        use_container_width=True,
                    ):
                        try:
                            next_preview = next_preview_after_delete(image_names, preview_index)
                            delete_files([preview_path], task_root)
                            bump_artifact_revision(outputs_root)
                            st.session_state[selection_key] = [
                                name
                                for name in st.session_state[selection_key]
                                if name != preview_name
                            ]
                            if next_preview:
                                st.session_state[preview_key] = next_preview
                                next_page = preview_index // PAGE_SIZE + 1
                                st.session_state[page_state_key] = next_page
                                st.session_state[page_widget_key] = next_page
                            else:
                                st.session_state.pop(preview_key, None)
                            st.session_state.pop(preview_delete_key, None)
                            st.toast(f"已删除 {preview_name}", icon="🗑️")
                            st.rerun()
                        except (OSError, ValueError) as exc:
                            st.error(f"删除失败: {exc}")
                    if cancel_col.button(
                        "取消",
                        key=f"artifact_preview_delete_cancel_{preview_key}_{preview_name}",
                        use_container_width=True,
                    ):
                        st.session_state.pop(preview_delete_key, None)
                        st.rerun()
                previous_col, position_col, next_col = st.columns([1, 2, 1])
                if previous_col.button(
                    "← 上一张",
                    key=f"artifact_previous_{preview_key}",
                    disabled=preview_index == 0,
                    use_container_width=True,
                ):
                    target_index = preview_index - 1
                    st.session_state[preview_key] = image_names[target_index]
                    st.session_state[page_state_key] = target_index // PAGE_SIZE + 1
                    st.session_state[page_widget_key] = target_index // PAGE_SIZE + 1
                    st.rerun()
                position_col.markdown(
                    f"<div style='text-align:center;padding-top:.5rem'>"
                    f"{preview_index + 1} / {len(images)}</div>",
                    unsafe_allow_html=True,
                )
                if next_col.button(
                    "下一张 →",
                    key=f"artifact_next_{preview_key}",
                    disabled=preview_index == len(images) - 1,
                    use_container_width=True,
                ):
                    target_index = preview_index + 1
                    st.session_state[preview_key] = image_names[target_index]
                    st.session_state[page_state_key] = target_index // PAGE_SIZE + 1
                    st.session_state[page_widget_key] = target_index // PAGE_SIZE + 1
                    st.rerun()

        start = (int(page) - 1) * PAGE_SIZE
        page_images = images[start : start + PAGE_SIZE]
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
            for column, image_path in zip(columns, page_images[row_start : row_start + 4]):
                with column:
                    stat = image_path.stat()
                    thumbnail = _thumbnail_data_uri(str(image_path), stat.st_mtime_ns, stat.st_size)
                    is_selected = image_path.name in st.session_state[selection_key]
                    marker = "✅ " if is_selected else ""
                    button_content = (
                        f"![preview]({thumbnail})\n\n{marker}{image_path.name}"
                        if thumbnail
                        else f"⚠️ 无法预览\n\n{marker}{image_path.name}"
                    )
                    if st.button(
                        button_content,
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
                format_func=lambda name: (
                    f"{name}（{_format_size(file_by_name[name].stat().st_size)}）"
                ),
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
        _centered_table(
            [
                {
                    "文件名": path.name,
                    "类型": path.suffix.lower() or "文件",
                    "大小": _format_size(path.stat().st_size),
                }
                for path in other_files
            ]
        )

    if not selected:
        return
    st.divider()
    total_size = sum(path.stat().st_size for path in selected)
    st.caption(f"已选择 {len(selected)} 个文件，共 {_format_size(total_size)}")

    if should_bundle(selected):
        signature = hashlib.sha1(
            "\0".join(
                f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}" for path in selected
            ).encode("utf-8")
        ).hexdigest()[:12]
        prepared_key = f"artifact_prepared_zip_{signature}"
        if st.button(
            "📦 准备所选文件 ZIP",
            key=f"artifact_prepare_zip_{signature}",
            use_container_width=True,
        ):
            progress = st.progress(0.0, text="正在准备 ZIP")
            try:
                zip_path = prepare_files_zip(
                    selected,
                    task_root,
                    f"{task_name}_artifacts",
                    progress_cb=lambda current, total: progress.progress(
                        current / max(total, 1), text=f"正在压缩 {current}/{total}"
                    ),
                )
            except (OSError, ValueError) as exc:
                st.error(f"ZIP 准备失败: {exc}")
            else:
                st.session_state[prepared_key] = str(zip_path)
                progress.progress(1.0, text="ZIP 已准备完成")
        prepared_path = st.session_state.get(prepared_key)
        if prepared_path and Path(prepared_path).is_file():
            st.download_button(
                "下载已准备的 ZIP",
                data=deferred_file_bytes(Path(prepared_path)),
                file_name=f"{task_name}_artifacts.zip",
                mime="application/zip",
                key=f"artifact_zip_download_{signature}",
                on_click="ignore",
                use_container_width=True,
            )
    else:
        st.markdown("#### 下载所选文件")
        for index, path in enumerate(selected):
            st.download_button(
                f"下载 {path.name}",
                data=deferred_file_bytes(path, task_root),
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
        if yes_col.button(
            "确认删除", key="artifact_delete_confirm", type="primary", use_container_width=True
        ):
            delete_progress = st.progress(0.0, text="正在删除所选文件")
            deleted_names: list[str] = []

            def report_delete(current: int, total: int, name: str) -> None:
                deleted_names.append(name)
                delete_progress.progress(
                    current / max(total, 1), text=f"正在删除 {current}/{total}: {name}"
                )

            try:
                count = delete_files(
                    [Path(path) for path in pending],
                    task_root,
                    progress_cb=report_delete,
                )
            except (OSError, ValueError) as exc:
                bump_artifact_revision(outputs_root)
                st.error(f"删除中断：已删除 {len(deleted_names)} 个，失败 1 个。原因：{exc}")
            else:
                bump_artifact_revision(outputs_root)
                st.session_state.pop("artifact_delete_pending", None)
                st.session_state["artifact_refresh_token"] = refresh_token + 1
                delete_progress.progress(1.0, text=f"删除完成：成功 {count} 个，失败 0 个")
                st.toast(f"已删除 {count} 个文件并清理空目录", icon="🗑️")
                st.rerun()
        if no_col.button("取消", key="artifact_delete_cancel", use_container_width=True):
            st.session_state.pop("artifact_delete_pending", None)
            st.rerun()
