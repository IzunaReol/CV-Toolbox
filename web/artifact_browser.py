"""outputs 任务工件浏览、预览、下载与安全删除。"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

import streamlit as st

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


def list_task_roots(outputs_root: Path) -> list[Path]:
    """列出允许展示的顶层任务目录，隐藏所有内部目录。"""
    root = Path(outputs_root).resolve()
    if not root.exists():
        return []
    return sorted(
        (
            path for path in root.iterdir()
            if path.is_dir() and not path.is_symlink() and not path.name.startswith("_")
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
    """递归列出任务内部目录；任务内的 _uploaded 允许展示。"""
    root = Path(task_root).resolve(strict=True)
    directories = [root]
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            resolved = path.resolve(strict=True)
            if _is_within(resolved, root):
                directories.append(resolved)
    return sorted(set(directories), key=lambda path: path.relative_to(root).as_posix())


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


def render_artifact_browser(outputs_root: Path) -> None:
    """渲染第五种运行模式：任务工件浏览器。"""
    st.subheader("📁 文件浏览")
    st.caption("浏览 outputs 中的任务工件；内部模型、源视频和下载缓存不会显示。")

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
    task_name = st.selectbox("任务", task_names, key="artifact_task")
    try:
        task_root = resolve_task_root(outputs_root, task_name)
        directories = list_directories(task_root)
    except (OSError, ValueError) as exc:
        st.error(f"无法打开任务目录: {exc}")
        return

    relative_dirs = ["." if path == task_root else path.relative_to(task_root).as_posix() for path in directories]
    relative_dir = st.selectbox("目录", relative_dirs, key=f"artifact_dir_{task_name}")
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
    selected_names = st.multiselect(
        "选择文件",
        list(file_by_name),
        format_func=lambda name: f"{name}（{_format_size(file_by_name[name].stat().st_size)}）",
        key=_selection_key(task_name, relative_dir, refresh_token),
    )
    selected = [file_by_name[name] for name in selected_names]

    images = [path for path in files if path.suffix.lower() in IMAGE_EXTENSIONS]
    other_files = [path for path in files if path.suffix.lower() not in IMAGE_EXTENSIONS]
    if images:
        st.markdown("#### 图片预览")
        page_count = max((len(images) + PAGE_SIZE - 1) // PAGE_SIZE, 1)
        page = st.number_input(
            "页码", min_value=1, max_value=page_count, value=1, step=1,
            key=f"artifact_page_{task_name}_{relative_dir}_{refresh_token}",
        )
        start = (int(page) - 1) * PAGE_SIZE
        page_images = images[start:start + PAGE_SIZE]
        for row_start in range(0, len(page_images), 4):
            columns = st.columns(4)
            for column, image_path in zip(columns, page_images[row_start:row_start + 4]):
                with column:
                    st.image(str(image_path), caption=image_path.name, width="stretch")
        st.caption(f"第 {int(page)}/{page_count} 页，共 {len(images)} 张图片")

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
