"""Streamlit WEB UI：视频处理流水线与任务工件浏览器。

启动方式（在项目根下）：
    streamlit run web/app.py --server.maxUploadSize 1024

v1.0.6 变更：
  0. 「🧹 清空缓存」改为**只重置页面 UI 控件 + session_state**，**不删任何本地文件**
     （磁盘文件删除仍由 v1.0.3 的「🧹 清空本地文件」两步按钮负责）
  1. 仅合成的源视频上传控件逻辑反转：「原视频帧率」时显示，「自定义」时隐藏
  2. widget 全部加显式 `key`，配合白名单 reset 真正恢复初始状态
  3. 结果区下载按钮的 IO 走 `@st.cache_data`，rerun 不再重复读盘
     → 不再出现「按钮置灰 → 消失 → 再出现」；减少 `WinError 10054` 触发

v1.0.6 hotfix（用户报告"清空缓存不彻底"+"上传卡死"后）：
  0. 删除「🔄 更换」按钮（上传卡在 uploading 时该按钮根本渲染不出来）
  1. 「🧹 清空缓存」改名为「🧹 重置页面」，并新增全局 `_reset_token`：
    点「重置页面」时 token++，所有 file_uploader 拿到全新 widget_key，
    即使上次上传卡在 99%，重置后再进入也像第一次打开页面
  2. _reset_model_state 保留作为「更换/删除模型」时的轻量 reset；不再通过
    on_reset 回调（已删除）；改为在 sidebar 里直接 prev_path != new_path 检测

v1.0.5 变更：
  0. 去掉上传后的「✅ 已选择」状态行（仅保留「🔄 更换」按钮）
  1. 仅抽帧 / 仅推理 / 仅合成 模式不展示「下载全部 (ZIP)」按钮
  2. 更换或清空模型后，重置「标注类别」列表（class_names 缓存 + widget 状态）
  3. 侧栏「清空会话」改名为「清空缓存」：清空页面缓存 + 已上传模型/文件

v1.0.4 变更：
  0. 每个 file_uploader 旁新增「🔄 更换」按钮（版本号重置 widget key）
  1. spinner 包裹流水线；README 新增 FAQ
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

# 让 helpers / pipeline 在任意调用方式下都能导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

try:
    from .artifact_browser import render_artifact_browser
    from .helpers import (
        COLOR_NAME_ZH_TO_EN,
        deferred_file_bytes,
        deferred_frames_zip,
        deferred_infer_zip,
        deferred_session_zip,
        label_table_to_dict,
        parse_positive_float_text,
        parse_positive_int_text,
        safe_stem,
        save_uploaded_images,
    )
    from .job_manager import active_status, cancel_batch, results_from_status, submit_pipeline
    from .pipeline import (
        OUTPUTS_DIR,
        VideoResult,
        cache_uploaded_model,
        get_model_class_names,
        parse_color,
        read_video_meta,
        save_uploaded_video,
    )
except ImportError:  # 当作顶层脚本运行（streamlit run web/app.py）时回落
    from artifact_browser import render_artifact_browser
    from helpers import (
        COLOR_NAME_ZH_TO_EN,
        deferred_file_bytes,
        deferred_frames_zip,
        deferred_infer_zip,
        deferred_session_zip,
        label_table_to_dict,
        parse_positive_float_text,
        parse_positive_int_text,
        safe_stem,
        save_uploaded_images,
    )
    from job_manager import active_status, cancel_batch, results_from_status, submit_pipeline
    from pipeline import (
        OUTPUTS_DIR,
        VideoResult,
        cache_uploaded_model,
        get_model_class_names,
        parse_color,
        read_video_meta,
        save_uploaded_video,
    )


DEFAULT_BOX_COLOR = (0, 0, 255)  # BGR 红色
COLOR_ZH_OPTIONS = list(COLOR_NAME_ZH_TO_EN.keys()) + ["自定义"]
MODE_OPTIONS = ["全流程", "仅抽帧", "仅推理", "仅合成", "文件浏览"]
MODE_KEY_MAP = {
    "全流程": "full",
    "仅抽帧": "extract",
    "仅推理": "infer",
    "仅合成": "encode",
    "文件浏览": "browse",
}
STAGE_LABELS = {
    "queued": "等待执行",
    "start": "准备处理",
    "full": "全流程",
    "extract": "视频抽帧",
    "infer": "模型推理",
    "encode": "视频合成",
    "done": "处理完成",
    "completed": "处理完成",
    "cancelled": "已取消",
    "cancelling": "正在取消",
    "failed": "处理失败",
    "error": "处理失败",
    "interrupted": "已中断",
}
DISPLAY_TIMEZONE = timezone(timedelta(hours=8), name="Asia/Shanghai")


def _format_started_at(value: str | None) -> str:
    if not value:
        return "未知"
    try:
        return (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            .astimezone(DISPLAY_TIMEZONE)
            .strftime("%Y-%m-%d %H:%M:%S")
        )
    except ValueError:
        return value


def _should_show_session_zip(results: dict[str, VideoResult]) -> bool:
    """全流程至少包含两个成功视频时才提供批量 ZIP。"""
    return sum(1 for result in results.values() if _is_full_result(result)) > 1


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


def _app_version() -> str:
    """从 pyproject.toml 读取 version 字段，渲染成 "vX.Y.Z"。

    解析失败时回退到 "v?.?.?"，避免 UI caption 报错。
    """
    import re

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M)
    except OSError:
        return "v?.?.?"
    return f"v{m.group(1)}" if m else "v?.?.?"


def _timestamp_id(prefix: str) -> str:
    """生成 prefix_YYYYMMDD_HHMMSS 形式的唯一 job id（仅推理 / 仅合成用）。"""
    from datetime import datetime

    return f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"


def _delete_dir_contents(path: Path, *, skip_names: set[str] | None = None) -> None:
    """递归删除 path 下所有内容，保留 path 本身（用于清空运行时数据）。

    skip_names: 跳过 path 下指定 name 的直接子项；用于保留 _models 等非生成物。
    """
    if not path.exists():
        return
    skip = skip_names or set()
    for entry in path.iterdir():
        if entry.name in skip:
            continue
        try:
            if entry.is_dir():
                import shutil

                shutil.rmtree(entry)
            else:
                entry.unlink()
        except OSError:
            pass


def _file_uploader_with_reset(
    label: str,
    type,
    base_key: str,
    *,
    accept_multiple: bool = False,
    help: str | None = None,
    container=None,
    **kwargs,
):
    """file_uploader 的稳定封装：widget_key 内嵌全局 _reset_token。

    v1.0.6 hotfix：去掉「🔄 更换」按钮（上传卡住时根本渲染不出来）。
    改为 widget_key 嵌入 `st.session_state["_reset_token"]`：当用户点
    「🧹 重置页面」时把 token++，所有 file_uploader 拿到全新 key，
    Streamlit 视为新 widget → 即使上次上传卡在 99%，下次进入也像第一次打开页面。

    container: 渲染目标容器，传 `st.sidebar` 可在侧栏内排版。
    返回值与 `st.file_uploader` 行为一致：单文件返回 UploadedFile 或 None，
    多文件返回 list[UploadedFile]（即使空也是 []）。
    """
    c = container if container is not None else st
    widget_key = f"{base_key}_v{_reset_suffix()}"
    uploaded = c.file_uploader(
        label,
        type=type,
        key=widget_key,
        accept_multiple_files=accept_multiple,
        help=help,
        **kwargs,
    )
    return uploaded


def _is_full_result(r) -> bool:
    """判断 VideoResult 是否来自全流程模式（有 mp4 + frames_dir + annotated_dir）。"""
    return bool(
        getattr(r, "output_video", None)
        and getattr(r, "frames_dir", None)
        and getattr(r, "annotated_dir", None)
    )


# ---------- session_state 初始化 ----------


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("results", {})
    ss.setdefault("model_path", None)
    ss.setdefault("cache_model", False)
    ss.setdefault("last_zip", b"")
    ss.setdefault("running", False)
    ss.setdefault("class_names", {})  # {int(cid): str(name)}
    ss.setdefault("class_names_key", None)  # 缓存键：(str(path), mtime)
    ss.setdefault("model_upload_id", None)
    ss.setdefault("cached_model_path", None)
    ss.setdefault("artifact_refresh_token", 0)
    ss.setdefault("_reset_token", 0)  # v1.0.6 hotfix: 全局重置 token


def _reset_suffix() -> str:
    """返回当前 reset_token 的字符串后缀，供所有 widget_key 嵌入：f\"{base}_v{_reset_suffix()}\"。

    v1.0.6 hotfix 关键：Streamlit widget_id 由 (script_path, line, col, widget_key) 决定，
    只有 widget_key 变 → widget_id 才变 → 旧的 widget state（即使 ss[key] 已清）也不会被复用。
    每次「🧹 重置页面」_reset_token++，所有 widget 拿到全新 key，UI 真正回到初始默认。
    """
    return str(st.session_state.get("_reset_token", 0))


def _get_cached_class_names(model_path: Path) -> dict[int, str]:
    """按 (path, mtime) 缓存模型类别读取，避免每次 rerun 都加载 YOLO。"""
    try:
        mtime = model_path.stat().st_mtime
    except OSError:
        return {}
    key = (str(model_path), mtime)
    if st.session_state.get("class_names_key") == key:
        return st.session_state.get("class_names", {})
    try:
        names = get_model_class_names(model_path)
    except Exception as exc:
        st.warning(f"读取模型类别失败: {exc}")
        names = {}
    st.session_state["class_names"] = names
    st.session_state["class_names_key"] = key
    return names


def _reset_model_state() -> None:
    """清空与模型相关的 session_state：model_path / class_names / 类别输入框。

    在「更换/删除模型」或「重置页面」时调用，确保 Step 2 的「标注类别」列表
    不会保留上一个模型的残留（class_names 缓存 + 每个 lbl_<cid> widget）。
    """
    ss = st.session_state
    ss["model_path"] = None
    ss["class_names"] = {}
    ss["class_names_key"] = None
    ss["model_upload_id"] = None
    ss["cached_model_path"] = None
    # 清掉 Step 2 的类别输入框 widget state（key 是 lbl_<cid>）
    for k in [
        k
        for k in ss.keys()
        if isinstance(k, str) and (k.startswith("lbl_") or k.startswith("infer_classes_v"))
    ]:
        del ss[k]


# v1.0.6 hotfix: 「🧹 重置页面」白名单。按白名单清 session_state，避免误删
# Streamlit 自身内部键（如 _streamlit_*、表单内部状态等）导致下次 rerun 报奇异错误。
# 所有 widget_key 在 reset 后嵌入 `_v<token>`；通过正则 `suffixed_widget_re` 在
# `_clear_all_state` 中匹配任意 cycle 的残留 key。下方白名单只列非 widget-derived key。
_KNOWN_KEYS = frozenset(
    {
        # 主流程派生 state
        "results",
        "model_path",
        "cache_model",
        "device",
        "running",
        "last_zip",
        "class_names",
        "class_names_key",
        "model_upload_id",
        "cached_model_path",
        "artifact_refresh_token",
        "artifact_delete_pending",
        "artifact_task",
        "artifact_refresh",
        "artifact_delete",
        "artifact_delete_confirm",
        "artifact_delete_cancel",
        "active_batch_id",
        "synced_batch_id",
        # 侧栏按钮 + 确认流（这些是触发器，不需要 reset 嵌入）
        "confirm_clear_files",
        "cf_yes",
        "cf_no",
        "btn_clear_files",  # 清空本地文件按钮（固定 key）
        "clear_results",  # 结果区清空按钮
        "start_btn",  # ▶ 开始处理按钮
        # 全局 reset token；重置后由调用方设为 1
        "_reset_token",
        # toast 队列（清空后下次可再次设置）
        "_toast_msg",
        # v1.0.6 hotfix#3: 硬刷新 flag，按钮 on_click 置位、main() 顶部消费
        "_reload_after_rerun",
        # 旧 cycle 的「重置页面」按钮 key（btn_reset_page_v<n>）靠
        # suffixed_widget_re 正则清；新 cycle 的 key 由 _reset_token 决定。
    }
)


def _clear_all_state() -> None:
    """按白名单清 session_state（不动 Streamlit 内部键），让 UI 回到初始默认。

    v1.0.6 hotfix: 「🧹 重置页面」使用此函数；磁盘文件**不动**（uploads/、
    outputs/、outputs/_models/ 都不删），由「🧹 清空本地文件」按钮负责。
    调用方负责随后把 `_reset_token` 设回 1，让所有 file_uploader 重建为空。
    """
    import re as _re

    # v1.0.6 hotfix: 所有 widget_key 内嵌 `_v<token>`；reset 后旧 cycle 留下的
    # `_v<n>` 残留 key（如 mode_radio_v0）也要清掉，否则下次 reset 越攒越多。
    suffixed_widget_re = _re.compile(
        r"^(mode_radio|interval_input|conf_input|iou_input|color_zh|"
        r"custom_color|unified_label|classes_all|classes_none|fps_choice|fps_custom|cache_model|"
        r"device|model_uploader|videos_uploader|infer_images|encode_images|"
        r"encode_source_video|btn_reset_page)_v\d+$"
    )
    ss = st.session_state
    for k in list(ss.keys()):
        if not isinstance(k, str):
            if k in _KNOWN_KEYS:
                del ss[k]
            continue
        # 动态前缀/模式匹配
        if (
            k.startswith("lbl_")  # Step 2 类别输入框
            or k.startswith("infer_classes_v")
            or k.startswith("artifact_")
            or k.startswith("_dl_cache_")  # 下载缓存
            or suffixed_widget_re.match(k)  # 任意 cycle 的 _v<n> widget state
        ):
            del ss[k]
            continue
        if k in _KNOWN_KEYS:
            del ss[k]


# ---------- 侧栏 ----------


def _device_options() -> list[str]:
    return ["auto", "cpu", "cuda"]


def _sidebar() -> None:
    st.sidebar.header("⚙️ 全局设置")
    uploaded_model = _file_uploader_with_reset(
        "选择模型 (.pt)",
        type=["pt"],
        base_key="model_uploader",
        help="单选一个 YOLO 权重文件",
        container=st.sidebar,
    )
    if uploaded_model is not None:
        upload_id = getattr(
            uploaded_model,
            "file_id",
            f"{uploaded_model.name}:{getattr(uploaded_model, 'size', '')}",
        )
        if st.session_state.get("cache_model"):
            if st.session_state.get("model_upload_id") != upload_id or not st.session_state.get(
                "cached_model_path"
            ):
                data = uploaded_model.read()
                new_path = cache_uploaded_model(data, uploaded_model.name)
                st.session_state["cached_model_path"] = str(new_path)
            else:
                new_path = Path(st.session_state["cached_model_path"])
            cached_msg = f"已缓存到 {new_path}"
        else:
            tmp = (
                OUTPUTS_DIR
                / "_models"
                / safe_stem(Path(uploaded_model.name).stem)
                / uploaded_model.name
            )
            tmp.parent.mkdir(parents=True, exist_ok=True)
            # v1.0.6.2: 只有文件缺失或大小变化才读+写。之前每次 rerun 都无条件
            # read()+write_bytes() 整个模型（50~130MB），导致每次交互（含上传图片
            # 触发的 rerun）都做一次大内存读 + 大磁盘写，阻塞主线程，把上传触发的
            # rerun 拖到「页面卡死、下载按钮迟迟不渲染」。用 size 属性判断即可，
            # 命中时连 read() 都不做，rerun 变成纯渲染。
            model_size = getattr(uploaded_model, "size", None)
            if (
                st.session_state.get("model_upload_id") != upload_id
                or model_size is None
                or not tmp.exists()
                or tmp.stat().st_size != model_size
            ):
                tmp.write_bytes(uploaded_model.read())
            new_path = tmp
            cached_msg = "未跨会话缓存"
        # 模型换文件时清空「标注类别」缓存 + 旧的 model_path
        prev_path = st.session_state.get("model_path")
        previous_upload_id = st.session_state.get("model_upload_id")
        if prev_path != new_path or previous_upload_id != upload_id:
            _reset_model_state()
        st.session_state["model_path"] = new_path
        st.session_state["model_upload_id"] = upload_id
        if st.session_state.get("cache_model"):
            st.session_state["cached_model_path"] = str(new_path)
        st.sidebar.caption(f"模型: {uploaded_model.name}（{cached_msg}）")

    cache_model = st.sidebar.checkbox(
        "跨会话缓存上传的模型",
        value=st.session_state.get("cache_model", False),
        key=f"cache_model_v{_reset_suffix()}",
        help="勾选后会把模型按 SHA1 存到 uploads/models/，下次会话可复用",
    )
    # widget key 带 reset 后缀；同步到稳定的业务 key 供下游读取。
    st.session_state["cache_model"] = cache_model
    device = st.sidebar.selectbox(
        "推理设备",
        _device_options(),
        index=0,
        key=f"device_v{_reset_suffix()}",
    )
    st.session_state["device"] = device

    st.sidebar.divider()

    # ---- 清空本地文件（两次确认：先点按钮，再点「确认删除」）----
    if st.sidebar.button(
        "🧹 清空本地文件",
        key="btn_clear_files",
        help="删除 outputs/ 下全部生成文件（保留 _models/ 上传的模型）",
    ):
        st.session_state["confirm_clear_files"] = True
    if st.session_state.get("confirm_clear_files"):
        st.sidebar.warning("确认要删除全部的生成文件吗？")
        col_yes, col_no = st.sidebar.columns(2)
        with col_yes:
            if st.button("确认删除", key="cf_yes", type="primary", use_container_width=True):
                # 保留 _models/（用户上传的模型文件，非生成物）
                _delete_dir_contents(OUTPUTS_DIR, skip_names={"_models"})
                st.session_state["results"] = {}
                st.session_state["confirm_clear_files"] = False
                st.toast("已删除全部生成文件", icon="🧹")
                st.rerun()
        with col_no:
            if st.button("取消", key="cf_no", use_container_width=True):
                st.session_state["confirm_clear_files"] = False
                st.rerun()

    # ---- 重置页面：仅重置页面 UI + session_state + 强制所有 file_uploader 重建为空 ----
    # v1.0.6 hotfix: 通过 `_reset_token` 让所有 file_uploader 的 widget_key 变掉，
    # 即使上次上传卡在 99%，重置后再进入也像第一次打开页面（uploads/ / outputs/
    # 等磁盘文件不动；磁盘清理由上方「清空本地文件」按钮负责）。
    #
    # v1.0.6 hotfix#2: 点下按钮**立刻**走浏览器级硬刷新（window.location.reload()），
    # 而不是仅 st.rerun()。这样能彻底丢掉 Streamlit 在浏览器里缓存的 widget state、
    # WebSocket 连接、以及任何 in-flight 的 file_uploader 请求。
    #
    # v1.0.6 hotfix#3: 用 on_click 回调 + flag 模式，避免 button 在同一 key 下
    # 第二次点不再触发 callback（Streamlit button widget value 已经"消费"过了，
    # 重复点击不会重新执行 callback）。flag 由 main() 顶部消费，注入 JS 后立刻
    # 删除——下次点击再次置位、再次消费，保证按钮可以反复点。
    def _on_reset_page() -> None:
        """重置按钮回调：清 ss + 置位 token + 设 flag 等待 main() 顶部注入 JS 硬刷新。"""
        _clear_all_state()
        st.session_state["_reset_token"] = 1
        st.session_state["_reload_after_rerun"] = True

    st.sidebar.button(
        "🧹 重置页面",
        key=f"btn_reset_page_v{_reset_suffix()}",
        help="重置所有页面 UI 控件与会话状态；不删除任何本地文件。"
        "上传 / 下载卡住时点此按钮即可硬刷新页面",
        on_click=_on_reset_page,
    )


# ---------- 三步的 step 渲染器 ----------


def _step_extract(mode_key: str) -> dict:
    """Step1：抽帧。mode_key != 'full' 且 mode_key != 'extract' 时返回空 dict。"""
    if mode_key not in ("full", "extract"):
        return {}
    with st.expander("🎞 Step 1 · 视频抽帧", expanded=True):
        videos = _file_uploader_with_reset(
            "上传视频（可多选）",
            type=["mp4", "mov", "avi", "mkv"],
            base_key="videos_uploader",
            accept_multiple=True,
            help="可一次选多个；上传卡住时点侧栏「🧹 重置页面」",
        )
        raw_interval = st.text_input(
            "抽帧间隔（每隔多少帧抽 1 帧，正整数）",
            value="1",
            key=f"interval_input_v{_reset_suffix()}",
        )
        interval, fell_back = parse_positive_int_text(raw_interval, default=1)
        if fell_back and raw_interval.strip():
            st.warning(f"抽帧间隔「{raw_interval}」无效，已回退为 1")
        st.caption(f"已选择 {len(videos) if videos else 0} 个视频")
        return {"videos": videos or [], "interval": interval}


def _step_infer(mode_key: str) -> dict:
    """Step2：推理。mode_key != 'full' 且 mode_key != 'infer' 时返回空 dict。"""
    if mode_key not in ("full", "infer"):
        return {}
    with st.expander("🤖 Step 2 · 模型推理与标注", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            raw_conf = st.text_input(
                "置信度阈值 (0~1)", value="0.25", key=f"conf_input_v{_reset_suffix()}"
            )
        with col2:
            raw_iou = st.text_input(
                "NMS IoU 阈值 (0~1)", value="0.45", key=f"iou_input_v{_reset_suffix()}"
            )
        conf, c_fell = parse_positive_float_text(raw_conf, default=0.25)
        if c_fell and raw_conf.strip():
            st.warning(f"置信度「{raw_conf}」无效，已回退为 0.25")
        iou, i_fell = parse_positive_float_text(raw_iou, default=0.45)
        if i_fell and raw_iou.strip():
            st.warning(f"IoU「{raw_iou}」无效，已回退为 0.45")

        color_zh = st.selectbox(
            "标注框颜色", COLOR_ZH_OPTIONS, index=0, key=f"color_zh_v{_reset_suffix()}"
        )
        if color_zh == "自定义":
            hex_color = st.color_picker(
                "自定义颜色", "#FF0000", key=f"custom_color_v{_reset_suffix()}"
            )
            box_color = parse_color(hex_color, default=DEFAULT_BOX_COLOR)
        else:
            box_color = parse_color(COLOR_NAME_ZH_TO_EN[color_zh], default=DEFAULT_BOX_COLOR)
        st.caption(f"当前 BGR: {box_color}")

        # ---- 模型类别过滤 + 自定义标注名称 ----
        model_path_str = st.session_state.get("model_path")
        model_path = Path(model_path_str) if model_path_str else None
        class_names = (
            _get_cached_class_names(model_path) if model_path and model_path.exists() else {}
        )

        label_rows: list[tuple[int, str, str]] = []
        selected_classes: list[int] | None = None
        if class_names:
            sorted_items = sorted(class_names.items())
            all_class_ids = [class_id for class_id, _ in sorted_items]
            selector_key = f"infer_classes_v{_reset_suffix()}"
            st.session_state.setdefault(selector_key, all_class_ids)
            all_col, none_col = st.columns(2)
            if all_col.button(
                "全部选择", key=f"classes_all_v{_reset_suffix()}", use_container_width=True
            ):
                st.session_state[selector_key] = all_class_ids
                st.rerun()
            if none_col.button(
                "全部取消", key=f"classes_none_v{_reset_suffix()}", use_container_width=True
            ):
                st.session_state[selector_key] = []
                st.rerun()
            selected_classes = st.multiselect(
                "推理类别",
                options=all_class_ids,
                format_func=lambda cid: f"{cid}: {class_names[cid]}",
                key=selector_key,
                help="模型只检测这里选中的类别",
            )
            if not selected_classes:
                st.warning("请至少选择一个推理类别")

            unified = st.text_input(
                "统一标注名称（留空则按下方每类自定义）",
                placeholder="例如: 目标 / Object",
                key=f"unified_label_v{_reset_suffix()}",
                help="非空时覆盖已选择类别的名称",
            )

            selected_items = [item for item in sorted_items if item[0] in selected_classes]
            n = len(selected_items)
            cols_per_row = 2 if n <= 6 else 3
            for start in range(0, n, cols_per_row):
                row_items = selected_items[start : start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, (cid, default_name) in zip(cols, row_items):
                    with col:
                        # key 里加 cid，保证每个类别一个独立 widget
                        val = st.text_input(
                            f"类别 {cid}（默认 {default_name}）",
                            value=default_name,
                            key=f"lbl_{cid}",
                        )
                        label_rows.append((cid, val, default_name))
        else:
            st.caption("尚未加载模型，无法显示类别列表")
            unified = ""

        label_map = label_table_to_dict(label_rows, unified) if label_rows else None

        return {
            "conf": conf,
            "iou": iou,
            "box_color": box_color,
            "label_map": label_map,
            "selected_classes": selected_classes,
        }


def _step_encode(mode_key: str) -> dict:
    """Step3：合成。mode_key != 'full' 且 mode_key != 'encode' 时返回空 dict。"""
    if mode_key not in ("full", "encode"):
        return {}
    with st.expander("🎬 Step 3 · 合成视频", expanded=True):
        fps_choice = st.selectbox(
            "帧率", ["原视频帧率", "自定义"], index=0, key=f"fps_choice_v{_reset_suffix()}"
        )
        fps = None
        if fps_choice == "自定义":
            fps = st.number_input("自定义帧率", 1, 120, 30, key=f"fps_custom_v{_reset_suffix()}")
        return {"fps_choice": fps_choice, "fps": fps}


# ---------- 单 stage 模式额外输入 ----------


def _collect_extract_extras(mode_key: str) -> dict:
    """仅抽帧模式不需要额外输入。"""
    return {}


def _collect_infer_extras(mode_key: str) -> dict:
    """仅推理模式：上传一张或多张图片（按文件名排序后批量推理）。"""
    if mode_key != "infer":
        return {}
    images = _file_uploader_with_reset(
        "上传图片（可多选，按文件名排序后推理）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        base_key="infer_images",
        accept_multiple=True,
        help="选择一张或多张图片文件，结果在 outputs/infer_<时间戳>/annotated/images/",
    )
    return {"images": images or []}


def _collect_encode_extras(mode_key: str, steps: dict) -> dict:
    """仅合成模式：上传多张图片 + 可选源视频（仅 fps=原视频帧率 时显示）。"""
    if mode_key != "encode":
        return {}
    images = _file_uploader_with_reset(
        "上传图片（可多选，按文件名顺序合成）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        base_key="encode_images",
        accept_multiple=True,
        help="按文件名排序后合成 mp4",
    )
    fps_choice = steps.get("fps_choice", "原视频帧率")
    source_video = None
    # v1.0.6: 反转逻辑 — 「原视频帧率」时才需要源视频（读 fps）；
    # 「自定义」时直接用 number_input 即可，源视频上传栏隐藏。
    if fps_choice == "原视频帧率":
        source_video = _file_uploader_with_reset(
            "源视频（用于读取原始帧率，可选）",
            type=["mp4", "mov", "avi", "mkv"],
            base_key="encode_source_video",
            help="上传后系统读取原始帧率；不传则按 30 fps 回退；上传卡住时点侧栏「🧹 重置页面」",
        )
    return {"images": images or [], "source_video": source_video, "fps_choice": fps_choice}


def _run_pipeline_ui(mode_key: str, steps: dict, infer_extras: dict, encode_extras: dict) -> None:
    """按 mode 分发到 run_pipeline(...) 的不同入口。"""
    device = st.session_state.get("device", "auto")
    model_path_str = st.session_state.get("model_path")
    model_path = Path(model_path_str) if model_path_str else None

    # ---- 准备 per-video 输入 ----
    video_paths: list[Path] = []
    if mode_key in ("full", "extract"):
        videos = steps.get("videos", [])
        if not videos:
            st.warning("请先在 Step 1 上传至少一个视频")
            return
        for vf in videos:
            data = vf.read()
            p = save_uploaded_video(data, vf.name)
            video_paths.append(p)
            vf.seek(0)

    if mode_key in ("full", "infer") and (model_path is None or not model_path.exists()):
        st.warning("请先在左侧上传并选择一个模型")
        return

    # 推断展示用的 stems（按 mode 不同，stems 来源不同）
    job_stems: dict[str, dict] = {}  # stem -> 该 job 的额外元数据
    if mode_key in ("full", "extract"):
        # uploads/<stem>/video.mp4 的父目录名才是唯一的 stem（p.stem 永远是 "video"）
        for p, uploaded in zip(video_paths, videos):
            stem = safe_stem(p.parent.name)
            job_stems[stem] = {"raw_video": p, "input_name": uploaded.name}
    elif mode_key == "infer":
        images = infer_extras.get("images", [])
        if not images:
            st.warning("请先上传至少一张图片")
            return
        stem = _timestamp_id("infer")
        # 先把图片落盘到 outputs/<stem>/_uploaded/，作为 frames_dir
        upload_dir = save_uploaded_images(images, stem, outputs_root=OUTPUTS_DIR)
        job_stems[stem] = {
            "frames_dir": upload_dir,
            "input_names": [image.name for image in images],
        }
    elif mode_key == "encode":
        images = encode_extras.get("images", [])
        if not images:
            st.warning("请先上传至少一张图片")
            return
        stem = _timestamp_id("compose")
        upload_dir = save_uploaded_images(images, stem, outputs_root=OUTPUTS_DIR)
        # 源视频：仅在 fps=自定义 时由 UI 提供；fps=原视频帧率 时直接用默认 30 fps
        sv_upload = encode_extras.get("source_video")
        encode_fps = steps.get("fps")
        if encode_fps is None and sv_upload is not None:
            tmp_sv = OUTPUTS_DIR / "_src" / safe_stem(sv_upload.name) / "source.mp4"
            tmp_sv.parent.mkdir(parents=True, exist_ok=True)
            tmp_sv.write_bytes(sv_upload.read())
            try:
                sv_fps, _ = read_video_meta(tmp_sv)
                encode_fps = max(int(round(sv_fps)), 1)
            except Exception as exc:
                st.error(f"读取源视频帧率失败: {exc}")
                return
            try:
                sv_upload.seek(0)
            except Exception:
                pass
        job_stems[stem] = {
            "annotated_dir": upload_dir,
            "fps": encode_fps,
            "input_names": [image.name for image in images],
        }
    else:
        st.error(f"未知运行模式: {mode_key}")
        return

    display_stems = list(job_stems.keys())

    # ---- 构造 run_pipeline 参数 ----
    common = dict(
        model_path=model_path,
        frame_interval=steps.get("interval", 1),
        conf=steps.get("conf", 0.25),
        iou=steps.get("iou", 0.45),
        device=device,
        box_color=steps.get("box_color", DEFAULT_BOX_COLOR),
        label_map=steps.get("label_map"),
        selected_classes=steps.get("selected_classes"),
        fps=steps.get("fps"),
    )

    # 单 job 的三种 mode: infer / encode / extract → 传入对应的输入目录
    if mode_key == "full":
        # full 模式按 video_paths 走旧路径
        kwargs = {**common, "mode": "full", "video_paths": video_paths}
    elif mode_key == "extract":
        kwargs = {**common, "mode": "extract", "video_paths": video_paths}
    elif mode_key == "infer":
        # 把每个 stem 的 frames_dir 注入；但 v1.0.2 单 job 单 stem，直接传
        stem = display_stems[0]
        frames_dir = job_stems[stem]["frames_dir"]
        kwargs = {**common, "mode": "infer", "video_paths": [], "frames_dir": frames_dir}
    elif mode_key == "encode":
        stem = display_stems[0]
        annotated_dir = job_stems[stem]["annotated_dir"]
        encode_fps = job_stems[stem]["fps"]
        kwargs = {
            **common,
            "mode": "encode",
            "video_paths": [],
            "annotated_dir": annotated_dir,
            "fps": encode_fps,
        }
    else:
        st.error(f"未知运行模式: {mode_key}")
        return

    try:
        batch_id = _timestamp_id("batch")
        task_config = {
            "frame_interval": steps.get("interval", 1),
            "confidence": steps.get("conf", 0.25),
            "iou": steps.get("iou", 0.45),
            "device": device,
            "box_color": list(steps.get("box_color", DEFAULT_BOX_COLOR)),
            "label_map": steps.get("label_map") or {},
            "selected_classes": steps.get("selected_classes"),
            "fps": kwargs.get("fps"),
            "model_name": model_path.name if model_path else None,
        }
        task_inputs = {}
        for stem, meta in job_stems.items():
            raw_video = meta.get("raw_video")
            task_inputs[stem] = meta.get("input_names") or [
                meta.get("input_name") or (raw_video.name if raw_video else stem)
            ]
        kwargs["task_context"] = {
            "batch_id": batch_id,
            "config": task_config,
            "inputs": task_inputs,
        }
        submit_pipeline(
            batch_id=batch_id,
            outputs_root=OUTPUTS_DIR,
            kwargs=kwargs,
            stems=display_stems,
        )
        st.session_state["running"] = True
        st.session_state["active_batch_id"] = batch_id
        st.toast("任务已提交到后台", icon="🚀")
        st.rerun()
    except Exception as exc:
        st.error(f"无法启动后台任务: {exc}")


@st.fragment(run_every="1s")
def _background_status_panel() -> None:
    """轮询磁盘任务状态；后台线程不直接调用 Streamlit。"""
    status = active_status(OUTPUTS_DIR)
    if not status:
        st.session_state["running"] = False
        return
    state = str(status.get("status", ""))
    running = state in {"queued", "running", "cancelling"}
    batch_id = str(status.get("batch_id", ""))
    session_batch_id = str(st.session_state.get("active_batch_id", ""))
    if running and not session_batch_id:
        # 刷新或重新打开页面时，只自动关联仍在执行的任务。已结束的历史批次
        # 不应在「重置页面」后再次出现。
        st.session_state["active_batch_id"] = batch_id
        session_batch_id = batch_id
    if not running and session_batch_id != batch_id:
        st.session_state["running"] = False
        return
    st.session_state["running"] = running
    if not running and st.session_state.get("synced_batch_id") != status.get("batch_id"):
        for result in results_from_status(status):
            st.session_state["results"][result.stem] = result
        st.session_state["synced_batch_id"] = status.get("batch_id")
        st.toast(str(status.get("message", "后台批次已结束")), icon="✅")
        st.rerun()

    labels = {
        "queued": "等待执行",
        "running": "处理中",
        "cancelling": "正在取消",
        "completed": "已完成",
        "cancelled": "已取消",
        "failed": "存在失败",
        "interrupted": "已中断",
    }
    with st.container(border=True):
        st.markdown(f"#### 任务进度 · {labels.get(state, state)}")
        task_name = status.get("current_task") or next(iter(status.get("stems") or []), "-")
        task_mode = str(status.get("mode", "未知"))
        st.markdown(f"**任务名称：{task_name}**")
        st.markdown(f"任务类型：{STAGE_LABELS.get(task_mode, task_mode)}")
        st.markdown(
            f"任务开始时间：{_format_started_at(status.get('started_at') or status.get('created_at'))}"
        )
        total = max(int(status.get("total") or 0), 1)
        current = min(max(int(status.get("current") or 0), 0), total)
        st.progress(current / total, text=str(status.get("message", "")))
        st.caption(
            f"当前阶段：{STAGE_LABELS.get(str(status.get('stage', '')), status.get('stage', '-'))}　"
            f"批次：{status.get('batch_id', '-')}"
        )
        if running:
            if st.button(
                "停止当前批次",
                key=f"cancel_batch_{status.get('batch_id')}",
                disabled=state == "cancelling",
                use_container_width=True,
            ):
                if cancel_batch(OUTPUTS_DIR, str(status.get("batch_id"))):
                    st.toast("已发送取消请求", icon="⏹️")
                else:
                    st.warning("任务已经结束，无法取消")


# ---------- 结果区 ----------


def _results_panel() -> None:
    results: dict[str, VideoResult] = st.session_state.get("results", {})
    header_cols = st.columns([5, 1])
    with header_cols[0]:
        st.subheader("📥 处理结果")
    with header_cols[1]:
        if st.button(
            "清空结果",
            key="clear_results",
            use_container_width=True,
            help="仅清空当前会话展示的结果（磁盘文件保留）",
        ):
            st.session_state["results"] = {}
            st.rerun()

    if not results:
        st.caption("尚无结果。先上传参数并点击「开始处理」。")
        return

    # v1.0.5: 仅全流程模式展示「📦 下载全部 (ZIP)」；单 stage 模式各自单 stem 下载即可
    show_zip = _should_show_session_zip(results)

    for stem in sorted(results.keys()):
        r = results[stem]
        with st.container(border=True):
            task_mode = STAGE_LABELS.get(r.mode, r.mode)
            st.markdown(f"**任务名称：{stem}**")
            st.markdown(f"任务类型：{task_mode}")
            st.markdown(f"任务开始时间：{_format_started_at(r.started_at)}")
            if r.error:
                if r.status == "cancelled":
                    st.warning("任务已取消，已生成的部分工件保留在文件浏览中。")
                else:
                    st.error(f"失败: {r.error}")
                continue

            # v1.0.3: 全流程模式只展示最终视频；不展示中间抽帧 / 标注目录
            is_full = _is_full_result(r)

            if r.stats:
                stats_cols = st.columns(4)
                stats_cols[0].metric("推理图片", r.stats.get("total_images", 0))
                stats_cols[1].metric("命中图片", r.stats.get("matched_images", 0))
                stats_cols[2].metric("读取失败", r.stats.get("failed_images", 0))
                stats_cols[3].metric(
                    "检测目标",
                    sum((r.stats.get("class_counts") or {}).values()),
                )
                if r.stats.get("class_counts"):
                    rows = []
                    for index, (raw_name, count) in enumerate(
                        (r.stats["class_counts"] or {}).items(), 1
                    ):
                        category = str(raw_name).split(":", 1)[-1]
                        rows.append({"序号": index, "类别": category, "数量": count})
                    _centered_table(rows)

            # ---- 视频产物（full / encode）----
            if r.output_video and r.output_video.exists():
                # 下载数据使用延迟回调：页面渲染时只读取文件大小，用户点击后
                # 才在下载线程读盘；on_click="ignore" 避免下载触发整页 rerun。
                mp4_path = r.output_video
                col_v, col_btn = st.columns([3, 1])
                col_v.markdown(
                    f"输出: `{mp4_path}`  ({mp4_path.stat().st_size / 1024 / 1024:.2f} MB)"
                )
                col_btn.download_button(
                    "下载视频",
                    data=deferred_file_bytes(mp4_path),
                    file_name=mp4_path.name,
                    mime="video/mp4",
                    key=f"dl_video_{stem}",
                    on_click="ignore",
                    use_container_width=True,
                )

            # ---- 抽帧产物（仅 extract）----
            if not is_full and r.frames_dir and r.frames_dir.exists():
                n_frames = sum(1 for _ in r.frames_dir.glob("*.jpg"))
                col_f, col_btn = st.columns([3, 1])
                col_f.markdown(f"抽帧目录: `{r.frames_dir}`（{n_frames} 张）")
                col_btn.download_button(
                    "下载抽帧 ZIP",
                    data=deferred_frames_zip(r.frames_dir),
                    file_name=f"{stem}_frames.zip",
                    mime="application/zip",
                    key=f"dl_frames_{stem}",
                    on_click="ignore",
                    use_container_width=True,
                )

            # ---- 推理产物（仅 infer）----
            if not is_full and r.annotated_dir and r.annotated_dir.exists():
                n_ann = sum(1 for _ in r.annotated_dir.glob("*.jpg"))
                col_a, col_btn = st.columns([3, 1])
                col_a.markdown(f"标注目录: `{r.annotated_dir}`（{n_ann} 张）")
                col_btn.download_button(
                    "下载推理 ZIP",
                    data=deferred_infer_zip(r.annotated_dir),
                    file_name=f"{stem}_infer.zip",
                    mime="application/zip",
                    key=f"dl_infer_{stem}",
                    on_click="ignore",
                    use_container_width=True,
                )

    if not show_zip:
        return
    st.divider()
    # v1.0.2: 只打包当前会话在 st.session_state["results"] 里的 stem
    st.download_button(
        "📦 下载全部 (ZIP)",
        data=deferred_session_zip(results),
        file_name="cv_session.zip",
        mime="application/zip",
        on_click="ignore",
        use_container_width=True,
    )


# ---------- 主入口 ----------


def main() -> None:
    st.set_page_config(
        page_title="CV工具箱",
        page_icon="🧰",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _sidebar()

    # v1.0.6: 消费「重置页面」按钮的 toast（必须在 rerun 后第一时间消费，否则错过显示时机）
    if "_toast_msg" in st.session_state:
        msg, icon = st.session_state.pop("_toast_msg")
        st.toast(msg, icon=icon)

    # v1.0.6 hotfix#3: 消费「重置页面」按钮的硬刷新 flag。
    # 按钮 on_click 回调里只设 flag + 清状态；真正的页面刷新通过
    # <meta http-equiv="refresh"> 在浏览器侧触发，**不依赖 JS**。
    #
    # 为什么不用 st.components.v1.html 或 st.markdown 里的 <script>？
    # components.v1.html 渲染的是 0×0 <iframe>，iframe 内的 JS `window.location.reload()`
    # 只能 reload iframe 自身，**不会 reload 外层页面**。
    # st.markdown(unsafe_allow_html=True) 里嵌入的 <script> 在 Streamlit 1.30+ 里
    # 也会被 sandbox 化（脚本默认 CSP 不允许 inline 脚本），执行时机也不可控。
    #
    # <meta http-equiv="refresh" content="0"> 是浏览器**原生**支持的硬刷新指令，
    # 不依赖 JS 执行、不被任何 CSP 拦截、立即生效：浏览器解析到 meta tag 后
    # 立刻 reload 当前顶层 URL → WebSocket 断开 → 服务端 session 清零 →
    # 浏览器重新加载 → 等价于地址栏按回车，file_uploader 的 FileList 也会清掉。
    #
    # ⚠️ 已知限制：浏览器正在上传大文件（multipart POST 进行中）时，整个 JS 主线程
    # 被 throttle，WebSocket 消息积压、页面渲染冻结；直到当前上传完成/超时，
    # meta refresh 才会生效（看起来像"卡住不动"）。这种场景需要用户主动按
    # Ctrl+R / F5 硬刷新浏览器（sidebar 顶部有提示）。
    if st.session_state.pop("_reload_after_rerun", False):
        st.markdown(
            '<meta http-equiv="refresh" content="0">',
            unsafe_allow_html=True,
        )

    st.title("🧰 CV工具箱")
    st.caption(f"{_app_version()} · 视频抽帧 / 类别过滤推理 / 视频合成 / 工件管理")

    mode_zh = st.radio(
        "运行模式",
        MODE_OPTIONS,
        horizontal=True,
        index=0,
        key=f"mode_radio_v{_reset_suffix()}",
        help="运行完整流水线、单独步骤，或浏览 outputs 中的任务工件。",
    )
    mode_key = MODE_KEY_MAP[mode_zh]

    if mode_key == "browse":
        render_artifact_browser(OUTPUTS_DIR)
        return

    s1 = _step_extract(mode_key)
    s2 = _step_infer(mode_key)
    s3 = _step_encode(mode_key)

    # 单 stage 模式补的额外输入（encode 需要读取 s3 的 fps_choice）
    infer_extras = _collect_infer_extras(mode_key)
    encode_extras = _collect_encode_extras(mode_key, s3)

    st.divider()
    missing_classes = mode_key in ("full", "infer") and s2.get("selected_classes") == []
    run_clicked = st.button(
        "▶ 开始处理",
        type="primary",
        use_container_width=True,
        key="start_btn",
        disabled=st.session_state.get("running", False) or missing_classes,
    )
    if run_clicked:
        _run_pipeline_ui(mode_key, {**s1, **s2, **s3}, infer_extras, encode_extras)

    st.divider()
    _background_status_panel()
    _results_panel()


if __name__ == "__main__":
    main()
