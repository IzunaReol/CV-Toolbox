"""Streamlit WEB UI（v1.0.6）：包装抽帧 → 推理 → 合成视频 三段式流水线。

启动方式（在项目根下）：
    streamlit run web/app.py --server.maxUploadSize 1024

v1.0.6 变更：
  0. 「🧹 清空缓存」改为**只重置页面 UI 控件 + session_state**，**不删任何本地文件**
     （磁盘文件删除仍由 v1.0.3 的「🧹 清空本地文件」两步按钮负责）
  1. 仅合成的源视频上传控件逻辑反转：「原视频帧率」时显示，「自定义」时隐藏
  2. widget 全部加显式 `key`，配合白名单 reset 真正恢复初始状态
  3. 结果区下载按钮的 IO 走 `@st.cache_data`，rerun 不再重复读盘
     → 不再出现「按钮置灰 → 消失 → 再出现」；减少 `WinError 10054` 触发

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
from pathlib import Path

# 让 helpers / pipeline 在任意调用方式下都能导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

try:
    from .helpers import (
        COLOR_NAME_ZH_TO_EN,
        build_frames_zip,
        build_infer_zip,
        build_session_zip,
        label_table_to_dict,
        parse_positive_float_text,
        parse_positive_int_text,
        read_file_bytes_cached,
        safe_stem,
        save_uploaded_images,
    )
    from .pipeline import (
        OUTPUTS_DIR,
        UPLOADS_DIR,
        VideoResult,
        cache_uploaded_model,
        get_model_class_names,
        parse_color,
        read_video_meta,
        run_pipeline,
        save_uploaded_video,
    )
except ImportError:  # 当作顶层脚本运行（streamlit run web/app.py）时回落
    from helpers import (
        COLOR_NAME_ZH_TO_EN,
        build_frames_zip,
        build_infer_zip,
        build_session_zip,
        label_table_to_dict,
        parse_positive_float_text,
        parse_positive_int_text,
        read_file_bytes_cached,
        safe_stem,
        save_uploaded_images,
    )
    from pipeline import (
        OUTPUTS_DIR,
        UPLOADS_DIR,
        VideoResult,
        cache_uploaded_model,
        get_model_class_names,
        parse_color,
        read_video_meta,
        run_pipeline,
        save_uploaded_video,
    )


DEFAULT_BOX_COLOR = (0, 0, 255)  # BGR 红色
COLOR_ZH_OPTIONS = list(COLOR_NAME_ZH_TO_EN.keys()) + ["自定义"]
MODE_OPTIONS = ["全流程", "仅抽帧", "仅推理", "仅合成"]
MODE_KEY_MAP = {"全流程": "full", "仅抽帧": "extract", "仅推理": "infer", "仅合成": "encode"}


def _app_version() -> str:
    """从 pyproject.toml 读取 version 字段，渲染成 "vX.Y.Z"。

    解析失败时回退到 "v?.?.?"，避免 UI caption 报错。
    """
    import re
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        m = re.search(r'^version\s*=\s*"([^"]+)"',
                      pyproject.read_text(encoding="utf-8"), re.M)
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
    on_reset=None,
    **kwargs,
):
    """file_uploader 的可重置封装：上传后旁置一个「🔄 更换」按钮。

    为什么要包一层：Streamlit 的 `st.file_uploader` 在某些场景（用户取消上传、
    重复点开新选择框、网络抖动）会卡在 "uploading..." 客户端状态且没有重置入口。
    这里通过在 widget key 中嵌入一个自增版本号，按下「🔄 更换」时把版本号 +1，
    触发 Streamlit 重建一个新实例，相当于「换一个空 file_uploader」，给用户一个
    明确的恢复出口。

    on_reset: 触发重置时的回调（可选），用于清掉与该 uploader 联动的下游状态
              （例如：更换模型时清空「标注类别」缓存）。
    container: 渲染目标容器，传 `st.sidebar` 可在侧栏内排版。
    返回值与 `st.file_uploader` 行为一致：单文件返回 UploadedFile 或 None，
    多文件返回 list[UploadedFile]（即使空也是 []）。
    """
    c = container if container is not None else st
    version = st.session_state.get(f"_uploader_ver_{base_key}", 0)
    widget_key = f"{base_key}_v{version}"
    uploaded = c.file_uploader(
        label,
        type=type,
        key=widget_key,
        accept_multiple_files=accept_multiple,
        help=help,
        **kwargs,
    )
    # 已上传：旁置「🔄 更换」按钮（无 ✅ 状态行，避免布局杂乱）
    if uploaded:
        if c.button("🔄 更换", key=f"reset_{widget_key}",
                    use_container_width=True,
                    help="清空当前选择，可重新选择文件"):
            st.session_state[f"_uploader_ver_{base_key}"] = version + 1
            if on_reset is not None:
                on_reset()
            st.rerun()
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
    ss.setdefault("class_names", {})        # {int(cid): str(name)}
    ss.setdefault("class_names_key", None)  # 缓存键：(str(path), mtime)


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

    在「更换/删除模型」或「清空缓存」时调用，确保 Step 2 的「标注类别」列表
    不会保留上一个模型的残留（class_names 缓存 + 每个 lbl_<cid> widget）。
    """
    ss = st.session_state
    ss["model_path"] = None
    ss["class_names"] = {}
    ss["class_names_key"] = None
    # 清掉 Step 2 的类别输入框 widget state（key 是 lbl_<cid>）
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith("lbl_")]:
        del ss[k]


# v1.0.6: 「🧹 清空缓存」白名单。按白名单清 session_state，避免误删 Streamlit
# 自身内部键（如 _streamlit_*、表单内部状态等）导致下次 rerun 报奇异错误。
# 显式列出所有本应用管理的 key + 通过前缀匹配捕获动态 key（uploader 版本号、
# 类别输入框 lbl_<cid>、下载缓存 _dl_cache_<stem>__<kind>）。
_KNOWN_KEYS = frozenset({
    # 主流程派生 state
    "results", "model_path", "cache_model", "device", "running", "last_zip",
    "class_names", "class_names_key",
    # 显式 widget keys（与渲染处 key=... 一一对应）
    "mode_radio", "fps_choice", "fps_custom",
    "interval_input", "conf_input", "iou_input",
    "color_zh", "custom_color", "unified_label",
    "cache_model_checkbox",  # 旧版本名保留兜底
    "start_btn",
    # 侧栏按钮 + 确认流
    "confirm_clear_files", "cf_yes", "cf_no",
    "btn_clear_files", "btn_clear_cache",
    # uploader 内部版本号（基础名，具体版本在 helper 内 _uploader_ver_<base>）
    "_uploader_ver_model_uploader",
    "_uploader_ver_videos_uploader",
    "_uploader_ver_infer_images",
    "_uploader_ver_encode_images",
    "_uploader_ver_encode_source_video",
    # toast 队列（清空后下次可再次设置）
    "_toast_msg",
    # 兜底：所有 _dl_cache_*
})


def _clear_all_state() -> None:
    """按白名单清 session_state（不动 Streamlit 内部键），让 UI 回到初始默认。

    v1.0.6: 「🧹 清空缓存」使用此函数；磁盘文件**不动**（uploads/、outputs/、
    outputs/_models/ 都不删），由「🧹 清空本地文件」按钮负责。
    """
    ss = st.session_state
    for k in list(ss.keys()):
        # 匹配三种动态前缀：lbl_<cid>、_uploader_ver_<base>、_dl_cache_*
        if isinstance(k, str) and (
            k.startswith("lbl_")
            or k.startswith("_uploader_ver_")
            or k.startswith("_dl_cache_")
        ):
            del ss[k]
            continue
        if k in _KNOWN_KEYS:
            del ss[k]


# ---------- 侧栏 ----------

def _device_options() -> list[str]:
    try:
        import torch
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        cuda_count = 0
    options = ["auto", "cpu"]
    options.extend(f"cuda:{i}" for i in range(cuda_count))
    if cuda_count:
        options.insert(2, "cuda")
    return options


def _sidebar() -> None:
    st.sidebar.header("⚙️ 全局设置")
    uploaded_model = _file_uploader_with_reset(
        "选择模型 (.pt)", type=["pt"], base_key="model_uploader",
        help="单选一个 YOLO 权重文件",
        container=st.sidebar,
        on_reset=lambda: _reset_model_state(),
    )
    if uploaded_model is not None:
        data = uploaded_model.read()
        if st.session_state.get("cache_model"):
            new_path = cache_uploaded_model(data, uploaded_model.name)
            cached_msg = f"已缓存到 {new_path}"
        else:
            tmp = OUTPUTS_DIR / "_models" / safe_stem(
                Path(uploaded_model.name).stem) / uploaded_model.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            new_path = tmp
            cached_msg = "未跨会话缓存"
        # 模型换文件时清空「标注类别」缓存 + 旧的 model_path
        prev_path = st.session_state.get("model_path")
        if prev_path != new_path:
            _reset_model_state()
        st.session_state["model_path"] = new_path
        st.sidebar.caption(f"模型: {uploaded_model.name}（{cached_msg}）")

    st.session_state["cache_model"] = st.sidebar.checkbox(
        "跨会话缓存上传的模型", value=st.session_state.get("cache_model", False),
        key="cache_model",
        help="勾选后会把模型按 SHA1 存到 uploads/models/，下次会话可复用",
    )
    st.session_state["device"] = st.sidebar.selectbox(
        "推理设备", _device_options(), index=0, key="device")

    st.sidebar.divider()

    # ---- 清空本地文件（两次确认：先点按钮，再点「确认删除」）----
    if st.sidebar.button("🧹 清空本地文件",
                         key="btn_clear_files",
                         help="删除 outputs/ 下全部生成文件（保留 _models/ 上传的模型）"):
        st.session_state["confirm_clear_files"] = True
    if st.session_state.get("confirm_clear_files"):
        st.sidebar.warning("确认要删除全部的生成文件吗？")
        col_yes, col_no = st.sidebar.columns(2)
        with col_yes:
            if st.button("确认删除", key="cf_yes", type="primary",
                         use_container_width=True):
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

    # ---- 清空缓存：仅重置页面 UI + session_state；不删任何本地文件 ----
    # v1.0.6: 用户明确「只清空页面控件，恢复初始状态」；磁盘清理由上方「清空本地文件」按钮负责
    if st.sidebar.button("🧹 清空缓存",
                         key="btn_clear_cache",
                         help="仅重置页面 UI 控件与会话状态；不删除任何本地文件"):
        _clear_all_state()
        st.session_state["_toast_msg"] = ("已重置页面控件", "🧹")
        st.rerun()


# ---------- 三步的 step 渲染器 ----------

def _step_extract(mode_key: str) -> dict:
    """Step1：抽帧。mode_key != 'full' 且 mode_key != 'extract' 时返回空 dict。"""
    if mode_key not in ("full", "extract"):
        return {}
    with st.expander("🎞 Step 1 · 视频抽帧", expanded=True):
        videos = _file_uploader_with_reset(
            "上传视频（可多选）", type=["mp4", "mov", "avi", "mkv"],
            base_key="videos_uploader", accept_multiple=True,
            help="可一次选多个；上传卡住时点右侧「🔄 更换」",
        )
        raw_interval = st.text_input(
            "抽帧间隔（每隔多少帧抽 1 帧，正整数）", value="1",
            key="interval_input")
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
            raw_conf = st.text_input("置信度阈值 (0~1)", value="0.25",
                                     key="conf_input")
        with col2:
            raw_iou = st.text_input("NMS IoU 阈值 (0~1)", value="0.45",
                                    key="iou_input")
        conf, c_fell = parse_positive_float_text(raw_conf, default=0.25)
        if c_fell and raw_conf.strip():
            st.warning(f"置信度「{raw_conf}」无效，已回退为 0.25")
        iou, i_fell = parse_positive_float_text(raw_iou, default=0.45)
        if i_fell and raw_iou.strip():
            st.warning(f"IoU「{raw_iou}」无效，已回退为 0.45")

        color_zh = st.selectbox("标注框颜色", COLOR_ZH_OPTIONS, index=0,
                         key="color_zh")
        if color_zh == "自定义":
            hex_color = st.color_picker("自定义颜色", "#FF0000",
                                        key="custom_color")
            box_color = parse_color(hex_color, default=DEFAULT_BOX_COLOR)
        else:
            box_color = parse_color(
                COLOR_NAME_ZH_TO_EN[color_zh], default=DEFAULT_BOX_COLOR)
        st.caption(f"当前 BGR: {box_color}")

        # ---- 模型类别动态列表 + 统一标注名称 ----
        model_path_str = st.session_state.get("model_path")
        model_path = Path(model_path_str) if model_path_str else None
        class_names = (_get_cached_class_names(model_path)
                       if model_path and model_path.exists() else {})

        unified = st.text_input(
            "统一标注名称（留空则按下方每类自定义）",
            placeholder="例如: 目标 / Object",
            key="unified_label",
            help="非空时覆盖下方所有类别输入",
        )

        label_rows: list[tuple[int, str, str]] = []
        if class_names:
            sorted_items = sorted(class_names.items())
            n = len(sorted_items)
            cols_per_row = 2 if n <= 6 else 3
            # 把每类的 text_input 按 cols_per_row 一行布局
            for start in range(0, n, cols_per_row):
                row_items = sorted_items[start:start + cols_per_row]
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

        label_map = label_table_to_dict(label_rows, unified) if label_rows else None

        return {"conf": conf, "iou": iou, "box_color": box_color,
                "label_map": label_map}


def _step_encode(mode_key: str) -> dict:
    """Step3：合成。mode_key != 'full' 且 mode_key != 'encode' 时返回空 dict。"""
    if mode_key not in ("full", "encode"):
        return {}
    with st.expander("🎬 Step 3 · 合成视频", expanded=True):
        fps_choice = st.selectbox(
            "帧率", ["原视频帧率", "自定义"], index=0,
            key="fps_choice")
        fps = None
        if fps_choice == "自定义":
            fps = st.number_input("自定义帧率", 1, 120, 30,
                                  key="fps_custom")
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
        base_key="infer_images", accept_multiple=True,
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
        base_key="encode_images", accept_multiple=True,
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
            help="上传后系统读取原始帧率；不传则按 30 fps 回退；上传卡住可点右侧「🔄 更换」",
        )
    return {"images": images or [], "source_video": source_video,
            "fps_choice": fps_choice}


# ---------- 流水线执行 ----------

def _progress_callback_for(per_video_blocks: dict):
    """闭包：把 PipelineEvent 路由到对应的 per-video 进度块。"""
    def on_progress(ev) -> None:
        blk = per_video_blocks.get(ev.video_stem)
        if blk is None:
            return
        total = max(ev.total, 1)
        if ev.stage == "extract":
            ratio = 0.05 + 0.25 * (ev.current / total)
        elif ev.stage == "infer":
            ratio = 0.35 + 0.35 * (ev.current / total)
        elif ev.stage == "encode":
            ratio = 0.75 + 0.20 * (ev.current / total)
        elif ev.stage == "done":
            ratio = 1.0
        elif ev.stage == "error":
            ratio = 1.0
        elif ev.stage == "start":
            ratio = 0.02
        else:
            ratio = 0.0
        blk["bar"].progress(min(max(ratio, 0.0), 1.0),
                            text=f"[{ev.stage}] {ev.message}")
        blk["log"].text(ev.message)
        if ev.stage == "done":
            blk["box"].update(label=f"{ev.video_stem} ✅ 完成", state="complete")
        elif ev.stage == "error":
            blk["box"].update(label=f"{ev.video_stem} ❌ 失败", state="error")
    return on_progress


def _run_pipeline_ui(mode_key: str, steps: dict,
                      infer_extras: dict, encode_extras: dict) -> None:
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

    # ---- 进度容器：每个 per-job 用独立 container(border=True) 避免塌缩 ----
    # 推断展示用的 stems（按 mode 不同，stems 来源不同）
    job_stems: dict[str, dict] = {}  # stem -> 该 job 的额外元数据
    if mode_key in ("full", "extract"):
        # uploads/<stem>/video.mp4 的父目录名才是唯一的 stem（p.stem 永远是 "video"）
        for p in video_paths:
            stem = safe_stem(p.parent.name)
            job_stems[stem] = {"raw_video": p}
    elif mode_key == "infer":
        images = infer_extras.get("images", [])
        if not images:
            st.warning("请先上传至少一张图片")
            return
        stem = _timestamp_id("infer")
        # 先把图片落盘到 outputs/<stem>/_uploaded/，作为 frames_dir
        upload_dir = save_uploaded_images(images, stem, outputs_root=OUTPUTS_DIR)
        job_stems[stem] = {"frames_dir": upload_dir}
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
            tmp_sv = OUTPUTS_DIR / "_src" / safe_stem(
                sv_upload.name) / "source.mp4"
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
        job_stems[stem] = {"annotated_dir": upload_dir, "fps": encode_fps}
    else:
        st.error(f"未知运行模式: {mode_key}")
        return

    display_stems = list(job_stems.keys())
    per_video_blocks: dict[str, dict] = {}
    for stem in display_stems:
        # 外层 border container 是 bug #1 的关键修复
        with st.container(border=True):
            with st.status(f"处理 {stem}", expanded=True) as box:
                bar = st.progress(0.0, text="等待中")
                log = st.empty()
                per_video_blocks[stem] = {"box": box, "bar": bar, "log": log}

    on_progress = _progress_callback_for(per_video_blocks)

    # ---- 构造 run_pipeline 参数 ----
    common = dict(
        model_path=model_path,
        frame_interval=steps.get("interval", 1),
        conf=steps.get("conf", 0.25),
        iou=steps.get("iou", 0.45),
        device=device,
        box_color=steps.get("box_color", DEFAULT_BOX_COLOR),
        label_map=steps.get("label_map"),
        fps=steps.get("fps"),
        progress_cb=on_progress,
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
        kwargs = {**common, "mode": "infer", "video_paths": [],
                  "frames_dir": frames_dir}
    elif mode_key == "encode":
        stem = display_stems[0]
        annotated_dir = job_stems[stem]["annotated_dir"]
        encode_fps = job_stems[stem]["fps"]
        kwargs = {**common, "mode": "encode", "video_paths": [],
                  "annotated_dir": annotated_dir,
                  "fps": encode_fps}
    else:
        st.error(f"未知运行模式: {mode_key}")
        return

    st.session_state["running"] = True
    try:
        results = run_pipeline(**kwargs)
        for r in results:
            st.session_state["results"][r.stem] = r
        st.toast("处理结束", icon="✅")
    except Exception as exc:
        st.error(f"流水线异常: {exc}")
    finally:
        st.session_state["running"] = False


# ---------- 结果区 ----------

def _results_panel() -> None:
    header_cols = st.columns([5, 1])
    with header_cols[0]:
        st.subheader("📥 处理结果")
    with header_cols[1]:
        if st.button("清空结果", key="clear_results",
                     use_container_width=True,
                     help="仅清空当前会话展示的结果（磁盘文件保留）"):
            st.session_state["results"] = {}
            st.rerun()

    results: dict[str, VideoResult] = st.session_state.get("results", {})
    if not results:
        st.caption("尚无结果。先上传参数并点击「开始处理」。")
        return

    # v1.0.5: 仅全流程模式展示「📦 下载全部 (ZIP)」；单 stage 模式各自单 stem 下载即可
    show_zip = any(_is_full_result(r) for r in results.values())

    for stem in sorted(results.keys()):
        r = results[stem]
        with st.container(border=True):
            st.markdown(f"**{stem}**")
            if r.error:
                st.error(f"失败: {r.error}")
                continue

            # v1.0.3: 全流程模式只展示最终视频；不展示中间抽帧 / 标注目录
            is_full = _is_full_result(r)

            # ---- 视频产物（full / encode）----
            if r.output_video and r.output_video.exists():
                # v1.0.6: 用 cache_data 缓存按 (path, mtime_ns) 读盘，
                # rerun 不再触发大文件 IO 与 download_button 置灰
                mp4_path = r.output_video
                data = read_file_bytes_cached(
                    str(mp4_path), mp4_path.stat().st_mtime_ns)
                col_v, col_btn = st.columns([3, 1])
                col_v.markdown(
                    f"输出: `{mp4_path}`  ({len(data)/1024/1024:.2f} MB)")
                col_btn.download_button(
                    "下载视频", data=data,
                    file_name=mp4_path.name, mime="video/mp4",
                    key=f"dl_video_{stem}",
                    use_container_width=True,
                )

            # ---- 抽帧产物（仅 extract）----
            if not is_full and r.frames_dir and r.frames_dir.exists():
                n_frames = sum(1 for _ in r.frames_dir.glob("*.jpg"))
                col_f, col_btn = st.columns([3, 1])
                col_f.markdown(
                    f"抽帧目录: `{r.frames_dir}`（{n_frames} 张）")
                col_btn.download_button(
                    "下载抽帧 ZIP",
                    data=build_frames_zip(r.frames_dir),
                    file_name=f"{stem}_frames.zip",
                    mime="application/zip",
                    key=f"dl_frames_{stem}",
                    use_container_width=True,
                )

            # ---- 推理产物（仅 infer）----
            if not is_full and r.annotated_dir and r.annotated_dir.exists():
                n_ann = sum(1 for _ in r.annotated_dir.glob("*.jpg"))
                col_a, col_btn = st.columns([3, 1])
                col_a.markdown(
                    f"标注目录: `{r.annotated_dir}`（{n_ann} 张）")
                col_btn.download_button(
                    "下载推理 ZIP",
                    data=build_infer_zip(r.annotated_dir),
                    file_name=f"{stem}_infer.zip",
                    mime="application/zip",
                    key=f"dl_infer_{stem}",
                    use_container_width=True,
                )

    if not show_zip:
        return
    st.divider()
    # v1.0.2: 只打包当前会话在 st.session_state["results"] 里的 stem
    zip_bytes = build_session_zip(results)
    st.session_state["last_zip"] = zip_bytes
    st.download_button(
        "📦 下载全部 (ZIP)",
        data=zip_bytes,
        file_name="cv_session.zip",
        mime="application/zip",
        disabled=not zip_bytes,
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

    # v1.0.6: 消费「清空缓存」按钮的 toast（必须在 rerun 后第一时间消费，否则错过显示时机）
    if "_toast_msg" in st.session_state:
        msg, icon = st.session_state.pop("_toast_msg")
        st.toast(msg, icon=icon)

    st.title("🧰 CV工具箱")
    st.caption(f"{_app_version()} · 视频抽帧 / 模型推理 / 视频合成")

    mode_zh = st.radio(
        "运行模式", MODE_OPTIONS, horizontal=True, index=0,
        key="mode_radio",
        help="全流程: 抽帧→推理→合成；或单独跑其中一个步骤。",
    )
    mode_key = MODE_KEY_MAP[mode_zh]

    s1 = _step_extract(mode_key)
    s2 = _step_infer(mode_key)
    s3 = _step_encode(mode_key)

    # 单 stage 模式补的额外输入（encode 需要读取 s3 的 fps_choice）
    infer_extras = _collect_infer_extras(mode_key)
    encode_extras = _collect_encode_extras(mode_key, s3)

    st.divider()
    run_clicked = st.button(
        "▶ 开始处理", type="primary", use_container_width=True,
        key="start_btn",
        disabled=st.session_state.get("running", False),
    )
    if run_clicked:
        # 外层 spinner 给用户一个明确的「处理中」反馈；具体的 per-video 进度在内部
        with st.spinner("🚀 处理中，请稍候（具体进度见下方每个任务的卡片）..."):
            _run_pipeline_ui(mode_key,
                             {**s1, **s2, **s3},
                             infer_extras, encode_extras)

    st.divider()
    _results_panel()


if __name__ == "__main__":
    main()