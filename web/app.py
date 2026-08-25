"""Streamlit WEB UI（v1.0.1）：包装抽帧 → 推理 → 合成视频 三段式流水线。

启动方式（在项目根下）：
    streamlit run web/app.py --server.maxUploadSize 2048

v1.0.1 变更：
  0. 运行模式选择（全流程 / 仅抽帧 / 仅推理 / 仅合成）
  1. 修复多视频状态块塌缩（包 st.container(border=True)）
  3. uploads 内部文件名 v.mp4 → video.mp4
  4. 颜色下拉框中文化
  5. 抽帧间隔/置信度/IoU 改文本框
  6. 模型类别动态列表 + 统一标注名称
  7. 帧率下拉框
  8. 标题改为 "CV工具箱"
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
    uploaded_model = st.sidebar.file_uploader(
        "选择模型 (.pt)", type=["pt"], key="model_uploader",
        help="单选一个 YOLO 权重文件",
    )
    if uploaded_model is not None:
        data = uploaded_model.read()
        if st.session_state.get("cache_model"):
            st.session_state["model_path"] = cache_uploaded_model(
                data, uploaded_model.name)
            cached_msg = f"已缓存到 {st.session_state['model_path']}"
        else:
            tmp = OUTPUTS_DIR / "_models" / safe_stem(
                Path(uploaded_model.name).stem) / uploaded_model.name
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            st.session_state["model_path"] = tmp
            cached_msg = "未跨会话缓存"
        st.sidebar.caption(f"模型: {uploaded_model.name}（{cached_msg}）")

    st.session_state["cache_model"] = st.sidebar.checkbox(
        "跨会话缓存上传的模型", value=st.session_state["cache_model"],
        help="勾选后会把模型按 SHA1 存到 uploads/models/，下次会话可复用",
    )
    st.session_state["device"] = st.sidebar.selectbox(
        "推理设备", _device_options(), index=0)

    st.sidebar.divider()

    # ---- 清空本地文件（两次确认：先点按钮，再点「确认删除」）----
    if st.sidebar.button("🧹 清空本地文件",
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

    # ---- 清空会话：session_state + 已上传文件（视频 + 非缓存模型）----
    if st.sidebar.button("🗑 清空会话",
                         help="重置所有 UI 控件 + 删除 uploads/ 与 outputs/_models/ 下的上传文件"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        # 删除已上传的视频（uploads/<stem>/）与已缓存的模型（uploads/models/）
        _delete_dir_contents(UPLOADS_DIR)
        # 删除未缓存的模型（OUTPUTS/_models/），它只是本次会话的临时副本
        _delete_dir_contents(OUTPUTS_DIR / "_models")
        st.rerun()


# ---------- 三步的 step 渲染器 ----------

def _step_extract(mode_key: str) -> dict:
    """Step1：抽帧。mode_key != 'full' 且 mode_key != 'extract' 时返回空 dict。"""
    if mode_key not in ("full", "extract"):
        return {}
    with st.expander("🎞 Step 1 · 视频抽帧", expanded=True):
        videos = st.file_uploader(
            "上传视频（可多选）", type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=True, key="videos_uploader",
        )
        raw_interval = st.text_input(
            "抽帧间隔（每隔多少帧抽 1 帧，正整数）", value="1")
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
            raw_conf = st.text_input("置信度阈值 (0~1)", value="0.25")
        with col2:
            raw_iou = st.text_input("NMS IoU 阈值 (0~1)", value="0.45")
        conf, c_fell = parse_positive_float_text(raw_conf, default=0.25)
        if c_fell and raw_conf.strip():
            st.warning(f"置信度「{raw_conf}」无效，已回退为 0.25")
        iou, i_fell = parse_positive_float_text(raw_iou, default=0.45)
        if i_fell and raw_iou.strip():
            st.warning(f"IoU「{raw_iou}」无效，已回退为 0.45")

        color_zh = st.selectbox("标注框颜色", COLOR_ZH_OPTIONS, index=0)
        if color_zh == "自定义":
            hex_color = st.color_picker("自定义颜色", "#FF0000")
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
            "帧率", ["原视频帧率", "自定义"], index=0)
        fps = None
        if fps_choice == "自定义":
            fps = st.number_input("自定义帧率", 1, 120, 30)
        return {"fps_choice": fps_choice, "fps": fps}


# ---------- 单 stage 模式额外输入 ----------

def _collect_extract_extras(mode_key: str) -> dict:
    """仅抽帧模式不需要额外输入。"""
    return {}


def _collect_infer_extras(mode_key: str) -> dict:
    """仅推理模式：上传一张或多张图片（按文件名排序后批量推理）。"""
    if mode_key != "infer":
        return {}
    images = st.file_uploader(
        "上传图片（可多选，按文件名排序后推理）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="infer_images",
        help="选择一张或多张图片文件，结果在 outputs/infer_<时间戳>/annotated/images/",
    )
    return {"images": images or []}


def _collect_encode_extras(mode_key: str, steps: dict) -> dict:
    """仅合成模式：上传多张图片 + 可选源视频（仅 fps=自定义 时显示）。"""
    if mode_key != "encode":
        return {}
    images = st.file_uploader(
        "上传图片（可多选，按文件名顺序合成）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="encode_images",
        help="按文件名排序后合成 mp4",
    )
    fps_choice = steps.get("fps_choice", "原视频帧率")
    source_video = None
    # 仅在 fps=自定义 时显示源视频上传控件；fps=原视频帧率 时直接回退默认 30 fps
    if fps_choice == "自定义":
        source_video = st.file_uploader(
            "源视频（仅用于读取原始帧率，可选）",
            type=["mp4", "mov", "avi", "mkv"],
            accept_multiple_files=False,
            key="encode_source_video",
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
                data = r.output_video.read_bytes()
                col_v, col_btn = st.columns([3, 1])
                col_v.markdown(
                    f"输出: `{r.output_video}`  ({len(data)/1024/1024:.2f} MB)")
                col_btn.download_button(
                    "下载视频", data=data,
                    file_name=r.output_video.name, mime="video/mp4",
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

    st.title("🧰 CV工具箱")
    st.caption(f"{_app_version()} · 视频抽帧 / 模型推理 / 视频合成")

    mode_zh = st.radio(
        "运行模式", MODE_OPTIONS, horizontal=True, index=0,
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
        disabled=st.session_state.get("running", False),
    )
    if run_clicked:
        _run_pipeline_ui(mode_key,
                         {**s1, **s2, **s3},
                         infer_extras, encode_extras)

    st.divider()
    _results_panel()


if __name__ == "__main__":
    main()