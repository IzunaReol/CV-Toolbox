# 更新日志 / Changelog

本项目的版本演进记录。约定：每个版本列出**新增 (Added)**、**修复 (Fixed)**、**变更 (Changed)** 三类。
本项目的版本演进记录。Each release lists **Added / Fixed / Changed** sections.

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。


---

## [v1.0.6] - 2026-08-25

### 修复 (Fixed)
- 「🧹 清空缓存」按钮**行为变更**：从"清 `session_state` + 删 `uploads/` 与 `outputs/_models/`"改为**只重置页面 UI 控件 + session_state，不删任何本地文件**（磁盘文件删除仍由 v1.0.3 引入的「🧹 清空本地文件」两步按钮负责）。
- 仅合成模式的源视频上传控件逻辑反转：「原视频帧率」时**显示**源视频上传（用于读 fps）；「自定义 fps」时**隐藏**（直接用 `number_input` 输入帧率即可）。
- 处理结果区点击「下载视频」按钮后短暂置灰 → 几秒后消失 → 又出现一个相同按钮的现象：根因是每次 rerun 都重新读大文件 + 重新分配 `download_button` 的 download token。修复：`build_frames_zip / build_infer_zip / build_session_zip` 加 `@st.cache_data(show_spinner=False)`；新增 `read_file_bytes_cached(path, mtime_ns)` 缓存视频文件读取，按 `(path, mtime_ns)` 命中缓存，rerun 期间不再触发大文件 IO。
- 上传/下载期间终端偶发的 `_ProactorBasePipeTransport._call_connection_lost(...) ConnectionResetError: [WinError 10054]` 报错（Windows asyncio ProactorEventLoop 已知问题）频率显著降低：根因是每次 rerun 重复大文件 IO + 客户端断连同步发生。**v1.0.6 起**通过把结果区下载按钮的 IO 全部走 `@st.cache_data` 缓存，rerun 不再重读。

### 新增 (Added)
- 所有 widget 加显式 `key`（`mode_radio` / `fps_choice` / `fps_custom` / `interval_input` / `conf_input` / `iou_input` / `color_zh` / `custom_color` / `unified_label` / `cache_model` / `device` / `start_btn` / `btn_clear_files` / `btn_clear_cache`），确保 `del st.session_state[k]` 能精确清掉对应 widget state，下次 rerun 回到默认值。
- 新增 `_KNOWN_KEYS` 白名单 + `_clear_all_state()` 工具：按白名单清 session_state（不动 Streamlit 内部键如 `_streamlit_*`），避免 `for k in list(ss.keys()): del ss[k]` 这种激进方式删除后导致的奇异错误。
- 侧栏清空缓存按钮触发后，通过 `st.session_state["_toast_msg"]` 延迟到 `main()` 顶部消费 toast，保证「已重置页面控件」提示不被错过。

### v1.0.6 hotfix

#### 修复 (Fixed)
- **删除「🔄 更换」按钮**。该按钮位于上传框下方，但仅在 `if uploaded:` 时才渲染——上传卡在 "uploading..." 时按钮根本不存在，毫无用处。
- **新增全局 `_reset_token`**：所有 `file_uploader` 的 `widget_key` 嵌入 `_reset_token`，使「🧹 清空缓存」按钮能真正强制所有上传控件**重建为空**（Streamlit 把新 key 视为新 widget，不再复用旧上传状态）。即使上次上传卡在 99%，点重置后再进入也像第一次打开页面。

#### 变更 (Changed)
- 「🧹 清空缓存」按钮**更名为「🧹 重置页面」**（更准确表达"清掉上传状态 + UI 回到初始"）。
- `_file_uploader_with_reset` 的 `on_reset` 回调参数已移除（不再需要 widget-level reset hook）；模型的轻量 reset 由 sidebar 内 `prev_path != new_path` 检测 + `_reset_model_state()` 继续承担。

### v1.0.6 hotfix#2（用户报告"重置页面二次无效 + 仅合成上传图片卡死"后）

#### 修复 (Fixed)
- 「🧹 重置页面」按钮**第二次点击无效**：根因是按钮 widget 在固定 key 下重复点击不会重新触发回调。改为 `on_click` 回调 + `_reload_after_rerun` flag 模式，flag 在 `main()` 顶部消费并注入 `<meta http-equiv="refresh" content="0">`，触发浏览器硬刷新（等价于地址栏回车），彻底丢掉浏览器缓存的 widget state 与 in-flight 上传请求。
- **仅合成上传图片后页面卡死**（上传框一直加载、下载按钮不渲染、刷新无效，直到手动停止浏览器加载后积压请求才瞬间吐出）：根因是 `_sidebar` 每次 rerun 都无条件 `read()` + `write_bytes()` 重写整个模型文件（50~130MB），阻塞主线程。加 size 守卫，只有文件缺失或大小变化才读+写，命中时连 `read()` 都不做，rerun 变为纯渲染。

---

## [v1.0.5] - 2026-07-20

### 变更 (Changed)
- 去掉 `file_uploader` 上传后的「✅ 已选择：`<filename>`（`N.N` MB）」信息行（之前在 v1.0.4 新增，与 `🔄 更换` 按钮一起显得布局杂乱、多此一举）。仅保留「🔄 更换」按钮作为唯一反馈。
- 结果区「📦 下载全部 (ZIP)」按钮在仅抽帧 / 仅推理 / 仅合成三种单 stage 模式下**不再展示**。全流程模式仍展示该按钮（多视频下载场景需要）。
- 侧栏「🗑 清空会话」按钮更名为「🧹 清空缓存」（行为不变：清空 `session_state` + 删除 `uploads/` 与 `outputs/_models/` 下的上传文件）。

### 新增 (Added)
- 更换或删除侧栏模型时，Step 2 的「标注类别」列表（`class_names` 缓存 + 每个 `lbl_<cid>` widget 状态）会被自动清空，避免上一个模型的残留类别显示给当前模型。新增 `_reset_model_state()` 工具。
- `file_uploader` 封装新增 `on_reset` 回调钩子（`on_reset=lambda: _reset_model_state()`），供各调用点接入「重置时联动清空下游状态」逻辑。

---

## [v1.0.4] - 2026-07-20

### 新增 (Added)
- 每个 `file_uploader`（侧栏模型、Step 1 视频、仅推理图片、仅合成图片/源视频）旁新增「🔄 更换」按钮：用户上传卡住时点一下即可强制重置该上传控件，无需刷新整页。
- 上传完成后在上传框下方显示「✅ 已选择：`<filename>`（`N.N` MB）」状态，明确反馈"上传是否完成 / 文件多大"。
- `st.spinner("🚀 处理中...")` 包裹整个流水线执行，给用户更明确的"在跑"信号（具体 per-video 进度仍由内部 `st.status` 块展示）。
- README 新增「❓ 常见问题」章节，解释 `starlette.requests.ClientDisconnect` 是 Streamlit/uvicorn 服务端的**无害日志噪音**（用户取消上传时打印），不是程序崩溃。

### 变更 (Changed)
- 启动命令 `--server.maxUploadSize` 默认建议由 2048 改为 1024（普通 1 GB 视频/模型足够，更大文件上传更慢、更容易触发 ClientDisconnect 噪音）。

---

## [v1.0.3] - 2026-06-11

### 新增 (Added)
- 侧栏新增「🧹 清空本地文件」按钮：两次确认（先点按钮弹"确认要删除全部的生成文件吗？"，再点「确认删除」），实际删除 `outputs/` 下全部生成物（保留 `outputs/_models/` 用户上传的模型）。
- 标题副标题动态从 `pyproject.toml` 读取 version，避免发版后忘记同步。

### 变更 (Changed)
- 「清空会话」按钮：除清空 `session_state` 外，**也**删除 `uploads/` 下的已上传视频与缓存模型，以及 `outputs/_models/` 下的非缓存模型；点击后 `st.rerun()` 刷新右侧页面。
- 全流程模式的结果区**只**展示最终合成的视频；不再展示「抽帧目录 / 下载抽帧 ZIP」与「标注目录 / 下载推理 ZIP」。
- 「📦 下载全部 (ZIP)」在全流程模式下**只**打包最终视频，与 UI 展示对齐（之前会同时打包中间帧/标注目录）。
- 仅合成模式：帧率 =「原视频帧率」时**隐藏**源视频上传控件（之前在该选项下源视频为必填，UI 会报错；现在直接回退默认 30 fps）。仅帧率 =「自定义」时显示源视频上传，作为可选。

---

## [v1.0.2] - 2026-05-14

### 修复 (Fixed)
- 全流程的「下载全部 (ZIP)」会打包 `outputs/` 下历史轮次的视频。改为只打包当前会话在 `st.session_state["results"]` 里的 stem。

### 新增 (Added)
- 仅推理 / 仅合成 重构为独立功能：
  - 仅推理：上传一张或多张图片 → 自动写入 `outputs/infer_<时间戳>/_uploaded/` → 调 YOLO 推理 → 标注目录下载。
  - 仅合成：上传多张图片 + 可选源视频 → 按文件名顺序合成 → mp4 下载。
  - 仅合成在帧率选「原视频帧率」时，源视频变**必填**（UI 红字提示）。
- 处理结果区新增「清空结果」按钮，仅清空内存展示，磁盘文件保留。
- 仅抽帧 / 仅推理 结果区增加每 stem 独立的「下载抽帧 ZIP」/「下载推理 ZIP」按钮。

### 变更 (Changed)
- 仅推理 / 仅合成的 stem 一律改用时间戳自动生成（`infer_20260825_133800` / `compose_20260825_133800`），不再要求用户额外命名。
- 三个独立模式不再依赖「先抽帧才能推理」的串联输入，每个模式都是独立的端到端功能。

---

## [v1.0.1] - 2026-05-12

### 修复 (Fixed)
- **多视频上传只处理一个**：`save_uploaded_video` 把视频存为 `uploads/<stem>/video.mp4`，但 pipeline 用 `safe_stem(p.stem)` 推 stem 时拿到的是字面值 `"video"`（不是 `<stem>`），导致三个上传的 stem 全部相同 → 进度块互相覆盖、结果只剩一份。改为按 `p.parent.name` 推 stem。

### 新增 (Added)
- 模式选择器：全流程 / 仅抽帧 / 仅推理 / 仅合成 四选一（`st.radio`，水平排列）。
- Step 1 抽帧间隔、Step 2 置信度 / NMS IoU 改用文本输入框，非法值回退默认值并 warning。
- Step 2 标注框颜色下拉框中文化（红色 / 绿色 / 蓝色 ...），自定义颜色仍可调色板选取。
- Step 2 加载模型后，按模型内置类别动态渲染每类的「标注名称」输入框（默认 = 模型原名），顶部一个「统一标注名称」输入框非空时覆盖所有。
- Step 3 帧率下拉框：「原视频帧率」/「自定义」二选一。
- 多视频上传时，每个 per-video 状态块独立显示（外层包 `st.container(border=True)`，避免 Streamlit 状态块塌缩）。

### 变更 (Changed)
- 页面标题由「模型效果展示 — 视频抽帧 / 推理 / 合成」改为「🧰 CV工具箱」。
- 上传视频内部文件名 `v.mp4` → `video.mp4`（避免与字母 v 混淆）。

---

## [v1.0.0] - 2026-05-11

### 新增 (Added)
- **初始版本**：覆盖流程描述中的全部核心功能。
- CLI 三段式脚本（v1.0.3 后已重命名为 `scripts/` 与 ASCII 文件名）：
  - `脚本/1.视频抽帧.py` — 视频按帧间隔抽帧为 jpg → 现 `scripts/1_frame_extract.py`
  - `脚本/2.模型推理.py` — YOLO 推理 + 绘制中文标注框与 label → 现 `scripts/2_model_infer.py`
  - `脚本/3.图片转视频.py` — 按文件名顺序合成 mp4 → 现 `scripts/3_images_to_video.py`
- Streamlit WEB UI（`web/app.py`）：
  - 上传一个或多个视频文件
  - 选择本地 `.pt` 模型（仅允许单选）
  - 分别设置抽帧间隔 / 置信度 / IoU / 框颜色 / 标注名称 / 帧率
  - 点击「开始处理」后显示进度条
  - 处理完成后可下载处理后的视频文件；多视频时支持「下载全部 (ZIP)」
- 支持中文路径、Chinese label 显示、自定义颜色与 label map。

---

## 版本约定 / Versioning

本项目遵循 [语义化版本 (Semantic Versioning)](https://semver.org/lang/zh-CN/)：

- **MAJOR** (x.0.0)：不兼容的架构变更
- **MINOR** (0.x.0)：向下兼容的功能新增
- **PATCH** (0.0.x)：向下兼容的 bug 修复
