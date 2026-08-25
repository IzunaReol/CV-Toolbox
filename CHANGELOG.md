# 更新日志 / Changelog

本项目的版本演进记录。约定：每个版本列出**新增 (Added)**、**修复 (Fixed)**、**变更 (Changed)** 三类。
本项目的版本演进记录。Each release lists **Added / Fixed / Changed** sections.

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。


---

## [v1.0.4] - 2026-08-25

### 新增 (Added)
- 每个 `file_uploader`（侧栏模型、Step 1 视频、仅推理图片、仅合成图片/源视频）旁新增「🔄 更换」按钮：用户上传卡住时点一下即可强制重置该上传控件，无需刷新整页。
- 上传完成后在上传框下方显示「✅ 已选择：`<filename>`（`N.N` MB）」状态，明确反馈"上传是否完成 / 文件多大"。
- `st.spinner("🚀 处理中...")` 包裹整个流水线执行，给用户更明确的"在跑"信号（具体 per-video 进度仍由内部 `st.status` 块展示）。
- README 新增「❓ 常见问题」章节，解释 `starlette.requests.ClientDisconnect` 是 Streamlit/uvicorn 服务端的**无害日志噪音**（用户取消上传时打印），不是程序崩溃。

### 变更 (Changed)
- 启动命令 `--server.maxUploadSize` 默认建议由 2048 改为 1024（普通 1 GB 视频/模型足够，更大文件上传更慢、更容易触发 ClientDisconnect 噪音）。

---

## [v1.0.3] - 2026-07-30

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
