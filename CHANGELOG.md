# 更新日志 / Changelog

本项目的版本演进记录。约定：每个版本列出**新增 (Added)**、**修复 (Fixed)**、**变更 (Changed)** 三类。
本项目的版本演进记录。Each release lists **Added / Fixed / Changed** sections.

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

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
- CLI 三段式脚本：
  - `脚本/1.视频抽帧.py` — 视频按帧间隔抽帧为 jpg
  - `脚本/2.模型推理.py` — YOLO 推理 + 绘制中文标注框与 label
  - `脚本/3.图片转视频.py` — 按文件名顺序合成 mp4
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
