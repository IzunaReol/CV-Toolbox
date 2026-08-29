# 🧰 CV 工具箱 / CV Toolbox

> 计算机视觉模型效果展示项目 — **视频抽帧 → 指定类别推理 → 视频合成 → 工件管理** 一站式 CLI + Web UI。
> A CV model demo project with **frame extraction, class-filtered YOLO inference, video composition, and artifact management**.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%2B-00FFFF?logo=yolo)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v2.1.2-orange)](CHANGELOG.md)



https://github.com/user-attachments/assets/6073e336-b2b9-4734-bb0c-d556b88a1a94




---

## 📖 目录 / Table of Contents

- [中文介绍](#中文介绍)
  - [项目简介](#项目简介)
  - [功能特性](#功能特性)
  - [快速开始](#快速开始)
  - [项目结构](#项目结构)
  - [架构总览](#架构总览)
  - [使用示例](#使用示例)
- [English](#english)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Project Layout](#project-layout)
  - [Architecture](#architecture)
  - [Usage Examples](#usage-examples)
- [更多文档 / Further Reading](#更多文档--further-reading)
- [❓ 常见问题 / FAQ](#-常见问题--faq)
- [开源协议 / License](#开源协议--license)

---

## 中文介绍

### 项目简介

**CV 工具箱** 是一个用于演示计算机视觉模型推理效果的本地工具，覆盖视频处理全流程：

1. **抽帧**：从视频按设定间隔抽取关键帧。
2. **推理**：用 YOLO（Ultralytics）模型对每帧做目标检测，可选择一个或多个类别，并自定义标注框颜色与中文 label。
3. **合成**：把标注意图按原视频帧率合成为 MP4。
4. **管理**：在 Web UI 中浏览、预览、下载或删除 `outputs/` 下的任务工件。

提供两种使用方式：
- **CLI 脚本**（`scripts/`）：适合批处理、自动化、调试
- **WEB UI**（`web/app.py`）：基于 Streamlit，适合人工演示与参数调试

### 功能特性

| 模块 | 能力 |
|---|---|
| 🎞 **抽帧** | 单/多视频、按帧间隔、CJK 文件名安全 |
| 🤖 **推理** | YOLOv8+、按类别过滤、自定义颜色、中文 label、CJK 字体回退 |
| 🎬 **合成** | 按文件名排序合成、自动/自定义帧率、源视频帧率回退 |
| 🌐 **Web UI** | 上传/参数/进度/下载一条龙；支持全流程或单步运行 |
| 📥 **结果下载** | 每个任务独立下载；全流程多视频时提供当前批次 ZIP |
| 📁 **工件管理** | 浏览任务目录、分页预览图片、多选下载与二次确认删除 |
| 📊 **任务追踪** | 后台串行处理、刷新恢复、批次取消、统一任务信息与推理统计 |
| 🛠 **工程** | 中文路径、Unicode 文件名、Windows 兼容、模块化可复用 |

### 快速开始

#### 1. 环境准备

要求 **Python 3.10+**（`from __future__ import annotations` + 类型注解语法依赖）。

```bash
# 推荐使用虚拟环境
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS / Linux

# 安装依赖（仅 Web UI 需要 streamlit）
pip install -r web/requirements.txt
```

> 💡 若本地有 NVIDIA GPU 且已装 CUDA，可单独 `pip install torch --index-url https://download.pytorch.org/whl/cu121`，否则 `pip install torch` 会装 CPU 版。

#### 2. 启动 Web UI

```bash
streamlit run web/app.py --server.maxUploadSize 1024
```

浏览器打开 `http://localhost:8501`，按以下步骤使用：

1. **左侧栏** 上传 `.pt` 模型文件（可选「跨会话缓存」）。
2. **顶部 radio** 选择运行模式：`全流程 / 仅抽帧 / 仅推理 / 仅合成 / 文件浏览`。
3. **Step 1/2/3** 展开对应面板设置参数。
4. Step 2 可按模型类别多选，并可一键全部选择或全部取消。
5. 点击 **▶ 开始处理** 后任务进入后台；可查看进度、刷新页面继续跟踪或停止当前批次。
6. 处理完成后下载结果，或切换到 **文件浏览** 查看任务摘要、推理统计和历史工件。

#### 3. CLI 用法

见 [docs/WORKFLOW.md § 三段式 CLI 脚本](docs/WORKFLOW.md#三段式-cli-脚本--three-step-cli-scripts)。

### 项目结构

```
模型效果展示项目/
├── scripts/                    # CLI 三段式脚本
│   ├── 1_frame_extract.py
│   ├── 2_model_infer.py
│   └── 3_images_to_video.py
├── web/                        # Streamlit WEB UI
│   ├── app.py                  # UI 入口
│   ├── pipeline.py             # 编排层（加载脚本 + run_pipeline）
│   ├── helpers.py              # 工具函数（路径/ZIP/解析）
│   ├── artifact_browser.py     # 任务工件浏览、下载与删除
│   ├── job_manager.py          # 单线程后台任务、取消与状态恢复
│   ├── task_store.py           # 任务配置、统计与工件修订号
│   └── requirements.txt
├── docs/
│   └── WORKFLOW.md             # 工作流详细说明
├── uploads/                    # 运行时上传（gitignored）
├── outputs/                    # 运行时产物（gitignored）
├── CHANGELOG.md                # 版本历史
├── LICENSE                     # MIT
├── README.md                   # 本文件
└── .gitignore
```

### 架构总览

```
┌────────────────────────────────────────────────────────────┐
│                  Streamlit Web UI (app.py)                 │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │ Step 1 抽帧  │  │ Step 2 推理   │  │ Step 3 合成  │    │
│   └─────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         └──────────────────┴─────────────────┘            │
│                            │                              │
│                     pipeline.py                            │
│                  (run_pipeline mode=                       │
│                   full/extract/infer/encode)               │
│                            │                              │
│      ┌─────────────────────┼─────────────────────┐        │
│      ▼                     ▼                     ▼        │
│ scripts/1_frame_extract.py  scripts/2_model_infer.py  scripts/3_images_to_video.py
│  extract_frames()       infer()               create_video_from_images()
└────────────────────────────────────────────────────────────┘
```

Web UI 通过 `importlib.util.spec_from_file_location` 直接加载 `scripts/` 下的三个 CLI 脚本，调用其**纯函数**（不触发脚本的 `input()` 交互逻辑）。

「文件浏览」模式使用带修订号的缓存快照，只访问 `outputs/` 下的任务工件；模型、源视频、任务元数据和下载缓存等内部内容不会出现在文件列表中。大图支持导航、选择和单独下载，损坏图片不会阻塞其他工件。

### 使用示例

**Web UI 全流程**：
1. 上传 1~N 个视频（mp4/mov/avi/mkv）
2. 上传 `yolov8n.pt`
3. Step 1: 抽帧间隔 = 1（默认每帧抽）
4. Step 2: 红色框 + 类别 0 自定义为 "人"
5. Step 3: 帧率 = 原视频帧率（默认）
6. 开始处理 → 进度条 → 下载 ZIP

**CLI 链式调用**：
```bash
python scripts/1_frame_extract.py --video <path/to/your/video.mp4> --output outputs/demo/frames --interval 1
python scripts/2_model_infer.py   --model <path/to/your/model.pt>  --input outputs/demo/frames --output outputs/demo/annotated --color red --label-map "0:人"
python scripts/2_model_infer.py   --model <path/to/your/model.pt>  --input outputs/demo/frames --output outputs/demo/annotated --classes "person,car"
python scripts/3_images_to_video.py --input outputs/demo/annotated/images --output outputs/demo/demo.mp4 --fps 24
```

---

## English

### Introduction

**CV Toolbox** is a local toolkit for demonstrating computer-vision model inference end-to-end:

1. **Extract** frames from videos at a configurable interval.
2. **Infer** selected YOLO classes per frame, with customizable box color and **Chinese** labels.
3. **Compose** annotated frames back into an MP4 at the original video's frame rate.
4. **Manage** task artifacts with image previews, multi-file download, and confirmed deletion.

Two ways to use it:
- **CLI scripts** under [`scripts/`](scripts/) — for batch / automation / debugging
- **WEB UI** ([`web/app.py`](web/app.py)) — a Streamlit app for interactive demos

### Features

| Module | Capability |
|---|---|
| 🎞 **Frame extraction** | Single/batch videos, configurable interval, CJK-filename safe |
| 🤖 **Inference** | YOLOv8+, class filtering, custom colors, Chinese labels, CJK-font fallback |
| 🎬 **Compose** | Filename-sorted, auto/custom FPS, source-video FPS fallback |
| 🌐 **Web UI** | Upload / params / progress / download in one page; full-pipeline or single-step |
| 📥 **Result download** | Per-task downloads; a current-batch ZIP is offered for multi-video full runs |
| 📁 **Artifact browser** | Task navigation, paginated previews, multi-download, confirmed deletion |
| 📊 **Task tracking** | Background serial jobs, refresh recovery, batch cancellation, configs and inference stats |
| 🛠 **Engineering** | Chinese paths, Unicode filenames, Windows-friendly, modular & reusable |

### Quick Start

#### 1. Prerequisites

Requires **Python 3.10+**.

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS / Linux

pip install -r web/requirements.txt
```

#### 2. Launch the Web UI

```bash
streamlit run web/app.py --server.maxUploadSize 1024
```

Open `http://localhost:8501`. Flow:

1. **Sidebar** — upload a `.pt` model file (optionally enable cross-session cache).
2. **Top radio** — pick a processing mode or the artifact browser.
3. **Step 1/2/3** — fill in the parameters.
4. Click **▶ Start** to submit a background job; monitor, refresh, or cancel the active batch.
5. Download from **📥 Results**, or inspect task summaries and inference statistics in the artifact browser.

#### 3. CLI usage

See [`docs/WORKFLOW.md § Three-Step CLI Scripts`](docs/WORKFLOW.md#三段式-cli-脚本--three-step-cli-scripts).

### Project Layout

```
cv-toolbox/
├── scripts/                    # CLI scripts
│   ├── 1_frame_extract.py     # 1. Frame extraction
│   ├── 2_model_infer.py       # 2. Model inference
│   └── 3_images_to_video.py   # 3. Image-to-video
├── web/                        # Streamlit WEB UI
│   ├── app.py
│   ├── pipeline.py
│   ├── helpers.py
│   ├── artifact_browser.py
│   └── requirements.txt
├── docs/
│   └── WORKFLOW.md             # Workflow details
├── uploads/                    # Runtime uploads (gitignored)
├── outputs/                    # Runtime outputs (gitignored)
├── CHANGELOG.md
├── LICENSE
└── README.md
```

### Architecture

```
┌────────────────────────────────────────────────────────────┐
│                  Streamlit Web UI (app.py)                 │
│   ┌─────────────┐  ┌──────────────┐  ┌──────────────┐    │
│   │ Step 1       │  │ Step 2       │  │ Step 3       │    │
│   └─────┬───────┘  └──────┬───────┘  └──────┬───────┘    │
│         └──────────────────┴─────────────────┘            │
│                            │                              │
│                     pipeline.py                            │
│                  (run_pipeline mode=                       │
│                   full/extract/infer/encode)               │
│                            │                              │
│      ┌─────────────────────┼─────────────────────┐        │
│      ▼                     ▼                     ▼        │
│ scripts/1_frame_extract.py  scripts/2_model_infer.py  scripts/3_images_to_video.py
│  extract_frames()       infer()               create_video_from_images()
└────────────────────────────────────────────────────────────┘
```

The Web UI loads the three scripts under `scripts/` via `importlib.util.spec_from_file_location` and calls their **pure functions** directly (avoiding their `input()` interactive prompts).

Artifact browsing is isolated in `web/artifact_browser.py` and is restricted to task directories under `outputs/`; internal model, source-video, and download-cache directories are hidden.

### Usage Examples

**Web UI full pipeline**:
1. Upload 1..N videos (mp4/mov/avi/mkv)
2. Upload `yolov8n.pt`
3. Step 1: frame interval = 1
4. Step 2: red box + rename class 0 to "人"
5. Step 3: frame rate = original
6. Start → watch progress → download the result (multi-video runs also offer a ZIP)

**CLI chain**:
```bash
python scripts/1_frame_extract.py  --video <path/to/your/video.mp4> --output outputs/demo/frames --interval 1
python scripts/2_model_infer.py    --model <path/to/your/model.pt>  --input outputs/demo/frames --output outputs/demo/annotated --color red --label-map "0:人" --classes "person,car"
python scripts/3_images_to_video.py --input outputs/demo/annotated/images --output outputs/demo/demo.mp4 --fps 24
```

---

## 更多文档 / Further Reading

- 📘 [docs/WORKFLOW.md](docs/WORKFLOW.md) — 工作流详细说明 / Detailed workflow description
- 📝 [CHANGELOG.md](CHANGELOG.md) — 版本演进 / Version history
- ⚖️ [LICENSE](LICENSE) — MIT License

---

## ❓ 常见问题 / FAQ

### Q1. 上传文件时浏览器卡在 "uploading..." 不动 / 报 `ClientDisconnect` 错 / 终端堆栈刷屏

**这不是程序崩溃。** Streamlit / Starlette / Uvicorn 在用户**中断文件上传**时（手动刷新、点开了另一个 file_uploader、网络抖动、上传中点「开始处理」），服务端会以 `ERROR` 级别打印完整 traceback，看起来吓人但实际无害。完整堆栈末尾的 `starlette.requests.ClientDisconnect` 就是"客户端关掉连接"的意思。

- **每次取消 = 一条 traceback**，所以你会看到十几条几乎相同的堆栈
- 我们的 `app.py` 是在**上传完成后**才执行的；上传被中断时 Python 端根本不会跑
- 解决方法：等待浏览器自己恢复（一般几秒），或点上传框右侧的「🔄 更换」强制重置

### Q2. 上传后看不到文件名 / 一直显示 "uploading..."

1. 检查文件大小是否超过 `--server.maxUploadSize`（默认命令行 1024 MB）
2. 网络慢时大文件可能耗时数分钟——**不要在 uploading 状态时点刷新**，会触发 Q1 的情况
3. **v1.0.6 hotfix 起**：上传卡住直接点侧栏「🧹 重置页面」按钮，会让所有 file_uploader 重建为空（即使卡在 99% 也能恢复）

### Q3. 上传了一个新文件，但页面状态没刷新

Streamlit 的 file_uploader 在新文件到达时会自动触发一次 rerun。如果看起来"没反应"：
- 看是否浏览器标签页右上角有"重新连接"提示
- 试一下点「🔍 Rerun」按钮（Streamlit 菜单里的）
- 终极方法：点侧栏「🧹 重置页面」（不会删除本地文件），或刷新整个页面（注意：这会清空当前 session，请先下载结果）

### Q4. 推理速度很慢 / 第一次跑模型要等很久

- YOLO 模型首次加载会占用较多内存（几秒到几十秒）
- 切到「跨会话缓存上传的模型」会避免每次重新解析 `.pt` 头部
- 大视频（>1 GB）建议用「仅抽帧」先抽出关键帧，再「仅推理」单独跑

### Q5. 想批量跑多个视频

当前 Web UI 全流程模式支持多选上传，每个视频独立跑、独立出结果、独立可下载。CLI 方式可写 shell 循环：

```bash
for f in videos/*.mp4; do
  python scripts/1_frame_extract.py --video "$f" --output "outputs/$(basename "$f" .mp4)/frames" --interval 5
done
```

### Q6. 中文 label 渲染成方块/问号

说明系统缺 CJK 字体。脚本会按 `msyh.ttc` (Windows) → `PingFang.ttc` (macOS) → `NotoSansCJK` (Linux) 顺序查找。Linux 服务器请 `apt install fonts-noto-cjk` 或把字体放到 `/usr/share/fonts/` 后 `fc-cache -fv`。

### Q7. 报错 / 卡死 / 异常如何排查

- 看终端（启动 `streamlit run ...` 的那个窗口）的最新输出
- 看浏览器开发者工具的 Console（F12 → Console 标签）
- `outputs/` 下的中间产物可以判断卡在哪一步
- 仍然解决不了请附上报错截图 + 命令行版本 + Python 版本（`python --version`）开 issue

### Q8. 上传 / 下载时终端报 `_ProactorBasePipeTransport._call_connection_lost` / `WinError 10054`

这是 Windows asyncio ProactorEventLoop 在客户端**主动断开 WebSocket / multipart 连接**时的回调错误（Streamlit 1.x 在 Windows 上已知问题）。无害，连接被关闭是用户操作（手动刷新、点开另一个 widget、上传中点别的按钮）的结果。**v1.0.6 起**通过把结果区下载按钮的 IO 全部走 `@st.cache_data` 缓存，rerun 期间不再重读大文件，触发频率已明显降低。v1.0.6 hotfix#2 进一步给侧栏模型落盘加了 size 守卫（只有文件缺失或大小变化才读+写），rerun 期间不再重写 50~130MB 模型，卡死与断连进一步减少。如仍偶发，刷新页面即可。

### Q9. 点「🧹 重置页面」会不会把 outputs/ 下的视频删掉？上传卡死怎么办？

**不会删任何文件。** v1.0.6 hotfix 起「重置页面」（原「清空缓存」）**只重置页面 UI 控件与 `session_state`**：通过全局 `_reset_token` 让所有 `file_uploader` 拿到全新 widget_key，再注入 `<meta http-equiv="refresh" content="0">` 触发浏览器**硬刷新**（等价于地址栏回车），把浏览器缓存的 widget state 与 in-flight 上传请求一起丢掉。**不删除任何本地文件**（uploads/、outputs/、outputs/_models/ 都不动）。要删磁盘产物请用侧栏上方的「🧹 清空本地文件」按钮。

**上传图片卡死**（uploading… 一直转、点任何按钮无反应、刷新也没用）的根因是每次 rerun 都重读并重写整个模型文件（50~130MB）阻塞主线程。v1.0.6 hotfix#2 已加 size 守卫：只有文件缺失或大小变化才读+写，命中时连 `read()` 都不做，rerun 变为纯渲染，卡死已消除。

### Q10. 点击下载后页面无响应，刷新后为什么会同时下载多个文件？

旧版下载按钮会在每次点击时重新执行整个 Streamlit 页面；页面重跑期间还会同步读取视频、压缩 ZIP，连续点击产生的请求可能因此排队。现在下载改为延迟执行：页面渲染时不读取大文件，点击后才在独立下载线程读取或生成 ZIP，而且下载不再触发整页重跑。生成过的 ZIP 会缓存在 `outputs/_downloads/`，源文件未变化时直接复用。

若刷新页面中断了尚未结束的上传，后台仍可能记录一次 `starlette.requests.ClientDisconnect`。它表示浏览器主动断开了上传请求，不代表模型推理或文件处理失败。

---

## 开源协议 / License

本项目基于 [MIT License](LICENSE) 开源。
This project is licensed under the [MIT License](LICENSE).
