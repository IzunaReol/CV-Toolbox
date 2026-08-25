# 🧰 CV 工具箱 / CV Toolbox

> 计算机视觉模型效果展示项目 — **视频抽帧 → 模型推理 → 视频合成** 一站式 CLI + Web UI。
> A CV model demo project — **frame extraction → YOLO inference → video composition** with CLI scripts and a Streamlit WEB UI.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36%2B-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Ultralytics](https://img.shields.io/badge/YOLO-v8%2B-00FFFF?logo=yolo)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-v1.0.4-orange)](CHANGELOG.md)

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
2. **推理**：用 YOLO（Ultralytics）模型对每帧做目标检测，可自定义标注框颜色与中文 label。
3. **合成**：把标注意图按原视频帧率合成为 MP4。

提供两种使用方式：
- **CLI 脚本**（`scripts/`）：适合批处理、自动化、调试
- **WEB UI**（`web/app.py`）：基于 Streamlit，适合人工演示与参数调试

### 功能特性

| 模块 | 能力 |
|---|---|
| 🎞 **抽帧** | 单/多视频、按帧间隔、CJK 文件名安全 |
| 🤖 **推理** | YOLOv8+、自定义颜色、中文 label、CJK 字体回退、Unicode 路径安全 |
| 🎬 **合成** | 按文件名排序合成、自动/自定义帧率、源视频帧率回退 |
| 🌐 **Web UI** | 上传/参数/进度/下载一条龙；支持全流程或单步运行 |
| 📥 **结果下载** | 每 stem 独立下载 + 当前会话 ZIP（不会混入历史产物） |
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
2. **顶部 radio** 选择运行模式：`全流程 / 仅抽帧 / 仅推理 / 仅合成`。
3. **Step 1/2/3** 展开对应面板设置参数。
4. 点击 **▶ 开始处理**，下方进度条实时更新。
5. 处理完成后在 **📥 处理结果** 区下载视频或 ZIP。

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
python scripts/3_images_to_video.py --input outputs/demo/annotated/images --output outputs/demo/demo.mp4 --fps 24
```

---

## English

### Introduction

**CV Toolbox** is a local toolkit for demonstrating computer-vision model inference end-to-end:

1. **Extract** frames from videos at a configurable interval.
2. **Infer** with a YOLO (Ultralytics) detector per frame, with customizable box color and **Chinese** labels.
3. **Compose** annotated frames back into an MP4 at the original video's frame rate.

Two ways to use it:
- **CLI scripts** under [`scripts/`](scripts/) — for batch / automation / debugging
- **WEB UI** ([`web/app.py`](web/app.py)) — a Streamlit app for interactive demos

### Features

| Module | Capability |
|---|---|
| 🎞 **Frame extraction** | Single/batch videos, configurable interval, CJK-filename safe |
| 🤖 **Inference** | YOLOv8+, custom colors, Chinese labels, CJK-font fallback, Unicode path safe |
| 🎬 **Compose** | Filename-sorted, auto/custom FPS, source-video FPS fallback |
| 🌐 **Web UI** | Upload / params / progress / download in one page; full-pipeline or single-step |
| 📥 **Result download** | Per-stem individual download + **current-session** ZIP (no historical bleed) |
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
2. **Top radio** — pick mode: `Full pipeline / Extract only / Infer only / Compose only`.
3. **Step 1/2/3** — fill in the parameters.
4. Click **▶ Start**, watch progress bars below.
5. Download from **📥 Results** panel (per stem or session ZIP).

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

### Usage Examples

**Web UI full pipeline**:
1. Upload 1..N videos (mp4/mov/avi/mkv)
2. Upload `yolov8n.pt`
3. Step 1: frame interval = 1
4. Step 2: red box + rename class 0 to "人"
5. Step 3: frame rate = original
6. Start → watch progress → download ZIP

**CLI chain**:
```bash
python scripts/1_frame_extract.py  --video <path/to/your/video.mp4> --output outputs/demo/frames --interval 1
python scripts/2_model_infer.py    --model <path/to/your/model.pt>  --input outputs/demo/frames --output outputs/demo/annotated --color red --label-map "0:人"
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
3. 实在卡住，点上传框右侧的「🔄 更换」按钮可强制重置

### Q3. 上传了一个新文件，但页面状态没刷新

Streamlit 的 file_uploader 在新文件到达时会自动触发一次 rerun。如果看起来"没反应"：
- 看是否浏览器标签页右上角有"重新连接"提示
- 试一下点「🔍 Rerun」按钮（Streamlit 菜单里的）
- 终极方法：刷新整个页面（注意：这会清空当前 session，请先下载结果）

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

---

## 开源协议 / License

本项目基于 [MIT License](LICENSE) 开源。
This project is licensed under the [MIT License](LICENSE).
