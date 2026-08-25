# 工作流说明 / Workflow

> 本文档定义 **CV 模型效果展示项目** 的输入输出契约与业务流程。
> 详细版本演进见 [CHANGELOG.md](../CHANGELOG.md)；项目总体说明见 [README.md](../README.md)。

---

## 项目定位 / Project Overview

本项目是一个**计算机视觉模型效果展示项目**，可用于：

- 视频抽帧 (video → frames)
- 图片标注 (frames → annotated frames with bounding boxes & labels)
- 视频合成 (annotated frames → MP4)

默认基础模型为 **YOLOv8**（`yolov8n.pt`），但只要符合 Ultralytics YOLO 接口的 `.pt` 文件都可使用。

---

## 业务流程 / Pipeline

整个工作流分为 3 步：

```
┌──────────────┐    ┌──────────────┐    ┌──────────────────┐
│  原始视频     │    │  抽帧 JPG     │    │  YOLO 推理结果    │
│  (mp4/avi/    │ →  │  (每 N 帧    │ →  │  (带 bbox +      │ →  ┌──────────────┐
│   mov/mkv)    │    │   抽一张)     │    │   label 标注)    │    │  最终 MP4    │
└──────────────┘    └──────────────┘    └──────────────────┘    │  (原帧率)    │
                                                                 └──────────────┘
  Step 1                Step 2                 Step 3
```

### Step 1 · 视频抽帧 (Frame Extraction)

- 输入：原始视频文件
- 输出：与视频帧率一致的抽帧图片（默认间隔 1，即每帧都抽）
- 命名：`frame_NNNNNN.jpg`（多视频时带前缀避免冲突）

### Step 2 · 模型推理 (YOLO Inference)

- 输入：抽帧得到的图片目录
- 输出：在原图上叠加 bbox 与 label 的可视化图片
- 标注名称默认取模型内置名称（如 `person`），用户可在 Web UI 中自定义（支持单类快捷或 `0:吸烟,1:打火机` 形式）
- 中文 label 通过 PIL + 系统 CJK 字体渲染（绕过 OpenCV Hershey 字体的中文不支持问题）

### Step 3 · 视频合成 (Encode)

- 输入：标注目录
- 输出：按**文件名升序**合成、保持原始视频帧率的 MP4
- 命名：`<视频原名>.mp4`（与 Step 1 输入同名）

---

## 三段式 CLI 脚本 / Three-Step CLI Scripts

每个脚本既能**命令行参数**调用，也能**交互式**输入参数运行（命令行不全时进入交互模式）。

| # | 脚本 | 输入 | 输出 |
|---|---|---|---|
| 1 | [`脚本/1.视频抽帧.py`](../脚本/1.视频抽帧.py) | 视频路径、抽帧间隔 | 图片目录 |
| 2 | [`脚本/2.模型推理.py`](../脚本/2.模型推理.py) | 模型 `.pt`、图片目录 | 标注目录 |
| 3 | [`脚本/3.图片转视频.py`](../脚本/3.图片转视频.py) | 图片目录、帧率 | `output.mp4` |

调用示例 / Quick examples:

```bash
# Step 1
python 脚本/1.视频抽帧.py --video test/男人抽烟.mp4 --output outputs/男人抽烟/frames --interval 1

# Step 2
python 脚本/2.模型推理.py \
    --model test/yolov8n.pt \
    --input outputs/男人抽烟/frames \
    --output outputs/男人抽烟/annotated \
    --color red --label-map "0:人"

# Step 3
python 脚本/3.图片转视频.py --input outputs/男人抽烟/annotated/images --output outputs/男人抽烟/男人抽烟.mp4 --fps 24
```

---

## WEB UI 功能 / WEB UI Features

`web/app.py` 是一个 Streamlit 应用，覆盖以下能力：

1. **上传视频文件**：单选或多选（`mp4 / mov / avi / mkv`）。
2. **选择模型**：从侧栏上传 `.pt` 文件；可勾选「跨会话缓存」按 SHA1 持久化到 `uploads/models/`。
3. **分步参数设置**：
   - **Step 1**：抽帧间隔（文本框，非法值回退 + warning）。
   - **Step 2**：置信度阈值、NMS IoU 阈值、标注框颜色（下拉框中文选项 + 自定义调色板）、标注名称（按模型类别动态渲染每类输入框 + 顶部「统一标注名称」覆盖）。
   - **Step 3**：帧率（下拉框「原视频帧率」/「自定义」二选一）。
4. **运行模式**：全流程 / 仅抽帧 / 仅推理 / 仅合成 四选一（水平 `st.radio`）。
5. **进度展示**：每个 per-video 用 `st.container(border=True) + st.status` 独立显示进度条和日志。
6. **结果下载**：
   - 每 stem 独立的「下载视频 / 下载抽帧 ZIP / 下载推理 ZIP」按钮。
   - 底部全局「下载全部 (ZIP)」：仅打包**当前会话**在 `st.session_state["results"]` 里的产物，**不会**混入 `outputs/` 下历史轮次的视频。
7. **清空结果**：仅清空内存中的结果列表，磁盘文件保留。

---

## 输入输出约定 / I/O Conventions

### 路径与文件名

- **中文路径**：Windows 下 OpenCV `cv2.imread` 无法直接读取含中文的路径。脚本统一通过 `np.fromfile + cv2.imdecode` 实现 Unicode 安全读取（见 `脚本/2.模型推理.py::imread_unicode`）。
- **目录命名清洗**：上传文件名经 `web/helpers.py::safe_stem` 清洗 Windows 非法字符 `\\/:*?"<>|`。
- **磁盘布局**：

```
<project_root>/
├── uploads/                  # 用户上传的素材（gitignored）
│   ├── <stem>/video.mp4      # 上传的视频（v1.0.1+ 改名）
│   └── models/<name>_<sha>.pt
└── outputs/                  # 流水线产物（gitignored）
    ├── <stem>/frames/*.jpg
    ├── <stem>/annotated/images/*.jpg
    └── <stem>/<stem>.mp4
```

仅推理 / 仅合成的临时上传目录为 `outputs/<stem>/_uploaded/`。

### 推理参数 / Inference Parameters

| 参数 | 默认值 | 含义 |
|---|---|---|
| `conf_thres` | 0.25 | 置信度阈值，低于此值的检测会被过滤 |
| `iou_thres` | 0.45 | NMS IoU 阈值，越低 → 重叠框抑制越严格 |
| `device` | `auto` | 推理设备（`cpu` / `cuda` / `cuda:0` / 自动选择） |
| `box_color` | 红色 (0,0,255) | BGR 元组，OpenCV 顺序 |
| `label_map` | `{}` | `{class_id: display_name}`，如 `{"0":"吸烟"}` |

---

## 已知限制 / Known Limitations

- **中文 label 渲染依赖系统字体**：自动查找 `msyh.ttc` (Windows)、`PingFang.ttc` (macOS)、`NotoSansCJK-Regular.ttc` (Linux)。若系统无 CJK 字体需手动指定。
- **仅推理 / 仅合成的进度条按单 job 展示**：一次上传 = 一个时间戳 stem，进度只显示一个块。多 batch 需要串行。
- **源视频帧率推断**：仅合成模式下若上传了源视频，从源视频读取 fps；若未上传且未自定义 fps，回退 30 fps。
