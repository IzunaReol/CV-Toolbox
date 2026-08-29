# 工作流说明 / Workflow

> 本文档定义 **CV 模型效果展示项目** 的输入输出契约与业务流程。
> 详细版本演进见 [CHANGELOG.md](../CHANGELOG.md)；项目总体说明见 [README.md](../README.md)。

---

## 项目定位 / Project Overview

本项目是一个**计算机视觉模型效果展示项目**，可用于：

- 视频抽帧 (video → frames)
- 指定类别图片标注 (frames → selected-class annotated frames)
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
- 可选择一个或多个模型类别进行推理，未选择的类别不会进入检测结果；不指定时保持全类别推理。
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
| 1 | [`scripts/1_frame_extract.py`](../scripts/1_frame_extract.py) | 视频路径、抽帧间隔 | 图片目录 |
| 2 | [`scripts/2_model_infer.py`](../scripts/2_model_infer.py) | 模型 `.pt`、图片目录 | 标注目录 |
| 3 | [`scripts/3_images_to_video.py`](../scripts/3_images_to_video.py) | 图片目录、帧率 | `output.mp4` |

调用示例 / Quick examples:

```bash
# Step 1
python scripts/1_frame_extract.py --video <path/to/your/video.mp4> --output outputs/demo/frames --interval 1

# Step 2
python scripts/2_model_infer.py \
    --model <path/to/your/model.pt> \
    --input outputs/demo/frames \
    --output outputs/demo/annotated \
    --color red --label-map "0:人" \
    --classes "person,car"

# Step 3
python scripts/3_images_to_video.py --input outputs/demo/annotated/images --output outputs/demo/demo.mp4 --fps 24
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
4. **运行模式**：全流程 / 仅抽帧 / 仅推理 / 仅合成 / 文件浏览 五选一。
5. **任务进度**：同一时间只运行一个批次，批次内视频串行处理；页面每秒读取磁盘状态并以中文展示阶段和进度。任务进度与处理结果统一展示任务名称、任务类型和开始时间。文件浏览模式不显示任务进度区域。
6. **结果下载**：
   - 每 stem 独立的「下载视频 / 下载抽帧 ZIP / 下载推理 ZIP」按钮。
   - 全流程上传多个视频时，底部提供「下载全部 (ZIP)」，只打包**当前会话**的产物；单个视频不额外提供 ZIP。
7. **清空结果**：仅清空内存中的结果列表，磁盘文件保留。
8. **类别过滤**：模型加载后默认全部选择，支持多选、全部选择和全部取消；未选择类别时不能开始推理。
9. **任务控制**：刷新页面后继续跟踪后台批次；取消会停止当前处理并跳过批次剩余视频，已生成工件保留。
10. **文件浏览**：浏览任务摘要、配置和推理统计；图片每页预览 24 张，支持大图导航、多选下载和二次确认删除。删除当前预览图后自动切换至下一张。内部目录不会展示。

### 文件浏览下载规则

- 选择不超过 5 个文件且总大小不超过 20 MiB 时，逐个下载。
- 文件数超过 5 个或总大小超过 20 MiB 时，先显示 ZIP 准备进度，完成后提供下载。
- 删除显示逐文件进度；完成后自动清理空目录和无工件的空任务。
- 只允许选择和删除公开工件；所有操作均限制在当前任务目录内，并拒绝符号链接。

### 任务状态与统计

- 状态依次为 `queued / running / completed`，异常分为 `failed / cancelled / interrupted`。
- 服务仍运行时，页面刷新会从 `outputs/_jobs/` 恢复任务跟踪；服务重启后遗留运行状态标记为 `interrupted`。
- `outputs/<task>/_meta/task.json` 保存任务配置、状态和摘要。
- 推理任务同时保存 `inference_stats.json` 和逐图 `inference_images.csv`；内部元数据不会进入普通文件浏览和工件下载。

---

## 输入输出约定 / I/O Conventions

### 路径与文件名

- **中文路径**：Windows 下 OpenCV 的直接图片读写可能报告成功但未实际落盘。推理通过 `np.fromfile + cv2.imdecode` 读取，抽帧通过 `cv2.imencode + tofile` 写入，并在计数前验证文件已经生成。
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
    ├── <stem>/<stem>.mp4
    ├── <stem>/_meta/          # 任务配置与推理统计（文件浏览中隐藏）
    ├── _jobs/*.json           # 后台批次状态（文件浏览中隐藏）
    └── _downloads/*.zip      # 延迟生成的下载缓存（文件浏览中隐藏）
```

仅推理 / 仅合成的临时上传目录为 `outputs/<stem>/_uploaded/`。

### 推理参数 / Inference Parameters

| 参数 | 默认值 | 含义 |
|---|---|---|
| `conf_thres` | 0.25 | 置信度阈值，低于此值的检测会被过滤 |
| `iou_thres` | 0.45 | NMS IoU 阈值，越低 → 重叠框抑制越严格 |
| `device` | `auto` | Web 推理设备（`auto` / `cpu` / `cuda`） |
| `box_color` | 红色 (0,0,255) | BGR 元组，OpenCV 顺序 |
| `label_map` | `{}` | `{class_id: display_name}`，如 `{"0":"吸烟"}` |
| `classes` | `None` | 需要推理的类别 ID 列表；`None` 表示全部类别 |

---

## 已知限制 / Known Limitations

- **中文 label 渲染依赖系统字体**：自动查找 `msyh.ttc` (Windows)、`PingFang.ttc` (macOS)、`NotoSansCJK-Regular.ttc` (Linux)。若系统无 CJK 字体需手动指定。
- **任务串行执行**：同一时间只运行一个批次，避免多个模型任务争用 GPU 显存。
- **源视频帧率推断**：仅合成模式下若上传了源视频，从源视频读取 fps；若未上传且未自定义 fps，回退 30 fps。
