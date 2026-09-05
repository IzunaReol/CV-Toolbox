# 🧰 CV 工具箱 / CV Toolbox

用来展示 YOLO 检测效果的小工具：上传视频，抽帧、推理，再把标注后的图片合成视频。可以在网页里操作，也可以分步运行命令行脚本。

Extract frames, run YOLO detection, and turn the annotated images back into a video. Includes a local web UI and CLI scripts.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?logo=streamlit)](web/requirements.txt)
[![Version](https://img.shields.io/badge/Version-v2.2.0-orange)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[快速开始](#快速开始) · [使用说明](#使用说明) · [常见问题](#常见问题) · [English](#english)

## 演示

https://github.com/user-attachments/assets/6073e336-b2b9-4734-bb0c-d556b88a1a94

演示视频的界面可能与当前版本略有不同。

## 能做什么

- 一次处理一个或多个视频，也可以只抽帧、只推理或只合成。
- 选择要检测的类别，修改框的颜色和标签，支持中文。
- 查看进度、取消任务，浏览和下载以前的结果。
- 不需要中间图片时，可以只保存最终视频和统计。

v2.2.0 给上传框加了“清空已上传”，修复了同名文件和历史帧混用的问题，也减少了模型加载和图片读写。详细改动见 [更新日志](CHANGELOG.md)。

## 快速开始

需要 **Python 3.10+**。在项目根目录创建并激活虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux：

```bash
source .venv/bin/activate
```

安装依赖并启动：

```bash
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py --server.maxUploadSize 1024
```

打开 [本地页面](http://localhost:8501)。如果是从旧版升级，先重启服务。

默认使用 `auto`：CUDA 可用时用 GPU，否则用 CPU。想用 GPU，需要先装好支持 CUDA 的 PyTorch。

## 使用说明

### Web UI

1. 选择 **全流程 / 仅抽帧 / 仅推理 / 仅合成 / 文件浏览**。
2. 上传视频或图片。需要推理时，再从侧栏上传 `.pt` 检测模型。
3. 设置抽帧间隔、推理类别、阈值、标注样式和合成帧率。
4. 全流程可勾选 **仅保存最终视频（减少磁盘占用）**。
5. 点击 **开始处理**，在任务区域查看进度；结束后下载结果。

视频支持 MP4、MOV、AVI、MKV；图片支持 JPG、JPEG、PNG、BMP、WebP、TIF、TIFF。

### 上传与清空

| 操作 | 效果 |
|---|---|
| 清空已上传 | 只清空对应上传框，可立即重新上传；其他上传框与结果保留 |
| 清空模型上传 | 同时清掉模型选择和类别状态；已经提交的批次继续使用原权重 |
| 清空结果 | 清空当前页面结果列表，磁盘文件保留 |
| 重置页面 | 重置页面参数和上传控件，不删除磁盘文件 |
| 清空本地文件 | 确认后清理 `outputs/`，保留 `_models/`；运行中不允许删除 |

上传过的文件仍保存在本地。历史结果在 **文件浏览** 中管理，原始素材在 `uploads/` 目录。

### 帧率与输出

- **原视频帧率**：直接使用源 FPS，不按抽帧间隔换算。例如 30 FPS 视频每 5 帧取一帧，再按 30 FPS 合成，时长约为原来的五分之一。
- **自定义帧率**：支持 29.97、23.976 等小数值。
- **仅合成**：选择原视频帧率时可上传源视频读取 FPS；未提供时回退到 30 FPS。
- 标准全流程保留抽帧和标注图片；流式全流程只保留成功视频与推理统计。
- 合成后会检查视频能否打开、帧数和首尾帧是否正常。失败或取消时保留已有成品。

### 命令行

以下三条命令依次完成抽帧、推理、合成。将 `input.mp4` 和 `model.pt` 替换为自己的文件，类别 `0` 及标签按模型调整：

```bash
python scripts/1_frame_extract.py --video "input.mp4" --output "outputs/demo/frames" --interval 1
python scripts/2_model_infer.py --model "model.pt" --input "outputs/demo/frames" --output "outputs/demo/annotated" --classes "0" --color red --label-map "0:人"
python scripts/3_images_to_video.py --input "outputs/demo/annotated/images" --output "outputs/demo/demo.mp4" --fps 29.97
```

重跑时将 `demo` 改为新的目录名。抽帧和推理会拒绝复用已有结果图片的目录，防止混入历史帧。CLI 合成使用 `--fps` 指定值，未指定时为 30，不会自动查找源视频。

更多参数和文件保存位置见 [工作流说明](docs/WORKFLOW.md)。

## 项目结构

```text
CV-Toolbox/
├── scripts/                    # 三段式 CLI
│   ├── 1_frame_extract.py
│   ├── 2_model_infer.py
│   └── 3_images_to_video.py
├── web/
│   ├── app.py                  # Streamlit 界面
│   ├── pipeline.py             # 标准与流式编排
│   ├── media.py                # 图片格式、视频写入校验
│   ├── helpers.py              # 上传、下载、参数工具
│   ├── job_manager.py          # 后台任务与状态
│   ├── task_store.py           # 任务配置与推理统计
│   ├── artifact_browser.py     # 工件浏览与删除
│   └── requirements.txt
├── tests/                      # 功能、媒体 IO 与界面回归测试
├── docs/
│   ├── WORKFLOW.md
│   ├── V2.2.0_ACCEPTANCE.md
│   └── releases/v2.2.0.md
├── uploads/                    # 运行时素材，不纳入 Git
├── outputs/                    # 运行时结果，不纳入 Git
├── pyproject.toml
├── CHANGELOG.md
└── LICENSE
```

## 常见问题

**上传没有完成或界面暂时无响应？**

先检查文件大小是否超过启动时设置的上传限制，并等待传输完成。页面仍可操作时，可点击对应框的“清空已上传”重新选择；整个页面失去响应时再尝试重置页面或刷新。若问题持续，查看服务终端日志。

**日志出现 `ClientDisconnect` 或连接重置？**

取消上传或刷新页面时可能出现这类日志。先看任务是否还在运行；如果任务也失败了，再检查后面的报错。

**清空上传或刷新后，后台任务会丢失吗？**

任务开始后会读取本地文件，清空上传不会取消它。只要服务还开着，刷新页面后就能继续看进度；重启服务则需要重新提交任务。

**推理慢、磁盘占用大？**

先看是否在用 CPU。可以适当增大抽帧间隔，或者勾选“仅保存最终视频”来省磁盘空间。这里的模型缓存指保存权重文件；旧结果需要自己清理。

**中文标签显示方块？**

中文绘制依赖系统字体。代码会查找微软雅黑、苹方、Noto Sans CJK 等常见字体；缺失时需安装相应字体后重试。

**为什么视频时长变短，或没有声音？**

按源 FPS 合成采样帧会缩短时长。当前输出为无音轨 MP4，不重建可变帧率时间戳。标准流程经过 JPEG 压缩，流式流程直接使用解码帧，两者像素及检测结果可能略有差异。

## 开发与验证

```bash
python -m unittest discover -s tests -p "test_*.py"
python -m compileall -q web scripts tests
```

代码规范检查需先安装开发工具：

```bash
python -m pip install ruff
python -m ruff check web scripts tests
python -m ruff format --check web scripts tests
```

测试会生成图片和 MP4，并检查界面操作。检测部分用了替代模型，不下载权重，也不测 GPU 速度。[CI 配置](.github/workflows/lint.yml)覆盖 Python 3.10、3.11、3.12。

## English

CV Toolbox runs locally with a Streamlit UI. Upload a video and a YOLO detection model, choose the classes to detect, then download the annotated video. Each step can also run separately.

```bash
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py --server.maxUploadSize 1024
```

Open [localhost:8501](http://localhost:8501) to get started. Python 3.10+ is required.

Each upload field has a **清空已上传** (clear uploads) button, so you can replace files without refreshing the page. Clearing an upload does not stop a submitted job. Jobs run one batch at a time; a page refresh keeps the job running, but restarting the server interrupts it.

Original FPS uses the source frame rate as-is. If you sample every N frames, the output will be about 1/N of the original length. You can set a custom FPS, including fractional values such as 29.97.

Enable “仅保存最终视频” to skip intermediate images and keep only the video and statistics. Output videos have no audio. Chinese labels need a CJK font installed on the system.

For CLI usage, follow the [example above](#命令行). Use a new output directory when repeating extraction or inference. More details are in the [workflow](docs/WORKFLOW.md).

## 文档与许可

- [工作流与输入输出约定](docs/WORKFLOW.md)
- [更新日志](CHANGELOG.md)
- [v2.2.0 发布说明](docs/releases/v2.2.0.md)
- [v2.2.0 验收与发布清单](docs/V2.2.0_ACCEPTANCE.md)
- [MIT License](LICENSE)
