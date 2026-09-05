# 🧰 CV 工具箱 / CV Toolbox

视频抽帧 → YOLO 类别过滤推理 → 视频合成 → 工件管理，提供本地 Web UI 和命令行工具。

A local toolbox for frame extraction, class-filtered YOLO detection, video composition, and artifact management.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](pyproject.toml)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.62-FF4B4B?logo=streamlit)](web/requirements.txt)
[![Version](https://img.shields.io/badge/Version-v2.2.0-orange)](CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[快速开始](#快速开始) · [使用说明](#使用说明) · [常见问题](#常见问题) · [English](#english)

## 演示

https://github.com/user-attachments/assets/6073e336-b2b9-4734-bb0c-d556b88a1a94

演示用于了解基本流程，当前界面以本仓库代码为准。

## 功能

| 功能 | 说明 |
|---|---|
| 视频抽帧 | 单个或多个视频，按帧间隔采样，支持中文文件名 |
| 模型推理 | YOLO 检测模型、类别多选、框颜色与中文标签 |
| 视频合成 | 按文件名排序，使用源视频帧率或自定义小数帧率 |
| 流式处理 | 全流程可选只保存最终视频和统计，减少中间图片读写 |
| 上传管理 | 每个上传框独立“清空已上传”，无需刷新即可重新选择 |
| 后台任务 | 批次内串行处理、进度查看、页面刷新后继续跟踪、取消批次 |
| 结果管理 | 单任务下载、多视频全流程 ZIP、分页图片预览、确认删除 |
| 结果隔离 | 同名素材独立保存，模型按内容寻址，避免混用历史结果 |

**v2.2.0** 还增加了视频写入校验、模型批次复用、字体缓存和进度写入节流。详见 [版本更新日志](CHANGELOG.md) 与 [v2.2.0 发布说明](docs/releases/v2.2.0.md)。

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

打开 [本地 Web UI](http://localhost:8501)。升级后请重启服务，确保后台线程加载新代码。

GPU 推理需要当前 PyTorch 环境支持 CUDA。界面默认 `auto`，CUDA 不可用时使用 CPU；模型缓存不会改变计算设备。

## 使用说明

### Web UI

1. 选择 **全流程 / 仅抽帧 / 仅推理 / 仅合成 / 文件浏览**。
2. 涉及推理时，在侧栏上传 `.pt` 检测模型；上传对应的视频或图片。
3. 设置抽帧间隔、推理类别、阈值、标注样式和合成帧率。
4. 全流程可勾选 **仅保存最终视频（减少磁盘占用）**。
5. 点击 **开始处理**，在任务区域查看进度；结束后下载结果。

支持的视频上传格式：MP4、MOV、AVI、MKV。图片上传、推理和下载统一支持 JPG、JPEG、PNG、BMP、WebP、TIF、TIFF。

### 上传与清空

| 操作 | 效果 |
|---|---|
| 清空已上传 | 只清空对应上传框，可立即重新上传；其他上传框与结果保留 |
| 清空模型上传 | 同时清掉模型选择和类别状态；已经提交的批次继续使用原权重 |
| 清空结果 | 清空当前页面结果列表，磁盘文件保留 |
| 重置页面 | 重置页面参数和上传控件，不删除磁盘文件 |
| 清空本地文件 | 确认后清理 `outputs/`，保留 `_models/`；运行中不允许删除 |

“清空已上传”不会删除已落盘的素材或结果。历史任务可在 **文件浏览** 中管理；上传素材位于 `uploads/`，不在该浏览器中展示。

### 帧率与输出

- **原视频帧率**：直接使用源 FPS，不按抽帧间隔换算。例如 30 FPS 视频每 5 帧取一帧，再按 30 FPS 合成，时长约为原来的五分之一。
- **自定义帧率**：支持 29.97、23.976 等小数值。
- **仅合成**：选择原视频帧率时可上传源视频读取 FPS；未提供时回退到 30 FPS。
- 标准全流程保留抽帧和标注图片；流式全流程只保留成功视频与推理统计。
- 视频通过编码器、文件、帧数和首尾帧校验后才成为成品；失败或取消不会覆盖已有成品。

### 命令行

以下三条命令依次完成抽帧、推理、合成。将 `input.mp4` 和 `model.pt` 替换为自己的文件，类别 `0` 及标签按模型调整：

```bash
python scripts/1_frame_extract.py --video "input.mp4" --output "outputs/demo/frames" --interval 1
python scripts/2_model_infer.py --model "model.pt" --input "outputs/demo/frames" --output "outputs/demo/annotated" --classes "0" --color red --label-map "0:人"
python scripts/3_images_to_video.py --input "outputs/demo/annotated/images" --output "outputs/demo/demo.mp4" --fps 29.97
```

重跑时将 `demo` 改为新的目录名。抽帧和推理会拒绝复用已有结果图片的目录，防止混入历史帧。CLI 合成使用 `--fps` 指定值，未指定时为 30，不会自动查找源视频。

详细参数、目录约定和取消行为见 [工作流说明](docs/WORKFLOW.md)。

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

如果发生在取消上传或刷新页面后，先检查后台任务是否仍正常运行。连接日志本身不能证明推理失败，也不能排除其他故障；应结合任务状态与后续错误定位。

**清空上传或刷新后，后台任务会丢失吗？**

已提交批次使用磁盘输入快照，清空上传不会取消它。服务仍运行时，刷新页面可以继续跟踪活动批次；重启服务会中断原任务，需要重新提交。

**推理慢、磁盘占用大？**

确认实际使用的推理设备；按展示需求增大抽帧间隔，或启用仅保存最终视频。同一批次会复用模型，但跨会话模型缓存保存的是权重文件，不保证跨批次保留 GPU 模型。历史输出需要按需清理。

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

回归测试包含真实图片/MP4 读写与 Streamlit 界面测试，使用替代检测模型，不下载权重，也不代表真实 GPU 性能测试。[CI 配置](.github/workflows/lint.yml)覆盖 Python 3.10、3.11、3.12。

## English

CV Toolbox provides a Streamlit UI and three CLI scripts for video frame extraction, YOLO object detection, video composition, and artifact browsing. Run commands from the repository root.

```bash
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py --server.maxUploadSize 1024
```

Open [the local UI](http://localhost:8501), choose a processing mode, upload your inputs, configure detection classes and FPS, and start the batch.

- Each upload field has an independent **清空已上传** (clear uploads) button. Clearing the model also resets class selection; submitted jobs keep their original inputs.
- Original FPS means the source FPS, without adjustment for frame sampling. Sampling every N frames makes the output approximately 1/N of the original duration.
- Optional streaming mode saves only the final video and inference statistics, avoiding intermediate images.
- JPG/JPEG/PNG/BMP/WebP/TIF/TIFF are supported throughout image input and result downloads.
- Uploads and repeated runs use separate task directories. Model weights are stored by SHA-256 content hash; a batch reuses one model instance.
- Video output is validated before replacing the destination. Cancellation removes incomplete videos; standard-mode intermediate images remain available.
- Jobs run serially. Refreshing the page reconnects to an active batch; restarting the server interrupts it. Deletion is blocked while a batch is running.
- Output is silent MP4. Variable frame rate timestamps and source audio are not preserved. Chinese labels require a compatible system font.

For CLI usage, follow the [three-command example](#命令行). Use new output directories when repeating extraction or inference. See the [workflow](docs/WORKFLOW.md) and [v2.2.0 release notes](docs/releases/v2.2.0.md) for details.

## 文档与许可

- [工作流与输入输出约定](docs/WORKFLOW.md)
- [更新日志](CHANGELOG.md)
- [v2.2.0 发布说明](docs/releases/v2.2.0.md)
- [v2.2.0 验收与发布清单](docs/V2.2.0_ACCEPTANCE.md)
- [MIT License](LICENSE)
