# python 2_model_infer.py --model <path/to/your/model.pt> --input <path/to/frames_dir> --output <path/to/output_dir> --device cuda --color red --label-map "0:吸烟,1:打火机"
# 直接运行也可，按提示依次输入模型路径、图片目录、标注框颜色、自定义标注名称即可

import argparse
import re
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# 常用颜色名 -> BGR（OpenCV 使用 BGR 顺序）
COLOR_NAMES = {
    'red':     (0, 0, 255),
    'green':   (0, 255, 0),
    'blue':    (255, 0, 0),
    'yellow':  (0, 255, 255),
    'cyan':    (255, 255, 0),
    'magenta': (255, 0, 255),
    'white':   (255, 255, 255),
    'black':   (0, 0, 0),
    'orange':  (0, 165, 255),
    'purple':  (128, 0, 128),
}

DEFAULT_BOX_COLOR = COLOR_NAMES['red']  # 默认红色 (BGR: 0,0,255)

def imread_unicode(path):
    """以 numpy.fromfile + cv2.imdecode 读取图片，绕过 OpenCV 对中文路径的支持问题。"""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None

def find_cjk_font(size: int = 18):
    """在常见系统位置查找支持中文（CJK）的字体文件，返回 PIL ImageFont。

    优先级：微软雅黑 > 黑体 > 宋体 > 苹方 > Noto Sans CJK > DejaVuSans。
    若都找不到，返回一个内置的位图字体（中文将回退为方块，仍可保证不报错）。
    """
    # Windows + macOS + Linux 常见中文/全 Unicode 字体路径
    candidate_paths = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\Deng.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidate_paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                continue
    # 兜底：PIL 自带字体（不含中文）
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()

def get_device(device_str: str):
    """自动选择合适的设备"""
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device_str == 'cpu':
        return 'cpu'
    if device_str.startswith('cuda') or device_str.isdigit():
        if torch.cuda.is_available():
            return device_str
        else:
            print("⚠️ CUDA 不可用，回退到 CPU")
            return 'cpu'
    return 'cpu'

def prompt_existing_path(prompt_text: str) -> Path:
    """交互式提示用户输入一个已存在的路径，无效则反复询问"""
    while True:
        raw = input(prompt_text).strip().strip('"').strip("'")
        if not raw:
            print("路径不能为空，请重新输入")
            continue
        p = Path(raw)
        if not p.exists():
            print(f"路径不存在: {p}，请重新输入")
            continue
        return p

def parse_color(value, default=DEFAULT_BOX_COLOR):
    """解析用户输入的颜色，支持：颜色名 / BGR 三元组 / #RRGGBB 十六进制。失败时返回默认颜色。"""
    if value is None:
        return default
    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return default

    # 颜色名
    key = raw.lower()
    if key in COLOR_NAMES:
        return COLOR_NAMES[key]

    # 十六进制 #RRGGBB 或 RRGGBB（按 RGB 转 BGR）
    hex_match = re.fullmatch(r'#?([0-9a-fA-F]{6})', key)
    if hex_match:
        hex_val = hex_match.group(1)
        r = int(hex_val[0:2], 16)
        g = int(hex_val[2:4], 16)
        b = int(hex_val[4:6], 16)
        return (b, g, r)  # 转 BGR

    # BGR 三元组，如 "0,0,255" 或 "0;0;255"
    parts = re.split(r'[\s,;]+', raw)
    if len(parts) == 3:
        try:
            bgr = tuple(int(max(0, min(255, int(p)))) for p in parts)
            return bgr
        except ValueError:
            pass

    print(f"无法识别的颜色: {value}，使用默认红色")
    return default

def draw_boxes(image_bgr, results, box_color, show_label=True, show_conf=True, label_map=None):
    """在原图上手动绘制检测框与label，便于自定义颜色与名称（支持中文）。

    使用 PIL 进行文字渲染，以正确显示中文/全 Unicode 字符。
    label_map: 可选 dict[int|str, str]，用于覆盖模型默认的类别名；
               未提供的类别仍使用模型返回的名称。
    """
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return image_bgr

    xyxy = boxes.xyxy.cpu().numpy().astype(int)
    cls_ids = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    names = results.names
    label_map = label_map or {}

    h = image_bgr.shape[0]
    # 字体大小按图片高度自适应
    font_size = max(14, min(28, h // 40))
    font = find_cjk_font(size=font_size)

    # BGR -> PIL RGB
    pil_img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # OpenCV BGR -> PIL RGB
    fill_rgb = (int(box_color[2]), int(box_color[1]), int(box_color[0]))
    text_color = (255, 255, 255) if sum(box_color) < 380 else (0, 0, 0)

    for (x1, y1, x2, y2), cls_id, conf in zip(xyxy, cls_ids, confs):
        # 矩形框
        draw.rectangle([x1, y1, x2, y2], outline=fill_rgb, width=2)

        # 组合 label 文本：优先使用自定义名称，再退回模型名称
        parts = []
        if show_label:
            custom_name = label_map.get(cls_id)
            if custom_name is None:
                custom_name = label_map.get(str(cls_id))
            if custom_name is None:
                custom_name = names.get(cls_id, str(cls_id))
            parts.append(str(custom_name))
        if show_conf:
            parts.append(f"{conf:.2f}")
        if not parts:
            continue  # 既不要 label 也不要 conf，仅画框

        label = " ".join(parts)

        # 计算文本尺寸 (PIL 用 textbbox)
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]
            offset_y = tb[1]
        except AttributeError:
            tw, th = draw.textsize(label, font=font)
            offset_y = 0

        pad_x, pad_y = 4, 2
        ty1 = max(y1 - th - pad_y * 2, 0)
        ty2 = y1
        # 文本底色填充
        draw.rectangle([x1, ty1, x1 + tw + pad_x * 2, ty2], fill=fill_rgb)
        draw.text((x1 + pad_x, ty1 + pad_y - offset_y), label,
                  fill=text_color, font=font)

    # PIL RGB -> BGR numpy
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def infer(
    model_path: str,
    input_dir: str,
    output_dir: str = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    device: str = 'auto',
    box_color: tuple = DEFAULT_BOX_COLOR,
    show_label: bool = True,
    show_conf: bool = True,
    label_map: dict = None
):
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_path}")

    # 默认输出目录：<输入目录名>_annotated
    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.name}_annotated"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 必带：保存带标注框和label的可视化图片
    img_out_dir = output_path / "images"
    img_out_dir.mkdir(exist_ok=True)

    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    target_device = get_device(device)
    model.to(target_device)
    print(f"推理设备: {target_device}")
    print(f"标注框颜色 (BGR): {box_color}")
    print(f"显示类别名: {show_label}    显示置信度: {show_conf}")
    if label_map:
        print(f"自定义标注名称映射: {label_map}")
    else:
        print("自定义标注名称映射: 无（使用模型默认名称）")

    # 记录模型原始名称，方便用户参考
    raw_names = getattr(model, 'names', None)
    if label_map is None and raw_names:
        print(f"模型默认类别名（可作为自定义映射的参考）: {dict(raw_names)}")

    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.webp')
    img_files = [f for f in input_path.iterdir() if f.suffix.lower() in img_exts]
    if not img_files:
        print(f"警告: 在 {input_path} 中未找到任何图片文件")
        return

    print(f"找到 {len(img_files)} 张图片，开始推理并绘制标注...")

    annotated_count = 0
    for idx, img_file in enumerate(img_files, 1):
        print(f"处理 [{idx}/{len(img_files)}]: {img_file.name}")
        results = model(img_file, conf=conf_thres, iou=iou_thres, verbose=False)

        # 读取原图后用 draw_boxes 自定义颜色绘制（使用 unicode 安全读取）
        img_bgr = imread_unicode(img_file)
        if img_bgr is None:
            print(f"  读取图片失败，跳过: {img_file.name}")
            continue
        plotted = draw_boxes(img_bgr, results[0], box_color=box_color,
                             show_label=show_label, show_conf=show_conf,
                             label_map=label_map)

        out_img_path = img_out_dir / img_file.name
        cv2.imwrite(str(out_img_path), plotted)

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            annotated_count += 1

    print("=" * 50)
    print(f"推理完成！共处理 {len(img_files)} 张图片，其中 {annotated_count} 张检测到目标")
    print(f"带标注框与label的可视化结果保存在: {img_out_dir}")

def parse_positive_float(value: str, default: float) -> float:
    """解析 [0,1] 范围内的浮点数，无效则返回默认值"""
    try:
        parsed = float(value)
        if parsed <= 0 or parsed > 1:
            print(f"输入值 {value} 超出 (0,1] 范围，使用默认值 {default}")
            return default
        return parsed
    except (ValueError, TypeError):
        print(f"输入值 {value} 不是有效数字，使用默认值 {default}")
        return default

def parse_label_map(value, model_names=None):
    """解析自定义标注名称映射。

    支持三种形式：
      1. JSON 字符串：'{"0":"吸烟","1":"打火机"}'
      2. 简写 k:v 列表：'0:吸烟,1:打火机' （分隔符可换成分号或空格）
      3. JSON 文件路径：'/path/to/map.json'
    返回 dict[str,str] 或 None（无输入/解析失败时回退到模型默认名称）。

    便捷简写：若输入是单个普通字符串（不含 : { , ;），且模型仅有一个类别，
    则自动当作 {0: 该字符串} 处理。
    """
    if value is None:
        return None
    raw = str(value).strip().strip('"').strip("'")
    if not raw:
        return None

    # 可能是 JSON 文件路径
    p = Path(raw)
    if p.exists() and p.is_file():
        try:
            import json
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            print(f"标签映射文件内容不是字典格式: {p}")
        except Exception as e:
            print(f"读取标签映射文件失败: {e}")
        return None

    # 先尝试 JSON 字符串
    try:
        import json
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (ValueError, TypeError):
        pass

    # 再尝试简写 0:吸烟,1:打火机（支持中英文冒号、分号、空格分隔）
    try:
        result = {}
        for item in re.split(r'[,;]\s*', raw):
            if not item.strip():
                continue
            normalized = item.replace('：', ':')
            if ':' not in normalized:
                continue
            key, _, val = normalized.partition(':')
            key = key.strip()
            val = val.strip()
            if key and val:
                result[key] = val
        if result:
            return result
    except Exception:
        pass

    # 便捷简写：单个普通名称 -> 当作 {0: name}（无论类别数量）
    if not re.search(r'[:{;,\[\]]', raw):
        # 单个纯净字符串（不含分隔符/JSON标记），按 class 0 处理
        if model_names and 0 in model_names:
            print(f"将 '{raw}' 应用到类别 0（{model_names[0]}）")
            print("  多类别模型如需精确映射，请使用 0:人,1:车 这种格式")
        else:
            print(f"将 '{raw}' 应用到类别 0")
        return {'0': raw}

    # 解析失败：给出更明确的错误提示
    print(f"无法解析自定义标注名称映射: '{value}'")
    print("  期望格式示例:")
    print("    单类别简写:    人")
    print("    多类别简写:    0:吸烟,1:打火机")
    print("    JSON 字符串:  {\"0\":\"吸烟\",\"1\":\"打火机\"}")
    print("    JSON 文件路径: D:/maps/names.json")
    print("  将使用模型默认名称")
    return None

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 推理脚本（交互式输入，输出带标注框的可视化结果）")
    parser.add_argument("--model", type=str, default=None, help="训练好的模型路径(.pt)")
    parser.add_argument("--input", type=str, default=None, help="输入图片文件夹路径")
    parser.add_argument("--output", type=str, default=None, help="输出结果文件夹路径（默认: <输入目录名>_annotated/images）")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 阈值 (默认 0.45)")
    parser.add_argument("--device", type=str, default='auto', help="推理设备: 'auto', 'cpu', 'cuda', '0', '1' 等")
    parser.add_argument("--color", type=str, default=None,
                        help="标注框颜色，支持颜色名(red/green/blue/yellow/cyan/...)、BGR三元组(0,0,255)、#RRGGBB。默认 red")
    parser.add_argument("--label-map", type=str, default=None,
                        help="自定义标注名称映射。支持 '0:吸烟,1:打火机'、JSON 字符串、JSON 文件路径。默认使用模型返回的名称")
    args = parser.parse_args()

    # 检查库
    try:
        from ultralytics import YOLO  # noqa: F401
        print("Ultralytics 可用")
    except ImportError:
        print("未找到 ultralytics，请安装: pip install ultralytics")
        return

    # 模型路径：优先用命令行参数，否则交互输入
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f"模型文件不存在: {model_path}")
            return
    else:
        model_path = prompt_existing_path("请输入模型文件路径(.pt): ")

    # 输入图片目录：优先用命令行参数，否则交互输入
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"输入文件夹不存在: {input_path}")
            return
    else:
        input_path = prompt_existing_path("请输入图片目录路径: ")

    # 提前加载模型以获取类别名称，便于在交互时给出参考与支持单类别简写
    print(f"正在加载模型: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    model_names = getattr(model, 'names', None)

    # 置信度与 IoU 阈值：交互式可覆盖
    conf_thres = args.conf
    iou_thres = args.iou
    box_color = parse_color(args.color, default=DEFAULT_BOX_COLOR)
    label_map = parse_label_map(args.label_map, model_names=model_names)

    if len(sys.argv) <= 1:
        if model_names:
            print(f"模型类别参考: {dict(model_names)}")
        conf_raw = input(f"请输入置信度阈值 (0~1, 默认 {conf_thres}): ").strip()
        if conf_raw:
            conf_thres = parse_positive_float(conf_raw, conf_thres)
        iou_raw = input(f"请输入 NMS IoU 阈值 (0~1, 默认 {iou_thres}): ").strip()
        if iou_raw:
            iou_thres = parse_positive_float(iou_raw, iou_thres)
        color_raw = input(
            f"请输入标注框颜色（颜色名/BGR/#RRGGBB，默认 red）: ").strip()
        if color_raw:
            box_color = parse_color(color_raw, default=box_color)
        map_raw = input(
            "请输入自定义标注名称映射（格式: 单类别直接输入名称如\"人\"；多类别用 0:吸烟,1:打火机；也可输入 JSON 或 JSON 文件路径。回车使用模型默认名称）: ").strip()
        if map_raw:
            label_map = parse_label_map(map_raw, model_names=model_names)

    infer(
        model_path=str(model_path),
        input_dir=str(input_path),
        output_dir=args.output,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        device=args.device,
        box_color=box_color,
        label_map=label_map
    )

if __name__ == "__main__":
    main()