import os
import torch
from nodes import MAX_RESOLUTION
import torchvision.transforms.v2 as T
from PIL import Image, ImageDraw, ImageFont, ImageColor, ImageFilter
from comfy.utils import ProgressBar
import numpy as np

# 尝试自动定位字体目录
FONTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "font")
if not os.path.exists(FONTS_DIR):
    try:
        os.makedirs(FONTS_DIR, exist_ok=True)
    except:
        pass

class YC_SubtitleNode:
    """
    字幕序列节点 (高性能优化版)
    
    优化点：
    1. 引入缓存机制：相同的字幕段落只渲染一次。
    2. 移除逐帧 PIL 转换：直接在 Tensor 层面进行图像合成。
    3. 极大提升处理速度并降低内存抖动。
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        fonts = []
        if os.path.exists(FONTS_DIR):
            fonts = sorted([f for f in os.listdir(FONTS_DIR) if f.endswith('.ttf') or f.endswith('.otf')])
        if not fonts:
            fonts = ["default"]
        
        return {
            "required": {
                "images": ("IMAGE",), # [Batch, H, W, C]
                "subtitle_text": ("STRING", {
                    "multiline": True,
                    "default": "第一段字幕(30帧)||第二段字幕(60帧)",
                }),
                "frame_durations": ("STRING", {
                    "multiline": True, 
                    "default": "30|15|60",
                }),
                "delimiter": ("STRING", {"default": "|"}),
                "font": (fonts, {"default": fonts[0] if fonts else "default"}),
                "font_size": ("INT", {"default": 48, "min": 8, "max": 500}),
                "text_color": ("STRING", {"default": "#FFFFFF"}),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 20}),
                "stroke_color": ("STRING", {"default": "#000000"}),
                "background_color": ("STRING", {"default": "#00000000"}),
                "horizontal_align": (["left", "center", "right"], {"default": "center"}),
                "vertical_align": (["top", "center", "bottom"], {"default": "bottom"}),
                "offset_x": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "offset_y": ("INT", {"default": -50, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "shadow_enabled": (["disabled", "enabled"], {"default": "enabled"}),
                "shadow_distance": ("INT", {"default": 2, "min": 0, "max": 50}),
                "shadow_blur": ("INT", {"default": 3, "min": 0, "max": 50}),
                "shadow_expand": ("INT", {"default": 0, "min": 0, "max": 30}),
                "shadow_color": ("STRING", {"default": "#000000"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "YC_VideoCutHelper/Subtitle"
    
    def parse_subtitle_text(self, text, delimiter):
        if not text:
            return []
        segments = text.split(delimiter)
        return [s.strip() for s in segments]
    
    def parse_frame_durations(self, frame_str, delimiter, count_needed):
        if not frame_str:
            return [30] * count_needed
        try:
            durations = [int(x.strip()) for x in frame_str.split(delimiter) if x.strip().isdigit()]
        except ValueError:
            print("[YC_Subtitle] 帧数格式错误，使用了默认值30")
            durations = []
        if not durations:
            return [30] * count_needed
        if len(durations) < count_needed:
            durations.extend([durations[-1]] * (count_needed - len(durations)))
        return durations

    def get_font(self, font_name, font_size):
        if font_name != "default" and os.path.exists(FONTS_DIR):
            font_path = os.path.join(FONTS_DIR, font_name)
            if os.path.exists(font_path):
                try:
                    return ImageFont.truetype(font_path, font_size)
                except:
                    pass
        try:
            return ImageFont.load_default()
        except:
            return ImageFont.load_default()

    def get_text_size(self, text, font, stroke_width=0):
        try:
            left, top, right, bottom = font.getbbox(text, stroke_width=stroke_width)
            return right - left, bottom - top
        except TypeError:
            try:
                left, top, right, bottom = font.getbbox(text)
                return (right - left) + stroke_width * 2, (bottom - top) + stroke_width * 2
            except:
                return font.getsize(text)
        except:
            return font.getsize(text)

    # 核心优化：只生成一张包含字幕的透明 Tensor，不合成到原图
    def create_subtitle_mask(self, width, height, text, font, text_color_str, stroke_width, stroke_color_str, background_color_str,
                           horizontal_align, vertical_align, offset_x, offset_y,
                           shadow_enabled, shadow_distance, shadow_blur, shadow_expand, shadow_color_str):
        
        # 1. 基础设置
        lines = text.split("\n")
        try:
            ascent, descent = font.getmetrics()
            line_spacing = ascent + descent
        except:
            line_spacing = font.size * 1.2
        line_spacing += stroke_width
            
        line_dims = []
        for line in lines:
            if not line.strip():
                line_dims.append((0, 0))
            else:
                line_dims.append(self.get_text_size(line, font, stroke_width))
        
        content_height = len(lines) * line_spacing

        if vertical_align == "top":
            start_y = offset_y
        elif vertical_align == "center":
            start_y = (height - content_height) / 2 + offset_y
        else:  # bottom
            start_y = height - content_height + offset_y
            
        def parse_color(c_str, default_alpha=255):
            try:
                if c_str.startswith('#'):
                    c_str = c_str.strip()
                    if len(c_str) == 7: return ImageColor.getrgb(c_str) + (default_alpha,)
                    if len(c_str) == 9: return ImageColor.getrgb(c_str[:7]) + (int(c_str[7:9], 16),)
                return ImageColor.getrgb(c_str) + (default_alpha,)
            except:
                return (255, 255, 255, default_alpha)

        text_rgba = parse_color(text_color_str, 255)
        stroke_rgba = parse_color(stroke_color_str, 255)
        bg_rgba = parse_color(background_color_str, 0)
        shadow_rgba = parse_color(shadow_color_str, 255)

        # 2. 创建 PIL 图层
        layer = Image.new('RGBA', (width, height), color=bg_rgba)
        
        # 3. 绘制阴影
        if shadow_enabled == "enabled" and shadow_distance > 0:
            shadow_layer = Image.new('RGBA', (width, height), (0,0,0,0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            curr_y = start_y
            shadow_stroke_width = stroke_width + shadow_expand
            for i, line in enumerate(lines):
                if line.strip():
                    w, h = line_dims[i]
                    if horizontal_align == "left": x = offset_x
                    elif horizontal_align == "center": x = (width - w) / 2 + offset_x
                    else: x = width - w + offset_x
                    
                    shadow_draw.text((x + shadow_distance, curr_y + shadow_distance), line, font=font, 
                                   fill=shadow_rgba, stroke_width=shadow_stroke_width, stroke_fill=shadow_rgba)
                curr_y += line_spacing
            if shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
            layer = Image.alpha_composite(layer, shadow_layer)

        # 4. 绘制正文
        draw = ImageDraw.Draw(layer)
        curr_y = start_y
        for i, line in enumerate(lines):
            if line.strip():
                w, h = line_dims[i]
                if horizontal_align == "left": x = offset_x
                elif horizontal_align == "center": x = (width - w) / 2 + offset_x
                else: x = width - w + offset_x
                
                draw.text((x, curr_y), line, font=font, fill=text_rgba, 
                          stroke_width=stroke_width, stroke_fill=stroke_rgba)
            curr_y += line_spacing
            
        # 5. 关键优化：转换为 Tensor，归一化到 [0, 1]
        # PIL (RGBA) -> Numpy -> Tensor [H, W, 4]
        mask_np = np.array(layer).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np) # [H, W, 4]
        
        return mask_tensor

    def execute(self, images, subtitle_text, frame_durations, delimiter,
                font, font_size, text_color, stroke_width, stroke_color, background_color,
                horizontal_align, vertical_align, offset_x, offset_y,
                shadow_enabled, shadow_distance, shadow_blur, shadow_expand, shadow_color):
        
        # 1. 解析参数
        segments = self.parse_subtitle_text(subtitle_text, delimiter)
        if not segments:
            return (images,)
        durations = self.parse_frame_durations(frame_durations, delimiter, len(segments))
        
        batch_size, height, width, channels = images.shape
        font_obj = self.get_font(font, font_size)
        
        # 2. 预渲染缓存 (Cache Pre-rendering)
        # 找出所有不重复的非空字幕文本，先渲染成遮罩 Tensor
        unique_texts = set([s for s in segments if s and s.strip()])
        text_cache = {}
        
        print(f"[YC_Subtitle] 正在预渲染 {len(unique_texts)} 个唯一的字幕遮罩...")
        
        for text in unique_texts:
            mask_tensor = self.create_subtitle_mask(
                width, height, text, font_obj, 
                text_color, stroke_width, stroke_color, background_color,
                horizontal_align, vertical_align, offset_x, offset_y,
                shadow_enabled, shadow_distance, shadow_blur, shadow_expand, shadow_color
            )
            # 确保 mask 在与 images 相同的设备上 (CPU/GPU)
            if images.device != mask_tensor.device:
                mask_tensor = mask_tensor.to(images.device)
            text_cache[text] = mask_tensor

        print(f"[YC_Subtitle] 预渲染完成。开始合成视频帧...")

        # 3. 建立帧索引映射
        # result_images 直接克隆输入，避免修改原始数据（ComfyUI原则）
        result_images = images.clone() 
        
        current_frame_idx = 0
        pbar = ProgressBar(len(segments))
        
        for i, seg_text in enumerate(segments):
            duration = durations[i]
            start_frame = current_frame_idx
            end_frame = min(current_frame_idx + duration, batch_size)
            
            # 如果这几帧有字幕，且字幕不为空
            if seg_text and seg_text.strip() and start_frame < batch_size:
                # 获取缓存的遮罩: [H, W, 4]
                mask = text_cache[seg_text]
                
                # 分离 RGB 和 Alpha
                # overlay_rgb: [H, W, 3]
                # overlay_alpha: [H, W, 1]
                overlay_rgb = mask[:, :, :3]
                overlay_alpha = mask[:, :, 3:4]
                
                # 4. 批量张量合成 (Vectorized Compositing)
                # 我们一次性处理这一段的所有帧 [Start:End, H, W, C]
                # 公式: Target = Source * (1 - Alpha) + Overlay * Alpha
                
                # 利用广播机制：
                # frame_slice: [N, H, W, 3]
                # overlay_alpha: [H, W, 1] -> 广播为 [N, H, W, 1] -> [N, H, W, 3]
                # overlay_rgb: [H, W, 3] -> 广播为 [N, H, W, 3]
                
                frame_slice = result_images[start_frame:end_frame]
                
                # 执行合成运算
                # 注意：inplace 操作比创建新 tensor 更省显存
                # frame_slice * (1 - alpha)
                frame_slice.mul_(1.0 - overlay_alpha) 
                # + overlay * alpha
                frame_slice.add_(overlay_rgb * overlay_alpha)
                
            current_frame_idx = end_frame
            pbar.update(1)
            
            if current_frame_idx >= batch_size:
                break
                
        return (result_images,)

NODE_CLASS_MAPPINGS = {
    "YC_Subtitle": YC_SubtitleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YC_Subtitle": "🎬 YC Subtitle (Optimized)",
}