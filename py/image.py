import torch
import numpy as np
from PIL import Image

# 导入ComfyUI工具函数
def pil2tensor(image):
    """PIL图像转tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def tensor2pil(image):
    """tensor转PIL图像"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def create_crop_mask(original_width, original_height, left, top, right, bottom, invert_mask=False):
    """创建裁剪遮罩，可选择遮罩方向
    
    Args:
        original_width: 原始图像宽度
        original_height: 原始图像高度
        left: 左边裁剪像素
        top: 上边裁剪像素
        right: 右边裁剪像素
        bottom: 下边裁剪像素
        invert_mask: 是否反向遮罩
            - False: 保留区域为白色(1.0)，裁剪区域为黑色(0.0)
            - True: 保留区域为黑色(0.0)，裁剪区域为白色(1.0)
    """
    # 计算裁剪后的尺寸
    new_width = original_width - left - right
    new_height = original_height - top - bottom
    
    # 确保尺寸有效
    if new_width <= 0 or new_height <= 0:
        raise ValueError(f"裁剪后尺寸无效: {new_width}x{new_height}")
    
    # 根据反向设置决定默认颜色
    default_color = 255 if invert_mask else 0
    fill_color = 0 if invert_mask else 255
    
    # 创建PIL遮罩图像（使用原始尺寸）
    mask_image = Image.new('L', (original_width, original_height), default_color)
    
    # 在保留区域绘制填充色矩形
    # 保留区域是 (left, top) 到 (left + new_width, top + new_height)
    mask_image.paste(fill_color, (left, top, left + new_width, top + new_height))
    
    # 转换为numpy数组并归一化到0.0-1.0
    mask_np = np.array(mask_image).astype(np.float32) / 255.0
    
    # 转换为torch tensor并添加批次维度
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)  # [1, height, width]
    
    return mask_tensor



class YCImagePushPullLens:
    """推拉镜头效果节点 - 从图像中心逐渐放大或缩小，输出多帧
    支持单张图片生成多帧，也支持多帧图片分别处理"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "frames": ("INT", {"default": 30, "min": 2, "max": 120, "step": 1}),
                "start_crop_ratio": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05}),
                "middle_crop_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "end_crop_ratio": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 1.0, "step": 0.05}),
                "middle_frame": ("INT", {"default": 15, "min": 1, "max": 120, "step": 1}),
                "input_mode": (["single_to_multi", "multi_to_multi"], {"default": "single_to_multi"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("frames", "masks")
    FUNCTION = "push_pull_lens"
    CATEGORY = "YC_VideoCutHelper/Image"

    def push_pull_lens(self, image, frames, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame, input_mode):
        """推拉镜头处理 - 支持单张图片生成多帧和多帧图片分别处理"""
        ret_images = []
        ret_masks = []
        
        if input_mode == "single_to_multi":
            # 原有功能：单张图片生成多帧
            return self._single_to_multi(image, frames, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame)
        else:
            # 新功能：多帧图片分别处理
            return self._multi_to_multi(image, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame)
    
    def _single_to_multi(self, image, frames, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame):
        """原有功能：单张图片生成多帧"""
        ret_images = []
        ret_masks = []
        
        # 处理batch
        for img in image:
            # 转换为PIL图像
            pil_img = tensor2pil(torch.unsqueeze(img, 0))
            
            # 获取原始图像尺寸
            original_width, original_height = pil_img.size
            
            # 计算每帧的裁切系数
            crop_ratios = self._calculate_crop_ratios(frames, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame)
            
            # 为每一帧生成裁切图像
            for frame_idx, crop_ratio in enumerate(crop_ratios):
                # 计算当前帧的裁切参数（从中心裁切）
                crop_width = int(original_width * crop_ratio)
                crop_height = int(original_height * crop_ratio)
                
                # 计算裁切区域的左上角坐标（保持居中）
                left = (original_width - crop_width) // 2
                top = (original_height - crop_height) // 2
                right = original_width - (left + crop_width)  # 右边剩余的像素数量
                bottom = original_height - (top + crop_height)  # 下边剩余的像素数量
                
                # 执行裁切
                cropped_pil = pil_img.crop((left, top, left + crop_width, top + crop_height))
                
                # 调整到原始尺寸（保持宽高比）
                result_pil = cropped_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
                
                # 转换回tensor
                result_tensor = pil2tensor(result_pil)
                ret_images.append(result_tensor)
                
                # 创建对应的遮罩（显示裁切区域）
                mask_tensor = create_crop_mask(original_width, original_height, left, top, right, bottom, False)
                ret_masks.append(mask_tensor)
        
        # 返回结果
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
    
    def _multi_to_multi(self, image, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame):
        """新功能：多帧图片分别处理，每帧应用不同的裁切比例"""
        ret_images = []
        ret_masks = []
        
        # 获取输入帧数
        input_frames = image.shape[0]
        
        # 计算每帧的裁切系数
        crop_ratios = self._calculate_crop_ratios(input_frames, start_crop_ratio, middle_crop_ratio, end_crop_ratio, middle_frame)
        
        # 为每一帧生成裁切图像
        for frame_idx in range(input_frames):
            # 获取当前帧
            current_frame = image[frame_idx:frame_idx+1]  # 保持批次维度
            
            # 转换为PIL图像
            pil_img = tensor2pil(current_frame)
            
            # 获取原始图像尺寸
            original_width, original_height = pil_img.size
            
            # 获取当前帧的裁切系数
            crop_ratio = crop_ratios[frame_idx]
            
            # 计算当前帧的裁切参数（从中心裁切）
            crop_width = int(original_width * crop_ratio)
            crop_height = int(original_height * crop_ratio)
            
            # 计算裁切区域的左上角坐标（保持居中）
            left = (original_width - crop_width) // 2
            top = (original_height - crop_height) // 2
            right = original_width - (left + crop_width)  # 右边剩余的像素数量
            bottom = original_height - (top + crop_height)  # 下边剩余的像素数量
            
            # 执行裁切
            cropped_pil = pil_img.crop((left, top, left + crop_width, top + crop_height))
            
            # 调整到原始尺寸（保持宽高比）
            result_pil = cropped_pil.resize((original_width, original_height), Image.Resampling.LANCZOS)
            
            # 转换回tensor
            result_tensor = pil2tensor(result_pil)
            ret_images.append(result_tensor)
            
            # 创建对应的遮罩（显示裁切区域）
            mask_tensor = create_crop_mask(original_width, original_height, left, top, right, bottom, False)
            ret_masks.append(mask_tensor)
        
        # 返回结果
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
    
    def _calculate_crop_ratios(self, frames, start_ratio, middle_ratio, end_ratio, middle_frame):
        """计算每帧的裁切系数 - 支持三段式变化"""
        ratios = []
        
        # 确保中间帧数不超过总帧数
        middle_frame = min(middle_frame, frames - 1)
        
        # 计算每段的帧数
        first_segment_frames = middle_frame
        second_segment_frames = frames - middle_frame
        
        # 第一段：从起始比例到中间比例
        for i in range(first_segment_frames):
            if first_segment_frames > 1:
                progress = i / (first_segment_frames - 1)
                current_ratio = start_ratio + (middle_ratio - start_ratio) * progress
            else:
                current_ratio = start_ratio
            ratios.append(current_ratio)
        
        # 第二段：从中间比例到结束比例
        for i in range(second_segment_frames):
            if second_segment_frames > 1:
                progress = i / (second_segment_frames - 1)
                current_ratio = middle_ratio + (end_ratio - middle_ratio) * progress
            else:
                current_ratio = middle_ratio
            ratios.append(current_ratio)
        
        return ratios


class YCImageOverlayBlend:
    """多帧图片叠加混合节点 - 将多帧图片与底色叠加，裁剪系数转换为透明程度"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),  # 来自推拉镜头节点的多帧图片
                "background_color": ("STRING", {"default": "#000000", "multiline": False}),
                "start_alpha_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "middle_alpha_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "end_alpha_ratio": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "middle_frame": ("INT", {"default": 15, "min": 1, "max": 120, "step": 1}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light"], {"default": "normal"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("blended_frames", "alpha_masks")
    FUNCTION = "overlay_blend"
    CATEGORY = "YC_VideoCutHelper/Image"

    def overlay_blend(self, frames, background_color, start_alpha_ratio, middle_alpha_ratio, end_alpha_ratio, middle_frame, blend_mode):
        """多帧图片叠加混合处理"""
        ret_images = []
        ret_masks = []
        
        # 解析背景颜色
        bg_color = self._parse_color(background_color)
        
        # 计算每帧的透明系数
        alpha_ratios = self._calculate_alpha_ratios(frames.shape[0], start_alpha_ratio, middle_alpha_ratio, end_alpha_ratio, middle_frame)
        
        # 为每一帧生成叠加混合图像
        for frame_idx, alpha_ratio in enumerate(alpha_ratios):
            # 获取当前帧
            current_frame = frames[frame_idx:frame_idx+1]  # 保持批次维度
            
            # 创建背景图像
            bg_tensor = self._create_background_tensor(current_frame.shape, bg_color)
            
            # 应用透明度和混合模式
            blended_tensor = self._apply_blend(current_frame, bg_tensor, alpha_ratio, blend_mode)
            
            ret_images.append(blended_tensor)
            
            # 创建对应的透明度遮罩
            alpha_mask = self._create_alpha_mask(current_frame.shape, alpha_ratio)
            ret_masks.append(alpha_mask)
        
        # 返回结果
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0))
    
    def _parse_color(self, color_str):
        """解析颜色字符串为RGB值"""
        if color_str.startswith('#'):
            # 十六进制颜色
            color_str = color_str[1:]
            if len(color_str) == 6:
                r = int(color_str[0:2], 16) / 255.0
                g = int(color_str[2:4], 16) / 255.0
                b = int(color_str[4:6], 16) / 255.0
                return (r, g, b)
        
        # 默认黑色
        return (0.0, 0.0, 0.0)
    
    def _create_background_tensor(self, frame_shape, bg_color):
        """创建背景tensor"""
        batch_size, channels, height, width = frame_shape
        bg_tensor = torch.zeros(frame_shape, dtype=torch.float32)
        
        # 设置RGB通道
        for c in range(min(3, channels)):
            bg_tensor[:, c, :, :] = bg_color[c]
        
        return bg_tensor
    
    def _apply_blend(self, foreground, background, alpha, blend_mode):
        """应用混合模式和透明度"""
        # 应用透明度
        blended = foreground * alpha + background * (1 - alpha)
        
        # 应用混合模式
        if blend_mode == "multiply":
            blended = foreground * background
        elif blend_mode == "screen":
            blended = 1 - (1 - foreground) * (1 - background)
        elif blend_mode == "overlay":
            # 简化版overlay混合
            blended = torch.where(background < 0.5, 
                                2 * foreground * background,
                                1 - 2 * (1 - foreground) * (1 - background))
        elif blend_mode == "soft_light":
            # 简化版soft light混合
            blended = torch.where(background < 0.5,
                                background * (2 * foreground - 1),
                                background * (2 * foreground - 1) + 2 * background * (1 - foreground))
        
        return blended
    
    def _create_alpha_mask(self, frame_shape, alpha_ratio):
        """创建透明度遮罩"""
        batch_size, channels, height, width = frame_shape
        mask = torch.full((1, height, width), alpha_ratio, dtype=torch.float32)
        return mask
    
    def _calculate_alpha_ratios(self, frames, start_alpha, middle_alpha, end_alpha, middle_frame):
        """计算每帧的透明系数 - 支持三段式变化"""
        ratios = []
        
        # 确保中间帧数不超过总帧数
        middle_frame = min(middle_frame, frames - 1)
        
        # 计算每段的帧数
        first_segment_frames = middle_frame
        second_segment_frames = frames - middle_frame
        
        # 第一段：从起始透明度到中间透明度
        for i in range(first_segment_frames):
            if first_segment_frames > 1:
                progress = i / (first_segment_frames - 1)
                current_alpha = start_alpha + (middle_alpha - start_alpha) * progress
            else:
                current_alpha = start_alpha
            ratios.append(current_alpha)
        
        # 第二段：从中间透明度到结束透明度
        for i in range(second_segment_frames):
            if second_segment_frames > 1:
                progress = i / (second_segment_frames - 1)
                current_alpha = middle_alpha + (end_alpha - middle_alpha) * progress
            else:
                current_alpha = middle_alpha
            ratios.append(current_alpha)
        
        return ratios


class YCImageBatchBlend:
    """多批次图像混合节点 - 支持BCHW格式的多批次图像混合"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 第一组图像批次
                "image2": ("IMAGE",),  # 第二组图像批次
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "add", "subtract"], {"default": "normal"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_images",)
    FUNCTION = "batch_blend"
    CATEGORY = "YC_VideoCutHelper/Image"

    def batch_blend(self, image1, image2, blend_factor, blend_mode):
        """多批次图像混合处理"""
        # 确保两个图像批次具有相同的帧数
        frames1 = image1.shape[0]
        frames2 = image2.shape[0]
        
        # 如果帧数不同，取较小的帧数
        min_frames = min(frames1, frames2)
        
        # 截取相同帧数进行混合
        img1 = image1[:min_frames]
        img2 = image2[:min_frames]
        
        # 确保通道数匹配
        channels1 = img1.shape[1]
        channels2 = img2.shape[1]
        
        # 如果通道数不同，统一到3通道
        if channels1 != channels2:
            if channels1 == 4:  # RGBA
                img1 = img1[:, :3, :, :]  # 只取RGB通道
            elif channels1 == 1:  # 灰度
                img1 = img1.repeat(1, 3, 1, 1)  # 复制到3通道
                
            if channels2 == 4:  # RGBA
                img2 = img2[:, :3, :, :]  # 只取RGB通道
            elif channels2 == 1:  # 灰度
                img2 = img2.repeat(1, 3, 1, 1)  # 复制到3通道
        
        # 应用混合模式
        if blend_mode == "normal":
            blended = self._normal_blend(img1, img2, blend_factor)
        elif blend_mode == "multiply":
            blended = self._multiply_blend(img1, img2)
        elif blend_mode == "screen":
            blended = self._screen_blend(img1, img2)
        elif blend_mode == "overlay":
            blended = self._overlay_blend(img1, img2)
        elif blend_mode == "soft_light":
            blended = self._soft_light_blend(img1, img2)
        elif blend_mode == "add":
            blended = self._add_blend(img1, img2)
        elif blend_mode == "subtract":
            blended = self._subtract_blend(img1, img2)
        else:
            blended = self._normal_blend(img1, img2, blend_factor)
        
        return (blended,)
    
    def _normal_blend(self, img1, img2, factor):
        """标准混合模式"""
        return img1 * (1 - factor) + img2 * factor
    
    def _multiply_blend(self, img1, img2):
        """正片叠底混合"""
        return img1 * img2
    
    def _screen_blend(self, img1, img2):
        """滤色混合"""
        return 1 - (1 - img1) * (1 - img2)
    
    def _overlay_blend(self, img1, img2):
        """叠加混合"""
        # 使用img1作为混合层，img2作为基底层
        return torch.where(img2 < 0.5,
                          2 * img1 * img2,
                          1 - 2 * (1 - img1) * (1 - img2))
    
    def _soft_light_blend(self, img1, img2):
        """柔光混合"""
        # 使用img1作为混合层，img2作为基底层
        return torch.where(img2 < 0.5,
                          img2 * (2 * img1 - 1),
                          img2 * (2 * img1 - 1) + 2 * img2 * (1 - img1))
    
    def _add_blend(self, img1, img2):
        """加法混合"""
        return torch.clamp(img1 + img2, 0, 1)
    
    def _subtract_blend(self, img1, img2):
        """减法混合"""
        return torch.clamp(img1 - img2, 0, 1)


NODE_CLASS_MAPPINGS = {
    "YCImagePushPullLens": YCImagePushPullLens,
    "YCImageOverlayBlend": YCImageOverlayBlend,
    "YCImageBatchBlend": YCImageBatchBlend,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YCImagePushPullLens": "Image Push Pull Lens",
    "YCImageOverlayBlend": "Image Overlay Blend",
    "YCImageBatchBlend": "Image Batch Blend",
}
