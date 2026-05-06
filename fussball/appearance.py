"""
OSNet / ResNet50 外观特征提取器
支持两种骨干网络：OSNet-x0.75 和 ResNet50
优先使用仓库内微调权重，缺失时回退到 ImageNet 预训练
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .project import DEFAULT_OSNET_WEIGHTS

DEFAULT_RESNET_WEIGHTS = None  # 使用 ImageNet 预训练


class AppearanceExtractor:
    """
    外观特征提取器。
    支持两种模型：
        - 'osnet': OSNet-x0.75 (512维)
        - 'resnet50': ResNet50 (2048维)
    
    优先加载本地微调权重；若文件不存在则回退到 ImageNet 预训练。
    """

    def __init__(self, 
                 model_name='osnet',      # 'osnet' 或 'resnet50'
                 device=None, 
                 weights_path=None):
        
        self.model_name = model_name.lower()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 构建模型
        if self.model_name == 'osnet':
            self._build_osnet(weights_path)
        elif self.model_name == 'resnet50':
            self._build_resnet50(weights_path)
        else:
            raise ValueError(f"不支持的模型类型: {model_name}. 请选择 'osnet' 或 'resnet50'")

        self.model.eval()
        self.model.to(self.device)

        # 图像预处理
        if self.model_name == 'osnet':
            # OSNet 使用 256x128 输入
            self.transform = T.Compose([
                T.Resize((256, 128)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ])
        else:
            # ResNet50 使用 224x224 输入
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
            ])

        print(f"[Re-ID] 模型初始化完成: {self.model_name}, 特征维度={self.feat_dim}, 设备={self.device}")

    def _build_osnet(self, weights_path):
        """构建 OSNet-x0.75 模型"""
        resolved_weights = Path(weights_path).expanduser() if weights_path else DEFAULT_OSNET_WEIGHTS

        try:
            import torchreid
        except Exception as exc:
            raise ImportError(
                "运行 OSNet 外观模型需要先安装并正确配置 torchreid 依赖。"
                "请在当前环境执行: pip install torchreid gdown yacs"
            ) from exc

        self.model = torchreid.models.build_model(
            name='osnet_x0_75',
            num_classes=1,
            pretrained=False,
        )

        # 加载权重
        if resolved_weights.exists():
            print(f"[Re-ID] 加载 OSNet-x0.75 权重: {resolved_weights}")
            state = torch.load(str(resolved_weights), map_location='cpu', weights_only=False)
            
            # 处理不同的权重格式
            if isinstance(state, dict):
                if 'state_dict' in state and isinstance(state['state_dict'], dict):
                    state = state['state_dict']
                
                # 移除 'module.' 前缀和分类头
                filtered = {}
                for key, value in state.items():
                    clean_key = key.replace('module.', '')
                    if clean_key.startswith('classifier'):
                        continue
                    filtered[clean_key] = value
                
                # 尝试加载，允许缺失分类头
                missing, unexpected = self.model.load_state_dict(filtered, strict=False)
                if missing:
                    print(f"  [INFO] missing keys: {len(missing)} (通常是分类头)")
                if unexpected:
                    print(f"  [INFO] unexpected keys: {len(unexpected)}")
            else:
                # 直接是 state_dict
                self.model.load_state_dict(state, strict=False)
            print("  ✓ OSNet 权重加载完成")
        else:
            if weights_path:
                print(f"[Re-ID] 指定权重不存在: {resolved_weights}")
            else:
                print(f"[Re-ID] 默认权重不存在: {resolved_weights}")
            print("[Re-ID] 回退到 ImageNet 预训练")
            
            self.model = torchreid.models.build_model(
                name='osnet_x0_75',
                num_classes=1,
                pretrained=True,
            )

        # 移除分类头（如果还存在）
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Identity()
        
        self.feat_dim = 512

    def _build_resnet50(self, weights_path):
        """构建 ResNet50 模型"""
        from torchvision import models
        
        if weights_path and Path(weights_path).exists():
            print(f"[Re-ID] 加载 ResNet50 权重: {weights_path}")
            self.model = models.resnet50(pretrained=False)
            state = torch.load(weights_path, map_location='cpu', weights_only=False)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            self.model.load_state_dict(state, strict=False)
        else:
            print("[Re-ID] 使用 ImageNet 预训练的 ResNet50")
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 移除分类头，使用全局平均池化后的特征
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.feat_dim = 2048

    @torch.no_grad()
    def extract(self, frame_bgr, bboxes):
        """
        批量提取边界框对应的外观特征
        
        Args:
            frame_bgr: BGR 格式的图像 (H, W, 3)
            bboxes: 边界框列表 [[x1, y1, x2, y2], ...]
        
        Returns:
            features: (N, feat_dim) 归一化的特征向量，已 L2 归一化
        """
        if len(bboxes) == 0:
            return np.zeros((0, self.feat_dim), dtype=np.float32)

        crops = []
        for bbox in bboxes:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                # 无效边界框，返回零向量
                if self.model_name == 'osnet':
                    crops.append(torch.zeros(3, 256, 128))
                else:
                    crops.append(torch.zeros(3, 224, 224))
                continue
            
            crop = frame_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            crops.append(self.transform(pil_img))

        # 批量推理
        batch = torch.stack(crops).to(self.device)
        features = self.model(batch)
        
        # 展平特征（如果还有额外维度）
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        
        features = features.cpu().numpy().astype(np.float32)
        
        # L2 归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-9
        features = features / norms
        
        return features

    def extract_single(self, frame_bgr, bbox):
        """
        提取单个边界框的特征
        
        Args:
            frame_bgr: BGR 格式的图像
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            feature: (feat_dim,) 归一化的特征向量
        """
        features = self.extract(frame_bgr, [bbox])
        if len(features) > 0:
            return features[0]
        return np.zeros(self.feat_dim, dtype=np.float32)


# 便捷函数：快速创建提取器
def create_extractor(model_name='osnet', device=None, weights_path=None):
    """创建 AppearanceExtractor 实例的便捷函数"""
    return AppearanceExtractor(
        model_name=model_name,
        device=device,
        weights_path=weights_path
    )


# 使用示例
if __name__ == '__main__':
    # 测试代码
    import time
    
    # 创建测试图像
    test_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    test_bboxes = [[100, 100, 200, 300], [500, 200, 600, 400]]
    
    # 测试 OSNet
    print("=" * 50)
    print("测试 OSNet 模型")
    extractor = AppearanceExtractor(model_name='osnet')
    
    start = time.time()
    features = extractor.extract(test_frame, test_bboxes)
    elapsed = time.time() - start
    
    print(f"特征形状: {features.shape}")
    print(f"特征范数: {np.linalg.norm(features, axis=1)}")
    print(f"推理时间: {elapsed:.3f}s")
    
    # 测试 ResNet50
    print("\n" + "=" * 50)
    print("测试 ResNet50 模型")
    extractor2 = AppearanceExtractor(model_name='resnet50')
    
    start = time.time()
    features2 = extractor2.extract(test_frame, test_bboxes)
    elapsed = time.time() - start
    
    print(f"特征形状: {features2.shape}")
    print(f"特征范数: {np.linalg.norm(features2, axis=1)}")
    print(f"推理时间: {elapsed:.3f}s")