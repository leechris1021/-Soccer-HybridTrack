"""
Detection 检测框封装
"""
import numpy as np


class Detection:
    """
    单个检测框，包含位置、置信度、类别与外观特征。
    bbox: [x1, y1, x2, y2] (tlbr 格式)
    """

    def __init__(self, bbox, confidence, det_class, feature=None):
        self.bbox = np.array(bbox, dtype=float)   # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.det_class = det_class
        self.feature = feature  # 归一化后的外观特征向量

    def to_tlwh(self):
        """返回 [x, y, w, h]（左上角 + 宽高）"""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])

    def to_xyah(self):
        """返回 [cx, cy, w, h]（中心点 + 宽高），用于卡尔曼滤波"""
        tlwh = self.to_tlwh()
        cx = tlwh[0] + tlwh[2] / 2
        cy = tlwh[1] + tlwh[3] / 2
        return np.array([cx, cy, tlwh[2], tlwh[3]])
