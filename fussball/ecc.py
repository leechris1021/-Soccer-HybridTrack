"""
ECC 摄像机运动补偿模块
"""
import cv2
import numpy as np


class ECCMotionCompensation:
    """增强相关系数 (ECC) 运动补偿，用于修正摄像机移动"""
    
    def __init__(self, scale=0.75, max_iter=200, eps=1e-4):
        self.scale = scale
        self.max_iter = max_iter
        self.eps = eps
        self.prev_frame = None
        self.H = np.eye(3, dtype=np.float32)
        
    def compensate(self, frame):
        """计算当前帧相对于上一帧的变换矩阵"""
        if self.prev_frame is None:
            self.prev_frame = frame
            return self.H
            
        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.scale < 1.0:
            h, w = prev_gray.shape
            new_size = (int(w * self.scale), int(h * self.scale))
            prev_gray = cv2.resize(prev_gray, new_size)
            curr_gray = cv2.resize(curr_gray, new_size)
            
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.max_iter, self.eps)
        
        try:
            _, warp_matrix = cv2.findTransformECC(
                prev_gray, curr_gray, warp_matrix, warp_mode, criteria, None, 1
            )
            self.H = np.vstack([warp_matrix, [0, 0, 1]])
        except cv2.error:
            self.H = np.eye(3, dtype=np.float32)
            
        self.prev_frame = frame
        return self.H
        
    def apply(self, points):
        """将变换矩阵应用到坐标点"""
        if points.shape[0] == 0:
            return points
        points_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed = points_homo @ self.H.T
        return transformed[:, :2]
        
    def apply_to_bbox(self, bbox):
        """将变换矩阵应用到边界框 [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = bbox
        points = np.array([[x1, y1], [x2, y2]])
        transformed = self.apply(points)
        return [transformed[0, 0], transformed[0, 1], transformed[1, 0], transformed[1, 1]]
    
    def reset(self):
        """重置状态"""
        self.prev_frame = None
        self.H = np.eye(3, dtype=np.float32)
