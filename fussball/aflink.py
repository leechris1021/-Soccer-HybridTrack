"""
AFLink 轨迹碎片合并模块
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


class AFLink:
    """轨迹碎片合并 (Appearance-Free Link)"""
    
    def __init__(self, max_gap=60, sim_thresh=0.7, dist_thresh=200):
        self.max_gap = max_gap
        self.sim_thresh = sim_thresh
        self.dist_thresh = dist_thresh
        
    def link(self, tracks_data):
        """
        合并断裂的轨迹片段
        tracks_data: list of dicts, 每个 dict 包含:
            - track_id: 轨迹 ID
            - frames: 帧号列表
            - boxes: 边界框列表 [(x1, y1, x2, y2), ...]
        """
        if len(tracks_data) < 2:
            return {}
        
        n = len(tracks_data)
        cost_matrix = np.ones((n, n)) * 1e9
        
        for i, track_a in enumerate(tracks_data):
            for j, track_b in enumerate(tracks_data):
                if i == j:
                    continue
                
                # 检查时间间隙
                end_frame_a = track_a['frames'][-1]
                start_frame_b = track_b['frames'][0]
                gap = start_frame_b - end_frame_a
                
                if gap <= 0 or gap > self.max_gap:
                    continue
                
                # 计算空间距离
                end_box_a = track_a['boxes'][-1]
                start_box_b = track_b['boxes'][0]
                dist = self._compute_distance(end_box_a, start_box_b)
                
                if dist > self.dist_thresh:
                    continue
                
                # 计算运动相似度
                sim = self._compute_motion_similarity(track_a, track_b)
                
                if sim < self.sim_thresh:
                    continue
                
                cost_matrix[i, j] = 1 - sim
        
        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        merge_map = {}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1 - self.sim_thresh:
                merge_map[tracks_data[r]['track_id']] = tracks_data[c]['track_id']
        
        return merge_map
    
    def _compute_distance(self, box_a, box_b):
        """计算两个边界框的中心点欧氏距离"""
        cx_a = (box_a[0] + box_a[2]) / 2
        cy_a = (box_a[1] + box_a[3]) / 2
        cx_b = (box_b[0] + box_b[2]) / 2
        cy_b = (box_b[1] + box_b[3]) / 2
        return np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
    
    def _compute_motion_similarity(self, track_a, track_b):
        """基于速度预测计算运动相似度"""
        boxes_a = track_a['boxes']
        
        if len(boxes_a) >= 2:
            # 计算最后几帧的平均速度
            v_list = []
            for k in range(min(5, len(boxes_a) - 1)):
                box1 = boxes_a[-2 - k]
                box2 = boxes_a[-1 - k]
                v_list.append([box2[0] - box1[0], box2[1] - box1[1]])
            v_a = np.mean(v_list, axis=0)
        else:
            v_a = np.array([0, 0])
        
        gap = track_b['frames'][0] - track_a['frames'][-1]
        last_box = boxes_a[-1]
        pred_x = last_box[0] + v_a[0] * gap
        pred_y = last_box[1] + v_a[1] * gap
        
        first_box_b = track_b['boxes'][0]
        dist = np.sqrt((pred_x - first_box_b[0]) ** 2 + (pred_y - first_box_b[1]) ** 2)
        
        return np.exp(-dist / 100)
