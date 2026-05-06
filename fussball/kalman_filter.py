"""
扩展卡尔曼滤波器 (Extended Kalman Filter)
状态向量: [x, y, w, h, vx, vy, vw, vh]
"""
import numpy as np


class ExtendedKalmanFilter:
    """
    针对边界框目标跟踪的扩展卡尔曼滤波器。
    状态向量: x = [cx, cy, w, h, vx, vy, vw, vh]^T
    观测向量: z = [cx, cy, w, h]^T
    """

    def __init__(self):
        ndim = 4  # 观测维度
        dt = 1.0  # 时间步长（帧）

        # 状态转移矩阵 F (8x8), 匀速运动模型
        self.F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # 观测矩阵 H (4x8)
        self.H = np.eye(ndim, 2 * ndim)

        # 过程噪声协方差 Q
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

        # 测量噪声协方差 R
        self._std_weight_pos_meas = 1.0 / 20

    def initiate(self, measurement):
        """
        初始化卡尔曼状态。
        measurement: [cx, cy, w, h]
        返回: (mean, covariance)
        """
        mean_pos = measurement.copy()
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        预测下一帧状态。
        返回: (predicted_mean, predicted_covariance)
        """
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
        ]
        Q = np.diag(np.square(np.concatenate([std_pos, std_vel])))

        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + Q
        return mean, covariance

    def project(self, mean, covariance):
        """
        将状态投影到观测空间。
        返回: (projected_mean, projected_covariance)
        """
        std = [
            self._std_weight_pos_meas * mean[2],
            self._std_weight_pos_meas * mean[3],
            self._std_weight_pos_meas * mean[2],
            self._std_weight_pos_meas * mean[3],
        ]
        R = np.diag(np.square(std))
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + R
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """
        用观测值更新状态。
        measurement: [cx, cy, w, h]
        返回: (updated_mean, updated_covariance)
        标准卡尔曼更新: K = P H^T (H P H^T + R)^{-1}
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        # P H^T : (8x8)@(8x4)^T -> (8x4)... H.T shape=(8,4), covariance shape=(8,8)
        # K = (P H^T) @ inv(H P H^T + R)  shape: (8,4)
        PHt = covariance @ self.H.T                      # (8, 4)
        kalman_gain = PHt @ np.linalg.inv(projected_cov) # (8, 4)

        innovation = measurement - projected_mean        # (4,)
        new_mean = mean + kalman_gain @ innovation       # (8,)
        new_covariance = covariance - kalman_gain @ self.H @ covariance  # (8,8)
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements):
        """
        计算马氏距离（用于运动特征代价）。
        measurements: shape (N, 4)
        返回: shape (N,) 马氏距离
        """
        projected_mean, projected_cov = self.project(mean, covariance)
        diff = measurements - projected_mean
        chol = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(chol, diff.T)
        return np.sum(z ** 2, axis=0)
