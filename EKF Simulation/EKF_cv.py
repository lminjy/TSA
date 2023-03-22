import numpy as np
import matplotlib.pyplot as plt

from cv_gen import F, G, Q, T, segment_num, segment_len, actual_track
from observation import Sensor, Observation, Track

class filter:

    def __init__(self,segment_num, segment_len, T, sensor, F, G, R, z, z_s):
        self.segment_num = segment_num
        self.segment_len = segment_len
        self.T = T
        self.F = F
        self.G = G
        self.Q = Q
        self.sensor = sensor
        self.R = R
        self.z = z
        self.z_s = z_s
        self.H = None
        self.X_estimate = None
        self.P = None


    def EKF_CV(self, Xekf, P_pre, Z):
        # 状态预测
        Xekf_pre = self.F @ Xekf
        # 协方差阵预测
        P1 = self.F @ P_pre @ self.F.T + self.G @ self.Q @ self.G.T
        # 量测预测
        Z_pre = np.array([np.sqrt(pow((Xekf_pre[2]-self.sensor.y), 2) + pow((Xekf_pre[0]-self.sensor.x), 2)), np.arctan2((Xekf_pre[2] - self.sensor.y), (Xekf_pre[0] - self.sensor.x))])
        # 泰勒一阶近似量测矩阵
        self.H = np.array([[(Xekf_pre[0] - self.sensor.x) / np.sqrt(pow((Xekf_pre[2] - self.sensor.y), 2) + pow((Xekf_pre[0] - self.sensor.x), 2)), 0,
             (Xekf_pre[2] - self.sensor.y) / np.sqrt(pow((Xekf_pre[2] - self.sensor.y), 2) + pow((Xekf_pre[0] - self.sensor.x), 2)), 0],
                     [-(Xekf_pre[2] - self.sensor.y) / (pow((Xekf_pre[2] - self.sensor.y), 2) + pow((Xekf_pre[0] - self.sensor.x), 2)), 0,
                      (Xekf_pre[0] - self.sensor.x) / (pow((Xekf_pre[2] - self.sensor.y), 2) + pow((Xekf_pre[0] - self.sensor.x), 2)), 0]])
        # 卡尔曼最优增益
        K = P1 @ self.H.T @ np.linalg.pinv(self.H @ P1 @ self.H.T + self.R)
        # 状态更新，最优估计
        Xekf = Xekf_pre + K @ (Z - Z_pre)
        # 协方差阵更新，最优估计
        P_pre = P1 - K @ self.H @ P1
        return Xekf, P_pre

    def evaluate(self, mc):
        error_Kalman = np.zeros([self.segment_num, self.segment_len])
        error_Kalman_v = np.zeros([self.segment_num, self.segment_len])

        self.X_estimate = np.zeros([self.segment_num, 4, self.segment_len])
        self.P = np.zeros([self.segment_num, 4, 4, self.segment_len])

        for k in range(mc):

            # 滤波初始化
            self.X_estimate[:, :, 0] = np.array([self.z_s[:, 0, 0], (self.z_s[:, 0, 1] - self.z_s[:, 0, 0])/self.T, self.z_s[:, 1, 0], (self.z_s[:, 1, 1] - self.z_s[:, 1, 0])/self.T]).T
            self.P[:, :, :, 0] = np.tile(np.diag([1000**2, 50**2, 1000**2, 50**2]), (segment_num, 1, 1))
            for i in range(1, segment_len):
                for j in range(segment_num):
                    self.X_estimate[j, :, i], self.P[j, :, :, i] = self.EKF_CV(self.X_estimate[j, :, i-1], self.P[j, :, :, i-1], self.z[j, :, i])

            # 比较误差
            for i in range(segment_len):
                error_Kalman[:, i] += pow((self.X_estimate[:, 0, i] - actual_track[:, 0, i]), 2) + pow((self.X_estimate[:, 2, i] - actual_track[:, 2, i]), 2)
                error_Kalman_v[:, i] += pow((self.X_estimate[:, 1, i] - actual_track[:, 1, i]), 2) + pow((self.X_estimate[:, 3, i] - actual_track[:, 3, i]), 2)



        error_Kalman = error_Kalman / mc
        error_Kalman = np.sqrt(error_Kalman)

        error_Kalman_v = error_Kalman_v / mc
        error_Kalman_v = np.sqrt(error_Kalman_v)

        plt.figure()
        linespec = np.tile(np.arange(segment_len), [segment_num, 1])
        plt.plot(linespec.T, error_Kalman.T, marker='*', markersize=3.0)
        plt.figure()
        plt.plot(linespec.T, error_Kalman_v.T, marker='*', markersize=3.0)

# 雷达1量测初始化
sensor1 = Sensor(-2500, -6000, 100, 0.1*np.pi/180)
v_r1 = sensor1.sigma_dist
v_theta1 = sensor1.sigma_angle
# 量测协方差矩阵
R1 = np.diag([pow(v_r1, 2), pow(v_theta1, 2)])

# 雷达1的量测
observation1 = Observation(sensor1, R1)
z1, z_s1 = observation1.get(actual_track)


# 雷达2量测初始化
sensor2 = Sensor(2500, -6000, 50, 0.2*np.pi/180)
v_r2 = sensor2.sigma_dist
v_theta2 = sensor2.sigma_angle
# 量测协方差矩阵
R2 = np.diag([pow(v_r2, 2), pow(v_theta2, 2)])

# 雷达2的量测
observation2 = Observation(sensor2, R2)
z2, z_s2 = observation2.get(actual_track)

# 模特卡洛模拟次数
mc = 100

filter1 = filter(segment_num, segment_len, T, sensor1, F, G, R1, z1, z_s1)
filter1.evaluate(mc)
track1 = Track(z_s1, filter1.X_estimate, filter1.P, sensor1)
track1.show(actual_track)

filter2 = filter(segment_num, segment_len, T, sensor2, F, G, R2, z2, z_s2)
filter2.evaluate(mc)
track2 = Track(z_s1, filter2.X_estimate, filter2.P, sensor2)
track2.show(actual_track)

plt.figure()
plt.plot(sensor1.x, sensor1.y, marker='o', markersize=5.0)
plt.plot(sensor2.x, sensor2.y, marker='o', markersize=5.0)
plt.plot(actual_track[:, 0, :].T, actual_track[:, 2, :].T, linewidth=1.0, linestyle='-', marker='o', markersize=3.0)
plt.plot(track1.X_estimate[:, 0, :].T, track1.X_estimate[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*', markersize=3.0)
plt.plot(track1.X_estimate[:, 0, :].T, track1.X_estimate[:, 2, :].T, linewidth=3.0, linestyle='-')
plt.show()

