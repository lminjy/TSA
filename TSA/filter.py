import numpy as np
import matplotlib.pyplot as plt

class Filter:
    def __init__(self, SystemModel):
        self.F = SystemModel.F
        self.m = SystemModel.m

        self.G = SystemModel.G
        self.Q = SystemModel.Q

        # self.H = SystemModel.H
        self.H = np.zeros([SystemModel.n, SystemModel.m])
        self.n = SystemModel.n

        self.R = SystemModel.R

        self.T = SystemModel.T

        self.sensor = SystemModel.sensor
        self.segment_num = SystemModel.segment_num
        self.segment_len = SystemModel.segment_len

    def initSequence(self, x0, P0):
        self.X_estimate = np.zeros([self.segment_num, self.m, self.segment_len])
        self.P = np.zeros([self.segment_num, self.m, self.m, self.segment_len])
        self.X_estimate[:, :, 0] = x0
        self.P[:, :, :, 0] = P0

        self.Xekf_pre = np.zeros([self.segment_num, self.m, self.segment_len])
        self.P_pre = np.zeros([self.segment_num, self.m, self.m, self.segment_len])

    def prediction(self, j, i):
        # 状态预测
        self.Xekf_pre[j, :, i] = self.F @ self.X_estimate[j, :, i-1]
        # 协方差阵预测
        self.P_pre[j, :, :, i] = self.F @ self.P[j, :, :, i-1] @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, Z, j, i):
        Z_pre = np.array([np.sqrt(pow((self.Xekf_pre[j, 2, i]-self.sensor.y), 2) + pow((self.Xekf_pre[j, 0, i]-self.sensor.x), 2)),
                          np.arctan2((self.Xekf_pre[j, 2, i] - self.sensor.y), (self.Xekf_pre[j, 0, i] - self.sensor.x))])
        y = Z - Z_pre

        # 泰勒一阶近似量测矩阵
        r = pow((self.Xekf_pre[j, 2, i] - self.sensor.y), 2) + pow((self.Xekf_pre[j, 0, i] - self.sensor.x), 2)
        self.H[0, 0] = (self.Xekf_pre[j, 0, i] - self.sensor.x) / np.sqrt(r)
        self.H[0, 2] = (self.Xekf_pre[j, 2, i] - self.sensor.y) / np.sqrt(r)
        self.H[1, 0] = -(self.Xekf_pre[j, 2, i] - self.sensor.y) / r
        self.H[1, 2] = (self.Xekf_pre[j, 0, i] - self.sensor.x) / r

        S = self.H @ self.P_pre[j, :, :, i] @ self.H.T + self.R

        self.K = self.P_pre[j, :, :, i] @ self.H.T @ np.linalg.pinv(S)

        self.X_estimate[j, :, i] = self.Xekf_pre[j, :, i] + self.K @ y
        self.P[j, :, :, i] = self.P_pre[j, :, :, i] - self.K @ self.H @ self.P_pre[j, :, :, i]

    def EKF_CV(self, Z):
        for i in range(1, self.segment_len):
            for j in range(self.segment_num):
                self.prediction(j, i)
                self.update(Z[j, :, i], j, i)

    def prediction_without_observation(self, interval, xs, Ps):
        T_bwd = -self.T
        F_bwd = np.array([[1, T_bwd, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, T_bwd],
                          [0, 0, 0, 1]])
        self.X_estimate_bwd = np.zeros([self.segment_num, self.m, interval+1])
        self.P_bwd = np.zeros([self.segment_num, self.m, self.m, interval+1])
        self.X_estimate_bwd[:, :, 0] = xs
        self.P_bwd[:, :, :, 0] = Ps
        for i in range(1, interval+1):
            for j in range(self.segment_num):
                self.X_estimate_bwd[j, :, i] = F_bwd @ self.X_estimate_bwd[j, :, i - 1]
                # 协方差阵预测
                self.P_bwd[j, :, :, i] = F_bwd @ self.P_bwd[j, :, :, i - 1] @ F_bwd + self.G @ self.Q @ self.G.T

    def RTS_CV(self):
        self.X_estimate_s = np.zeros([self.segment_num, self.m, self.segment_len])
        self.P_s = np.zeros([self.segment_num, self.m, self.m, self.segment_len])
        self.X_estimate_s[:, :, -1] = self.X_estimate[:, :, -1]
        self.P_s[:, :, :, -1] = self.P[:, :, :, -1]

        for i in range(self.segment_len - 1, 0, -1):
            for j in range(self.segment_num):
                G_s = self.P[j, :, :, i-1] @ self.F.T @ np.linalg.pinv(self.P_pre[j, :, :, i])
                self.X_estimate_s[j, :, i-1] = self.X_estimate[j, :, i-1] + G_s @ (self.X_estimate_s[j, :, i] - self.Xekf_pre[j, :, i])
                self.P_s[j, :, :, i - 1] = self.P[j, :, :, i - 1] + G_s @ (self.P_s[j, :, :, i] - self.P[j, :, :, i]) @ G_s.T


    def evaluate(self, x0, P0, Z, ground_truth):
        mc = 50
        self.error_Kalman = np.zeros([self.segment_num, self.segment_len])
        self.error_Kalman_v = np.zeros([self.segment_num, self.segment_len])

        for k in range(mc):
            self.initSequence(x0, P0)
            self.EKF_CV(Z)

            # 比较误差
            for i in range(self.segment_len):
                self.error_Kalman[:, i] += pow((self.X_estimate[:, 0, i] - ground_truth[:, 0, i]), 2) + pow((self.X_estimate[:, 2, i] - ground_truth[:, 2, i]), 2)
                self.error_Kalman_v[:, i] += pow((self.X_estimate[:, 1, i] - ground_truth[:, 1, i]), 2) + pow((self.X_estimate[:, 3, i] - ground_truth[:, 3, i]), 2)

        self.error_Kalman = self.error_Kalman / mc
        self.error_Kalman = np.sqrt(self.error_Kalman)

        self.error_Kalman_v = self.error_Kalman_v / mc
        self.error_Kalman_v = np.sqrt(self.error_Kalman_v)

    def show(self):
        plt.figure()
        linespec = np.tile(np.arange(self.segment_len), [self.segment_num, 1])
        plt.plot(linespec.T, self.error_Kalman.T, marker='*', markersize=3.0)
        plt.figure()
        plt.plot(linespec.T, self.error_Kalman_v.T, marker='*', markersize=3.0)
        plt.show()

class SystemModel:

    def __init__(self, F, m, G, Q, n, R, T, segment_num, segment_len, sensor):
        self.F = F
        self.m = m

        self.G = G
        self.Q = Q

        # self.H = SystemModel.H
        self.n = n

        self.R = R

        self.T = T

        self.sensor = sensor
        self.segment_num = segment_num
        self.segment_len = segment_len
