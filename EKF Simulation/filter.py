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
        self.Xekf = x0
        self.Pekf = P0

    def prediction(self):
        # 状态预测
        self.Xekf_pre = self.F @ self.Xekf
        # 协方差阵预测
        self.P_pre = self.F @ self.Pekf @ self.F.T + self.G @ self.Q @ self.G.T

    def update(self, Z):
        Z_pre = np.array([np.sqrt(pow((self.Xekf_pre[2]-self.sensor.y), 2) + pow((self.Xekf_pre[0]-self.sensor.x), 2)),
                          np.arctan2((self.Xekf_pre[2] - self.sensor.y), (self.Xekf_pre[0] - self.sensor.x))])
        y = Z - Z_pre

        # 泰勒一阶近似量测矩阵
        r = pow((self.Xekf_pre[2] - self.sensor.y), 2) + pow((self.Xekf_pre[0] - self.sensor.x), 2)
        self.H[0, 0] = (self.Xekf_pre[0] - self.sensor.x) / np.sqrt(r)
        self.H[0, 2] = (self.Xekf_pre[2] - self.sensor.y) / np.sqrt(r)
        self.H[1, 0] = -(self.Xekf_pre[2] - self.sensor.y) / r
        self.H[1, 2] = (self.Xekf_pre[0] - self.sensor.x) / r

        S = self.H @ self.P_pre @ self.H.T + self.R

        self.K = self.Pekf @ self.H.T @ np.linalg.pinv(S)

        self.Xekf = self.Xekf_pre + self.K @ y
        self.Pekf = self.P_pre - self.K @ self.H @ self.P_pre

    def EKF_CV(self, Z):
        self.prediction()
        self.update(Z)
        return self.Xekf, self.Pekf

    def evaluate(self, x0, P0, Z, ground_truth):
        mc = 50
        self.error_Kalman = np.zeros([self.segment_num, self.segment_len])
        self.error_Kalman_v = np.zeros([self.segment_num, self.segment_len])

        self.X_estimate = np.zeros([self.segment_num, self.m, self.segment_len])
        self.P = np.zeros([self.segment_num, self.m, self.m, self.segment_len])

        for k in range(mc):
            for j in range(self.segment_num):
                self.initSequence(x0[j, :], P0[j, :, :])
                self.X_estimate[j, :, 0], self.P[j, :, :, 0] = self.Xekf, self.Pekf
                for i in range(self.segment_len):
                    self.X_estimate[j, :, i], self.P[j, :, :, i] = self.EKF_CV(Z[j, :, i])

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




