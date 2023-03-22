import numpy as np
import matplotlib.pyplot as plt

from cv_gen import segment_num, segment_len, T, m, n, F, G, Q
from observation import Sensor, Observation
from filter import Filter

class SystemModel:

    def __init__(self, R, sensor):
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


groud_truth = np.load('cv.npy')

# 雷达1量测初始化
sensor1 = Sensor(-2500, -6000, 50, 0.1*np.pi/180)

v_r1 = sensor1.sigma_dist
v_theta1 = sensor1.sigma_angle
# 量测协方差矩阵
R1 = np.diag([pow(v_r1, 2), pow(v_theta1, 2)])

# 雷达1的量测
observation1 = Observation(sensor1, R1)
z1, z_s1 = observation1.get(groud_truth)
# plt.plot(groud_truth[:, 0, :].T, groud_truth[:, 2, :].T)
# plt.scatter(z_s1[:, 0, :].T, z_s1[:, 1, :].T)
# plt.show()

model1 = SystemModel(R1, sensor1)
filter1 = Filter(model1)
x0 = np.array([z_s1[:, 0, 0], (z_s1[:, 0, 1] - z_s1[:, 0, 0])/T, z_s1[:, 1, 0], (z_s1[:, 1, 1] - z_s1[:, 1, 0])/T])
P0 = np.tile(np.diag([1000**2, 50**2, 1000**2, 50**2]), (segment_num, 1, 1))
filter1.evaluate(x0.T, P0, z1, groud_truth)
filter1.show()

plt.figure()
plt.plot(sensor1.x, sensor1.y, marker='o', markersize=5.0)
plt.scatter(z_s1[:, 0, :].T, z_s1[:, 1, :].T)
plt.plot(groud_truth[:, 0, :].T, groud_truth[:, 2, :].T, linewidth=1.0, linestyle='-')
plt.plot(filter1.X_estimate[:, 0, :].T, filter1.X_estimate[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*', markersize=3.0)
plt.show()

