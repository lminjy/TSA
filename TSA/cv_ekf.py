import numpy as np
import matplotlib.pyplot as plt

from cv_gen import segment_num, segment_len, T, m, n, F, G, Q
from observation import Sensor, Observation, Track
from filter import Filter, SystemModel

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

# model1 = SystemModel(F, m, G, Q, n, R1, T, segment_num, segment_len, sensor1)
# filter1 = Filter(model1)
# x0 = np.array([z_s1[:, 0, 0], (z_s1[:, 0, 1] - z_s1[:, 0, 0])/T, z_s1[:, 1, 0], (z_s1[:, 1, 1] - z_s1[:, 1, 0])/T])
# P0 = np.tile(np.diag([1000**2, 50**2, 1000**2, 50**2]), (segment_num, 1, 1))
# filter1.evaluate(x0.T, P0, z1, groud_truth)
# filter1.RTS_CV()

# filter1.show()

# plt.figure()
# plt.scatter(z_s1[:, 0, :].T, z_s1[:, 1, :].T)
# plt.plot(groud_truth[:, 0, :].T, groud_truth[:, 2, :].T, linewidth=1.0, linestyle='-')
# plt.plot(filter1.X_estimate_s[:, 0, :].T, filter1.X_estimate_s[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*', markersize=3.0)
# plt.show()

te = 20
z1o = z1[:, :, :te]
z_s1o = z_s1[:, :, :te]

model1o = SystemModel(F, m, G, Q, n, R1, T, segment_num, te, sensor1)
filter1o = Filter(model1o)
x0 = np.array([z_s1[:, 0, 0], (z_s1[:, 0, 1] - z_s1[:, 0, 0])/T, z_s1[:, 1, 0], (z_s1[:, 1, 1] - z_s1[:, 1, 0])/T])
P0 = np.tile(np.diag([1000**2, 50**2, 1000**2, 50**2]), (segment_num, 1, 1))
filter1o.initSequence(x0.T, P0)
filter1o.EKF_CV(z1o)

# plt.figure()
# plt.scatter(z_s1[:, 0, :].T, z_s1[:, 1, :].T)
# plt.plot(groud_truth[:, 0, :].T, groud_truth[:, 2, :].T, linewidth=1.0, linestyle='-')
# plt.plot(filter1o.X_estimate[:, 0, :].T, filter1o.X_estimate[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*', markersize=3.0)
# plt.show()

ts = 30
z1n = z1[:, :, ts:]
z_s1n = z_s1[:, :, ts:]

model1n = SystemModel(F, m, G, Q, n, R1, T, segment_num, segment_len - ts, sensor1)
filter1n = Filter(model1n)
x0 = np.array([z_s1[:, 0, ts], (z_s1[:, 0, ts+1] - z_s1[:, 0, ts])/T, z_s1[:, 1, ts], (z_s1[:, 1, ts+1] - z_s1[:, 1, ts])/T])
P0 = np.tile(np.diag([1000**2, 50**2, 1000**2, 50**2]), (segment_num, 1, 1))
filter1n.initSequence(x0.T, P0)
filter1n.EKF_CV(z1n)
filter1n.RTS_CV()

# plt.figure()
# plt.scatter(z_s1[:, 0, :].T, z_s1[:, 1, :].T)
# plt.plot(groud_truth[:, 0, :].T, groud_truth[:, 2, :].T, linewidth=1.0, linestyle='-')
# plt.plot(filter1n.X_estimate_s[:, 0, :].T, filter1n.X_estimate_s[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*', markersize=3.0)
# plt.show()

xs = filter1n.X_estimate_s[:, :, 0]
Ps = filter1n.P_s[:, :, :, 0]
interval = ts - te
filter1n.prediction_without_observation(interval, xs, Ps)

# plt.figure()
# plt.scatter(z_s1[:, 0, :].T, z_s1[:, 1, :].T)
# plt.plot(groud_truth[:, 0, :].T, groud_truth[:, 2, :].T, linewidth=1.0, linestyle='-')
# plt.plot(filter1n.X_estimate_bwd[:, 0, :].T, filter1n.X_estimate_bwd[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*', markersize=3.0)
# plt.show()

