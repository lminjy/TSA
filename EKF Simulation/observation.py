import numpy as np
import matplotlib.pyplot as plt
from cv_gen import segment_num, segment_len, n

# 传感器类（2D雷达）
class Sensor:

    # 传感器x，y坐标，测距误差sigma_dist，测角误差sigma_angle
    def __init__(self, x, y, sigma_dist, sigma_angle):

        self.x = x
        self.y = y
        self.sigma_dist = sigma_dist
        self.sigma_angle = sigma_angle

    def show(self):
        plt.plot(self.x, self.y, marker='*', markersize=3.0)
        plt.show()

# 量测协方差阵生成
def Trans_R(sensor, X_state):
    rho = np.sqrt(pow(sensor.x - X_state[0], 2) + pow(sensor.y - X_state[2], 2))
    theta = np.arctan2(sensor.y - X_state[2], sensor.x - X_state[0])
    sigma_x = pow(sensor.sigma_dist * np.cos(theta), 2) + pow(rho * sensor.sigma_angle * np.sin(theta), 2)
    sigma_y = pow(sensor.sigma_dist * np.sin(theta), 2) + pow(rho * sensor.sigma_angle * np.cos(theta), 2)
    sigma_xy = (pow(sensor.sigma_dist, 2) - pow(rho * sensor.sigma_angle, 2)) * np.sin(theta) * np.cos(theta)
    return np.array([[sigma_x, sigma_xy], [sigma_xy, sigma_y]])

class Observation:

    def __init__(self, sensor, R):
        self.sensor = sensor
        self.R = R
        self.z = np.zeros([segment_num, n, segment_len])
        self.z_s = np.zeros([segment_num, n, segment_len])

    # 生成量测
    def get(self, track):
        v_mu = np.array([0, 0])
        v = np.random.multivariate_normal(v_mu, self.R, [segment_num, segment_len])

        # sensor量测
        self.z[:, 0, :] = np.sqrt(pow(track[:, 0, :] - self.sensor.x, 2) + pow(track[:, 2, :] - self.sensor.y, 2)) + v[:, :, 0]
        self.z[:, 1, :] = np.arctan2(track[:, 2, :] - self.sensor.y, track[:, 0, :] - self.sensor.x) + v[:, :, 1]

        # 量测转换到直角坐标系
        self.z_s[:, 0, :] = self.sensor.x + self.z[:, 0, :] * np.cos(self.z[:, 1, :])
        self.z_s[:, 1, :] = self.sensor.y + self.z[:, 0, :] * np.sin(self.z[:, 1, :])
        return self.z, self.z_s

class Track:

    def __init__(self, z_s, X_estimate, P, sensor):
        self.z_s = z_s
        self.X_estimate = X_estimate
        self.P = P
        self.sensor = sensor

    def show(self, actual_track):
        plt.figure()
        plt.plot(self.sensor.x, self.sensor.y, marker='o', markersize=3.0)
        plt.plot(actual_track[:, 0, :].T, actual_track[:, 2, :].T, linewidth=1.0, linestyle='-')
        plt.plot(self.z_s[:, 0, :].T, self.z_s[:, 1, :].T, marker='*', markersize=3.0)
        plt.plot(self.X_estimate[:, 0, :].T, self.X_estimate[:, 2, :].T, linewidth=1.0, linestyle='-.', marker='*',
                 markersize=3.0)
        plt.show()