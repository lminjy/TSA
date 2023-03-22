import numpy as np
import matplotlib.pyplot as plt

# 航迹数目
segment_num = 3
# 航迹时长
segment_len = 50
# 状态向量维度
m = 4
# 量测向量维度
n = 2

# 采样周期
T = 2

# 状态转移矩阵
F = np.array([[1, T, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, T],
              [0, 0, 0, 1]])

# 过程噪声转移矩阵
G = np.array([[pow(T, 2)*0.5, 0],
              [T, 0],
              [0, pow(T, 2)*0.5],
              [0, T]])

# 过程噪声均值
w_mu = np.array([0, 0])

# 过程噪声协方差矩阵
sigma_x = 0.01
sigma_y = 0.01
Q = np.diag([sigma_x, sigma_y])

# 区域大小
area = np.array([-5000, 5000, -5000, 5000])

# # 生成航迹初始状态
# x = np.array([1e5, 1e5, 1e5])
# y = np.array([1e3, 2e3, 3e3])
# vx = np.array([200, 200, 200])
# vy = np.array([60, 40, 20])
# # 状态：位置x,速度vx；位置y,速度vy
# X_state = np.array([x, vx, y, vy])

# 生成航迹初始状态
x = area[0] + (area[1] - area[0])*np.random.rand(segment_num)
y = area[2] + (area[3] - area[2])*np.random.rand(segment_num)
v = 50 + (100-50)*np.random.rand(segment_num)
v_angle = 2*np.pi*np.random.rand(segment_num)
vx = v * np.cos(v_angle)
vy = v * np.sin(v_angle)
# 状态：位置x,速度vx；位置y,速度vy
X_state = np.array([x, vx, y, vy])

# 生成真实航迹
actual_track = np.zeros([segment_num, m, segment_len])
actual_track[:, :, 0] = X_state.T
for i in range(1, segment_len):
    # X_state = F @ X_state
    # actual_track[:, :, i] = X_state.T
    w = np.random.multivariate_normal(w_mu, Q, segment_num)
    if segment_num == 1:
        w = w.reshape(-1, 1)
        X_state = F @ X_state + G @ w
    else:
        X_state = F @ X_state + G @ w.T
    actual_track[:, :, i] = X_state.T


plt.plot(actual_track[:, 0, :].T, actual_track[:, 2, :].T, marker='*') # 行向量
plt.show()

# (segment_num, state_n, segment_len)
np.save('cv', actual_track)
