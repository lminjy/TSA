import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

from cv_ekf import filter1o, filter1n, m

old_track_num = len(filter1o.X_estimate)
new_track_num = len(filter1n.X_estimate)

## association
x_delta = np.zeros([old_track_num, new_track_num, m])
P_cov = np.zeros([old_track_num, new_track_num, m, m])
cost = np.zeros([old_track_num, new_track_num])
for i in range(old_track_num):
    for j in range(new_track_num):
        x_delta[i, j, :] = filter1n.X_estimate_bwd[j, :, -1] - filter1o.X_estimate[i, :, -1]
        P_cov[i, j, :, :] = filter1n.P_bwd[j, :, :, -1] + filter1n.P_bwd[i, :, :, -1]
        cost[i, j] = 0.5 * (x_delta[i, j, :].T @ np.linalg.pinv(P_cov[i, j, :, :]) @ x_delta[i, j, :] + np.log(2 * np.pi * np.linalg.det(P_cov[i, j, :, :])))

x = cp.Variable((old_track_num, new_track_num), integer=True)
obj = cp.Minimize(cp.sum(cp.multiply(cost, x)))
con= [0 <= x, x <= 1, cp.sum(x, axis=0, keepdims=True)==1,
             cp.sum(x, axis=1, keepdims=True)==1]
prob = cp.Problem(obj, con)
prob.solve(solver='GLPK_MI')
print("最优解为：\n", x.value)

