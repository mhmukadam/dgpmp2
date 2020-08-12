import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(".."))
from diff_gpmp2.gp_factor import GPFactor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#If we start at (x0, y0) with velocity vx, vy we should end up near x1 = x0 + vx*delta_t, y1 = y0 + vy*delta_t with noise Q_c
num_states = 5
dof = 2
delta_t = 1
Q_c = np.eye(dof)
gp_fac = GPFactor(dof, delta_t, Q_c)
phi = gp_fac.calc_phi()
Q = gp_fac.get_cov()
Qinv = gp_fac.get_inv_cov()

x0 = 0
y0 = 0
vx = 10
vy = 10

xs = np.linspace(x0 + vx*delta_t - 2.0, x0 + vx*delta_t + 2.0, 50)
ys = np.linspace(y0 + vy*delta_t - 2.0, y0 + vy*delta_t + 2.0, 50)
theta_0  = np.array([x0, y0, vx, vy])
theta_1  = np.array([[x,y, vx, vy] for x in xs for y in ys])

prob_vals = []
for theta in theta_1:
  prob_vals.append(gp_fac.eval_prob(theta_0, theta))
fig = plt.figure()
ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
xs = [th[0] for th in theta_1]
ys = [th[1] for th in theta_1] 
ax.plot_trisurf(xs, ys, prob_vals)

print 'Phi: ', phi
print 'Q: ', Q
print 'Qinv: ', Qinv
plt.show()