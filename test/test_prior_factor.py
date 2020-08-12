#!/usr/bin/env python
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(".."))
from diff_gpmp2.prior_factor import PriorFactor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#We want to fix the location of a state at x0, y0 with velocity zero
x0 = 0.0
y0 = 0.0
ndims = 4
cov = 0.002*np.eye(ndims)
mean = np.array([x0, y0, 0, 0])
prior_factor = PriorFactor(ndims, mean, cov)

xs = np.linspace(x0 - 0.1, x0 + 0.1, 70)
ys = np.linspace(y0 - 0.1, y0 + 0.1, 70)
theta_1  = np.array([[x,y, 0.0, 0.0] for x in xs for y in ys])

prob_vals = []
for theta in theta_1:
  prob_vals.append(prior_factor.eval_prob(theta))
fig = plt.figure()
ax = Axes3D(fig)#fig.add_subplot(111, projection='3d')
xs = [th[0] for th in theta_1]
ys = [th[1] for th in theta_1] 
ax.plot_trisurf(xs, ys, prob_vals)

print 'Covariance: ', cov
print 'Covinv: ', prior_factor.get_inv_cov()
# print 'Prob density integral', np.sum(prob_vals)/len(prob_vals)

plt.show()