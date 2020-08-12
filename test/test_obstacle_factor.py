import sys, os
import numpy as np
import pprint
sys.path.insert(0, os.path.abspath(".."))
from diff_gpmp2.obstacle_factor import ObstacleFactor
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(threshold=np.nan, linewidth=np.inf)
pp = pprint.PrettyPrinter()
#Load the environment
ENV_FILE = os.path.abspath("../diff_gpmp2/env/test_env.png")
env = Env2D()
env_params = dict()
env_params['y_lims'] = [-20, 20]
env_params['x_lims'] = [-20, 20]
env.initialize(ENV_FILE, env_params)
env.calculate_signed_distance_transform()
env.plot_signed_distance_transform()

#2D Point robot model
sphere_radius = 2
robot = PointRobot2D(sphere_radius)

dof = 2
state_dim = 4
eps = 2
cov = np.eye(robot.nlinks)

obs_factor = ObstacleFactor(state_dim, cov, eps, env, robot)
err, H = obs_factor.get_error(np.array([13, 5, 0, 0]))


plt.show()
