#!/usr/bin/env python
import sys
import os
sys.path.insert(0, "..")
import matplotlib.pyplot as plt
import numpy as np
import pprint
import scipy
import time
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D
from diff_gpmp2.gpmp2_planner import GPMP2Planner


np.set_printoptions(threshold=np.nan, linewidth=np.inf, formatter={'float_kind':'{:f}'.format})
pp = pprint.PrettyPrinter()
#Load the environment
ENV_FILE = os.path.abspath("../diff_gpmp2/env/simple_2d/1.png")
env = Env2D()
env_params = dict()
env_params['y_lims'] = [-20, 20]
env_params['x_lims'] = [-20, 20]
env.initialize(ENV_FILE, env_params)
env.calculate_signed_distance_transform()
env.plot_signed_distance_transform()

dof = 2
total_time_sec = 1
total_time_step = 3
total_check_step = 50
planner_params = dict()
planner_params['dof'] = dof
planner_params['state_dim'] = 2*dof #(x, y, xdot, ydot)
planner_params['total_time_sec'] = total_time_sec; #Total time of trajectory
planner_params['total_time_step'] = total_time_step; #Number of timesteps
planner_params['total_check_step'] = total_check_step; 
planner_params['use_gp_inter'] = False #use_gp_interpolation

#2D Point robot model
sphere_radius = 1
robot = PointRobot2D(sphere_radius)

#GP paramters
Q_c = np.eye(dof)
K_s = 0.0001*np.eye(2*dof) #Covariance on start
K_g = 0.0001*np.eye(2*dof) #Covariance on goal
gp_params = dict()
gp_params['Q_c'] = Q_c
gp_params['K_s'] = K_s
gp_params['K_g'] = K_g 

#Obstacle cost settings
obs_params = dict()
obs_params['cost_sigma'] = 0.3
obs_params['epsilon_dist'] = 2

if dof == 1:
  start_conf = np.array([-15])
  start_vel = np.array([0])
  goal_conf = np.array([17])
  goal_vel = np.array([0])
  avg_vel = (goal_conf - start_conf)/total_time_sec
  start = np.concatenate((start_conf, start_vel))
  goal = np.concatenate((goal_conf, goal_vel))

elif dof == 2:
  start_conf = np.array([-15, -8])
  start_vel = np.array([0, 0])
  goal_conf = np.array([17, 14])
  goal_vel = np.array([0, 0])
  avg_vel = (goal_conf - start_conf)/total_time_sec
  start = np.concatenate((start_conf, start_vel))
  goal = np.concatenate((goal_conf, goal_vel))

th_init = np.zeros((planner_params['state_dim'], total_time_step+1)) #Straight  line at constant velocity
for i in range(total_time_step+1):
  th_init[0:dof,i] = start_conf*(total_time_step - i)*1./total_time_step*1. + goal_conf * i*1./total_time_step*1.
  th_init[dof:2*dof,i] = avg_vel
  
planner = GPMP2Planner(gp_params, obs_params, planner_params, env, robot)
optim_params = dict()
optim_params['method'] = 'gauss_newton'
optim_params['plan_time'] = np.inf
optim_params['max_iters'] = 1
optim_params['tol_err'] = 1e-2
optim_params['tol_delta'] = 0#1e-2

th_final, k, time_taken = planner.plan(start, goal, th_init, optim_params)


# print('A')
# pp.pprint(A)
# print('b')
# pp.pprint(b)
# print('K')
# pp.pprint(np.round(K,2).diagonal())
