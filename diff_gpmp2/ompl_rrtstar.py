#!/usr/bin/env python
import numpy as np
import torch
from ompl import base as ob
from ompl import geometric as og
from diff_gpmp2.env.env_2d import Env2D
from diff_gpmp2.robot_models import PointRobot2D




class RRTStar(object):
  def __init__(self, space, bounds, env, robot, planner_params, epsilon_dist):
    self.space = space
    self.bounds = bounds
    self.env = env
    self.robot = robot
    self.eps = robot.get_sphere_radii()[0] + epsilon_dist
    self.ss = og.SimpleSetup(self.space)
    self.planner = og.RRTstar(self.ss.getSpaceInformation())
    self.ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.isStateValid))
    self.ss.setPlanner(self.planner)
    self.num_states = planner_params['total_time_step'] + 1

  def plan(self, start_conf, goal_conf, plan_time):
    # create a simple setup object   
    start = ob.State(self.space)
    goal = ob.State(self.space)
    start[0] = start_conf[0][0].item()
    start[1] = start_conf[0][1].item()
    
    goal[0] = goal_conf[0][0].item()
    goal[1] = goal_conf[0][1].item()
    self.ss.setStartAndGoalStates(start, goal)
    self.ss.setup()
    solved = self.ss.solve(plan_time)
    if solved:
      self.ss.simplifySolution()
    
    ompl_path = self.ss.getSolutionPath()
    ompl_path.interpolate(self.num_states)
    path_states = []
    for s in ompl_path.getStates():
      path_states.append([s[0], s[1]])

    return np.array(path_states)

  def isStateValid(self, state):
    state_torch = torch.tensor([state[0], state[1]])
    return self.env.is_feasible(state_torch, self.eps)


if __name__ == "__main__":
    plan()