#! /usr/bin/env python
"""Utilities for normalizing angles and finding angular distances

Author: Mohak Bhardwaj
Date: Oct. 12, 2017

"""
import numpy as np

def normalize_angle_positive(angle):
  """Given an angle (in radians), normalizes the angle to lie [0, 2*pi)

  @param: angle in radians
  @returns: Normalized angle in randians
  """
  PI = np.pi
  return ((angle%(2.0*PI)) + 2.0*PI)%(2.0*PI)


def normalize_angle(angle):
  """Given an angle (in radians), normalizes the angle to lie (-pi, pi]

  @param: angle in radians
  @returns: Normalized angle in randians
  """
  PI = np.pi
  ang = normalize_angle_positive(angle)
  if ang > PI:
    ang -= 2*PI
  return ang

def angular_distance(ang1, ang2):
  """Given two angles it returns the angular distnace between them normalized between (-pi, pi]
  """
  return normalize_angle(ang2-ang1)
