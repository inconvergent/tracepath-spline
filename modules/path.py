# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from numpy import cos
from numpy import sin
from numpy import pi
from numpy import arctan2
from numpy import linspace
from numpy import array
from numpy import row_stack
from numpy import column_stack

from helpers import myrandom

from scipy import interpolate

from scipy.spatial import cKDTree


class Path(object):

  def __init__(self,xy,r):

    self.xy = xy
    self.r = r
    self.tree = cKDTree(xy)

  def trace(self,the):

    r = self.r
    numxy = len(self.xy)

    all_near_inds = self.tree.query_ball_point(self.xy,r)

    near_last = []
    circles = []
    for k,inds in enumerate(all_near_inds):

      ii = set(inds)
      isect = ii.intersection(near_last)

      if len(isect)<5:
        near_last = inds
        circles.append((k,inds))

    ## attach last node.
    if circles[-1][0] < numxy-1:

      circles.append((numxy-1,all_near_inds[-1]))

    alphas = []
    for k,inds in circles:

      ## TODO: test average angle?
      inds_s = array(sorted(inds),'int')
      xy_diff_sum = self.xy[inds_s[-1],:]-self.xy[inds_s[0],:]

      #xy_diff = self.xy[inds_s[1:],:]-self.xy[inds_s[:-1],:]
      #xy_diff_sum = np.sum(xy_diff,axis=0)

      alpha = arctan2(xy_diff_sum[1],xy_diff_sum[0])
      alphas.append(alpha)

    alphas = array(alphas) + the

    xy_circles = row_stack([self.xy[k,:] for k,_ in circles])
    dx = cos(alphas)
    dy = sin(alphas)
    xy_new  = xy_circles[:,:] + column_stack((dx*r,dy*r))

    self.xy_circles = xy_circles
    self.xy_new = xy_new

  def noise(self):

    alpha_noise = myrandom(len(self.xy_new))*pi
    noise = column_stack([cos(alpha_noise),
                          sin(alpha_noise)])*self.r*0.13
    self.xy_new += noise

  def interpolate(self,num_p_multiplier):

    num_points = len(self.xy_circles)*num_p_multiplier

    tck,u = interpolate.splprep([self.xy_new[:,0],
                                 self.xy_new[:,1]],s=0)

    unew = linspace(0,1,num_points)
    out = interpolate.splev(unew,tck)

    self.xy_interpolated = column_stack(out)

