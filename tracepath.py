#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, linspace, zeros, array,\
                  concatenate, delete

from numpy.random import random, normal, randint, shuffle
import numpy as np
import cairo
from time import time as time
from operator import itemgetter
from scipy.spatial import cKDTree
import gtk, gobject

#np.random.seed(1)

PI = pi
TWOPI = pi*2.

SIZE = 1000
ONE = 1./SIZE

BACK = 1.

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

DIST_NEAR_INDICES = np.inf
NUM_NEAR_INDICES = 30
SHIFT_INDICES = 5

W = 0.9
PIX_BETWEEN = 11

#START_X = (1.-W)*0.5
#START_Y = (1.-W)*0.5
START_X = 0.5
START_Y = (1.-W)*0.5

NUMMAX = int(2*SIZE)
NUM_LINES = int(SIZE*W/PIX_BETWEEN)
H = W/NUM_LINES

FILENAME = 'aa'

TURTLE_ANGLE_NOISE = pi*0.05
INIT_TURTLE_ANGLE_NOISE = 0.05

def myrandom(size):

  #res = normal(size=size)
  res = 1.-2.*random(size=size)

  return res


def turtle(sthe,sx,sy,steps):

  XY = zeros((steps,2),'float')
  THE = zeros(steps,'float')

  XY[0,0] = sx
  XY[0,1] = sy
  THE[0] = sthe
  the = sthe

  noise = myrandom(size=steps)*INIT_TURTLE_ANGLE_NOISE
  for k in xrange(1,steps):

    x = XY[k-1,0] + cos(the)*ONE
    y = XY[k-1,1] + sin(the)*ONE
    XY[k,0] = x
    XY[k,1] = y
    THE[k] = the
    the += noise[k]

    if x>X_MAX or x<X_MIN or y>Y_MAX or y<Y_MIN:
      XY = XY[:k,:]
      THE = THE[:k]
      break

  return THE, XY

#def get_near_indices(tree,xy,d,k):

  #dist,data_inds = tree.query(xy,k=k,distance_upper_bound=d,eps=ONE)

  #return dist, data_inds.flatten()


#def alignment(the,dist):

  #dx = cos(the)
  #dy = sin(the)

  #### inverse proporional distance scale
  ##dx = np.sum(dx/dist)
  ##dy = np.sum(dy/dist)

  ### linear distance scale
  #dx = np.sum(dx*(1.-dist))
  #dy = np.sum(dy*(1.-dist))

  #dd = (dx*dx+dy*dy)**0.5

  #return dx/dd,dy/dd


class Render(object):

  def __init__(self,n):

    self.n = n
    self.__init_cairo()

  def __init_cairo(self):

    sur = cairo.ImageSurface(cairo.FORMAT_ARGB32,self.n,self.n)
    ctx = cairo.Context(sur)
    ctx.scale(self.n,self.n)
    ctx.set_source_rgb(BACK,BACK,BACK)
    ctx.rectangle(0,0,1,1)
    ctx.fill()

    self.sur = sur
    self.ctx = ctx

  def line(self,xy):

    self.ctx.set_source_rgba(0,0,0,0.6)
    self.ctx.set_line_width(ONE*3.)

    self.ctx.move_to(xy[0,0],xy[0,1])
    for (x,y) in xy[1:]:
      self.ctx.line_to(x,y)
    self.ctx.stroke()

  def circle(self,xy,r):

    self.ctx.arc(xy[0],xy[1],r,0,TWOPI)
    self.ctx.stroke()


class Path(object):

  def __init__(self,xy):

    self.xy = xy
    self.tree = cKDTree(xy)

  def trace(self,r,the):

    all_near_inds = self.tree.query_ball_point(self.xy,r)

    near_last = []
    circles = []
    for k,inds in enumerate(all_near_inds):

      ii = set(inds)
      isect = ii.intersection(near_last)

      print k,inds,isect

      if len(isect)<1:
        near_last = inds
        circles.append((k,inds))

    self.circles = circles

    xy_circles = [self.xy[k,:] for k,_ in circles]

    return xy_circles


def main():

  render = Render(SIZE)

  the,xy = turtle(0.5*PI,START_X,START_Y,NUMMAX)
  render.line(xy)

  path = Path(xy)

  circles = path.trace(5*ONE,PI)

  for xy in circles:
    render.circle(xy,5*ONE)


  render.sur.write_to_png('{:s}_final.png'.format(FILENAME))


if __name__ == '__main__':
  if False:
    import pstats
    import cProfile
    OUT = 'profile'
    pfilename = '{:s}.profile'.format(OUT)
    cProfile.run('main()',pfilename)
    p = pstats.Stats(pfilename)
    p.strip_dirs().sort_stats('cumulative').print_stats()
  else:
    main()

