#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import cos, sin, pi, arctan2, sqrt,\
                  square, linspace, zeros, array,\
                  concatenate, delete, row_stack, column_stack,\
                  cumsum

from numpy.random import random, normal, randint, shuffle
from scipy import interpolate

import numpy as np
import cairo, Image
from time import time as time
from scipy.spatial import cKDTree


PI = pi
TWOPI = pi*2.
PIHALF = pi*0.5

SIZE = 2000
ONE = 1./SIZE

BACK = 1.

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

DIST_NEAR_INDICES = np.inf

W = 0.9
PIX_BETWEEN = 10
#PIXNOISE = ONE*2
#PIXMIN = ONE*3

START_X = (1.-W)*0.5
STOP_X = 1.-START_X

START_Y = (1.-W)*0.5
STOP_Y = 1.-START_Y

NUMMAX = int(2*SIZE)
NUM_LINES = int(SIZE*W/PIX_BETWEEN)
H = W/NUM_LINES

FILENAME = './img/img2'
COLOR_PATH = '../colors/shimmering.gif'

INIT_TURTLE_ANGLE_NOISE = 0.
NOISE_SCALE = 2*ONE ## use ~2 for SIZE=20000

GRAINS = 60
ALPHA = 0.1


def myrandom(size):

  #res = normal(size=size)
  res = 1.-2.*random(size=size)

  return res


def turtle(sthe,sx,sy,steps):

  noise = myrandom(size=steps)*INIT_TURTLE_ANGLE_NOISE
  noise[0] = 0.
  the = sthe+cumsum(noise)

  dx = cos(the)*ONE
  dy = sin(the)*ONE

  xy = column_stack(( START_X + cumsum(dx),START_Y + cumsum(dy)))

  return the, xy


def get_limit_indices(xy,top,bottom):

  start = 0
  stop = len(xy)

  top_ymask = (xy[:,1]<top).nonzero()[0]
  if top_ymask.any():
    start = top_ymask.max()

  bottom_ymask = (xy[:,1]>bottom).nonzero()[0]
  if bottom_ymask.any():
    stop = bottom_ymask.min()

  return start, stop


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

    self.__get_colors(COLOR_PATH)
    self.n_colors = len(self.colors)

  def __get_colors(self,f):
    scale = 1./255.
    im = Image.open(f)
    w,h = im.size
    rgbim = im.convert('RGB')
    res = []
    for i in xrange(0,w):
      for j in xrange(0,h):
        r,g,b = rgbim.getpixel((i,j))
        res.append((r*scale,g*scale,b*scale))

    shuffle(res)
    self.colors = res

  def line(self,xy):

    self.ctx.move_to(xy[0,0],xy[0,1])
    for (x,y) in xy[1:]:
      self.ctx.line_to(x,y)
    self.ctx.stroke()

  def circles(self,xy,rr):

    for r,(x,y) in zip(rr,xy):
      self.ctx.arc(x,y,r,0,TWOPI)
      self.ctx.fill()

  def circle(self,xy,r):

    self.ctx.arc(xy[0],xy[1],r,0,TWOPI)
    self.ctx.stroke()

  def sand_paint(self,left,right):

    num_points = max(left.shape[0],right.shape[0])
    #num_points /= 2
    num_points = int(W*SIZE)*5

    left_tck,left_u = interpolate.splprep([left[:,0],\
                                           left[:,1]],s=0)

    steps = np.linspace(0,1,num_points)
    left_out = interpolate.splev(steps,left_tck)
    left_res = column_stack(left_out)

    right_tck,right_u = interpolate.splprep([right[:,0],\
                                             right[:,1]],s=0)

    steps = np.linspace(0,1,num_points)
    right_out = interpolate.splev(steps,right_tck)
    right_res = column_stack(right_out)

    dxx = right_res[:,0]-left_res[:,0]
    dyy = right_res[:,1]-left_res[:,1]
    dd = square(dxx)+square(dyy)
    sqrt(dd,dd)
    aa = arctan2(dyy,dxx)

    #r,g,b = self.colors[ int(random()*self.n_colors) ]
    #self.ctx.set_source_rgba(r,g,b,ALPHA)

    for i,((lx,ly),(rx,ry)) in enumerate(zip(left_res,right_res)):

      #r,g,b = self.colors[ int(((lx+rx)/2*SIZE)%self.n_colors) ]
      #self.ctx.set_source_rgba(r,g,b,ALPHA)

      r,g,b = self.colors[ i%self.n_colors ]
      self.ctx.set_source_rgba(r,g,b,ALPHA)

      a = aa[i]
      d = dd[i]
      scales = random(GRAINS)*d
      xp = lx + scales*cos(a)
      yp = ly + scales*sin(a)

      for x,y in zip(xp,yp):
        self.ctx.rectangle(x,y,ONE,ONE)
        self.ctx.fill()

class Path(object):

  def __init__(self,xy):

    self.xy = xy
    self.tree = cKDTree(xy)

  def trace(self,r,the):

    all_near_inds = self.tree.query_ball_point(self.xy,r)

    numxy = len(self.xy)

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
    noise = column_stack([cos(alpha_noise),\
                          sin(alpha_noise)])*NOISE_SCALE
    self.xy_new += noise

  def interpolate(self,num_p_multiplier):

    num_points = len(self.xy_circles)*num_p_multiplier

    tck,u = interpolate.splprep([self.xy_new[:,0],\
                                 self.xy_new[:,1]],s=0)

    unew = np.linspace(0,1,num_points)
    out = interpolate.splev(unew,tck)

    self.xy_interpolated = column_stack(out)

def main():

  render = Render(SIZE)

  render.ctx.set_line_width(ONE*2.)
  render.ctx.set_source_rgb(0,0,0)

  the,xy = turtle(0.5*PI,START_X,START_Y,NUMMAX)

  pix = PIX_BETWEEN*ONE
  
  draw_start,draw_stop = get_limit_indices(xy,top=START_Y,bottom=STOP_Y)
  last_xy = xy[draw_start:draw_stop,:]

  for i in xrange(NUM_LINES):

    #pix += myrandom(1)*PIXNOISE
    #if pix < PIXMIN:
      #pix = PIXMIN

    path = Path(xy)
    path.trace(pix,-PIHALF)
    path.noise()
    path.interpolate(PIX_BETWEEN*2)

    xy = path.xy_interpolated

    print 'num',i,'tot', NUM_LINES, 'points', len(path.xy_circles)

    #line_rad = random(size=draw_stop-draw_start)*LINE_RAD
    #render.circles(xy[draw_start:draw_stop,:],a[draw_start:draw_stop])

    ## remove nodes above and below canvas
    canvas_start,canvas_stop = get_limit_indices(xy,top=0.,bottom=1.)
    xy = xy[canvas_start:canvas_stop,:]

    ## render nodes above STOP_Y and below START_Y
    draw_start,draw_stop = get_limit_indices(xy,top=START_Y,bottom=STOP_Y)
    #render.ctx.set_source_rgba(0,0,0,0.3)
    #render.line(xy[draw_start:draw_stop,:])

    render.sand_paint(last_xy,xy[draw_start:draw_stop,:]) 
    last_xy = xy[draw_start:draw_stop,:]

    if (xy[:,0]>STOP_X).any():
      break

    if not i%50:

      if i>0:

        fn = '{:s}_{:05d}.png'.format(FILENAME,i)
        print fn
        render.sur.write_to_png(fn)

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

