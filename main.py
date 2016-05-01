#!/usr/bin/python
# -*- coding: utf-8 -*-

from numpy import pi
from numpy import sqrt
from numpy import linspace
from numpy import column_stack
from numpy import ones

from itertools import count


PI = pi
TWOPI = pi*2.
PIHALF = pi*0.5

SIZE = 1500
ONE = 1./SIZE

LINE_WIDTH = 1.5*ONE

BACK = [1,1,1,1]
FRONT = [0,0,0,0.9]

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

W = 0.9
PIX_BETWEEN = 20

START_X = (1.-W)*0.5
STOP_X = 1.-START_X

START_Y = (1.-W)*0.5
STOP_Y = 1.-START_Y

NMAX = int(2*SIZE)

GRAINS = 60
ALPHA = 0.1

FRONT = [0,0,0,0.7]
BACK = [1,1,1,1]


def main():

  from modules.path import Path

  from modules.helpers import get_limit_indices
  from modules.export import Exporter

  from render import Render
  from fn import Fn

  fn = Fn(prefix='./res/')

  exporter = Exporter()


  render = Render(SIZE, BACK, FRONT)
  render.set_line_width(LINE_WIDTH)

  #the = 0.5*PI*ones(NMAX)
  xy = column_stack((ones(NMAX)*START_X,linspace(START_Y,STOP_Y,NMAX)))

  draw_start,draw_stop = get_limit_indices(xy,top=START_Y,bottom=STOP_Y)
  # last_xy = xy[draw_start:draw_stop,:]

  # draw_circles = render.circles

  for i in count():

    ## gradient-like distribution of lines
    pix = sqrt(1+i)*ONE

    ## linear distribution of lines
    #pix = PIX_BETWEEN*ONE

    path = Path(xy,pix)
    path.trace(-PIHALF)
    path.noise()
    path.interpolate(int(pix/ONE)*2)

    xy = path.xy_interpolated

    ## remove nodes above and below canvas
    canvas_start,canvas_stop = get_limit_indices(xy,top=0.,bottom=1.)
    xy = xy[canvas_start:canvas_stop,:]

    ## render nodes above STOP_Y and below START_Y
    draw_start,draw_stop = get_limit_indices(xy,top=START_Y,bottom=STOP_Y)
    #draw_circles(xy[draw_start:draw_stop,:],\
                 #ones(draw_stop-draw_start)*ONE)
    render.path(xy[draw_start:draw_stop,:])

    exporter.add(xy[draw_start:draw_stop,:])

    xmax = xy[:,0].max()
    if (xmax>STOP_X):
      break

    print 'num',i,'points', len(path.xy_circles),'x', xmax

  name = fn.name()

  render.transparent_pix()
  render.write_to_png(name+'.png')
  exporter.export(name+'.2obj')


if __name__ == '__main__':

    main()

