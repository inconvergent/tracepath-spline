# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from numpy.random import random

def myrandom(size):
  res = 1.-2.*random(size=size)
  return res

def get_limit_indices(xy, top, bottom):

  start = 0
  stop = len(xy)

  top_ymask = (xy[:,1]<top).nonzero()[0]
  if top_ymask.any():
    start = top_ymask.max()

  bottom_ymask = (xy[:,1]>bottom).nonzero()[0]
  if bottom_ymask.any():
    stop = bottom_ymask.min()

  return start, stop

