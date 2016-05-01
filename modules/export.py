
from numpy import arange
from numpy import row_stack
from dddUtils.ioOBJ import export_2d

class Exporter(object):

  def __init__(self):

    self.vertices = []
    self.num = 0
    self.lines = []

  def add(self, vertices):

    self.vertices.append(vertices)
    n = len(vertices)

    print(n)
    self.lines.append(arange(self.num, self.num+n))
    self.num += n

  def export(self, fn):

    vertices = row_stack(self.vertices)
    lines = self.lines
    export_2d('lines', fn, verts=vertices, lines=lines)

