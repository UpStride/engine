from .... import generic_layers
from ....generic_layers import *


UPSTRIDE_TYPE = 3
BLADES_INDEXES = ['', '1', '2', '3', '12', '13', '23', '123']
GEOMETRICAL_DEF = (2, 0, 0)

# If you wish to overwrite some layers, please implements them here

class Conv2D(generic_layers.Conv2D):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class Dense(generic_layers.Dense):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class Conv2DTranspose(generic_layers.Conv2DTranspose):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class DepthwiseConv2D(generic_layers.DepthwiseConv2D):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class Dropout(generic_layers.Dropout):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class BatchNormalization(generic_layers.BatchNormalization):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)

class TF2Upstride(generic_layers.TF2Upstride):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class Upstride2TF(generic_layers.Upstride2TF):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)

