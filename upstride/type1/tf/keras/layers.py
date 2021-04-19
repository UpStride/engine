from upstride import generic_layers
from upstride.generic_layers import *
from upstride.batchnorm import BatchNormalizationC
from upstride.uptypes_utilities import UPTYPE1

# If you wish to overwrite some layers, please implement them here

class Conv2D(generic_layers.Conv2D):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class Dense(generic_layers.Dense):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class Conv2DTranspose(generic_layers.Conv2DTranspose):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class DepthwiseConv2D(generic_layers.DepthwiseConv2D):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class Dropout(generic_layers.Dropout):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class BatchNormalization(generic_layers.BatchNormalization):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class TF2Upstride(generic_layers.TF2Upstride):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)


class Upstride2TF(generic_layers.Upstride2TF):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE1, *args, **kwargs)

