from .... import generic_layers
from ....generic_layers import *
from .activations import *
from ....batchnorm import BatchNormalizationC

generic_layers.upstride_type = 1
generic_layers.blade_indexes = ["", "12"]
generic_layers.geometrical_def = (2, 0, 0)

# If you wish to overwrite some layers, please implements them here


class Conv2D(generic_layers.Conv2D):
  def __init__(self, *argv, **kwargs):
    super().__init__(*argv, **kwargs)


class DepthwiseConv2D(generic_layers.DepthwiseConv2D):
  def __init__(self, *argv, **kwargs):
    super().__init__(*argv, **kwargs)
