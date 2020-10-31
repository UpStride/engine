"""
Complex initialization is described here: https://arxiv.org/abs/1705.09792
Implementation is done following this open-source:
 https://github.com/ChihebTrabelsi/deep_complex_networks/edit/master/complexnn/init.py
"""

import numpy as np
from numpy.random import RandomState
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.initializers import Initializer
#from tensorflow.keras.utils.generic_utils import (serialize_keras_object, deserialize_keras_object)


class CInitializer(Initializer):

  # The standard complex initialization
  def __init__(self, blade_idx=None, seed=1337):

    self.seed = seed
    if blade_idx == None:
      raise ValueError('Missing value for argument <blade_idx>. This value is necessary '
        'for correctly initializing the weight of the network.')
    else:
      self.blade_idx = blade_idx

  def set_scale(self, fan_in, fan_out):
    return 1.

  def __call__(self, shape, dtype=None):
    # shape --> [filter_height, filter_width, in_channels, out_channels]

    fan_in, fan_out = shape[0], shape[1]
    kernel_shape = tuple(shape[:2])

    s = self.set_scale(fan_in, fan_out)
    rng = RandomState(self.seed)
    modulus = rng.rayleigh(scale=s, size=shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=shape)

    if self.blade_idx == 0:
      weight = modulus * np.cos(phase) # real part of weights
    elif self.blade_idx == 1:
      weight = modulus * np.sin(phase) # imaginary part of weights
    else:
      raise ValueError(f'Unexpected value blade_idx = {self.blade_idx} for the chosen '
        'upstride type.')

    return weight

class CInitializerGlorot(CInitializer):
  def set_scale(self, fan_in, fan_out):
    return 1. / (fan_in + fan_out)

class CInitializerHe(CInitializer):
  def set_scale(self, fan_in, fan_out):
    return 1. / fan_in



init_aliases_dict = {'complex_glorot': CInitializerGlorot,\
                     'complex_he': CInitializerHe}
