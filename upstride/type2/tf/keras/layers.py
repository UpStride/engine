from .... import generic_layers
from ....generic_layers import *
from ....batchnorm import BatchNormalizationH
import numpy as np
from ....convolutional import Conv2DParcollet as GenericConv2DParcollet

from ....generic_layers import UPTYPE2
UPSTRIDE_TYPE = 2
BLADES_INDEXES = ['', '12', '13', '23']
GEOMETRICAL_DEF = (3, 0, 0)

# If you wish to overwrite some layers, please implements them here

class Conv2D(generic_layers.Conv2D):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class DepthwiseConv2D(generic_layers.DepthwiseConv2D):
  def __init__(self, *args, **kwargs):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, *args, **kwargs)


class Conv2DParcollet(GenericConv2DParcollet):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE2, *args, **kwargs)


class TF2UpstrideJoint(tf.keras.layers.Layer):
  # TODO add documentation and tests
  def __init__(self, blade_indexes) -> None:
    super().__init__()
    self.image_data_format = tf.keras.backend.image_data_format()

  def call(self, input_tensor):
    if self.image_data_format == 'channels_last':
      red = tf.expand_dims(input_tensor[:, :, :, 0], -1)
      green = tf.expand_dims(input_tensor[:, :, :, 1], -1)
      blue = tf.expand_dims(input_tensor[:, :, :, 2], -1)
      zeros = tf.zeros_like(red)
    else:
      red = tf.expand_dims(input_tensor[:, 0, :, :], 1)
      green = tf.expand_dims(input_tensor[:, 1, :, :], 1)
      blue = tf.expand_dims(input_tensor[:, 2, :, :], 1)
      zeros = tf.zeros_like(red)
    return tf.concat([zeros, red, green, blue], axis=0)


class TF2UpstrideGrayscale(tf.keras.layers.Layer):
  # TODO document and test
  def __init__(self, blade_indexes) -> None:
    super().__init__()
    self.image_data_format = tf.keras.backend.image_data_format()  # can be 'channels_last' or 'channels_first'

  def call(self, input_tensor):
    if self.image_data_format == 'channels_last':
      red = tf.expand_dims(input_tensor[:, :, :, 0], -1)
      green = tf.expand_dims(input_tensor[:, :, :, 1], -1)
      blue = tf.expand_dims(input_tensor[:, :, :, 2], -1)
      grayscale = tf.image.rgb_to_grayscale(input_tensor)
    else:
      red = tf.expand_dims(input_tensor[:, 0, :, :], 1)
      green = tf.expand_dims(input_tensor[:, 1, :, :], 1)
      blue = tf.expand_dims(input_tensor[:, 2, :, :], 1)
      # rgb_to_grayscale function is only channel last
      x = tf.transpose(input_tensor, [0, 2, 3, 1])
      grayscale = tf.image.rgb_to_grayscale(x)
      grayscale = tf.transpose(grayscale, [0, 3, 1, 2])
    return tf.concat([grayscale, red, green, blue], axis=0)


class TF2Upstride(generic_layers.TF2Upstride):
  def __init__(self, strategy=''):
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, strategy)

  def add_strategies(self):
    self.strategies['joint'] = TF2UpstrideJoint
    self.strategies['grayscale'] = TF2UpstrideGrayscale


class Upstride2TF(generic_layers.Upstride2TF):
  # TODO add paper reference for norm strategy and add documentation
  """convert multivector back to real values.
  """

  def __init__(self, strategy=''):
    """
    args:
      strategy: can be the same strategy as defined in generic_layers.Upstride2TF or "norm_{number}" with number a integer number or the string "inf"
    """
    super().__init__(UPSTRIDE_TYPE, BLADES_INDEXES, GEOMETRICAL_DEF, strategy)
    self.strategies['norm'] = self.norm

    self.norm_order = None
    if self.strategy_name.startswith("norm"):
      norm_order = self.strategy_name.split('_')[-1]
      if norm_order == 'inf':
        self.norm_order = np.inf
      else:
        self.norm_order = float(norm_order)  # can raise ValueError
        assert self.norm_order > 0
      self.strategy_name = 'norm'

  def norm(self, x):
    x = tf.split(x, self.multivector_length, axis=0)
    stacked_tensors = tf.stack(x, axis=-1)
    return tf.norm(stacked_tensors, axis=-1, ord=self.norm_order)


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
