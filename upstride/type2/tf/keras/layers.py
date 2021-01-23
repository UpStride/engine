from tensorflow.keras.layers import Layer
from .... import generic_layers
from ....generic_layers import *
from ....generic_layers import TF2Upstride as GenericTF2UP
from ....generic_layers import Upstride2TF as GenericUP2TF
from ....batchnorm import BatchNormalizationH
from .convolutional import Conv2D, DepthwiseConv2D
from .dense import Dense
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import conv_utils
import numpy as np

generic_layers.upstride_type = 2
generic_layers.blade_indexes = ["", "12", "23", "13"]
generic_layers.geometrical_def = (3, 0, 0)

# If you wish to overwrite some layers, please implements them here


class TF2UpstrideJoint(tf.keras.layers.Layer):
  def __init__(self) -> None:
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
  def __init__(self) -> None:
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


class TF2Upstride(GenericTF2UP):
  def __init__(self, strategy=''):
    super().__init__(strategy)

  def add_strategies(self):
    self.strategies['joint'] = TF2UpstrideJoint
    self.strategies['grayscale'] = TF2UpstrideGrayscale


class Attention(Layer):
  """ Normal Attention and Gated Attention which can map complex non-linear relations 
       (introduced in this paper https://arxiv.org/pdf/1802.04712.pdf) is 
       extended for aggregating components 
  """

  def __init__(self, hidden_size=64, final_size=1, is_gated_attention=False):
    super(Attention, self).__init__()
    self.initial_proj = tf.keras.layers.Dense(hidden_size, use_bias=False)
    self.final_linear = tf.keras.layers.Dense(final_size, use_bias=False)

    self.is_gated_attention = is_gated_attention
    if is_gated_attention:
      self.gated_proj = tf.keras.layers.Dense(hidden_size, use_bias=False)

  def call(self, tensor_list):
    """
      Args:
        tensor_list: list of tensor with shape [batch_size, channels] / [batch_size, height, width, channels] / [batch_size, sequence, channels]
      Returns:
        Attention scores with shape [batch_size, final_size, len(tensor_list)]
    """
    if type(tensor_list) is not list:
      raise TypeError(f"tensor_list must be a list of tensors but given {type(tensor_list)}")

    tensor_rank = len(tensor_list[0].get_shape())
    if not 2 <= tensor_rank <= 4:
      raise TypeError(f"tensor rank must be 2, 3 or 4, but provied {tensor_rank}")

    # Attention will be applied using fully connected layer, so if a tensor is provied of rank 3 or 4 than
    # their dimension will be reduced using average pooling
    reduced_tensor_list = []
    reduction_layer = None
    if tensor_rank == 3:
      reduction_layer = tf.keras.layers.GlobalAveragePooling1D()
    elif tensor_rank == 4:
      reduction_layer = tf.keras.layers.GlobalAveragePooling2D()

    # Apply reduction layer if required
    if reduction_layer:
      for tensor in tensor_list:
        reduced_tensor_list.append(reduction_layer(tensor))

    scores = []
    for i in range(len(tensor_list)):
      if not reduced_tensor_list:
        tensor = tensor_list[i]
      else:
        tensor = reduced_tensor_list[i]
      proj = tf.keras.activations.tanh(self.initial_proj(tensor))
      if self.is_gated_attention:
        proj *= tf.keras.activations.sigmoid(self.gated_proj(tensor))
      scores.append(self.final_linear(proj))

    # stack the component from the list and normalize across that  dimension using softmax
    scores = tf.stack(scores, axis=-1)
    scores = tf.keras.activations.softmax(scores, axis=-1)

    if tensor_rank == 3:
      # Expand sequential dimension
      scores = tf.expand_dims(scores, axis=1)
    elif tensor_rank == 4:
      # Expand spatial (height, width) dimension
      scores = tf.expand_dims(tf.expand_dims(scores, axis=1), axis=1)

    # stacking components across last dimensiona and apply weighted sum using attentionj scores
    aggregated_tensor = tf.math.reduce_sum(tf.stack(tensor_list, axis=-1) * scores,  axis=-1)

    return aggregated_tensor


class Upstride2TF(GenericUP2TF):
  """convert multivector back to real values.
  """

  def __init__(self, strategy=''):
    super().__init__(strategy)
    self.strategies['norm'] = self.norm
    self.strategies['attention'] = self.attention

    self.norm_order = None
    if self.strategy_name.startswith("norm"):
      norm_order = self.strategy_name.split('_')[-1]
      if norm_order == 'inf':
        self.norm_order = np.inf
      else:
        self.norm_order = float(norm_order)  # can raise ValueError
        assert self.norm_order > 0
      self.strategy_name = 'norm'

    self.gated_attention = False
    if self.strategy_name == 'gated_attention':
      self.strategy_name = 'attention'
      self.gated_attention = True

  def norm(self, x):
    x = tf.split(x, multivector_length(), axis=0)
    stacked_tensors = tf.stack(x, axis=-1)
    return tf.norm(stacked_tensors, axis=-1, ord=self.norm_order)

  def attention(self, x):
    x = tf.split(x, multivector_length(), axis=0)
    dim = x[0].get_shape()[1]
    return Attention(hidden_size=64, final_size=dim, is_gated_attention=self.gated_attention)(x)


class MaxNormPooling2D(Layer):
  """ Max Pooling layer for quaternions which considers the norm of quaternions to choose the quaternions
  which exhibits maximum norm within a small window of pool size.
  Arguments:
    pool_size: An integer or tuple/list of 2 integers: (pool_height, pool_width)
      specifying the size of the pooling window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the pooling operation.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    padding: A string. The padding method, either 'valid' or 'same'.
      Case-insensitive.
    norm_type: A string or integer. Order of the norm.
      Supported values are 'fro', 'euclidean', 1, 2, np.inf and
      any positive real number yielding the corresponding p-norm.
    data_format: A string, one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first` corresponds to
      inputs with shape `(batch, channels, height, width)`.
    name: A string, the name of the layer.
  """

  def __init__(self, pool_size, strides=None, padding='valid', norm_type='euclidean',
               data_format=None, name=None, **kwargs):
    super(MaxNormPooling2D, self).__init__(name=name, **kwargs)
    if data_format is None:
      data_format = backend.image_data_format()
    if strides is None:
      strides = pool_size
    self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
    self.padding = conv_utils.normalize_padding(padding)
    self.data_format = conv_utils.normalize_data_format(data_format)
    self.norm_type = norm_type

  def call(self, inputs):
    inputs = [tf.expand_dims(inputs[i], -1) for i in range(len(inputs))]
    inputs = tf.keras.layers.concatenate(inputs, axis=-1)

    if self.data_format == 'channels_last':
      pool_shape = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)
    else:
      pool_shape = (1, 1) + self.pool_size
      strides = (1, 1) + self.strides
    padding = self.padding.upper()

    norm = tf.norm(inputs, ord=self.norm_type, axis=-1)

    _, indices = tf.nn.max_pool_with_argmax(norm, ksize=pool_shape, strides=strides, padding=padding,
                                            include_batch_in_index=True)

    input_shape = tf.shape(norm)
    output_shape = tf.shape(indices)

    flat_input_size = tf.reduce_prod(input_shape)
    flat_indices_size = tf.reduce_prod(output_shape)

    indices = tf.reshape(indices, [flat_indices_size, 1])

    inputs = tf.reshape(inputs, [flat_input_size, 4])

    pooled = [tf.reshape(tf.gather_nd(inputs[:, i], indices), output_shape) for i in range(4)]

    return pooled

  def get_config(self):
    config = {
        'pool_size': self.pool_size,
        'padding': self.padding,
        'strides': self.strides,
        'data_format': self.data_format,
        'norm_type': self.norm_type
    }
    base_config = super(MaxNormPooling2D, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
