""" Users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""
import inspect
from typing import List, Tuple

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
# Here we import some keras layers, but we don't use them. This is on purpose : these layers
# Have the same definition between real and hypercomplex. So they are imported to be available
from tensorflow.keras.layers import (Activation, Add, AveragePooling2D,
                                     Concatenate, Cropping2D, Flatten,
                                     GlobalAveragePooling2D, GlobalMaxPooling2D,
                                     LeakyReLU, MaxPool2D, MaxPooling2D, Multiply,
                                     ReLU, Reshape, UpSampling2D, ZeroPadding2D)
from .initializers import InitializersFactory
from tensorflow.python.keras.utils.conv_utils import convert_data_format


def unit_multiplier(uptype, i: int, j: int) -> Tuple[int, int]:
  """ Given \beta_i and \beta_j, return (k,s) such as : \beta_i * \beta_j = s * \beta_k

  with:
      \beta_0 = 1, \beta_1 = i if upstride_type == 1
      \beta_0 = 1, \beta_1 = i, \beta_2 = j, \beta_3 = k if upstride_type == 2
      s in {-1, 1}

  for instance, upstride_type == 1,
  (0, 0) -> (0, 1) because \beta_0 * \beta_0 = 1 * 1 = 1 * \beta_0
  (0, 1) -> (1, 1) because \beta_0 * \beta_1 = 1 * \beta_1
  (1, 1) -> (0, -1) because \beta_1 * \beta_1 = i**2 = -1 = -1 * \beta_0
  """
  index1 = uptype.blade_indexes[i]
  index2 = uptype.blade_indexes[j]
  s, index = ga_multiply_get_index(uptype, index1, index2)
  return uptype.blade_indexes.index(index), s


def ga_multiply_get_index(uptype, index_1: str, index_2: str) -> Tuple[int, str]:
  """ Given \beta_{index_1}, \beta_{index_2} return (s, index) such as \beta_{index_1} * \beta_{index_2} = s * \beta_{index}
  """
  l1 = [int(i) for i in index_1]
  l2 = [int(i) for i in index_2]
  s = 1

  # as l1 and l2 are already sorted, we can just merge them and count the number of permutation needed
  i1, i2, length_l1 = 0, 0, len(l1)
  out_l = []
  while i1 < len(l1) and i2 < len(l2):
    if l1[i1] == l2[i2]:
      # then move the element of l2 near the element of l1 and remove them
      if length_l1 % 2 == 0:
        s *= -1
      # check the sign of the square
      s *= square_vector(uptype, l1[i1])
      length_l1 -= 1
      i1 += 1
      i2 += 1
    elif l1[i1] > l2[i2]:
      # then move the element of l2 before the element of l1
      if length_l1 % 2 == 1:
        s *= -1
      out_l.append(l2[i2])
      i2 += 1
    elif l1[i1] < l2[i2]:
      out_l.append(l1[i1])
      length_l1 -= 1
      i1 += 1
  out_l += l1[i1:] + l2[i2:]

  return s, "".join([str(i) for i in out_l])


def square_vector(uptype, index: int) -> int:
  """
  Geometrical_def is a triplet (A, B, C) defining a GA where:
  - the square of the A first elements is 1
  - the square of the B next elements is -1
  - the square of the C last elements is 0
  dev note : the + 1 is because the first index in the vector notation of a GA is... 1
  """
  if index < 1 or index > uptype.multivector_length:
    raise ValueError(f'Index {index} outside range [1, {uptype.multivector_length}] allowed for type {uptype.__str__}')
  if index <= uptype.geometrical_def[0]:
    return 1
  if index <= uptype.geometrical_def[0] + uptype.geometrical_def[1]:
    return -1
  return 0


class BiasLayer(tf.keras.layers.Layer):
  """ Keras layer that only adds a bias to the input. It implements the operation:

  output = input + bias

  This layer supports both channels first and channels last

  Args:
      bias_initializer: Initializer for the bias vector.
      bias_regularizer: Regularizer for the bias vector.
      bias_constraint: Constraint for the bias vector.
  """

  def __init__(self, rank=0, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None):
    super().__init__()
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.rank = rank
    self.axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1
    self.is_conv = rank != 0
    if self.is_conv:
      self._tf_data_format = convert_data_format(
        tf.keras.backend.image_data_format().lower(), self.rank + 2)

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)

    self.bias = self.add_weight(
        name='bias',
        shape=[input_shape[self.axis],],
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)

    self.built = True

  def call(self, inputs):
    outputs = tf.add(inputs, self.bias) if not self.is_conv else tf.nn.bias_add(
                  inputs, self.bias, data_format=self._tf_data_format)
    return outputs

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
        'bias_regularizer': tf.keras.initializers.serialize(self.bias_regularizer),
        'bias_constraint': tf.keras.initializers.serialize(self.bias_constraint),
    }
    base_config = super(BiasLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class UpstrideLayer(tf.keras.layers.Layer):
  def __init__(self, uptype, **kwargs):
    super().__init__(**kwargs)
    self.uptype = uptype
    self.axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1


class GenericLinear(UpstrideLayer):
  def __init__(self, layer, uptype, *args, **kwargs):
    """
    Args:
      layer: a subclass of tf.keras.layers.Layer
    """
    super().__init__(uptype)

    # convert all arguments to kwargs to ease processing
    kwargs = self.convert_all_args_to_kwargs(layer.__init__, args, kwargs)
    kwargs[map_tf_linear_op_to_kwarg_output_size[layer]] *= self.uptype.multivector_length
    kwargs, add_bias, bias_parameters = self.remove_bias_from_kwargs(kwargs)

    self.layer = self.get_layers(layer, self.uptype.uptype_id, **kwargs)
    self.bias = None
    if add_bias:
      self.bias = BiasLayer(getattr(self.layer, 'rank', 0), bias_parameters['bias_initializer'], bias_parameters['bias_regularizer'], bias_parameters['bias_constraint'])

  def call(self, input_tensor, training=False):
    """
    Implementation note : keeping "training=False" is important. Even if the training parameter doesn't make sense here because the behavior of the linear
    layer is the same in both training and eval mode, tensorflow expect this parameter to exist.

    see https://www.tensorflow.org/tutorials/customization/custom_layers for more information
    """
    x = self.layer(input_tensor)
    x = self.geometric_multiplication(x, bias=self.bias)
    return x

  def geometric_multiplication(self, linear_layer_output, inverse=False, bias=None):
    """
    Args:
      linear_layer_output: Tensor of shape (N * BS, O * N, ...) with BS the batch size and O the output size if this layer was a real one
      inverse: if True then the engine compute y = W.x and not x.W
      bias: the bias operation, if needed

    Returns: A tensor of shape (N * BS, O, ...)

    """
    multivector_len = self.uptype.multivector_length

    # first, let's split the output of the layer
    layer_outputs = tf.split(linear_layer_output, multivector_len, axis=0)
    # here layer_outputs is a list of output of multiplication of one blade per all kernel blade

    # Now we apply the bias. This can seem like a weird place but by applying it here, between the 2 split, we can do in
    # a single operation what else would take N operations
    if bias is not None:
      layer_outputs[0] = bias(layer_outputs[0])  # add the bias on one of these output

    cross_product_matrix = []
    for layer_out in layer_outputs:
      # previous implementation:
      # dev note: if it is a grouped convolution, then we need to reshape each layer_output from
      # (BS, O*N, ...) to (BS, N*O, ...) so that the split that follows acts on the upstride datatype
      # if getattr(self.layer, 'groups', 1) > 1 or getattr(self.layer, 'depth_multiplier', 0) > 0:
      #   if self.axis == -1:
      #     layer_out = tf.concat([layer_out[..., i::multivector_len] for i in range(multivector_len)], axis=-1)
      #   elif self.axis == 1:
      #     layer_out = tf.concat([layer_out[:, i::multivector_len, ...] for i in range(multivector_len)], axis=1)
      # cross_product_matrix.append(tf.split(layer_out, multivector_len, axis=self.axis))

      if self.axis == -1:
        layer_out = [layer_out[..., i::multivector_len] for i in range(multivector_len)]
      elif self.axis == 1:
        layer_out = [layer_out[:, i::multivector_len, ...] for i in range(multivector_len)]

      cross_product_matrix.append(layer_out)


    # cross_product_matrix is a matrix such as
    # cross_product_matrix[i][j] is the result of the multiplication of the
    # i input by the j kernel
    output = [None] * multivector_len
    for i in range(multivector_len):
      for j in range(multivector_len):
        if not inverse:
          k, s = unit_multiplier(self.uptype, i, j)
        else:
          k, s = unit_multiplier(self.uptype, j, i)

        # same as output[k] += s*self.layers[i](inputs[j]), but cleaner TensorFlow execution graph
        if s == 1:
          if output[k] is None:
            output[k] = cross_product_matrix[i][j]
          else:
            output[k] += cross_product_matrix[i][j]
        elif s == -1:
          if output[k] is None:
            output[k] = -cross_product_matrix[i][j]
          else:
            output[k] -= cross_product_matrix[i][j]
    return tf.concat(output, axis=0)


  def convert_all_args_to_kwargs(self, function, args, kwargs):
    """ This function use the information in the signature of the function
    to convert all elements of args to kwargs.
    Then it also add in kwargs the default parameters of the function
    """
    parameters = inspect.getfullargspec(function).args
    for i, arg in enumerate(args):
      kwargs[parameters[i + 1]] = arg  # + 1 because the first element of parameters is 'self'
    # add all default parameters to kwargs
    for key, value in inspect.signature(function).parameters.items():
      if key in ['self', 'kwargs']:
        continue
      if key not in kwargs:
        kwargs[key] = value.default
    return kwargs

  def remove_bias_from_kwargs(self, kwargs):
    add_bias = False
    if "use_bias" in kwargs:
      add_bias = kwargs["use_bias"]
      kwargs["use_bias"] = False
    bias_parameters = {}
    if add_bias:
      for param in ["bias_initializer", "bias_regularizer", "bias_constraint"]:
        bias_parameters[param] = kwargs[param]
    return kwargs, add_bias, bias_parameters

  def get_layers(self, layer: tf.keras.layers.Layer, upstride_type, **kwargs) -> Tuple[List[tf.keras.layers.Layer], bool, dict]:
    """ Instantiate layer with the correct initializer
    """

    init_factory = InitializersFactory()

    for possible_name in ["kernel_initializer", "depthwise_initializer"]:
      if (possible_name in kwargs) and init_factory.is_custom_init(kwargs[possible_name]):
        custom_init = init_factory.get_initializer(kwargs[possible_name], upstride_type)
        kwargs[possible_name] = custom_init

    return layer(**kwargs)


# All linear layers should be defined here
# TODO add more explainations here
map_tf_linear_op_to_kwarg_output_size = {
    tf.keras.layers.Conv2D: 'filters',
    tf.keras.layers.Conv2DTranspose: 'filters',
    tf.keras.layers.Dense: 'units',
    tf.keras.layers.DepthwiseConv2D: 'depth_multiplier',
}


class Conv2D(GenericLinear):
  def __init__(self, uptype, *args, **kwargs):
    super().__init__(tf.keras.layers.Conv2D, uptype, *args, **kwargs)


class Dense(GenericLinear):
  def __init__(self, uptype, *args, **kwargs):
    super().__init__(tf.keras.layers.Dense, uptype, *args,  **kwargs)


class Conv2DTranspose(GenericLinear):
  def __init__(self, uptype, *args, **kwargs):
    super().__init__(tf.keras.layers.Conv2DTranspose, uptype, *args,  **kwargs)


class DepthwiseConv2D(GenericLinear):
  def __init__(self, uptype, *args, **kwargs):
    super().__init__(tf.keras.layers.DepthwiseConv2D, uptype, *args,  **kwargs)


# and now non linear layers

class BatchNormalization(UpstrideLayer):
  def __init__(self, uptype, *args, **kwargs):
    super().__init__(uptype)
    self.bn = tf.keras.layers.BatchNormalization(*args, **kwargs)

  def call(self, input_tensor, training=False):
    x = tf.split(input_tensor, self.uptype.multivector_length, axis=0)
    x = tf.concat(x, axis=self.axis)
    x = self.bn(x)
    x = tf.split(x, self.uptype.multivector_length, axis=self.axis)
    x = tf.concat(x, axis=0)
    return x


class Dropout(tf.keras.layers.Dropout):
  """ Among the non linear operation, dropout is a exception.
  We may want to cancel all components of a hypercomplex number at the same time. This can be done using the noise_shape parameter
  - if we don't want the dropout to be synchronized along the blades, then standard dropout is enough
  - else we need to reorganise the components to create a new dimension for the blades, and changing the noise so we don't apply dropout along this
  dimension
  """

  def __init__(self, uptype, rate, noise_shape=None, seed=None, synchronized=False, **kwargs):
    super().__init__(rate, noise_shape, seed, **kwargs)
    self.synchronized = synchronized
    self.uptype = uptype

  def _get_noise_shape(self, inputs):
    # see https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/core.py#L144-L244
    # for more details
    if self.noise_shape is None:
      if not self.synchronized:
        return None
      else:
        concrete_inputs_shape = array_ops.shape(inputs)  # concrete_inputs_shape is a tf.Tensor
        # transform the first value in the list to 1, so we don't perform Dropout along this dim
        return ops.convert_to_tensor_v2(tf.concat([[1], concrete_inputs_shape[1:]], axis=0))

    concrete_inputs_shape = array_ops.shape(inputs)

    if not self.synchronized:
      noise_shape = []
    else:
      noise_shape = [1]
    for i, value in enumerate(self.noise_shape):
      noise_shape.append(concrete_inputs_shape[i] if value is None else value)
    return ops.convert_to_tensor_v2_with_dispatch(noise_shape)

  def call(self, input_tensor, training=False):
    if not self.synchronized:
      return super().call(input_tensor, training)
    else:
      input_shape = tf.shape(input_tensor)
      transform_shape = tf.concat([[self.uptype.multivector_length, -1], input_shape[1:]], axis=0)
      x = tf.reshape(input_tensor, transform_shape)
      x = super().call(x, training)
      x = tf.reshape(x, input_shape)
      return x


class TF2Upstride(UpstrideLayer):
  """ This function should be called to transform TF tensors to Upstride Tensors
  an upstride tensor has the same shape than a TF tensor, but the imaginary part is stack among BS

  For specific types, if you want to add new strategies, then inherit this class, and add strategies inside
  the self.strategies dic
  """

  def __init__(self, uptype, strategy='', **kwargs):
    super().__init__(uptype, **kwargs)
    # This dictionary map the strategy name to the function to call
    self.uptype = uptype
    self.strategies = {
        'learned': TF2UpstrideLearned,
        'basic': TF2UpstrideBasic,
        '': TF2UpstrideBasic,
    }
    self.strategy_name = strategy

    self.add_strategies()

    if self.strategy_name not in self.strategies:
      raise ValueError(f"unknown strategy: {self.strategy_name}")
    self.model = self.strategies[self.strategy_name](self.uptype.blade_indexes, **kwargs)

  def add_strategies(self):
    """ The purpose of this function is to be overritten in a sub class to add elements in self.strategies
    """
    pass

  def call(self, input_tensor):
    return self.model(input_tensor)


class TF2UpstrideLearned(tf.keras.layers.Layer):
  """ Learning module taken from this paper (https://arxiv.org/pdf/1712.04604.pdf)
  BN --> ReLU --> Conv --> BN --> ReLU --> Conv

  Args:
    x (tensor): Input to the network.
    channels (int): number of filters

  Returns:
    tensor: output of the network. Learned component of the multi-vector.
  """

  def __init__(self, blade_indexes, channels=3, kernel_size=3, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=None):
    super().__init__()
    self.axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1
    self.layers = []
    self.uptype.multivector_length = len(blade_indexes)
    for i in range(1, self.uptype.multivector_length):
      self.layers.append(tf.keras.Sequential([
          tf.keras.layers.BatchNormalization(axis=self.axis),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), padding='same', use_bias=use_bias,
                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer),
          tf.keras.layers.BatchNormalization(axis=self.axis),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), padding='same', use_bias=use_bias,
                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
      ]))
    self.concat = tf.keras.layers.Concatenate(axis=0)

  def call(self, input_tensor):
    outputs = [input_tensor]
    for layer in self.layers:
      outputs.append(layer(input_tensor))
    return self.concat(outputs)


class TF2UpstrideBasic(tf.keras.layers.Layer):
  def __init__(self, blade_indexes):
    super().__init__()
    self.multivector_length = len(blade_indexes)
    self.concat = tf.keras.layers.Concatenate(axis=0)

  def call(self, x):
    outputs = [x]
    for _ in range(1, self.multivector_length):
      outputs.append(tf.zeros_like(x))
    return self.concat(outputs)


class Upstride2TF(UpstrideLayer):
  """ Convert multivector back to real values.
  """

  def __init__(self, uptype, strategy='', **kwargs):
    super().__init__(uptype, **kwargs)
    self.strategies = {
        'basic': self.basic,
        'default': self.basic,
        '': self.basic,
        'concat': self.concat,
        'max_pool': self.max_pool,
        'avg_pool': self.avg_pool
    }
    self.strategy_name = strategy

  def call(self, input_tensor):
    if self.strategy_name not in self.strategies:
      raise ValueError(f"unknown strategy: {self.strategy_name}")
    return self.strategies[self.strategy_name](input_tensor)

  def basic(self, x):
    output = tf.split(x, self.uptype.multivector_length, axis=0)
    return output[0]

  def concat(self, x):
    x = tf.split(x, self.uptype.multivector_length, axis=0)
    return tf.concat(x, self.axis)

  def max_pool(self, x):
    x = tf.split(x, self.uptype.multivector_length, axis=0)
    x = tf.stack(x, axis=-1)
    return tf.math.reduce_max(x, axis=-1)

  def avg_pool(self, x):
    x = tf.split(x, self.uptype.multivector_length, axis=0)
    x = tf.stack(x, axis=-1)
    return tf.math.reduce_mean(x, axis=-1)
