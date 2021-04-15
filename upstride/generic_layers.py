"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
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
import dataclasses


@dataclasses.dataclass(unsafe_hash=True)
class UpstrideDatatype:
  uptype_id: int
  geometrical_def: tuple
  blade_indexes: tuple

  @property
  def dimension(self) -> int:
      return len(self.blade_indexes)

UPTYPE0 = UpstrideDatatype(0, (0, 0, 0), ('',))
UPTYPE1 = UpstrideDatatype(1, (2, 0, 0), ('', '12'))
UPTYPE2 = UpstrideDatatype(2, (3, 0, 0), ('', '12', '13', '23'))
UPTYPE3 = UpstrideDatatype(3, (3, 0, 0), ('', '1', '2', '3', '12', '13', '23', '123'))


def unit_multiplier(uptype, i: int, j: int) -> Tuple[int, int]:
  """given \beta_i and \beta_j, return (k,s) such as : \beta_i * \beta_j = s * \beta_k

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
  s, index = _ga_multiply_get_index(uptype, index1, index2)
  return uptype.blade_indexes.index(index), s

def _ga_multiply_get_index(uptype, index_1: str, index_2: str) -> Tuple[int, str]:
  """given \beta_{index_1}, \beta_{index_2} return (s, index) such as \beta_{index_1} * \beta_{index_2} = s * \beta_{index}
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
      if (length_l1 - 1) % 2 != 0:
        s *= -1
      # check the sign of the square
      s *= square_vector(uptype, l1[i1])
      length_l1 -= 1
      i1 += 1
      i2 += 1
    elif l1[i1] > l2[i2]:
      # then move the element of l2 before the element of l1
      if length_l1 % 2 != 0:
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
  # geometrical_def is a triplet (A, B, C) defining a GA where:
  # - the square of the A first elements is 1
  # - the square of the B next elements is -1
  # - the square of the C last elements is 0
  # dev note : the + 1 is because the first index in the vector notation of a GA is... 1
  if index < uptype.geometrical_def[0] + 1:
    return 1
  if index < uptype.geometrical_def[0] + uptype.geometrical_def[1] + 1:
    return -1
  return 0

def prepare_inputs(uptype, inputs, **kwargs):
  # TODO consider implementing uptype.interlace so that inputs.shape is (BS*N, I, ...) instead of
  # inputs.shape (N*BS, I, ...)
  inputs = tf.reshape(inputs, [uptype.dimension, -1, *inputs.shape[1:]]) # shape (N, BS, I, ...)
  # Given that in a grouped convolution with g groups the input is splitted in g chunks along the
  # channels dimension (cf. https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D),
  # then a special attention is required so that the multivector components do NOT get splitted
  # into different convolutions. The solution is to return a tensor of shape (BS, I*N, ...). As it
  # only matters for grouped convolution, the cheaper-to-compute tensor of shape (BS, N*I, ...) is
  # preferred whenever possible. This propagates to prepare_hyper_weight(), bias addition and
  # prepare_output().
  if kwargs.get('groups', 1) > 1:
    rest = list(range(3, tf.rank(inputs)))
    inputs = tf.transpose(inputs, perm=[1, 2, 0, *rest]) # shape (BS, I, N, ...)
    inputs = tf.reshape(inputs, [inputs.shape[0], -1, *inputs.shape[3:]]) # shape (BS, I*N, ...)
  else:
    rest = list(range(2, tf.rank(inputs)))
    inputs = tf.transpose(inputs, perm=[1, 0, *rest]) # shape (BS, N, I, ...)
    inputs = tf.reshape(inputs, [inputs.shape[0], -1, *inputs.shape[3:]]) # shape (BS, N*I, ...)
  return inputs

def prepare_output(uptype, output, **kwargs):
  if kwargs.get('groups', 1) > 1: # output.shape (BS, O*N, ...)
    output = tf.reshape(output, [output.shape[0], -1, uptype.dimension, *output.shape[2:]]) # shape (BS, O, N, ...)
    rest = list(range(3, tf.rank(output)))
    output = tf.transpose(output, perm=[2, 0, 1, *rest]) # shape (N, BS, O, ...)
  else: # output.shape (BS, N*O, ...)
    output = tf.reshape(output, [output.shape[0], uptype.dimension, -1, *output.shape[2:]]) # shape (BS, N, O, ...)
    rest = list(range(2, tf.rank(output)))
    output = tf.transpose(output, perm=[1, 0, *rest]) # shape (N, BS, O, ...)
  output = tf.reshape(output, [-1, *output.shape[2:]]) # shape (N*BS, O, ...)
  return output

def prepare_hyper_weight(uptype, weight, **kwargs):
  if uptype == UPTYPE2:
    kernels_4_r = tf.concat([weight[0], -weight[1], -weight[2], -weight[3]], axis=2)
    kernels_4_i = tf.concat([weight[1],  weight[0],  weight[3], -weight[2]], axis=2)
    kernels_4_j = tf.concat([weight[2], -weight[3],  weight[0],  weight[1]], axis=2)
    kernels_4_k = tf.concat([weight[3],  weight[2], -weight[1],  weight[0]], axis=2)
    hyper_weight = tf.concat([kernels_4_r, kernels_4_i, kernels_4_j, kernels_4_k], axis=3)
  elif uptype == UPTYPE1:
    kernels_2_r = tf.concat([weight[0], -weight[1]], axis=2)
    kernels_2_i = tf.concat([weight[1],  weight[0]], axis=2)
    hyper_weight = tf.concat([kernels_2_r, kernels_2_i], axis=3)
  else:
    # If a the type in use doesn't have a hard-coded hyper_weight, then
    # compute a single element A[i][j] for the matrix A that embeds the Hamilton product
    w_shape = weight.shape
    hyper_weight_list = []
    for i in range(uptype.dimension):
      hyper_weight_row = []
      for j in range(uptype.dimension):
        k, sign_j_i = unit_multiplier(uptype, i, j)
        _, sign_j_j = unit_multiplier(uptype, j, j)
        if sign_j_j == 0:
          raise ZeroDivisionError()
        hyper_weight_row.append(weight[k] * sign_j_i * sign_j_j)  # Given that sign_j_j is 1, 0 or -1,
        # a multiplication sign was preferred over the division. When it's 0, an error is raised.
      hyper_weight_list.append(tf.concat(hyper_weight_row, axis=2))
    hyper_weight = tf.concat(hyper_weight_list, axis=3)
  # hyper_weight.shape (..., N*I, N*O)
  if kwargs.get('groups', 1) > 1:
    shape = hyper_weight.shape
    updim = uptype.dimension
    shape = [*shape[:-2], updim, shape[-2]//updim, updim, shape[-1]//updim]
    rest = list(range(0, tf.rank(hyper_weight) - 2))
    hyper_weight = tf.reshape(hyper_weight, shape) # shape (..., N, I, N, O)
    rank = tf.rank(hyper_weight)
    hyper_weight = tf.transpose(hyper_weight, perm=[*rest])
  return hyper_weight

class BiasLayer(tf.keras.layers.Layer):
  """Keras layer that only adds a bias to the input. It implements the operation:

  output = input + bias

  This layer supports both channels first and channels last

  Args:
      bias_initializer: Initializer for the bias vector.
      bias_regularizer: Regularizer for the bias vector.
      bias_constraint: Constraint for the bias vector.
  """

  def __init__(self, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None):
    super().__init__()
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)
    self.axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)

    broadcast_beta_shape = [1] * len(input_shape)
    broadcast_beta_shape[self.axis] = input_shape[self.axis]

    self.bias = self.add_weight(
        name='bias',
        shape=broadcast_beta_shape,
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint,
        trainable=True,
        dtype=self.dtype)

    self.built = True

  def call(self, inputs):
    return tf.add(inputs, self.bias)

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
  def __init__(self, upstride_type, blade_indexes, geometrical_def, **kwargs):
    super().__init__(**kwargs)
    self.upstride_type = upstride_type
    self.blade_indexes = blade_indexes
    self.geometrical_def = geometrical_def
    self.multivector_length = len(self.blade_indexes)
    self.axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1


class GenericLinear(UpstrideLayer):
  def __init__(self, layer, upstride_type, blade_indexes, geometrical_def, *args, **kwargs):
    """
    Args:
      layer: a subclass of tf.keras.layers.Layer
    """
    super().__init__(upstride_type, blade_indexes, geometrical_def)
    self.multivector_length = len(self.blade_indexes)

    # convert all arguments to kwargs to ease processing
    kwargs = self.convert_all_args_to_kwargs(layer.__init__, args, kwargs)
    kwargs[map_tf_linear_op_to_kwarg_output_size[layer]] *= self.multivector_length
    kwargs, add_bias, bias_parameters = self.remove_bias_from_kwargs(kwargs)

    self.layer = self.get_layers(layer, self.upstride_type, **kwargs)
    self.bias = None
    if add_bias:
      self.bias = BiasLayer(bias_parameters['bias_initializer'], bias_parameters['bias_regularizer'], bias_parameters['bias_constraint'])

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
      linear_layer_output: Tensor of shape (N * BS, N * C, ...) with BS the batch size and C the output size if this layer was a real one
      inverse: if True then the engine compute y = W.x and not x.W
      bias: the bias operation, if needed

    Returns: A tensor of shape (N * BS, C, ...)

    """
    # first, let's split the output of the layer
    layer_outputs = tf.split(linear_layer_output, self.multivector_length, axis=0)
    # here layer_outputs is a list of output of multiplication of one blade per all kernel blade

    # Now we apply the bias. This can seem like a weird place but by applying it here, between the 2 split, we can do in
    # a single operation what else would take N operations
    if bias is not None:
      layer_outputs[0] = bias(layer_outputs[0])  # add the bias on one of these output

    cross_product_matrix = []
    for i in range(self.multivector_length):
      cross_product_matrix.append(tf.split(layer_outputs[i], self.multivector_length, axis=self.axis))

    # cross_product_matrix is a matrix such as
    # cross_product_matrix[i][j] is the result of the multiplication of the
    # i input by the j kernel
    output = [None] * self.multivector_length
    for i in range(self.multivector_length):
      for j in range(self.multivector_length):
        if not inverse:
          k, s = self.unit_multiplier(i, j)
        else:
          k, s = self.unit_multiplier(j, i)

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

  def unit_multiplier(self, i: int, j: int) -> Tuple[int, int]:
    """given \beta_i and \beta_j, return (k,s) such as : \beta_i * \beta_j = s * \beta_k

    with:
        \beta_0 = 1, \beta_1 = i if upstride_type == 1
        \beta_0 = 1, \beta_1 = i, \beta_2 = j, \beta_3 = k if upstride_type == 2
        s in {-1, 1}

    for instance, upstride_type == 1,
    (0, 0) -> (0, 1) because \beta_0 * \beta_0 = 1 * 1 = 1 * \beta_0
    (0, 1) -> (1, 1) because \beta_0 * \beta_1 = 1 * \beta_1
    (1, 1) -> (0, -1) because \beta_1 * \beta_1 = i**2 = -1 = -1 * \beta_0
    """
    index1 = self.blade_indexes[i]
    index2 = self.blade_indexes[j]
    s, index = self._ga_multiply_get_index(index1, index2)
    return self.blade_indexes.index(index), s

  def _ga_multiply_get_index(self, index_1: str, index_2: str) -> Tuple[int, str]:
    """given \beta_{index_1}, \beta_{index_2} return (s, index) such as \beta_{index_1} * \beta_{index_2} = s * \beta_{index}
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
        if (length_l1 - 1) % 2 != 0:
          s *= -1
        # check the sign of the square
        s *= self.square_vector(l1[i1])
        length_l1 -= 1
        i1 += 1
        i2 += 1
      elif l1[i1] > l2[i2]:
        # then move the element of l2 before the element of l1
        if length_l1 % 2 != 0:
          s *= -1
        out_l.append(l2[i2])
        i2 += 1
      elif l1[i1] < l2[i2]:
        out_l.append(l1[i1])
        length_l1 -= 1
        i1 += 1
    out_l += l1[i1:] + l2[i2:]

    return s, "".join([str(i) for i in out_l])

  def square_vector(self, index: int) -> int:
    # geometrical_def is a triplet (A, B, C) defining a GA where:
    # - the square of the A first elements is 1
    # - the square of the B next elements is -1
    # - the square of the C last elements is 0
    # dev note : the + 1 is because the first index in the vector notation of a GA is... 1
    if index < self.geometrical_def[0] + 1:
      return 1
    if index < self.geometrical_def[0] + self.geometrical_def[1] + 1:
      return -1
    return 0

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
    """instantiate layer with the correct initializer
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
  def __init__(self, upstride_type, blade_indexes, geometrical_def, *args, **kwargs):
    super().__init__(tf.keras.layers.Conv2D, upstride_type, blade_indexes, geometrical_def, *args, **kwargs)


class Dense(GenericLinear):
  def __init__(self, upstride_type, blade_indexes, geometrical_def, *args, **kwargs):
    super().__init__(tf.keras.layers.Dense, upstride_type, blade_indexes, geometrical_def, *args,  **kwargs)


class Conv2DTranspose(GenericLinear):
  def __init__(self, upstride_type, blade_indexes, geometrical_def, *args, **kwargs):
    super().__init__(tf.keras.layers.Conv2DTranspose, upstride_type, blade_indexes, geometrical_def, *args,  **kwargs)


class DepthwiseConv2D(GenericLinear):
  def __init__(self, upstride_type, blade_indexes, geometrical_def, *args, **kwargs):
    super().__init__(tf.keras.layers.DepthwiseConv2D, upstride_type, blade_indexes, geometrical_def, *args,  **kwargs)


# and now non linear layers

class BatchNormalization(UpstrideLayer):
  def __init__(self, upstride_type, blade_indexes, geometrical_def, *args, **kwargs):
    super().__init__(upstride_type, blade_indexes, geometrical_def)
    self.bn = tf.keras.layers.BatchNormalization(*args, **kwargs)

  def call(self, input_tensor, training=False):
    x = tf.split(input_tensor, self.multivector_length, axis=0)
    x = tf.concat(x, axis=self.axis)
    x = self.bn(x)
    x = tf.split(x, self.multivector_length, axis=self.axis)
    x = tf.concat(x, axis=0)
    return x


class Dropout(tf.keras.layers.Dropout):
  """ Among the non linear operation, dropout is a exception.
  We may want to cancel all components of a hypercomplex number at the same time. This can be done using the noise_shape parameter
  - if we don't want the dropout to be synchronized along the blades, then standard dropout is enough
  - else we need to reorganise the components to create a new dimension for the blades, and changing the noise so we don't apply dropout along this
  dimension
  """

  def __init__(self, upstride_type, blade_indexes, geometrical_def, rate, noise_shape=None, seed=None, synchronized=False, **kwargs):
    super().__init__(rate, noise_shape, seed, **kwargs)
    self.synchronized = synchronized
    self.multivector_length = len(blade_indexes)

  def _get_noise_shape(self, inputs):
    # see https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/layers/core.py#L144-L244
    # for more details
    if self.noise_shape is None:
      if not self.synchronized:
        return None
      else:
        concrete_inputs_shape = array_ops.shape(inputs)  # concrete_inputs_shape is a tf.Tensor
        # transform the first value in the list to 1, so we don't perform Dropout along this dim
        return ops.convert_to_tensor_v2_with_dispatch(tf.concat([[1], concrete_inputs_shape[1:]], axis=0))

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
      transform_shape = tf.concat([[self.multivector_length, -1], input_shape[1:]], axis=0)
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

  def __init__(self, upstride_type, blade_indexes, geometrical_def, strategy='', **kwargs):
    super().__init__(upstride_type, blade_indexes, geometrical_def, **kwargs)
    # This dictionary map the strategy name to the function to call
    self.strategies = {
        'learned': TF2UpstrideLearned,
        'basic': TF2UpstrideBasic,
        '': TF2UpstrideBasic,
    }
    self.strategy_name = strategy

    self.add_strategies()

    if self.strategy_name not in self.strategies:
      raise ValueError(f"unknown strategy: {self.strategy_name}")
    self.model = self.strategies[self.strategy_name](self.blade_indexes, **kwargs)

  def add_strategies(self):
    """ The purpose of this function is to be overritten in a sub class to add elements in self.strategies
    """
    pass

  def call(self, input_tensor):
    return self.model(input_tensor)


class TF2UpstrideLearned(tf.keras.layers.Layer):
  """
    Learning module taken from this paper (https://arxiv.org/pdf/1712.04604.pdf)
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
    self.multivector_length = len(blade_indexes)
    for i in range(1, self.multivector_length):
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
  """convert multivector back to real values.
  """

  def __init__(self, upstride_type, blade_indexes, geometrical_def, strategy='', **kwargs):
    super().__init__(upstride_type, blade_indexes, geometrical_def, **kwargs)
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
    output = tf.split(x, self.multivector_length, axis=0)
    return output[0]

  def concat(self, x):
    x = tf.split(x, self.multivector_length, axis=0)
    return tf.concat(x, self.axis)

  def max_pool(self, x):
    x = tf.split(x, self.multivector_length, axis=0)
    x = tf.stack(x, axis=-1)
    return tf.math.reduce_max(x, axis=-1)

  def avg_pool(self, x):
    x = tf.split(x, self.multivector_length, axis=0)
    x = tf.stack(x, axis=-1)
    return tf.math.reduce_mean(x, axis=-1)
