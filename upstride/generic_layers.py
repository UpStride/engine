"""users shouldn't import this package directly. instead import upstride.typeX.tf.keras.layers
"""
import functools
import inspect
from typing import List, Tuple
import tensorflow as tf

# from .convolutional import Conv2D as Conv2DConj for now conjugate is not used
from .initializers import InitializersFactory

# Definition of the GA, setup when calling upstride.type{1/2/3}.tf.keras.layers
upstride_type = 3
blade_indexes = ["", "1", "2", "3", "12", "13", "23", "123"]
geometrical_def = (3, 0, 0)

conjugate = False


def change_upstride_type(type: int, new_blade_indexes: List[str],  new_geometrical_def: Tuple[int, int, int]):
  """ Called in upstride.type{1/2/3}.tf.keras.layers to setup the algebra. In a near future we should stop using 
  global variables for this
  """
  global upstride_type, blade_indexes, geometrical_def
  upstride_type = type
  blade_indexes = new_blade_indexes
  geometrical_def = new_geometrical_def


def set_conjugaison_mult(b):
  global conjugate
  conjugate = b


def multivector_length() -> int:
  """map the upstride type to the number of dimensions in our GA
  """
  return len(blade_indexes)


def blade_index_to_position(index: str) -> int:
  return blade_indexes.index(index)


def square_vector(index: int) -> int:
  # geometrical_def is a triplet (A, B, C) defining a GA where:
  # - the square of the A first elements is 1
  # - the square of the B next elements is -1
  # - the square of the C last elements is 0
  # dev note : the + 1 is because the first index in the vector notation of a GA is... 1
  if index < geometrical_def[0] + 1:
    return 1
  if index < geometrical_def[0] + geometrical_def[1] + 1:
    return -1
  return 0


def _ga_multiply_get_index(index_1: str, index_2: str) -> Tuple[int, str]:
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
      s *= square_vector(l1[i1])
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


def unit_multiplier(i: int, j: int) -> Tuple[int, int]:
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
  index1 = blade_indexes[i]
  index2 = blade_indexes[j]
  s, index = _ga_multiply_get_index(index1, index2)
  return blade_index_to_position(index), s


def convert_all_args_to_kwargs(function, args, kwargs):
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


def remove_bias_from_kwargs(kwargs):
  add_bias = False
  if "use_bias" in kwargs:
    add_bias = kwargs["use_bias"]
    kwargs["use_bias"] = False
  bias_parameters = {}
  if add_bias:
    for param in ["bias_initializer", "bias_regularizer", "bias_constraint"]:
      bias_parameters[param] = kwargs[param]
  return kwargs, add_bias, bias_parameters


def get_layers(layer: tf.keras.layers.Layer, conj_layer: tf.keras.layers.Layer = None, **kwargs) -> Tuple[List[tf.keras.layers.Layer], bool, dict]:
  """instantiate layer with the correct initializer

  TODO the ability to compute conjugate has been removed, because not used for now
  """

  init_factory = InitializersFactory()

  for possible_name in ["kernel_initializer", "depthwise_initializer"]:
    if (possible_name in kwargs) and init_factory.is_custom_init(kwargs[possible_name]):
      custom_init = init_factory.get_initializer(kwargs[possible_name], upstride_type)
      kwargs[possible_name] = custom_init

  return layer(**kwargs)


def geometric_multiplication(linear_layer_output, inverse=False, bias=None):
  """
  Args:
    linear_layer_output: Tensor of shape (N * BS, N * C, ...) with BS the batch size and C the output size if this layer was a real one
    inverse: if True then the engine compute y = W.x and not x.W
    bias: the bias operation, if needed

  Returns: A tensor of shape (N * BS, C, ...)

  """
  # first, let's split the output of the layer
  layer_outputs = tf.split(linear_layer_output, multivector_length(), axis=0)
  # here layer_outputs is a list of output of multiplication of one blade per all kernel blade

  # Now we apply the bias. This can seem like a weird place but by applying it here, between the 2 split, we can do in
  # a single operation what else would take N operations
  if bias is not None:
    layer_outputs[0] = bias(layer_outputs[0])  # add the bias on one of these output

  cross_product_matrix = []
  for i in range(multivector_length()):
    cross_product_matrix.append(tf.split(layer_outputs[i], multivector_length(), axis=1))

  # cross_product_matrix is a matrix such as
  # cross_product_matrix[i][j] is the result of the multiplication of the
  # i input by the j kernel
  output = [None] * multivector_length()
  for i in range(multivector_length()):
    for j in range(multivector_length()):
      if not inverse:
        k, s = unit_multiplier(i, j)
      else:
        k, s = unit_multiplier(j, i)

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


class BiasLayer(tf.keras.layers.Layer):
  """Keras layer that only adds a bias to the input. It implements the operation:

  output = input + bias

  Note that this layer only work in channel first neural network

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

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)

    broadcast_beta_shape = [1] * len(input_shape)
    broadcast_beta_shape[1] = input_shape[1]

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


class GenericLinear(tf.keras.Model):
  def __init__(self, layer, *args, conj_layer=None, **kwargs):
    """
    Args:
      layer: a subclass of tf.keras.layers.Layer 
      conj_layer: the layer implementing the conjugaison operation matching the layer operation
    """
    super().__init__()

    # convert all arguments to kwargs to ease processing
    kwargs = convert_all_args_to_kwargs(layer.__init__, args, kwargs)
    kwargs[map_tf_linear_op_to_kwarg_output_size[layer]] *= multivector_length()
    kwargs, add_bias, bias_parameters = remove_bias_from_kwargs(kwargs)

    # if the layer can run conjugaison, then self.conj_layer is an instance of the conj layer, else none
    self.layer = get_layers(layer, conj_layer, **kwargs)
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
    x = geometric_multiplication(x, bias=self.bias)
    return x


class GenericNonLinear(tf.keras.Model):
  def __init__(self, layer, *args, stack_channels = False, **kwargs):
    """
    stack_channels is a boolean that control how to prepare teh data before appling the real operation. 
    - If false then keep the blades stack on the first axis (batch size)
    - If true then move the blades to the second axis (channel). Useful for BN for instance
    """
    super().__init__()
    self.stack_channels = stack_channels  # usefull for BN

    # convert all arguments to kwargs to ease processing
    kwargs = convert_all_args_to_kwargs(layer.__init__, args, kwargs)
    kwargs, add_bias, bias_parameters = remove_bias_from_kwargs(kwargs)

    self.layer = layer(**kwargs)
    self.bias = None
    if add_bias:
      self.bias = BiasLayer(self.bias_parameters['bias_initializer'], self.bias_parameters['bias_regularizer'], self.bias_parameters['bias_constraint'])

  def call(self, input_tensor, training=False):
    if self.stack_channels:
      input_tensor = tf.split(input_tensor, multivector_length(), axis=0)
      input_tensor = tf.concat(input_tensor, axis=1)
    x = self.layer(input_tensor)
    if self.stack_channels:
      x = tf.split(x, multivector_length(), axis=1)
      x = tf.concat(x, axis=0)
    return x


# All linear layers should be defined here
map_tf_linear_op_to_kwarg_output_size = {
    tf.keras.layers.Conv2D: 'filters',
    tf.keras.layers.Conv2DTranspose: 'filters',
    tf.keras.layers.Dense: 'units',
    tf.keras.layers.DepthwiseConv2D: 'depth_multiplier',
}


class Conv2D(GenericLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Conv2D, *args, **kwargs)


class Dense(GenericLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Dense, **kwargs)


class Conv2DTranspose(GenericLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Conv2DTranspose, **kwargs)


class DepthwiseConv2D(GenericLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.DepthwiseConv2D, **kwargs)


# TODO SeparableConv2D is probably an exception. Go through the math
# class SeparableConv2D(GenericLinear):
#   def __init__(self, *args, **kwargs):
#     super().__init__(tf.keras.layers.SeparableConv2D, **kwargs)


# and now non linear layers
class UpSampling2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.UpSampling2D, *args, **kwargs)


class MaxPooling2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.MaxPooling2D, *args, **kwargs)


class AveragePooling2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.AveragePooling2D, *args, **kwargs)


class MaxPool2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.MaxPool2D, *args, **kwargs)


class AveragePool2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.AveragePool2D, *args, **kwargs)


class GlobalMaxPooling2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.GlobalMaxPooling2D, *args, **kwargs)


class GlobalAveragePooling2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.GlobalAveragePooling2D, *args, **kwargs)


class Reshape(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Reshape, *args, **kwargs)


class BatchNormalization(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.BatchNormalization, *args, stack_channels = True, **kwargs)


class Activation(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Activation, *args, **kwargs)


class Flatten(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Flatten, *args, **kwargs)


class ZeroPadding2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.ZeroPadding2D, *args, **kwargs)


class Cropping2D(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Cropping2D, *args, **kwargs)


class ReLU(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.ReLU, *args, **kwargs)


class LeakyReLU(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.LeakyReLU, *args, **kwargs)


class Add(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Add, *args, **kwargs)
    self.list_as_input = True


class Multiply(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Multiply, *args, **kwargs)
    self.list_as_input = True


class Concatenate(GenericNonLinear):
  def __init__(self, *args, **kwargs):
    super().__init__(tf.keras.layers.Concatenate, *args, **kwargs)
    self.list_as_input = True


class Dropout(tf.keras.Model):
  """ Among the non linear operation, dropout is a exception.
  We may want to cancel all components of a hypercomplex number at the same time. This can't be done with a single Dropout.
  We need to define N Dropout.
  - if we want the dropout to be synchronized, then they should all have the same random seed
  - else we should give them different random seed
  """
  def __init__(self, rate, noise_shape=None, seed=None, synchronized=False, **kwargs):
    super().__init__()

    seeds = []
    if synchronized:
      # Prevent the seed to be different for the several Dropout operations
      if seed is not None:
        seeds = [seed] * multivector_length()
      else:
        seeds = [4242] * multivector_length()
    if not synchronized:
      if seed is not None:
        # Prevent the seed to be the same for the several Dropout operations
        seeds = [seed + i for i in range(multivector_length())]
      else:
        seeds = [None] * multivector_length()

    self.layers = []
    for i in range(multivector_length()):
      self.layers.append(tf.keras.layers.Dropout(rate, noise_shape, seeds[i]))

  def call(self, input_tensor, training=False):
    input_tensor = tf.split(input_tensor, multivector_length(), axis=0)
    x = []
    for i in range(multivector_length()):
      x.append(self.layers[i](input_tensor[i], training))
    x = tf.concat(x, axis=0)
    return x



class TF2Upstride(tf.keras.layers.Layer):
  """ This function should be called to transform TF tensors to Upstride Tensors
  an upstride tensor has the same shape than a TF tensor, but the imaginary part is stack among BS

  For specific types, if you want to add new strategies, then inherit this class, and add strategies inside
  the self.strategies dic
  """

  def __init__(self, strategy='', **args):
    super().__init__()
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
    self.model = self.strategies[self.strategy_name](**args)

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

  def __init__(self, channels=3, kernel_size=3, use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=None):
    super().__init__()
    self.layers = []
    for i in range(1, multivector_length()):
      self.layers.append(tf.keras.Sequential([
          tf.keras.layers.BatchNormalization(axis=1),
          tf.keras.layers.Activation('relu'),
          tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), padding='same', use_bias=use_bias,
                                 kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer),
          tf.keras.layers.BatchNormalization(axis=1),
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
  def __init__(self):
    super().__init__()
    self.concat = tf.keras.layers.Concatenate(axis=0)

  def call(self, x):
    outputs = [x]
    for _ in range(1, multivector_length()):
      outputs.append(tf.zeros_like(x))
    return self.concat(outputs)


class Upstride2TF(tf.keras.layers.Layer):
  """convert multivector back to real values. 
  """

  def __init__(self, strategy=''):
    super().__init__()
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
    output = tf.split(x, multivector_length(), axis=0)
    return output[0]

  def concat(self, x):
    x = tf.split(x, multivector_length(), axis=0)
    return tf.concat(x, 1)

  def max_pool(self, x):
    x = tf.split(x, multivector_length(), axis=0)
    x = tf.stack(x, axis=-1)
    return tf.math.reduce_max(x, axis=-1)

  def avg_pool(self, x):
    x = tf.split(x, multivector_length(), axis=0)
    x = tf.stack(x, axis=-1)
    return tf.math.reduce_mean(x, axis=-1)
