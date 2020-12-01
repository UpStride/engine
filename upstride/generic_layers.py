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
  @functools.lru_cache(maxsize=1)
  def get_dict():
    """return a dictionary that map the blade index to the position in the list encoding the multivector
    """
    d = {}
    for i, e in enumerate(blade_indexes):
      d[e] = i
    return d
  return get_dict()[index]


def square_vector(index: int) -> int:
  @functools.lru_cache(maxsize=1)
  def get_list():
    """return a list that map the indice to the square
    For instance, for geometrical_def = (2, 3, 1) this function will return
    [1, 1, -1, -1, -1, 0]
    """
    l = [0]
    possible_squares = [1, -1, 0]
    for i in range(3):
      l += [possible_squares[i]] * geometrical_def[i]
    return l
  return get_list()[index]


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


def convert_all_args_to_kwargs(function, argv, kwargs):
  """ This function use the information in the signature of the function
  to convert all elements of argv to kwargs.
  Then it also add in kwargs the default parameters of the function
  """
  parameters = inspect.getfullargspec(function).args
  for i, arg in enumerate(argv):
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


def get_layers(layer: tf.keras.layers.Layer, conj_layer: tf.keras.layers.Layer = None, *argv, **kwargs) -> Tuple[List[tf.keras.layers.Layer], bool, dict]:
  """instantiate layer several times to match the number needed by the GA definition

  Any parameter analysis need to be done here. For instance, we can't define several times 
  a layer with the same name, so we need to edit the name manually

  Args:
      layer (tf.keras.layers.Layer): a keras layer that we need to instantiate several times

  Returns:
      List[tf.keras.layers.Layer]: the list of keras layers
  """
  kwargs = convert_all_args_to_kwargs(layer.__init__, argv, kwargs)
  kwargs, add_bias, bias_parameters = remove_bias_from_kwargs(kwargs)

  # hyper-complex initialization
  init_factory = InitializersFactory()
  # kernel_arg_name = ""

  for possible_name in ["kernel_initializer", "depthwise_initializer"]:
    if (possible_name in kwargs) and init_factory.is_custom_init(kwargs[possible_name]):
      custom_init = init_factory.get_initializer(kwargs[possible_name], upstride_type)
      kwargs[possible_name] = custom_init

  return layer(**kwargs), add_bias, bias_parameters


def geometric_multiplication(layer_output, inverse=False, bias=None):
  # first, let's split the output of the layer
  layer_outputs = tf.split(layer_output, multivector_length(), axis=0)
  # here layer_outputs is a list of output of multiplication of one blade per all kernel blade
  if bias is not None:
    layer_outputs[0] = bias(layer_outputs[0]) # add the bias on one of these output
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

      # same as output[k] += s*self.layers[i](inputs[j]), but cleaner graph
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
  return tf.keras.layers.Concatenate(axis=0)(output)


class BiasLayer(tf.keras.layers.Layer):
  """Keras layer that only adds a bias to the input.

  code from https://github.com/tensorflow/agents/blob/v0.4.0/tf_agents/networks/bias_layer.py#L24-L81
  with some modifications when initializing the weight to use the same conf as other layers

  `BiasLayer` implements the operation:
  `output = input + bias`
  Arguments:
      bias_initializer: Initializer for the bias vector.
  Input shape:
      nD tensor with shape: `(batch_size, ..., input_dim)`. The most common
        situation would be a 2D input with shape `(batch_size, input_dim)`. Note
        a rank of at least 2 is required.
  Output shape:
      nD tensor with shape: `(batch_size, ..., input_dim)`. For instance, for a
        2D input with shape `(batch_size, input_dim)`, the output would have
        shape `(batch_size, input_dim)`.
  """

  def __init__(self, bias_initializer='zeros', bias_regularizer=None, bias_constraint=None, **kwargs):
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    super(BiasLayer, self).__init__(**kwargs)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    self.supports_masking = True
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)

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
    }
    base_config = super(BiasLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GenericLinear(tf.keras.Model):
  def __init__(self, layer, *argv, conj_layer=None, **kwargs):
    super().__init__()
    # if the layer can run conjugaison, then self.conj_layer is an instance of the conj layer, else none
    self.layer, self.add_bias, self.bias_parameters = get_layers(layer, conj_layer, *argv, **kwargs)
    self.bias = None
    if self.add_bias:
      self.bias = BiasLayer(self.bias_parameters['bias_initializer'], self.bias_parameters['bias_regularizer'], self.bias_parameters['bias_constraint'])

  def call(self, input_tensor, training=False):
    x = self.layer(input_tensor)
    x = geometric_multiplication(x, bias=self.bias)
    return x


class GenericNonLinear(tf.keras.Model):
  def __init__(self, layer, *argv, **kwargs):
    super().__init__()
    self.stack_channels = False # usefull for BN
    self.layer, self.add_bias, self.bias_parameters = get_layers(layer, None, *argv, **kwargs)

  def __call__(self, input_tensor, training=False):
    if self.stack_channels:
      input_tensor = tf.split(input_tensor, multivector_length(), axis=0)
      input_tensor = tf.concat(input_tensor, axis=1)
    x = self.layer(input_tensor)
    if self.stack_channels:
      x = tf.split(x, multivector_length(), axis=1)
      x = tf.concat(x, axis=0)
    return x


class Conv2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    kwargs = convert_all_args_to_kwargs(tf.keras.layers.Conv2D.__init__, argv, kwargs)
    kwargs['filters'] *= multivector_length()
    super().__init__(tf.keras.layers.Conv2D, **kwargs)


class Dense(GenericLinear):
  def __init__(self, *argv, **kwargs):
    kwargs = convert_all_args_to_kwargs(tf.keras.layers.Dense.__init__, argv, kwargs)
    kwargs['units'] *= multivector_length()
    super().__init__(tf.keras.layers.Dense, **kwargs)


class Conv2DTranspose(GenericLinear):
  def __init__(self, *argv, **kwargs):
    kwargs = convert_all_args_to_kwargs(tf.keras.layers.Conv2DTranspose.__init__, argv, kwargs)
    kwargs['filters'] *= multivector_length()
    super().__init__(tf.keras.layers.Conv2DTranspose, **kwargs)


class UpSampling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.UpSampling2D, *argv, **kwargs)


class DepthwiseConv2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    kwargs = convert_all_args_to_kwargs(tf.keras.layers.DepthwiseConv2D.__init__, argv, kwargs)
    kwargs['depth_multiplier'] *= multivector_length()
    super().__init__(tf.keras.layers.DepthwiseConv2D, **kwargs)


class DepthwiseConv2DTranspose(GenericLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.DepthwiseConv2DTranspose, *argv, **kwargs)


class SeparableConv2D(GenericLinear):
  def __init__(self, *argv, **kwargs):
    kwargs = convert_all_args_to_kwargs(tf.keras.layers.SeparableConv2D.__init__, argv, kwargs)
    kwargs['filters'] *= multivector_length()
    super().__init__(tf.keras.layers.SeparableConv2D, **kwargs)


class MaxPooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.MaxPooling2D, *argv, **kwargs)


class AveragePooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.AveragePooling2D, *argv, **kwargs)


class MaxPool2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.MaxPool2D, *argv, **kwargs)


class AveragePool2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.AveragePool2D, *argv, **kwargs)


class GlobalMaxPooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.GlobalMaxPooling2D, *argv, **kwargs)


class GlobalAveragePooling2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.GlobalAveragePooling2D, *argv, **kwargs)


class Reshape(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Reshape, *argv, **kwargs)


class BatchNormalization(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.BatchNormalization, *argv, **kwargs)
    self.stack_channels = True


class Activation(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Activation, *argv, **kwargs)


class Flatten(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Flatten, *argv, **kwargs)


class ZeroPadding2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.ZeroPadding2D, *argv, **kwargs)


class Cropping2D(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Cropping2D, *argv, **kwargs)


class ReLU(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.ReLU, *argv, **kwargs)


class LeakyReLU(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.LeakyReLU, *argv, **kwargs)


class Add(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Add, *argv, **kwargs)
    self.list_as_input = True


class Multiply(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Multiply, *argv, **kwargs)
    self.list_as_input = True


class Concatenate(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    super().__init__(tf.keras.layers.Concatenate, *argv, **kwargs)
    self.list_as_input = True


class Dropout(GenericNonLinear):
  def __init__(self, *argv, **kwargs):
    # TODO for now dropout manage real part and img part separately. better if manage both at the same time
    # solution : only one layer and concat before ? Or define all Dropout with the same seed ?
    super().__init__(tf.keras.layers.Dropout, *argv, **kwargs)


class TF2Upstride:
  """ assume this function should be call to transform TF tensors to Upstride Tensors
  an upstride tensor has the same shape than a TF tensor, but the imaginary part is stack among BS

  For specific types, if you want to add new strategies, then inherit this class, and add strategies inside
  the self.strategies dic
  """

  def __init__(self, strategy='learned', **args):
    # This dictionary map the strategy name to the function to call
    self.strategies = {
        'learned': self.learned_strategy,
        'basic': self.basic
    }
    self.strategy_name = strategy
    self.args = args

  def __call__(self, x):
    if self.strategy_name not in self.strategies:
      raise ValueError(f"unknown strategy: {self.strategy_name}")
    return self.strategies[self.strategy_name](x)

  def learned_strategy(self, x):
    """
    Learning module taken from this paper (https://arxiv.org/pdf/1712.04604.pdf)
    BN --> ReLU --> Conv --> BN --> ReLU --> Conv

    Args:
      x (tensor): Input to the network.
      channels (int): number of filters

    Returns:
      tensor: output of the network. Learned component of the multi-vector.
    """

    channels = self.args.get('channels', 3)
    kernel_size = self.args.get('kernel_size', 3)
    use_bias = self.args.get('use_bias', False)
    kernel_initializer = self.args.get('kernel_initializer','glorot_uniform')
    kernel_regularizer = self.args.get('kernel_regularizer')
    input = x
    outputs = [x]
    for _ in range(1, multivector_length()):
      x = tf.keras.layers.BatchNormalization(axis=1)(input)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), padding='same', use_bias=use_bias, 
          kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(x)
      x = tf.keras.layers.BatchNormalization(axis=1)(x)
      x = tf.keras.layers.Activation('relu')(x)
      x = tf.keras.layers.Conv2D(channels, (kernel_size, kernel_size), padding='same', use_bias=use_bias,
          kernel_initializer=kernel_initializer,kernel_regularizer=kernel_regularizer)(x)
      outputs.append(x)

    return tf.keras.layers.Concatenate(axis=0)(outputs)

  def basic(self, x):
    outputs = [x]
    for _ in range(1, multivector_length()):
      outputs.append(tf.zeros_like(x))
    return tf.keras.layers.Concatenate(axis=0)(outputs)


class Upstride2TF:
  """convert multivector back to real values. 
  """

  def __init__(self, strategy='basic'):
    self.strategies = {
        'basic': self.basic,
        'concat': self.concat,
        'max_pool': self.max_pool,
        'avg_pool': self.avg_pool
    }
    self.strategy_name = strategy

  def __call__(self, x):
    if self.strategy_name not in self.strategies:
      raise ValueError(f"unknown strategy: {self.strategy_name}")
    return self.strategies[self.strategy_name](x)

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