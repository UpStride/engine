import tensorflow as tf
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
#from .... import generic_layers
#from ....generic_layers import *
from tensorflow.python.keras import backend

from tensorflow.math import sin, cos, sinh, cosh, pow, multiply, scalar_mul
#from numpy import sin, cos, sinh, cosh, power, multiply


'''class Activation(Layer):
  """Applies an activation function to an output.

  Arguments:
    activation: Activation function, such as `tf.nn.relu`, or string name of
      built-in activation function, such as "relu".

  Usage:

  >>> layer = tf.keras.layers.Activation('relu')
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]
  >>> layer = tf.keras.layers.Activation(tf.nn.relu)
  >>> output = layer([-3.0, -1.0, 0.0, 2.0])
  >>> list(output.numpy())
  [0.0, 0.0, 0.0, 2.0]

  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the batch axis)
    when using this layer as the first layer in a model.

  Output shape:
    Same shape as input.
  """

  def __init__(self, activation, **kwargs):
    super(Activation, self).__init__(**kwargs)
    self.supports_masking = True
    self.activation = activations.get(activation)

  def call(self, inputs):
    return self.activation(inputs)

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {'activation': activations.serialize(self.activation)}
    base_config = super(Activation, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))'''


def cos_fn(z):
  a, b = z[0], z[1]

  Re_F = multiply(cos(a),cosh(b))
  Re_F += a

  Im_F = multiply(-sin(a),sinh(b))
  Im_F += b

  return [Re_F, Im_F]

def cos_fn_grad(z):
  """
  Backward pass (gradient) of the activation function: F(z)=cos(z)+1, with z=a+ib
  """

  a, b = z[0], z[1]
  gradF_a = -sin(a)*cosh(b)+1-cos(a)*sinh(b)
  gradF_b = cos(a)*sinh(b)-sin(a)*cosh(b)+1

  return [gradF_a, gradF_b]


def pow2_fn(x):
  a, b = x[0], x[1]

  Re_F = pow(a,2)-pow(b,2)
  Im_F = scalar_mul(2,multiply(a,b))
  
  return [Re_F, Im_F]
  
def pow2_fn_grad(z):
  """
  Backward pass (gradient) of the activation function: F(z)=z^2, with z=a+ib
  """

  a, b = z[0], z[1]
  gradF_a = 2*a+2*b
  gradF_b = 2*a-2*b

  return [gradF_a, gradF_b]


class ActivationCos(Layer):
  """
  Activation function for complex numbers z=a+ib
  Forward pass of the activation function: F(z)=cos(z)+1
  We can rewrite F(z) as F(a+ib)=[cos(a)cosh(b)+a]+i[-sin(a)sinh(b)+b]
  """
  def __init__(self, precomp_grad=False):
    super(ActivationCos, self).__init__()
    self.precomp_grad = precomp_grad

  #def build(self):
  #  pass

  def call(self, input):
    if self.precomp_grad:
      return cos_fn_with_grad(input)
    else:
      return cos_fn(input)


##TODO: add the learnable parameter and initialize it properly in TF

class ActivationPow2(Layer):
  """
  Activation function for complex numbers z=a+ib
  Forward pass of the activation function: F(z)=z^2
  We can rewrite F(z) as F(a+ib)=[a^2-b^2]+i[2ab]
  """

  def __init__(self, alpha=1.0, trainable=False, precomp_grad=False, **kargs):
    super(ActivationPow2, self).__init__(**kargs)
    self.alpha = alpha
    self.trainable = trainable
    self.precomp_grad = precomp_grad

  def build(self, input_shape):
    self.alpha_factor = K.variable(self.alpha,
                                  dtype=K.floatx(),
                                  name='alpha_factor')
    if self.trainable:
        self._trainable_weights.append(self.alpha_factor)

    super(ActivationPow2, self).build(input_shape) 

  def call(self, input):
    if self.precomp_grad:
      return pow2_fn_with_grad(input)
    else:
      return pow2_fn(input)
