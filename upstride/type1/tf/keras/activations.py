import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import activations
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers
from tensorflow.python.keras import backend as K
from tensorflow.math import sin, cos, sinh, cosh, pow, multiply, scalar_mul


def cos_fn(z):
  a, b = tf.split(z, 2, axis=0)
  real = multiply(cos(a),cosh(b))
  real += a
  imag = multiply(-sin(a),sinh(b))
  imag += b
  return tf.concat([real,imag], axis=0)

def cos_fn_grad(z):
  """
  Backward pass (gradient) of the activation function: F(z)=cos(z)+1, with z=a+ib
  """
  a, b = tf.split(z, 2, axis=0)
  gradF_a = -sin(a)*cosh(b)+1-cos(a)*sinh(b)
  gradF_b = cos(a)*sinh(b)-sin(a)*cosh(b)+1
  return tf.concat([gradF_a, gradF_b], axis=0)


def pow2_fn(x, alpha=1.0):
  a, b = tf.split(x, 2, axis=0)
  alpha = np.float(alpha)

  real = pow(a,2) - pow(b,2)
  real = scalar_mul( alpha, real )
  imag = scalar_mul( 2.0, multiply(a,b) )
  imag = scalar_mul( alpha, imag )
  
  return tf.concat([real, imag], axis=0)
  
def pow2_fn_grad(z, alpha=1.0):
  """
  Backward pass (gradient) of the activation function: F(z)=z^2, with z=a+ib
  """

  a, b = tf.split(z, 2, axis=0)
  gradF_a = alpha*(2*a+2*b)
  gradF_b = alpha*(2*a-2*b)

  return tf.concat([gradF_a, gradF_b], axis=0)


class ActivationCos(Layer):
  """
  Activation function for complex numbers z=a+ib
  Forward pass of the activation function: F(z)=cos(z)+1
  We can rewrite F(z) as F(a+ib)=[cos(a)cosh(b)+a]+i[-sin(a)sinh(b)+b]
  """
  def __init__(self):
    super(ActivationCos, self).__init__()

  #def build(self):
  #  pass

  def call(self, input):
    return cos_fn(input)


class ActivationPow2(Layer):
  """
  Activation function for complex numbers z=a+ib
  Forward pass of the activation function: F(z)=z^2
  We can rewrite F(z) as F(a+ib)=[a^2-b^2]+i[2ab]
  """

  def __init__(self, alpha=1.0, trainable=False):
    super(ActivationPow2, self).__init__()
    self.alpha = alpha
    self.trainable = trainable

  def build(self, input_shape):
    self.alpha_factor = K.variable(self.alpha,
                                  dtype=K.floatx(),
                                  name='alpha_factor')
    if self.trainable:
        self._trainable_weights.append(self.alpha_factor)

    super(ActivationPow2, self).build(input_shape)

  def call(self, input):
    return pow2_fn(input, self.alpha_factor)
