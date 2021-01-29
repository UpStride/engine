import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.math import sin, cos, atan2, pow, sqrt, greater_equal, multiply


def relu_linebound_fn(z, l):
  """
  lambda = l.z , i.e. the dot product between l and z
  out = max(lambda, 0)*z_norm*exp(I*z_phase) , where I in the imaginary unit
  """

  a, b = tf.split(z, 2, axis=0)

  # Dot product between each complex feature (treated as a 2D vector) 
  # and the vector 'l', representing the normal vector to the line. This operation
  # provides the distance between the line l and each complex point as lambda.
  z_new = tf.concat([tf.expand_dims(a,axis=0), tf.expand_dims(b,axis=0)], axis=0)
  lam = tf.tensordot(l, z_new, axes=1)
  
  condition = greater_equal(lam, 0)
  real = tf.where(condition, tf.tanh(lam)*a, 0)
  imag = tf.where(condition, tf.tanh(lam)*b, 0)

  return tf.concat([real, imag], axis=0)

def swish_linebound_fn(z, l):
  """
  lambda = l.z , i.e. the dot product between l and z
  out = sigmoid(lambda)*z_norm*exp(I*sigmoid(lambda)*z_phase) , where I in the imaginary unit
  """

  a, b = tf.split(z, 2, axis=0)
  phase = atan2(b,a)
  norm = sqrt(pow(a,2)+pow(b,2))

  # Dot product between each complex feature (treated as a 2D vector) 
  # and the vector 'l', representing the normal vector to the line. This operation
  # provides the distance between the line l and each complex point as lambda.
  z_new = tf.concat([tf.expand_dims(a,axis=0), tf.expand_dims(b,axis=0)], axis=0)
  lam = tf.tensordot(l, z_new, axes=1)
  
  smooth_norm = tf.nn.sigmoid(lam)*norm
  smooth_phase = tf.nn.sigmoid(lam)*phase

  real = multiply(smooth_norm,cos(smooth_phase))
  imag = multiply(smooth_norm,sin(smooth_phase))

  return tf.concat([real, imag], axis=0)

class LineBound(Layer):
  """
  Custom activation function that generalizes the concepts of standard activation functions, like ReLU and Swish, to complex-valued tensors.

  Args:
      l (list[float]): pair of floats defining the line
      strategy (str): name of the specific strategy to use in the linebound family of activations
      trainable (bool): define if l will be trained during backprop (True) or fixed (False)
      name (str): name one the layer in the TF graph

  Attributes:
      l (list[float]): pair of floats defining the line
      strategy (str): name of the specific strategy to use in the linebound family of activations
      trainable (bool): define if l will be trained during backprop (True) or fixed (False)
      activation (fn): function corresponding to the strategy passed as argument
  """


  def __init__(self, l=None, strategy='swish', trainable=False, name=None):
    super(LineBound, self).__init__(name=name)

    # safe assignment of default value for mutable objects
    if l:
      self.l = l
    else:
      self.l=[1.0, 1.0]

    self.trainable = trainable

    strategies_factory = {'relu': relu_linebound_fn, 
                          'swish': swish_linebound_fn}
    if strategy in strategies_factory.keys():
      self.activation = strategies_factory[strategy]
    else:
      raise ValueError(f'{strategy} is not a valid strategy for LineBound. Use relu or swish instead.')

  def build(self, input_shape):
    if self.trainable:
      self.l_factor = self.add_weight('kernel_l_factor',
                                    shape=(2,),
                                    trainable=True,
                                    initializer=tf.keras.initializers.RandomUniform(-1.0, 1.0),
                                    constraint=tf.keras.constraints.unit_norm(), 
                                    )
    else:
      self.l_factor = self.add_weight('kernel_l_factor',
                                    shape=(2,),
                                    trainable=False,
                                    initializer=tf.constant_initializer(self.l),
                                    )

  def call(self, input):
    return self.activation(input, self.l_factor)
