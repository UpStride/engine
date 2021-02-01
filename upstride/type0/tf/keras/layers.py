import tensorflow as tf
from tensorflow.keras.layers import *

class TF2Upstride(tf.keras.layers.Layer):
  """ Dummy class
  """

  def __init__(self, strategy='', **kwargs):
      super().__init__(**kwargs)

  def call(self, input_tensor):
    return input_tensor


class Upstride2TF(tf.keras.layers.Layer):
  """Dummy class 
  """

  def __init__(self, strategy=''):
    super().__init__()
