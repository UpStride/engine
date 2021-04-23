import tensorflow as tf
from tensorflow.keras.layers import *
from upstride.generic_layers import GeometricMatrixMultiply as GenericGeometricMatrixMultiply
from upstride.uptypes_utilities import UPTYPE0

class TF2Upstride(Layer):
  def __init__(self, strategy='', **kwargs):
    super().__init__()

class Upstride2TF(Layer):
  def __init__(self, strategy='', **kwargs):
    super().__init__()

class GeometricMatrixMultiply(GenericGeometricMatrixMultiply):
  def __init__(self, *args, **kwargs):
    super().__init__(UPTYPE0, *args, **kwargs)
