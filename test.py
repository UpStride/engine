import unittest
import tensorflow as tf
from upstride.test_generic_layers import *
from upstride.test_initializers import *
from upstride.test_batchnorm import *
from upstride.type1.tf.keras.test_activations import *
from upstride.type2.tf.keras.test_layers import *
from upstride.type2.tf.keras.test_utils import *


if __name__ == "__main__":
  # Channel first is the default for the engine
  tf.keras.backend.set_image_data_format('channels_first')
  unittest.main()
