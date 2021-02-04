import unittest
import tensorflow as tf
from upstride.test_generic_layers import *
from upstride.test_initializers import *
from upstride.test_batchnorm import *
from upstride.type1.tf.keras.test_activations import *
from upstride.type2.tf.keras.test_layers import *
from upstride.type2.tf.keras.test_utils import *
from src_test.test_tf import *
from src_test.test_type0 import *
from src_test.test_type1 import *
from src_test.test_type2 import *
from src_test.test_type3 import *

if __name__ == "__main__":
  # Channel first is the default for the engine
  tf.keras.backend.set_image_data_format('channels_first')
  unittest.main()
