import tempfile, os, shutil
import unittest
import tensorflow as tf
import numpy as np
from .layers import TF2Upstride, Upstride2TF, Conv2D, DepthwiseConv2D, Conv2DParcollet


class TestQuaternionTF2Upstride(unittest.TestCase):
  def test_TF2Upstride(self):
    inputs = tf.convert_to_tensor([[[[1.]], [[3]], [[4]]]])
    self.assertEqual(inputs.shape, (1, 3, 1, 1))
    o = TF2Upstride()(inputs)
    self.assertEqual(o.shape, (4, 3, 1, 1))

    o = TF2Upstride("joint")(inputs)
    self.assertEqual(o.shape, (4, 1, 1, 1))

    o = TF2Upstride("grayscale")(inputs)
    self.assertEqual(o.shape, (4, 1, 1, 1))


class TestQuaternionUpstride2TF(unittest.TestCase):
  def test_take_first(self):
    inputs = tf.random.uniform((4, 3, 32, 32))
    output = Upstride2TF()(inputs)
    self.assertEqual(inputs[0].numpy().tolist(), output[0].numpy().tolist())

  def test_concat_channels_first(self):
    inputs = tf.random.uniform((4, 3, 32, 32))
    output = Upstride2TF("concat")(inputs)
    self.assertEqual(output.shape, (1, 12, 32, 32))

  def test_concat_channels_last(self):
    tf.keras.backend.set_image_data_format("channels_last")
    inputs = tf.random.uniform((4, 32, 32, 3))
    output = Upstride2TF("concat")(inputs)
    self.assertEqual(output.shape, (1, 32, 32, 12))
    tf.keras.backend.set_image_data_format("channels_first")

  def test_max_pool(self):
    inputs = tf.random.uniform((4, 3, 32, 32))
    output = Upstride2TF("max_pool")(inputs)
    # check the elements after max pooling
    a = np.max(inputs.numpy(), 0).tolist()
    b = output.numpy()[0].tolist()
    self.assertEqual(a, b)

  def test_avg_pool(self):
    inputs = tf.random.uniform((4, 3, 32, 32))
    output = Upstride2TF("avg_pool")(inputs)
    a = np.mean(inputs.numpy(), 0)
    b = output.numpy()[0]
    self.assertTrue(np.allclose(a, b))

  def test_p_norm(self):
    inputs = tf.random.uniform((4, 3, 32, 32))
    output = Upstride2TF("norm_1")(inputs)

    # check the elements after average pooling
    a = np.linalg.norm(inputs.numpy(), ord=1, axis=0)
    b = output.numpy()[0]
    self.assertTrue(np.allclose(a, b))

    # check norm-2
    output = Upstride2TF("norm_2")(inputs)
    a = np.linalg.norm(inputs.numpy(), ord=2, axis=0)
    b = output.numpy()[0]
    self.assertTrue(np.allclose(a, b))

    # check norm-inf
    output = Upstride2TF("norm_inf")(inputs)
    a = np.linalg.norm(inputs.numpy(), ord=np.inf, axis=0)
    b = output.numpy()[0]
    self.assertTrue(np.allclose(a, b))


class TestConv2DQuaternion(unittest.TestCase):
  def test_conv2d(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = TF2Upstride('basic')(inputs)
    x = Conv2D(4, (3, 3), use_bias=True)(x)
    x = Upstride2TF("basic")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    self.assertEqual(len(model.layers), 4)
    self.assertEqual(model.count_params(), (9*4*3+4)*4)

  def test_export(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input((3, 224, 224))
    x = TF2Upstride('basic')(inputs)
    x = Conv2D(4, (3, 3), use_bias=True)(x)
    x = DepthwiseConv2D(4, (3, 3), use_bias=True)(x)
    x = Upstride2TF("basic")(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    dest = tempfile.mkdtemp()
    tf.saved_model.save(model, dest)
    listdir = os.listdir(dest)
    listdir.sort()
    self.assertEqual(listdir, ['assets', 'saved_model.pb', 'variables'])
    shutil.rmtree(dest)

def random_integer_tensor(shape, dtype=tf.float32):
  """ Generates a random tensor containing integer values
  """
  tensor = tf.random.uniform(shape, -4, +4, dtype=tf.int32)
  return tf.cast(tensor, dtype)

class TestConv2DAlgorithms(unittest.TestCase):
  # def test_conv2d_generalized_and_type2(self):
  #   inputs = random_integer_tensor(shape=(4*2, 5, 3, 3))
  #   # Define and build operations (by calling them a first time)
  #   use_bias = False
  #   ref_op = Conv2D(2, 2, use_bias=use_bias)
  #   test_op = Conv2DType2(2, 2, use_bias=use_bias)
  #   ref_op(inputs)
  #   test_op(inputs)

  #   # Override weights, taking into account that ref_weight is of shape (H, W, I, 4*O), whereas
  #   # test_weight is of shape (4, H, W, I, O)
  #   ref_weight = random_integer_tensor(shape=ref_op.get_weights()[0].shape) # shape (H, W, I, N*O)
  #   if use_bias:
  #     ref_bias = random_integer_tensor(shape=ref_op.get_weights()[1].shape)
  #     ref_op.set_weights([ref_weight, ref_bias])
  #   else:
  #     ref_op.set_weights([ref_weight])

  #   test_weight_and_bias = tf.split(ref_op.get_weights()[0], 4, axis=-1)
  #   if use_bias:
  #     biases = tf.split(ref_op.get_weights()[1], 4, axis=ref_op.bias.axis)
  #     reshaped_biases = [tf.reshape(bias, -1) for bias in biases]
  #     test_weight_and_bias += reshaped_biases
  #   test_op.set_weights(test_weight_and_bias)

  #   # Compute outputs
  #   ref_output = ref_op(inputs)
  #   test_output = test_op(inputs)

  #   # Compare outputs
  #   diff_sum = tf.reduce_max(ref_output - test_output)
  #   self.assertEqual(ref_output.shape, test_output.shape)
  #   self.assertEqual(diff_sum, 0)


  def run_conv2d_generalized_and_parcollet(self, groups=1):
    inputs = random_integer_tensor(shape=(4*2, 6, 3, 3))
    # Define and build operations (by calling them a first time)
    ref_op = Conv2D(4, 2, use_bias=True, groups=groups)
    test_op = Conv2DParcollet(4, 2, use_bias=True, groups=groups)
    ref_op(inputs)
    test_op(inputs)

    # Override weights, taking into account that ref_weight is of shape (H, W, I, 4*O), whereas
    # test_weight is of shape (4, H, W, I, O)
    ref_weight = random_integer_tensor(shape=ref_op.get_weights()[0].shape) # shape (H, W, I, N*O)
    ref_bias = random_integer_tensor(shape=ref_op.get_weights()[1].shape)
    ref_op.set_weights([ref_weight, ref_bias])
    test_weight = tf.reshape(ref_weight, (*ref_weight.shape[:-1], 4, -1)) # shape (H, W, I, N, O)
    test_weight = tf.transpose(test_weight, perm=[3, 0, 1, 2, 4]) # shape (N, H, W, I, O)
    test_bias_shape = [test_weight.shape[0], ref_bias.shape[ref_op.bias.axis]//test_weight.shape[0]]
    test_bias = tf.reshape(ref_bias, test_bias_shape) # shape (N, O)
    test_op.set_weights([test_weight, test_bias])

    # Compute outputs
    ref_output = ref_op(inputs)
    test_output = test_op(inputs)

    # Compare outputs
    diff_sum = tf.reduce_max(ref_output - test_output)
    self.assertEqual(ref_output.shape, test_output.shape)
    self.assertEqual(diff_sum, 0)

  def test_conv2d_generalized_and_parcollet(self):
    self.run_conv2d_generalized_and_parcollet(groups=1)
  #   self.run_conv2d_generalized_and_parcollet(groups=2)