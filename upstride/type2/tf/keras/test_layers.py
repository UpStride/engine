import tempfile, os, shutil
import unittest
import tensorflow as tf
import numpy as np
from .layers import TF2Upstride, Upstride2TF, Conv2D, DepthwiseConv2D


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

