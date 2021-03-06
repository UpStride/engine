import unittest
from collections import defaultdict
from functools import reduce

import numpy as np
import tensorflow as tf
# from src_test.bn_from_dcn import ComplexBatchNormalization

from upstride.type1.tf.keras import layers
from upstride.batchnorm import BatchNormalizationC, BatchNormalizationH
from .dcn.bn_from_dcn import ComplexBatchNormalization


class TestQuaternionBN(unittest.TestCase):
  def test_init(self):
    """ Basic test to see if we can call BN on a simple case
    """
    inputs = tf.convert_to_tensor([[[[1., 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                    [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]]])
    self.assertEqual(inputs.shape, (4, 2, 3, 5))
    bn_layer = BatchNormalizationH()
    outputs = bn_layer(inputs, training=True)
    self.assertEqual(outputs.shape, (4, 2, 3, 5))
    self.assertTrue(np.array_equal(outputs, np.zeros((4, 2, 3, 5))))

  def test_compute_sqrt_inv(self):
    """
    From a matrix M,
    Call compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    inputs = tf.random.normal(shape=(4, 2, 3, 5))
    bn_layer = BatchNormalizationH()
    # call it once to init everything
    bn_layer(inputs, training=False)
    inputs = tf.split(inputs, 4, axis=0)
    _, v = bn_layer.compute_mean_var(inputs)
    w = bn_layer.compute_sqrt_inv(v)

    for i in range(1):
      v_matrix = np.array([[v['rr'][i], v['ri'][i], v['rj'][i], v['rk'][i]],
                           [v['ri'][i], v['ii'][i], v['ij'][i], v['ik'][i]],
                           [v['rj'][i], v['ij'][i], v['jj'][i], v['jk'][i]],
                           [v['rk'][i], v['ik'][i], v['jk'][i], v['kk'][i]]])

      w_matrix = np.array([[w['rr'][i],          0, 0, 0],
                           [w['ri'][i], w['ii'][i], 0, 0],
                           [w['rj'][i], w['ij'][i], w['jj'][i], 0],
                           [w['rk'][i], w['ik'][i], w['jk'][i], w['kk'][i]]])
      should_be_id = np.dot(np.dot(w_matrix.T, w_matrix), v_matrix)
      for i in range(4):
        for j in range(4):
          if i == j:
            self.assertAlmostEqual(should_be_id[i][j], 1, 5)
          else:
            self.assertAlmostEqual(should_be_id[i][j], 0, 5)


class TestComplexBN(unittest.TestCase):
  def test_init(self):
    inputs = tf.convert_to_tensor([[[[1., 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
                                    [[2, 2, 2, 2, 2], [2, 2, 2, 2, 2], [2, 2, 2, 2, 2]]],
                                   [[[3, 3, 3, 3, 3], [3, 3, 3, 3, 3], [3, 3, 3, 3, 3]],
                                    [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]]])
    self.assertEqual(inputs.shape, (2, 2, 3, 5))
    bn_layer = BatchNormalizationC()
    outputs = bn_layer(inputs, training=True)
    self.assertEqual(outputs.shape, (2, 2, 3, 5))
    self.assertTrue(np.array_equal(outputs, np.zeros((2, 2, 3, 5))))

  def test_compute_sqrt_inv(self):
    """
    From a matrix M,
    Call a compute_sqrt_inv to compute the inverse of the square root of M, I
    multiply M by I**2 to check that this gives Id
    """
    inputs = tf.random.normal(shape=(10, 5, 3, 2))
    bn_layer = BatchNormalizationC()
    # call it once to init everything
    bn_layer(inputs, training=False)
    inputs = tf.split(inputs, 2, axis=0)
    _, v = bn_layer.compute_mean_var(inputs)
    w = bn_layer.compute_sqrt_inv(v)

    for i in range(5):
      v_matrix = np.array([[v['rr'][i], v['ri'][i]], [v['ri'][i], v['ii'][i]]])
      w_matrix = np.array([[w['rr'][i], w['ri'][i]], [w['ri'][i], w['ii'][i]]])
      should_be_id = np.dot(np.dot(w_matrix, w_matrix), v_matrix)
      self.assertAlmostEqual(should_be_id[0][0], 1, 6)
      self.assertAlmostEqual(should_be_id[1][1], 1, 6)
      self.assertAlmostEqual(should_be_id[1][0], 0, 6)
      self.assertAlmostEqual(should_be_id[0][1], 0, 6)


class Channel2Batch(tf.keras.layers.Layer):
  # our BN implementation expects the inputs of multivector component in the Batch
  # Hence we split the inputs on the channel axis and concat them to the Batch
  def call(self, inputs):
    x = tf.split(inputs, 2, axis=1)
    return tf.concat(x, axis=0)


class Batch2Channel(tf.keras.layers.Layer):
  # We do the reverse. Split the inputs on the batch and concat them on the channel.
  def call(self, inputs):
    x = tf.split(inputs, 2, axis=0)
    return tf.concat(x, axis=1)


class TestBatchNorm(unittest.TestCase):
  @classmethod
  def setUp(self):
    self.batch_size = 16
    self.bn_args = {
        'axis': 1,
        'center': True,
        'scale': True,
        'momentum': 0.9,
        'epsilon': 1e-4
    }
    self.epochs = 20
    self.loss = tf.keras.losses.MeanSquaredError()
    self.input_shape = (6, 3, 3)

  def get_dataset(self, list_of_batches):
    # gets the list of tensors, concats them on the batch
    # outputs tf.data.dataset object with specified batch size
    X = tf.concat(list_of_batches, axis=0)
    y = tf.reshape(X, (self.batch_size * len(list_of_batches), reduce(lambda x, y: x*y, self.input_shape)))
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(self.batch_size)
    return dataset

  def get_model(self, is_dcn_ours=True):
    inputs = tf.keras.layers.Input(shape=self.input_shape)
    if is_dcn_ours:
      x = Channel2Batch()(inputs)
      x = layers.BatchNormalizationC(**self.bn_args)(x)
      x = Batch2Channel()(x)
      x = layers.Flatten()(x)
    else:
      x = ComplexBatchNormalization(**self.bn_args)(inputs)
      x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    return model

  def custom_training_loop(self, dataset, model, optimizer):
    weight_dict = defaultdict(list)
    for e in range(self.epochs):
      # print(f"Epoch number {e}/{self.epochs}")
      for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
          y_hat = model(x_batch, training=True)
          loss_value = self.loss(y_batch, y_hat)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        for weight in model.weights:
          weight_name = weight.name.split('/')[1]
          weight_dict[weight_name].append([e, step, weight.numpy()[0]])  # picking the first value from the weight (values ares same)
    return weight_dict

  def compare_runs(self, list_of_batches, lr=0.01):
    dataset = self.get_dataset(list_of_batches)

    # Training for DCN ours
    tf.keras.backend.clear_session()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model_dcn_ours = self.get_model(is_dcn_ours=True)
    weights_dcn_ours = self.custom_training_loop(dataset, model_dcn_ours, optimizer)

    # Training for DCN source
    tf.keras.backend.clear_session()
    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model_dcn_source = self.get_model(is_dcn_ours=False)
    weights_dcn_source = self.custom_training_loop(dataset, model_dcn_source, optimizer)

    # For the purpose of comparison these are removed from the DCN ours, DCN source adds weight for the whole input shape
    del weights_dcn_ours['betai:0']
    del weights_dcn_ours['moving_meani:0']

    for (k1, v1), (k2, v2) in zip(sorted(weights_dcn_ours.items()), sorted(weights_dcn_source.items())):
      self.assertTrue(np.allclose(v1, v2, atol=1e-7))

  def test_dataset_ones_lr_zero(self):
    list_of_batches = []
    for i in range(1, 11):
      list_of_batches.append(tf.ones((self.batch_size, 6, 3, 3)) * i)
    self.compare_runs(list_of_batches, lr=0.0)

  def test_dataset_ones(self):
    list_of_batches = []
    for i in range(1, 11):
      list_of_batches.append(tf.ones((self.batch_size, 6, 3, 3)) * i)
    self.compare_runs(list_of_batches)

  def test_dataset_ones_2i(self):
    list_of_batches = []
    for i in range(1, 11):
      list_of_batches.append(tf.ones((self.batch_size, 6, 3, 3)) * (2*i))
    self.compare_runs(list_of_batches)

  def test_dataset_r_ones_i_zeros(self):
    list_of_batches = []
    for i in range(1, 11):
      re = tf.ones((self.batch_size, 3, 3, 3))
      im = tf.zeros((self.batch_size, 3, 3, 3))
      list_of_batches.append(tf.concat([re, im], axis=1))
    self.compare_runs(list_of_batches)

  def test_dataset_r_i_random_normal(self):
    list_of_batches = []
    for _ in range(10):
      re = tf.random.normal((self.batch_size, 3, 3, 3), seed=42)
      im = tf.random.normal((self.batch_size, 3, 3, 3), seed=42)
      list_of_batches.append(tf.concat([re, im], axis=1))
    self.compare_runs(list_of_batches)

  def test_dataset_r_i_random_normal_mean_stddev(self):
    list_of_batches = []
    for _ in range(10):
      re = tf.random.normal((self.batch_size, 3, 3, 3), seed=42)
      im = tf.random.normal((self.batch_size, 3, 3, 3), seed=42, mean=1., stddev=2.)
      list_of_batches.append(tf.concat([re, im], axis=1))
    self.compare_runs(list_of_batches)

  def test_dataset_r_i_random_uniform(self):
    list_of_batches = []
    for _ in range(10):
      re = tf.random.uniform((self.batch_size, 3, 3, 3), seed=42)
      im = tf.random.uniform((self.batch_size, 3, 3, 3), seed=42)
      list_of_batches.append(tf.concat([re, im], axis=1))
    self.compare_runs(list_of_batches)

  # For quick manual check
  # os.makedirs("DCN_ours",exist_ok=True)
  # os.makedirs("DCN_source",exist_ok=True)

  # for f in weights_dcn_ours.keys():
  #   filename = f"DCN_ours/{f}.csv"
  #   with open (filename, 'w') as fi:
  #     csv_writer = csv.writer(fi)
  #     csv_writer.writerow(['Step','Weight_value'])
  #     csv_writer.writerows(weights_dcn_ours[f])

  # for f in weights_dcn_source.keys():
  #   filename = f"DCN_source/{f}.csv"
  #   with open (filename, 'w') as fi:
  #     csv_writer = csv.writer(fi)
  #     csv_writer.writerow(['Step','Weight_value'])
  #     csv_writer.writerows(weights_dcn_source[f])
