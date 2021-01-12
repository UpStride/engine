# TO BE MOVED to test_batchnorm.py once the unittests are passing.
import unittest
from collections import defaultdict
from functools import reduce

import tensorflow as tf

import numpy as np 

from upstride.type1.tf.keras import layers
from bn_dcn_source import ComplexBatchNormalization

tf.keras.backend.set_image_data_format('channels_first')

class C_to_B(tf.keras.layers.Layer):
  # our BN implementation expects the inputs of multivector component in the Batch
  # Hence we split the inputs on the channel axis and concat them to the Batch 
  def call(self, inputs):
    x = tf.split(inputs, 2, axis=1)
    return tf.concat(x, axis=0)
    
class B_to_C(tf.keras.layers.Layer):
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
      'center' : True,
      'scale' : True,
      'momentum' : 0.9,
      'epsilon' : 1e-4
    }
    self.epochs = 20
    self.loss = tf.keras.losses.MeanSquaredError()
    self.optimizer = tf.keras.optimizers.Adam(lr=0.0)
    self.input_shape = (6,3,3)

  def get_dataset(self, l):
    # gets the list of tensors, concats them on the batch
    # outputs tf.data.dataset object with specified batch size
    X = tf.concat(l,axis=0)
    y = tf.reshape(X, (self.batch_size * len(l) ,reduce(lambda x,y: x*y, self.input_shape)))
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.batch(self.batch_size)
    return dataset
    
  def get_model(self,is_dcn_ours=True):
    inputs = tf.keras.layers.Input(shape=self.input_shape)
    if is_dcn_ours:
      x = C_to_B()(inputs)
      x = layers.BatchNormalizationC(**self.bn_args)(x)
      x = B_to_C()(x)
      x = layers.Flatten()(x)
    else:
      x = ComplexBatchNormalization(**self.bn_args)(inputs)
      x = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    return model

  def custom_training_loop(self,dataset,model):
    weight_dict = defaultdict(list)
    for e in range(self.epochs):
      # print(f"Epoch number {e}/{self.epochs}")
      for step, (x_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
          y_hat = model(x_batch, training=True)
          loss_value = self.loss(y_batch, y_hat)
        grads = tape.gradient(loss_value, model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,model.trainable_weights))
        for weight in model.weights:
          weight_name = weight.name.split('/')[1]
          weight_dict[weight_name].append([e,step,weight.numpy()[0]]) #picking the first value from the weight (values ares same)
    del grads, tape, model
    return weight_dict

  def compare_runs(self,l):
    dataset = self.get_dataset(l)
    model_dcn_ours = self.get_model(is_dcn_ours=True)
    weights_dcn_ours = self.custom_training_loop(dataset, model_dcn_ours)

    tf.keras.backend.clear_session()
    # del self.get_model
    # Training for DCN ours
    model_dcn_source = self.get_model(is_dcn_ours=False)
    weights_dcn_source = self.custom_training_loop(dataset, model_dcn_source)

    del weights_dcn_ours['betai:0']
    del weights_dcn_ours['moving_meani:0']
    print(weights_dcn_ours.keys()) 
    print(weights_dcn_source.keys())
    for (k1, v1), (k2, v2) in zip(sorted(weights_dcn_ours.items(),key=lambda x: x), sorted(weights_dcn_source.items(),key=lambda x: x)):
      # if k1 == k2:
        # print(k1,k2)
        # self.assertTrue(np.allclose(v1,v2))
      # #     print(k1,v1[0])
      # #     print(k2,v2[0])
      print(k1,v1[0])
      print(k2,v2[0])

  def test_dataset_dcn_source(self):
    l = []
    for i in range(1, 11): 
      # l.append(tf.ones((16,6,3,3)) * i)
      # # im = tf.ones((16,3,3,3)) * (2*i)
      re = tf.random.normal((self.batch_size,3,3,3), seed=42)
      im = tf.random.normal((self.batch_size,3,3,3), seed=42, mean=1. ,stddev=2.)
      l.append(tf.concat([re,im], axis=1))
    self.compare_runs(l)
    # FIXME which variable is causing to cause differences in the gamma and beta to stay and mess with 2 runs

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

if __name__ == "__main__":
  unittest.main()