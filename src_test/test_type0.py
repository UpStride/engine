import unittest
import tensorflow as tf

from upstride.type0.tf.keras import layers


class TestType0(unittest.TestCase):
  def test_network(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input(shape=(3, 32, 32))
    x = layers.TF2Upstride()(inputs)
    x = layers.Conv2D(8, (3, 3), name='test-names')(x)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.DepthwiseConv2D((3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(24, (3, 3))(x)
    x = layers.BatchNormalization(center=False, scale=False)(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(50)(x)
    x = layers.Upstride2TF()(x)
    x = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
