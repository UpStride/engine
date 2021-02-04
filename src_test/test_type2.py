import unittest
import tensorflow as tf
from upstride.type2.tf.keras import layers


class TestType2(unittest.TestCase):
  def test_network(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input(shape=(3, 24, 24))
    x = layers.TF2Upstride()(inputs)
    x = layers.Conv2D(4, (3, 3), name='test-names')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(10)(x)
    x = layers.Upstride2TF()(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
