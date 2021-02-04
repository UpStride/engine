import unittest
import tensorflow as tf
from upstride.type3.tf.keras import layers


class TestType3(unittest.TestCase):
  def test_network(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input(shape=(3, 224, 224))
    x = layers.TF2Upstride()(inputs)
    x = layers.Conv2D(6, (3, 3), name='test-names')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(6, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(6, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Upstride2TF()(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
