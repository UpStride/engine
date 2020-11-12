import unittest
import tensorflow as tf
from upstride.type1.tf.keras import layers


class TestUpstride(unittest.TestCase):
  def test_network(self):
    layers.set_conjugaison_mult(False)
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = layers.TF2Upstride(strategy='learned')(inputs)
    #self.assertEqual(len(x), 1)
    x = layers.Conv2D(8, (3, 3), name='test-names')(x)
    self.assertEqual(len(x), 2)
    x = layers.Activation('relu')(x)
    x = layers.BatchNormalizationC()(x)
    x = layers.DepthwiseConv2D(8, (3, 3), kernel_initializer='complex_glorot')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(16, (3, 3), kernel_initializer='complex_glorot')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(24, (3, 3), kernel_initializer='complex_he')(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(50, kernel_initializer='complex_he')(x)
    x = layers.Activation('relu')(x)
    x = layers.Dense(50, upstride2tf=True)(x)
    x = layers.Upstride2TF()(x)
    x = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    #tf.keras.utils.plot_model(model, to_file='model_type1.png', show_shapes=True)


if __name__ == "__main__":
  unittest.main()
