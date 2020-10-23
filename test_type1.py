import unittest
import tensorflow as tf
from upstride.type1.tf.keras import layers


class TestUpstride(unittest.TestCase):
  def test_network(self):
    layers.set_conjugaison_mult(False)
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = layers.TF2Upstride()(inputs)
    self.assertEqual(len(x), 1)
    x = layers.Conv2D(4, (3, 3))(x)
    self.assertEqual(len(x), 2)
    x = layers.ActivationPow2(alpha=2.0)(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3), upstride2tf=True)(x)
    x = layers.Upstride2TF()(x)
    x = layers.ActivationPow2()(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    #tf.keras.utils.plot_model(model, to_file='model_type1.png', show_shapes=True)


if __name__ == "__main__":
  unittest.main()
