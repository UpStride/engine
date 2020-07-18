import unittest
import tensorflow as tf
from upstride.type2.tf.keras import layers


class TestUpstride(unittest.TestCase):
  def test_network(self):
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = layers.TF2Upstride()(inputs)
    self.assertEqual(len(x), 1)
    x = layers.Conv2D(4, (3, 3))(x)
    self.assertEqual(len(x), 4)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(4, (3, 3))(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(100)(x)
    x = layers.Upstride2TF()(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_type2.png', show_shapes=True)


if __name__ == "__main__":
  unittest.main()
