import unittest
import tensorflow as tf


class TestUpstride(unittest.TestCase):
  def test_network(self):
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Conv2D(8, (3, 3))(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3))(x)
    x = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()
    tf.keras.utils.plot_model(model, to_file='model_tf.png', show_shapes=True)


if __name__ == "__main__":
  unittest.main()
