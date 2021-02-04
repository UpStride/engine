import unittest
import tensorflow as tf


class TestTF(unittest.TestCase):
  def test_network(self):
    tf.keras.backend.set_image_data_format('channels_first')
    inputs = tf.keras.layers.Input(shape=(3, 224, 224))
    x = tf.keras.layers.Conv2D(8, (3, 3))(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3))(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(8, (3, 3))(x)
    x = tf.keras.layers.Activation('relu')(x)

    model = tf.keras.Model(inputs=[inputs], outputs=[x])
    model.summary()


if __name__ == "__main__":
  unittest.main()
