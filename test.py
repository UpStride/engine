import unittest
import tensorflow as tf
from upstride.type1.tf.keras import layers


class TestUpstride(unittest.TestCase):
    def test_network(self):
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = layers.TF2Upstride()(inputs)
        self.assertEqual(len(x), 1)
        x = layers.Conv2D(6, (3, 3))(x)
        self.assertEqual(len(x), 2)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(6, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(6, (3, 3))(x)
        x = layers.Activation('relu')(x)
        x = layers.Upstride2TF()(x)

        model = tf.keras.Model(inputs=[inputs], outputs=[x])
        model.summary()
        tf.keras.utils.plot_model(model)
        # for l in model.layers:
        #     print(type(l))

        #     tf.keras.engine.input_layer.InputLayer
        #     tf.keras.layers.convolutional.Conv2D
        #     tf.keras.layers.core.Activation
        #     tf.keras.layers.convolutional.Conv2D
        #     tf.keras.layers.convolutional.Conv2D
        #     tf.keras.layers.core.Activation
        #     tf.keras.engine.base_layer.TensorFlowOpLayer
        #     tf.keras.layers.convolutional.Conv2D
        #     tf.keras.engine.base_layer.TensorFlowOpLayer
        #     tf.keras.engine.base_layer.TensorFlowOpLayer
        #     tf.keras.engine.base_layer.TensorFlowOpLayer


if __name__ == "__main__":
    unittest.main()
