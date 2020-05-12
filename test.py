import unittest
import tensorflow as tf
from upstride import generic_layers
from upstride.generic_layers import _ga_multiply_get_index, upstride_type, unit_multiplier, reorder


class TestGAMultiplication(unittest.TestCase):
    def test_ga_multiply_get_index(self):
        s, index = _ga_multiply_get_index("123", "12")
        self.assertEqual(s, -1)
        self.assertEqual(index, "3")
        s, index = _ga_multiply_get_index("13", "12")
        self.assertEqual(s, 1)
        self.assertEqual(index, "23")
        s, index = _ga_multiply_get_index("3", "2")
        self.assertEqual(s, -1)
        self.assertEqual(index, "23")
        s, index = _ga_multiply_get_index("32", "32")
        self.assertEqual(s, -1)
        self.assertEqual(index, "")
        s, index = _ga_multiply_get_index("2", "2")
        self.assertEqual(s, 1)
        self.assertEqual(index, "")

    def test_unit_multiplier(self):
        generic_layers.upstride_type = 3
        # order : (scalar, e1, e2, e3, e12, e13, e23, e123)
        self.assertEqual(unit_multiplier(0, 0), (0, 1))  # 1*1 = 1
        self.assertEqual(unit_multiplier(3, 3), (0, 1))  # e_3*e_3 = 1
        self.assertEqual(unit_multiplier(4, 5), (6, -1))  # e12 * e13 = -e23
        self.assertEqual(unit_multiplier(6, 7), (1, -1))  # e23 * e123 = -e1

    def test_reorder(self):
        inputs = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(reorder(inputs), [[1, 5], [2, 6], [3, 7], [4, 8]])


    def test_bias_undefined(self):
        generic_layers.upstride_type = 1
        inputs = tf.keras.layers.Input((224, 224, 3))
        x = generic_layers.TF2Upstride()(inputs)
        x = generic_layers.Conv2D(4, (3, 3))(x)
        x = generic_layers.Upstride2TF()(x)
        model = tf.keras.Model(inputs=[inputs], outputs=[x])
        self.assertEqual(len(model.layers), 3) # input, conv, bias
        self.assertEqual(model.count_params(), 9*4*3+4)

    def test_bias_defined_false(self):
        generic_layers.upstride_type = 1
        inputs = tf.keras.layers.Input((224, 224, 3))
        x = generic_layers.TF2Upstride()(inputs)
        x = generic_layers.Conv2D(4, (3, 3), use_bias=False)(x)
        x = generic_layers.Upstride2TF()(x)
        model = tf.keras.Model(inputs=[inputs], outputs=[x])
        self.assertEqual(len(model.layers), 2) # input, conv
        self.assertEqual(model.count_params(), 9*4*3)

    def test_bias_defined_true(self):
        generic_layers.upstride_type = 1
        inputs = tf.keras.layers.Input((224, 224, 3))
        x = generic_layers.TF2Upstride()(inputs)
        x = generic_layers.Conv2D(4, (3, 3), use_bias=True)(x)
        x = generic_layers.Upstride2TF()(x)
        model = tf.keras.Model(inputs=[inputs], outputs=[x])
        self.assertEqual(len(model.layers), 3) # input, conv, bias
        self.assertEqual(model.count_params(), 9*4*3+4)


if __name__ == "__main__":
    unittest.main()
