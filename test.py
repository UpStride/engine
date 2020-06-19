import unittest
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_math_ops
from upstride import generic_layers
from upstride.generic_layers import _ga_multiply_get_index, upstride_type, unit_multiplier, reorder
from upstride.type2.tf.keras.utils import quaternion_mult1, quaternion_mult2, multiply_by_a1, multiply_by_a2, quaternion_mult_naive
from upstride.type2.tf.keras.layers import TF2Upstride as QTF2Upstride
from upstride.type2.tf.keras.layers import BatchNormalization as QBatchNormalization


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
        generic_layers.change_upstride_type(3, ["", "1", "2", "3", "12", "13", "23", "123"], (3, 0, 0))
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
        self.assertEqual(len(model.layers), 3)  # input, conv, bias
        self.assertEqual(model.count_params(), 9*4*3+4)

    def test_bias_defined_false(self):
        generic_layers.upstride_type = 1
        inputs = tf.keras.layers.Input((224, 224, 3))
        x = generic_layers.TF2Upstride()(inputs)
        x = generic_layers.Conv2D(4, (3, 3), use_bias=False)(x)
        x = generic_layers.Upstride2TF()(x)
        model = tf.keras.Model(inputs=[inputs], outputs=[x])
        self.assertEqual(len(model.layers), 2)  # input, conv
        self.assertEqual(model.count_params(), 9*4*3)

    def test_bias_defined_true(self):
        generic_layers.upstride_type = 1
        inputs = tf.keras.layers.Input((224, 224, 3))
        x = generic_layers.TF2Upstride()(inputs)
        x = generic_layers.Conv2D(4, (3, 3), use_bias=True)(x)
        x = generic_layers.Upstride2TF()(x)
        model = tf.keras.Model(inputs=[inputs], outputs=[x])
        self.assertEqual(len(model.layers), 3)  # input, conv, bias
        self.assertEqual(model.count_params(), 9*4*3+4)


class TestQuaternionTF2Upstride(unittest.TestCase):
    def test_QTF2Upstride(self):
        inputs = tf.convert_to_tensor([[[[1, 3, 4]]]])
        o = QTF2Upstride()(inputs)
        self.assertEqual(type(o), list)
        self.assertEqual(o[0].shape, (1, 1, 1, 3))

        o = QTF2Upstride("rgbinimg")(inputs)
        self.assertEqual(type(o), list)
        self.assertEqual(o[0].shape, (1, 1, 1, 1))
        self.assertEqual(o[1].shape, (1, 1, 1, 1))
        self.assertEqual(o[2].shape, (1, 1, 1, 1))
        self.assertEqual(o[3].shape, (1, 1, 1, 1))


class TestQuaternionMult(unittest.TestCase):
    def test_multiply_by_a1(self):
        self.assertEqual(multiply_by_a1([1, 0, 0, 0]), [1, 1, 1, 1])
        self.assertEqual(multiply_by_a1([0, 1, 0, 0]), [1, -1, 1, -1])

    def test_multiply_by_a2(self):
        self.assertEqual(multiply_by_a2([1, 0, 0, 0]), [1, 1, 1, 1])
        self.assertEqual(multiply_by_a2([0, 1, 0, 0]), [1, -1, 1, -1])

    def test_quaternion_mult1(self):
        def op(x, y): return x*y
        self.assertEqual(quaternion_mult1(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
        self.assertEqual(quaternion_mult1(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
        self.assertEqual(quaternion_mult1(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
        self.assertEqual(quaternion_mult1(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
        self.assertEqual(quaternion_mult1(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

    def test_quaternion_mult2(self):
        def op(x, y): return x*y
        self.assertEqual(quaternion_mult2(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
        self.assertEqual(quaternion_mult2(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
        self.assertEqual(quaternion_mult2(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
        self.assertEqual(quaternion_mult2(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
        self.assertEqual(quaternion_mult2(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

    def test_quaternion_mult_naive(self):
        def op(x, y): return x*y
        self.assertEqual(quaternion_mult_naive(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
        self.assertEqual(quaternion_mult_naive(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
        self.assertEqual(quaternion_mult_naive(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
        self.assertEqual(quaternion_mult_naive(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
        self.assertEqual(quaternion_mult_naive(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

    # def test_big_matric_precision(self):
    #     shape = (1000, 1000)
    #     type = tf.dtypes.float16
    #     a = [tf.random.uniform(shape, minval=0, maxval=1, dtype=type, seed=None, name=None) for i in range(4)]
    #     b = [tf.random.uniform(shape, minval=0, maxval=1/1000, dtype=type, seed=None, name=None) for i in range(4)]
    #     c = quaternion_mult2(gen_math_ops.mat_mul, a, b)
    #     d = c[0] - (gen_math_ops.mat_mul(a[0], b[0])-gen_math_ops.mat_mul(a[1], b[1])-gen_math_ops.mat_mul(a[2], b[2])-gen_math_ops.mat_mul(a[3], b[3]))
    #     e = c[1] - (gen_math_ops.mat_mul(a[0], b[1])+gen_math_ops.mat_mul(a[1], b[0])+gen_math_ops.mat_mul(a[2], b[3])-gen_math_ops.mat_mul(a[3], b[2]))
    #     print(c)
    #     print(tf.math.reduce_max(tf.math.abs(d)))
    #     print(tf.math.reduce_max(tf.math.abs(e)))

    #     c = quaternion_mult1(gen_math_ops.mat_mul, a, b)
    #     d = c[0] - (gen_math_ops.mat_mul(a[0], b[0])-gen_math_ops.mat_mul(a[1], b[1])-gen_math_ops.mat_mul(a[2], b[2])-gen_math_ops.mat_mul(a[3], b[3]))
    #     e = c[1] - (gen_math_ops.mat_mul(a[0], b[1])+gen_math_ops.mat_mul(a[1], b[0])+gen_math_ops.mat_mul(a[2], b[3])-gen_math_ops.mat_mul(a[3], b[2]))
    #     print(tf.math.reduce_max(tf.math.abs(d)))
    #     print(tf.math.reduce_max(tf.math.abs(e)))


class TestQuaternionBN(unittest.TestCase):
    def test_init(self):
        generic_layers.change_upstride_type(2, ["", "12", "13", "23"], (3, 0, 0))
        inputs = tf.convert_to_tensor([[[[1., 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 4, 5, 6]],
                                        [[1, 3, 4, 5, 6], [1, 3, 4, 5, 6], [1, 3, 4, 5, 6]]]])
        self.assertEqual(inputs.shape, (1, 2, 3, 5))
        inputs = [inputs for _ in range(4)]
        bn_layer = QBatchNormalization()
        outputs = bn_layer(inputs, training=False)


if __name__ == "__main__":
    unittest.main()
