import unittest
from . import utils

class TestQuaternionMult(unittest.TestCase):
  def test_multiply_by_a1(self):
    self.assertEqual(utils.multiply_by_a1([1, 0, 0, 0]), [1, 1, 1, 1])
    self.assertEqual(utils.multiply_by_a1([0, 1, 0, 0]), [1, -1, 1, -1])

  def test_multiply_by_a2(self):
    self.assertEqual(utils.multiply_by_a2([1, 0, 0, 0]), [1, 1, 1, 1])
    self.assertEqual(utils.multiply_by_a2([0, 1, 0, 0]), [1, -1, 1, -1])

  def test_quaternion_mult1(self):
    def op(x, y): return x*y
    self.assertEqual(utils.quaternion_mult1(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
    self.assertEqual(utils.quaternion_mult1(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
    self.assertEqual(utils.quaternion_mult1(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
    self.assertEqual(utils.quaternion_mult1(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
    self.assertEqual(utils.quaternion_mult1(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

  def test_quaternion_mult2(self):
    def op(x, y): return x*y
    self.assertEqual(utils.quaternion_mult2(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
    self.assertEqual(utils.quaternion_mult2(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
    self.assertEqual(utils.quaternion_mult2(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
    self.assertEqual(utils.quaternion_mult2(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
    self.assertEqual(utils.quaternion_mult2(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

  def test_quaternion_mult_naive(self):
    def op(x, y): return x*y
    self.assertEqual(utils.quaternion_mult_naive(op,  [1, 0, 0, 0], [1, 0, 0, 0]), [1, 0, 0, 0])
    self.assertEqual(utils.quaternion_mult_naive(op,  [1, 0, 0, 0], [0, 2, 0, 0]), [0, 2, 0, 0])
    self.assertEqual(utils.quaternion_mult_naive(op,  [0, 2, 2, 0], [0, 2, 0, 0]), [-4, 0, 0, -4])
    self.assertEqual(utils.quaternion_mult_naive(op,  [1, 2, 0, 3], [0, 2, 2, 0]), [-4, -4, 8, 4])
    self.assertEqual(utils.quaternion_mult_naive(op,  [1, 2, 3, 4], [5, 6, 7, 8]), [-60, 12, 30, 24])

#   def test_quaternion_mult_cpp(self):
#     def op(x, y): return x*y
#     def convert_to_list(op_result): return [op_result[i] for i in range(len(op_result))]
#     self.assertEqual(convert_to_list(utils.quaternion_mult_cpp(op,  [1., 0., 0., 0.], [1., 0., 0., 0.])), [1., 0., 0., 0.])
#     self.assertEqual(convert_to_list(utils.quaternion_mult_cpp(op,  [1., 0., 0., 0.], [0., 2., 0., 0.])), [0., 2., 0., 0.])
#     self.assertEqual(convert_to_list(utils.quaternion_mult_cpp(op,  [0., 2., 2., 0.], [0., 2., 0., 0.])), [-4., 0., 0., -4.])
#     self.assertEqual(convert_to_list(utils.quaternion_mult_cpp(op,  [1., 2., 0., 3.], [0., 2., 2., 0.])), [-4., -4., 8., 4.])
#     self.assertEqual(convert_to_list(utils.quaternion_mult_cpp(op,  [1., 2., 3., 4.], [5., 6., 7., 8.])), [-60., 12., 30., 24.])
