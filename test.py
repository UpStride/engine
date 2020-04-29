import unittest
import tensorflow as tf
from upstride.generic_layers import _ga_multiply_get_index, upstride_type, unit_multiplier, reorder


class TestGAMultiplication(unittest.TestCase):
    def test_ga_multiply_get_index(self):
        even_n_permutation, index = _ga_multiply_get_index("123", "12")
        self.assertEqual(even_n_permutation, False)
        self.assertEqual(index, "3")
        even_n_permutation, index = _ga_multiply_get_index("13", "12")
        self.assertEqual(even_n_permutation, True)
        self.assertEqual(index, "23")
        even_n_permutation, index = _ga_multiply_get_index("3", "2")
        self.assertEqual(even_n_permutation, False)
        self.assertEqual(index, "23")
        even_n_permutation, index = _ga_multiply_get_index("232", "32")
        self.assertEqual(even_n_permutation, False)
        self.assertEqual(index, "2")
        even_n_permutation, index = _ga_multiply_get_index("2", "2")
        self.assertEqual(even_n_permutation, True)
        self.assertEqual(index, "")

    def test_unit_multiplier(self):
        global upstride_type
        upstride_type = 3
        # order : (scalar, e1, e2, e3, e12, e13, e23, e123)
        self.assertEqual(unit_multiplier(0, 0), (0, 1))  # 1*1 = 1
        self.assertEqual(unit_multiplier(3, 3), (0, 1))  # e_3*e_3 = 1
        self.assertEqual(unit_multiplier(4, 5), (6, -1))  # e12 * e13 = -e23
        self.assertEqual(unit_multiplier(6, 7), (1, -1))  # e23 * e123 = -e1

    def test_reorder(self):
        inputs = [[1, 2, 3, 4], [5, 6, 7, 8]]
        self.assertEqual(reorder(inputs), [[1, 5], [2, 6], [3, 7], [4, 8]])


if __name__ == "__main__":
    unittest.main()
