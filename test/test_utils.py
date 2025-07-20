import torch
import tinygrad
import unittest
import numpy as np
import jax.numpy as jnp

from viiew.utils import get_vrange, number_string


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.data = 6 ** np.linspace(-10, 20).reshape(-1, 10)
        self.rows = self.data.tolist()

    def test_positive_float_range(self):
        for nchars in range(5, 8):
            for x in self.data.flatten():
                self.assertEqual(len(number_string(x, nchars)), nchars)

    def test_negative_float_range(self):
        for nchars in range(6, 9):
            for x in self.data.flatten():
                self.assertEqual(len(number_string(-x, nchars)), nchars)

    def test_value_range(self):
        for is_table in (True, False):
            vmin_true = self.data.min(0 if is_table else None)
            vmax_true = self.data.max(0 if is_table else None)
            for framework in ['numpy', 'torch', 'jax', 'tinygrad']:
                data = {'numpy': self.data.copy(), 'torch': torch.Tensor(self.data),
                        'jax': jnp.asarray(self.data), 'tinygrad': tinygrad.Tensor(self.rows)}[framework]
                vmin, vmax = get_vrange(data, self.rows, colwise=is_table)
                if is_table:
                    for vt, v in zip(vmin_true, vmin):
                        self.assertAlmostEqual(1, (vt / v).item(), places=6)
                    for vt, v in zip(vmax_true, vmax):
                        self.assertAlmostEqual(1, (vt / v).item(), places=6)
                else:
                    #self.assertEqual(vmin_true.item(), vmin[0])
                    #self.assertEqual(vmax_true.item(), vmax[0])
                    self.assertAlmostEqual(1, vmin_true / vmin[0])
                    self.assertAlmostEqual(1, vmax_true / vmax[0])


if __name__ == "__main__":
    unittest.main()
