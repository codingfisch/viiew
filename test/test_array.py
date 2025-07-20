import io
import sys
import torch
import tinygrad
import unittest
import numpy as np
import pandas as pd
import jax.numpy as jnp
from pathlib import Path

from viiew.main import view_array
DATA_PATH = f'{Path(__file__).parents[1].resolve()}/test/data'


class TestMain(unittest.TestCase):
    def setUp(self):
        self.data = 6 ** np.linspace(-10, 20).reshape(-1, 10)
        self.data[::2] *= -1
        self.df = pd.DataFrame({'dog': {'legs': 4, 'weight': 50, 'desc': 'beautiful animal'},
                                'bee': {'legs': 6, 'weight': .005, 'desc': 'annoying'}}).T

    def test_view_array(self):
        for framework in ['python', 'torch', 'jax', 'tinygrad']:
            data = {'python': self.data.tolist(), 'torch': torch.Tensor(self.data),
                    'jax': jnp.asarray(self.data), 'tinygrad': tinygrad.Tensor(self.data.tolist())}[framework]
            for nchars in range(5, 7):  # nchars=8 gives different last digit for numpy vs. other frameworks
                for row0 in (0, 1):
                    for col0 in (0, 1):
                        for order in (-1, 0, 1, 2):
                            if order == 2:
                                order = 0
                                cidx = None
                            else:
                                cidx = 0
                            true_str = get_view_array_string(self.data, color=False, row0=row0, col0=col0,
                                                            nchars=nchars, cidx=cidx, order=order)
                            view_str = get_view_array_string(data, color=False, row0=row0, col0=col0,
                                                             nchars=nchars, cidx=cidx, order=order)
                            self.assertEqual(view_str, true_str)
                            # with open(f'{DATA_PATH}/array/n{nchars}r{row0}c{col0}o{order}.txt', 'w') as f:
                            #     f.write(true_str)

    def test_view_table(self):
        for nchars in range(5, 7):
            for row0 in (0, 1):
                for col0 in (0, 1):
                    for order in (-1, 0, 1, 2):
                        if order == 2:
                            order = 0
                            cidx = None
                        else:
                            cidx = 0
                        view_str = get_view_array_string(self.df, color=False, row0=row0, col0=col0,
                                                         nchars=nchars, cidx=cidx, order=order)
                        with open(f'{DATA_PATH}/table/n{nchars}r{row0}c{col0}o{order}.txt', 'r') as file:
                            true_str = file.read()
                        self.assertEqual(view_str, true_str)
                        # with open(f'{DATA_PATH}/table/n{nchars}r{row0}c{col0}o{order}.txt', 'w') as f:
                        #     f.write(view_str)


def get_view_array_string(data, **kwargs):
    captured_output = io.StringIO()
    sys.stdout = captured_output
    view_array(data, **kwargs)
    return captured_output.getvalue()


if __name__ == "__main__":
    unittest.main()
