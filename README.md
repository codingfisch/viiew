# viiew 🧐
`pip install viiew` to quickly `view(your_data)` in the terminal!

<p align="center">
  <img src="https://github.com/codingfisch/viiew/blob/main/demo.svg">
</p>

`your_data` can be a

- 2D [NumPy](https://github.com/numpy/numpy) array
- [pandas](https://github.com/pandas-dev/pandas) DataFrame
- 2D [PyTorch](https://github.com/pytorch/pytorch) Tensor
- 2D [jax](https://github.com/jax-ml/jax) array
- 2D [tinygrad](https://github.com/tinygrad/tinygrad) Tensor
- list of lists

As shown in the GIF, just call `view`...
```python
import numpy as np
from viiew import view

x = np.arange(256).reshape(16, 16)
view(x)
```
...and e.g. press `s` to scroll through `your_data` (see [**all keybindings**](https://github.com/codingfisch/viiew?tab=readme-ov-file#keybindings))!

`view` calls `view_array` and pressing `s` adds 1 to its `row0` argument.

<details>
  <summary><b>Click here</b>, to read about all arguments of `view` and `view_array` 📑</summary>

`view` and `view_array` take the arguments
- `data`: The data object to view (e.g., numpy array, pandas DataFrame, etc.)  
- `row0`: Starting row index (default: 0)  
- `col0`: Starting column index (default: 0)  
- `nrows`: Number of rows to display (default: 20)  
- `ncols`: Number of columns to display (default: 10)  
- `cidx`: Current column index for sorting (default: None)  
- `order`: Sorting order (0: none, 1: ascending, -1: descending) (default: 0)  
- `color`: Whether to use color coding for values (default: True)  
- `is_table`: Whether to treat the data as a table (auto-detected for pandas DataFrames) (default: None)  
- `expand`: Expansion level for columns (default: 0)  
- `nchars`: Number of characters per cell (default: 7)  
- `end`: String to append after each cell (default: ' ')  
</details>

## Keybindings
- **`w`**: Move up one row
- **`s`**: Move down one row
- **`a`**: Move left one column
- **`d`**: Move right one column
- **`o`**: Cycle through sorting orders (ascending, descending, none)
- **`t`**: Toggle table mode (column-wise colormap)
- **`c`**: Toggle color display
- **`r`**: Decrease number of rows
- **`R`**: Increase number of rows
- **`c`**: Decrease number of columns
- **`C`**: Increase number of columns
- **`e`**: Decrease column expansion
- **`E`**: Increase column expansion
- **`n`**: Decrease number of characters per cell
- **`N`**: Increase number of characters per cell
- **`W`**: Move up 10 rows
- **`S`**: Move down 10 rows
- **`A`**: Move left 10 columns
- **`D`**: Move right 10 columns
- **`q`**: Quit the viewer
