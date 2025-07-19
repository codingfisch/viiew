import shutil

from .utils import get_rgb, get_vrange, is_number, int_string, float_string, short_string, kitty_encode, get_pressed_key
FONTSIZE = 16


def view(data, row0=0, col0=0, nrows=50, ncols=10, cidx=None, order=0, color=True, is_table=None,
         expand=0, nchars=7, end=' '):
    if hasattr(data, 'shape'):
        n_rows, n_cols = data.shape[:2]
    else:
        n_rows, n_cols = len(data), len(data[0])
    cidx = col0 if cidx is None else max(0, min(cidx, n_cols - ncols - 1))
    rows = None
    while True:
        print('Use WASD to navigate and Q to quit. Full list of keybindings: https://github.com/codingfisch/viiew')
        rows = view_array(data, row0, col0, nrows, ncols, cidx, order, color, is_table, nchars, expand, end, rows)
        key = get_pressed_key()
        if key.lower() == 'q': break
        elif key.lower() == 'w': row0 = max(0, row0 - (1 + 9 * (key == 'W')))
        elif key.lower() == 's': row0 = min(max(0, n_rows - nrows), row0 + (1 + 9 * (key == 'S')))
        elif key.lower() == 'a':
            step = 1 + 9 * (key == 'A')
            cidx = max(0, cidx - step)
            if cidx < col0: col0 = max(0, col0 - step)
        elif key.lower() == 'd':
            step = 1 + 9 * (key == 'D')
            cidx = min(n_cols - 1, cidx + step)
            if cidx >= col0 + ncols: col0 = min(max(0, n_cols - ncols), col0 + step)
        elif key == 'o': order = -1 if order > 0 else order + 1
        elif key == 'c': color = not color
        elif key == 't': is_table = not is_table
        elif key == 'E': expand += 1
        elif key == 'e': expand = max(0, expand - 1)
        elif key == 'N': nchars += 1
        elif key == 'n': nchars = max(0, nchars - 1)
        lines = min(nrows, n_rows) + 2 + hasattr(data, 'columns')
        print(f'\033[A\033[{lines}A')
        if key in 'nf':
            print(lines * (shutil.get_terminal_size()[0] * ' ' + '\n'), end='')
            print(f'\033[A\033[{lines}A')


def view_array(data, row0=0, col0=0, nrows=50, ncols=10, cidx=None, order=0, color=True, is_table=None,
               nchars=7, expand=0, end=' ', rows=None):
    is_table = hasattr(data, 'columns') if is_table is None else is_table
    arr = data.values if hasattr(data, 'columns') else data if hasattr(data, 'min') else None
    if rows is None:
        rows = arr.tolist() if hasattr(arr, 'tolist') else data
    if arr is None: assert all(len(r) == len(rows[0]) for r in rows), 'All rows must have the same length'
    assert row0 < len(rows), f'row0 must be smaller than {len(rows)}'
    assert col0 < len(rows[0]), f'col0 must be smaller than {len(rows[0])}'
    if order:
        row_idx = sorted([[i, r[cidx]] for i, r in enumerate(rows)], key=lambda row: row[1], reverse=order == -1)
        row_idx = [row[0] for row in row_idx[row0:row0 + min(nrows, len(rows) - row0)]]
    else:
        row_idx = list(range(row0, row0 + min(nrows, len(rows) - row0)))
    col_idx = list(range(col0, col0 + min(ncols, len(rows[0]) - col0)))
    if color: vmin, vmax = get_vrange(arr, rows, colwise=is_table)
    idx_chars = max(len(str(idx)) for idx in row_idx)
    print(idx_chars * ' ', end=end)
    if hasattr(data, 'columns'): print(nchars * ' ', end=end)
    for j in col_idx:
        print(f'\033[{7 * (j == cidx)}m{j:{nchars + expand * (j == cidx)}d}\033[0m', end=end)
    if hasattr(data, 'columns'):
        print('\n' + (idx_chars + nchars) * ' ', end=end)
        for c in data.columns[col_idx]:
            print(short_string(c, nchars + expand * (c == cidx)), end=end)
    print()
    for i in row_idx:
        print(f'{i:{idx_chars}d}', end=end)
        if hasattr(data, 'columns'):
            print(short_string(str(data.index[i]), nchars), end=end)
        for j in col_idx:
            jchars = nchars + expand * (j == cidx)
            vrange = (vmin[j if is_table else 0], vmax[j if is_table else 0]) if color else None
            v = rows[i][j]
            s = (float_string(v, jchars) if isinstance(v, float) else
                 int_string(v, jchars) if isinstance(v, int) else
                 short_string(v, jchars))
            if color and all(is_number(x) for x in [v, *vrange]):
                rgb = get_rgb(v, vmin=vrange[0], vmax=vrange[1])
                print(f'\033[48;2;{rgb}m{s}\033[0m', end=end)
            else:
                print(s, end=end)
        print()
    return rows


def view_image(data):
    rows = data.tolist() if hasattr(data, 'tolist') else data
    if hasattr(data, 'tolist'): assert all(len(r) == len(rows[0]) for r in rows), 'All rows must have the same length'
    vmin, vmax = get_vrange(data, rows)
    vmin, vmax = vmin[0], vmax[0]
    print(kitty_encode(sum(rows, []), width=len(rows[0]), height=len(rows), vmin=vmin, vmax=vmax), flush=True)
