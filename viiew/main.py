from .utils import get_rgb, get_vrange, is_number, int_string, float_string, short_string, kitty_encode, get_pressed_key
FONTSIZE = 16


def view(data, row0=0, col0=0, nrows=50, ncols=10, cmode=None, sort_by=None, reverse=False, chars=7, end=' ',
         image=False, fontsize=FONTSIZE):
    n_rows, n_cols = len(data), len(data[0])
    has_columns = hasattr(data, 'columns')
    last_sort = None
    reverse_state = reverse
    while True:
        print('Use WASD to navigate, C to change cmode, 0-9 to sort by column and Q to quit')
        if image:
            view_image(data)
        else:
            view_table(data, row0, col0, nrows, ncols, cmode, sort_by, reverse, chars, end)
        key = get_pressed_key()
        if key == 'q': break
        elif key == 'w': row0 = max(0, row0 - 1)
        elif key == 's': row0 = min(max(0, n_rows - nrows), row0 + 1)
        elif key == 'a': col0 = max(0, col0 - 1)
        elif key == 'd': col0 = min(max(0, n_cols - ncols), col0 + 1)
        elif key == 'c': cmode = 0 if cmode == 2 else (cmode + 1) if cmode is not None else 0
        elif key in '0123456789':
            sort_col = int(key) + col0
            if sort_col < n_cols:
                if reverse and last_sort == sort_col:
                    sort_by = None
                    reverse_state = False
                    last_sort = None
                else:
                    sort_by = sort_col
                    reverse_state = not reverse_state if last_sort == sort_col else False
                    last_sort = sort_col
                reverse = reverse_state
        if not image:
            lines = min(nrows, n_rows) + 2 + has_columns
            print(f'\033[A\033[{lines}A')
            spaces = (ncols + 2) * chars * ' '
            print(lines * (spaces + '\n'), end='')
            print(f'\033[A\033[{lines}A')
        else:
            print('\033_Ga=d\033\\', end='')
            print(f'\033[A\033[{len(data) // fontsize}A')
        if key == 'i': image = not image


def view_table(data, row0=0, col0=0, nrows=50, ncols=10, cmode=None, sort_by=None, reverse=False, chars=7, end=' '):
    cmode = 2 if hasattr(data, 'columns') else 1 if cmode is None else cmode
    arr = data.values if hasattr(data, 'columns') else data if hasattr(data, 'min') else None
    rows = arr.tolist() if hasattr(arr, 'tolist') else data
    if arr is None: assert all(len(r) == len(rows[0]) for r in rows), 'All rows must have the same length'
    assert row0 < len(rows), f'row0 must be smaller than {len(rows)}'
    assert col0 < len(rows[0]), f'col0 must be smaller than {len(rows[0])}'
    if sort_by is not None:
        rows = sorted(rows, key=lambda x: x[sort_by], reverse=reverse)
    if cmode:
        vmin, vmax = get_vrange(arr, rows, colwise=cmode > 1)
    nocolor = cmode == 0 or not is_number(vmin) or not is_number(vmax)
    row_idx = list(range(row0, row0 + min(nrows, len(rows) - row0)))
    col_idx = list(range(col0, col0 + min(ncols, len(rows[0]) - col0)))
    idx_chars = len(str(row0 + nrows - 1))
    print(idx_chars * ' ', end=end)
    if hasattr(data, 'columns'): print(chars * ' ', end=end)
    for j in col_idx:
        print(f'{j:{chars}d}', end=end)
    if hasattr(data, 'columns'):
        print('\n' + (idx_chars + chars) * ' ', end=end)
        for c in data.columns[col_idx]:
            print(short_string(c, chars), end=end)
    print()
    for i in row_idx:
        print(f'{i:{idx_chars}d}', end=end)
        if hasattr(data, 'columns'):
            print(short_string(str(data.index[i]), chars), end=end)
        for j in col_idx:
            v = rows[i][j]
            s = (float_string(v, chars) if isinstance(v, float) else
                 int_string(v, chars) if isinstance(v, int) else
                 short_string(v, chars))
            if nocolor or not is_number(v):
                print(s, end=end)
            else:
                rgb = get_rgb(v, vmin=vmin[j if cmode == 2 else 0], vmax=vmax[j if cmode == 2 else 0])
                print(f'\033[48;2;{rgb}m{s}\033[0m', end=end)
        print()


def view_image(data):
    rows = data.tolist() if hasattr(data, 'tolist') else data
    if hasattr(data, 'tolist'): assert all(len(r) == len(rows[0]) for r in rows), 'All rows must have the same length'
    vmin, vmax = get_vrange(data, rows)
    vmin, vmax = vmin[0], vmax[0]
    print(kitty_encode(sum(rows, []), width=len(rows[0]), height=len(rows), vmin=vmin, vmax=vmax), flush=True)
