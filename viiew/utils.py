import sys, tty, math, base64, termios, platform


def get_pressed_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key


def kitty_encode(data, vmin, vmax, width, height):
    data = b''.join([3 * int(255 * (float(i) - vmin)/(vmax - vmin)).to_bytes() for i in data])  # data.flatten()
    return f'\033_Ga=T,f=24,s={width},v={height};{base64.b64encode(data).decode("ascii")}\033\\'


def float_string(v, chars):
    if not is_number(v):
        return short_string(str(v), chars)
    elif v > 10 ** chars:
        log_v = int(math.log10(v))
        return f'{v / 10 ** log_v:.{chars - 6}f}e+{int(math.log10(v)):02d}'
    elif v > 10 ** (chars - 2):
        return f'{int(v):{chars}d}'
    elif v > 10 ** -(chars - 2):
        return f'{v:{chars}.{chars - len(str(int(v))) - 1}f}'
    else:
        return f'{v:{chars}.{chars - 5}g}'


def int_string(v, chars):
    if -10 ** (chars - 1) < v < 10 ** chars:
        return f'{v:{chars}d}'
    else:
        n = 5 + (v < 0)
        assert chars >= n, f'This int needs at least {n} digits. Set chars to at least {n}!'
        log_v = int(math.log10(abs(v)))
        sign = '+' if v >= 1 else '-'
        string = f'{v / 10 ** log_v:.{chars - (6 + (v < 0))}f}e{sign}{log_v:02d}'
        return (chars * ' ' + string)[-chars:]


def short_string(v, chars):
    return v[:chars - 2] + '..' if len(v) > chars else (v + ' ' * chars)[:chars]


def get_vrange(arr, rows, colwise=0):
    if arr is None:
        vmin, vmax = [], []
        nrows, ncols = len(rows), len(rows[0])
        for i in range(ncols):
            col = [rows[j][i] for j in range(nrows)]
            vmin.append(min(col))
            vmax.append(max(col))
        vmin = vmin if colwise else [min(vmin)]
        vmax = vmax if colwise else [max(vmax)]
    else:
        vmin = (arr.amin(0) if hasattr(arr, 'amin') else arr.min(0)).tolist() if colwise else [arr.min().item()]
        vmax = (arr.amax(0) if hasattr(arr, 'amax') else arr.max(0)).tolist() if colwise else [arr.max().item()]
    return vmin, vmax


def get_rgb(v, vmin, vmax):
    n = (v - vmin) / vmax
    r, g, b = (0, 0, int(255 * (1 - 2 * n))) if n <= .5 else (int(255 * (2 * (n - 0.5))), 0, 0)
    return f'{r};{g};{b}'


def is_number(v):
    return not (str(v) in ['nan', 'inf', '-inf'] or isinstance(v, str))
