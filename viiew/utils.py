import sys, tty, base64, termios


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
    return f'\033_Ga=T,f=24,s={width},v={height};{base64.b64encode(data).decode('ascii')}\033\\'


def number_string(v, nchars):
    if not is_number(v):
        return short_string(str(v), nchars)
    else:
        if isinstance(v, int):
            f_str = f'{v:{nchars}d}'
            return f_str if len(f_str) == nchars else exp_string(v, nchars)
        else:
            f_str = f'{v:1.{nchars - 2 - (v < 0)}f}'[:nchars]
            if f_str == '0.' + (nchars - 2) * '0' or f_str == '-0.' + (nchars - 3) * '0' or '.' not in f_str:
                return exp_string(v, nchars)
            else:
                return f_str


def exp_string(v, nchars):
    space = ' ' if (v > 0 and nchars == 6) or (v < 0 and nchars == 7) else ''
    return space + f'{v:.{max(0, nchars - 6 - (v < 0))}e}'


def short_string(v, nchars):
    return v[:nchars - 2] + '..' if len(v) > nchars else (v + ' ' * nchars)[:nchars]


def get_vrange(arr, rows, colwise=False):
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
    n = max(0, min(v - vmin / vmax , 1)) if vmax != 0 else 0
    return (0, 0, int(255 * (1 - 2 * n))) if n <= .5 else (int(255 * (2 * (n - 0.5))), 0, 0)


def is_number(v):
    return not (str(v) in ['nan', 'inf', '-inf'] or isinstance(v, str))
