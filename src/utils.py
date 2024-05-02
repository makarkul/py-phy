import numpy as np

def read_file(fname, dtype):
    a = np.fromfile(fname, dtype=dtype)
    if dtype == np.int16:
        re = a[0::2]
        im = a[1::2]
        a = re + 1j*im
    return a

def write_file(fname, data, dtype):
    data.flatten().astype(dtype).tofile(fname)
