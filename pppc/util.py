import time
from functools import wraps


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        print('Delta-t for {}: {} s'.format(func.__name__, t1 - t0))
        return ret
    return wrapper


def class_timeit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        ret = func(self, *args, **kwargs)
        t1 = time.time()
        print('Delta-t for {}: {} s'.format(func.__name__, t1 - t0))
        return ret
    return wrapper
