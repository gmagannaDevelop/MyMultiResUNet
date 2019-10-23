
from functools import wraps
from time import time

def time_this(f):
    """
      Code snippet taken from : https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
      Modified to work on Python 3.7
      This simply wraps the function and takes the time it takes to compute on a single run. 
      
      One could log it and then perform statistical analysis, rather than using timeit which 
      disables the garbage collector.
    """
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        exec_time = te - ts
        print(f'func:{f.__name__} args:[{args}, {kw}] took: {exec_time:.4f} s')
        return result
    return wrap
##
