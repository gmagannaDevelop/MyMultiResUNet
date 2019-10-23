
import numpy as np
import timing
import multiresunet as mru

# The argument of timing.time_log() is a string
# contaning the log file (saves to current working directory)
# or the absolute path to the log file.
@timing.time_log('example_log.jsonl')
def _example_matrix_operation(mu: float, sigma: float, n: int = 1000):
    """
        Some dummy matrix operation.
    """
    y = np.array([np.random.normal(loc=mu, scale=sigma) for _ in range(n)])
    return y @ y.T
##

@timing.time_log('non_trivial_log.jsonl')
def print_type(foo):
    """
        print the type of an argument.
    """
    print(type(foo))

def main():
    """
        Example of how to use the time_log decorator.
        That is, after using the decorator on the function definition,
        just use the function normally.
    """
    
    for i in range(200):
        _ = _example_matrix_operation(i/40, i, n=5*i)

    y = mru.MultiResUNet()
    print_type(y)
##

if __name__ == "__main__":
    main()
