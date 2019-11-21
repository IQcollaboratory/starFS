import numpy as np


def flatten(x): 
    ''' deals with those pesky cases when you have an array or list with only one element!
    '''
    if isinstance(x, float): 
        return x
    elif isinstance(x, list): 
        if len(x) == 1: 
            return x[0] 
        else: 
            return x
    elif isinstance(x, np.ndarray): 
        if len(x) == 1: 
            return x[0]
        else: 
            return x 
    else: 
        return x



