import math
import numpy as np

def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if value < array[0] or value > array[-1]:
        return None

    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx