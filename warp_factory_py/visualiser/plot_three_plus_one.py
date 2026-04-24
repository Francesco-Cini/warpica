import numpy as np

def plotThreePlusOne(metric, slicedPlanes = None, sliceLocations = None, alpha = None):

    if slicedPlanes is None:
        slicedPlanes = np.array([1, 4])

    if sliceLocations is None:
        s = np.array([metric.tensor[1][1]])
        sliceLocations = np.round((s+1)/2)

    if alpha is None:
        alpha = 0.2