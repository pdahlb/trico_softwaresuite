import math
import scipy
import numpy as np
import errno
import os
import signal
import functools

class SymGaussFitting(object):
    
    def SymGaussFn(xy,A, xo, yo, sx, offset):
        x, y = xy
        xo = float(xo)
        yo = float(yo)   
        Nom = np.power((x-xo), 2)
        Den = 2*np.power(sx,2)
        T_X = np.divide(Nom,Den)
        Nom = np.power((y-yo), 2)
        T_Y = np.divide(Nom,Den)
        fullfn = np.multiply(A, np.exp(-(T_X + T_Y))) + offset
        return fullfn.ravel()

    def MillAlg(x, xo, t, w, phi, r, offset):
        decay = np.exp(-(np.divide((x-xo), t)))
        oscillation = np.sin(np.multiply(x, w) + phi)
        fullfn = offset + np.multiply(r, x) + np.multiply(decay, oscillation)
        return fullfn
