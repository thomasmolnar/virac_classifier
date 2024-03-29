from __future__ import division

import numpy as np
import nfft

"""
    Code taken from Jake Vanderplas' nfft package.
"""

def complex_exponential_sum(t, y, f0, Nf, df, method='auto'):
    """Compute a trigonometric sum over frequencies f = f0 + df * range(Nf)

    The computed sum is S_m = sum_k [y_k * exp(2 * i * f_m * t_k)]

    Parameters
    ----------
    t : array_like, length N
        Sorted array of times
    y : array_like, length N
        array of weights
    f0 : float
        minimum frequency
    Nf : integer
        number of frequencies. Must be an even number
    df : float
        frequency step. Must be smaller than 1 / (t.max() - t.min()).
    method : string, default='auto'
        method to use. Must be one of ['auto', 'direct', 'slow', 'nfft']

    Returns
    -------
    S : ndarray, length Nf
        resulting sums at each frequency
    """
    assert method in ['direct', 'nfft', 'slow', 'auto']
    freq = f0 + df * np.arange(Nf)

    t = np.asarray(t)
    assert t.ndim == 1

    Nf = int(Nf)
    assert Nf > 0
    assert Nf % 2 == 0

    y = np.asarray(y)
    y = np.broadcast_to(y, np.broadcast(y, t).shape)

    T = t.max() - t.min()
    F_O = 1. / (df * T)
    assert F_O > 1

    if method == 'direct':
        return np.dot(y, np.exp(2j * np.pi * freq * t[:, np.newaxis]))
    else:
        # shift the times to the range (-0.5, 0.5)
        Tstar = F_O * T
        t0 = t.min() - 0.5 * (Tstar - T)
        q = -0.5 + (t - t0) / Tstar

        # wrap phase into the weights
        x = y * np.exp(2j * np.pi * (f0 * Tstar + 0.5 * Nf) * q)
        Qn = simple_complex_exponential_sum(t=q,
                                            y=x,
                                            Nf=Nf // 2,
                                            method=method)

        # multiply by the phase that corrects for the above shifts
        return Qn * np.exp(2j * np.pi * (t0 + 0.5 * Tstar) * freq)


def simple_complex_exponential_sum(t, y, Nf, method='auto'):
    """Compute a trigonometric sum over frequencies f = range(-Nf, Nf)

    The computed sum is S_m = sum_k [y_k * exp(2 * i * n * f_m * t_k)]

    Parameters
    ----------
    t : array_like, length N
        Sorted array of times in the interval [-0.5, 0.5]
    y : array_like, broadcastable with t
        array of weights
    Nf : integer
        number of frequencies
    method : string, default='auto'
        method to use. Must be one of ['auto', 'slow', 'nfft']

    Returns
    -------
    S : ndarray, shape y.shape[:-1] + (2 * Nf,)
        resulting sums at each frequency
    """
    assert method in ['nfft', 'slow', 'auto']

    t = np.asarray(t)
    assert t.ndim == 1
    assert t.min() > -0.5 and t.max() < 0.5

    Nf = int(Nf)

    y = np.asarray(y)
    y = np.broadcast_to(y, np.broadcast(y, t).shape)
    outshape = y.shape[:-1] + (2 * Nf, )

    if method == 'slow':
        freq = np.arange(-Nf, Nf)
        results = np.dot(y, np.exp(2j * np.pi * freq * t[:, np.newaxis]))
    else:
        results = nfft.nfft_adjoint(t, y, 2 * Nf, sigma=3)
    return results.reshape(outshape)
