import numpy as np
import math
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from astropy.timeseries import LombScargle
try:
    from fourier_chi2 import find_period, fourier_poly_chi2_fit_cython
except:
    pass
from astropy.time import Time
from gatspy import periodic


def find_time_field(data):
    fld = 'HJD'
    for pp in ['HJD', 'JD', 'MJD', 'mjd', 'mjdobs']:
        if pp in data.keys():
            fld = pp
            break
    return fld


def jd_to_year(times):
    times_ = Time(times, format='jd')
    return times_.decimalyear


def wrap_df(data, fld, period=None, phase_zero=False):
    data[fld] -= data[fld].values[0] * phase_zero
    if period is not None:
        data[fld] = data[fld] % (2. * period)
    return data.sort_values(by=fld)


def autofrequency(times,
                  samples_per_peak=5,
                  nyquist_factor=5,
                  minimum_frequency=None,
                  maximum_frequency=None,
                  return_freq_limits=False):
    """
    TAKEN FROM ASTROPY IMPLEMENTATION
    
        Determine a suitable frequency grid for data.
    Note that this assumes the peak width is driven by the observational
    baseline, which is generally a good assumption when the baseline is
    much larger than the oscillation period.
    If you are searching for periods longer than the baseline of your
    observations, this may not perform well.
    Even with a large baseline, be aware that the maximum frequency
    returned is based on the concept of "average Nyquist frequency", which
    may not be useful for irregularly-sampled data. The maximum frequency
    can be adjusted via the nyquist_factor argument, or through the
    maximum_frequency argument.
    Parameters
    ----------
    times : array_like
        Array of times
    samples_per_peak : float (optional, default=5)
        The approximate number of desired samples across the typical peak
    nyquist_factor : float (optional, default=5)
        The multiple of the average nyquist frequency used to choose the
        maximum frequency if maximum_frequency is not provided.
    minimum_frequency : float (optional)
        If specified, then use this minimum frequency rather than one
        chosen based on the size of the baseline.
    maximum_frequency : float (optional)
        If specified, then use this maximum frequency rather than one
        chosen based on the average nyquist frequency.
    return_freq_limits : bool (optional)
        if True, return only the frequency limits rather than the full
        frequency grid.
    Returns
    -------
    frequency : ndarray
        The heuristically-determined optimal frequency bin
    """
    baseline = times.max() - times.min()
    n_samples = times.size

    df = 1.0 / baseline / samples_per_peak

    if minimum_frequency is None:
        minimum_frequency = 0.5 * df

    if maximum_frequency is None:
        avg_nyquist = 0.5 * n_samples / baseline
        maximum_frequency = nyquist_factor * avg_nyquist

    Nf = 1 + int(np.round((maximum_frequency - minimum_frequency) / df))

    if return_freq_limits:
        return minimum_frequency, minimum_frequency + df * (Nf - 1)
    else:
        return minimum_frequency + df * np.arange(Nf)


from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
from nfft_utils import complex_exponential_sum

def trig_sum_nfft(t, y, freq_factor, use_nfft, **kwargs):
    if kwargs['df'] * freq_factor * (t.max() - t.min()) > 1 or not use_nfft:
        S, C = trig_sum(t, y, freq_factor=freq_factor, **kwargs)
        
        return C + S * 1j
    
    else:
        return complex_exponential_sum(t * freq_factor, y, kwargs['f0'],
                                       kwargs['N'], kwargs['df'])


def fourier_chi2_fit(times,
                     mag,
                     inverr2,
                     f0,
                     f1,
                     Nf,
                     normalization='standard',
                     nterms=1,
                     use_nfft=True):
    """
        Faster implementation of astropy fastchi2 lombscargle.
        Faster because the linalg.solve is vectorized
        
        Parameters
        ----------
        times, mag, inverrs2 : array_like
            time, values and inverse errors squared for datapoints
        f0, f1, Nf : (float, float, int)
            parameters describing the frequency grid, f = np.linspace(f0,f1,Nf)
        nterms : float
            number of Fourier terms to use (default 1)
        use_nfft: bool
            if True, use NFFT library. This puts limitations on frequency grid so 
            defaults to no NFFT if conditions not satisfied.
        normalization: string
            how to normalize power (see astropy.timeseries.LombScargle)
    """

    df = (f1 - f0) / (Nf - 1)
    if df <= 0:
        raise ValueError("df must be positive")
    ws = np.sum(inverr2)
    meanmag = np.dot(inverr2, mag) / ws
    mag = mag - meanmag
    magw = inverr2 * mag
    magws = np.sum(magw)

    Q = np.empty((Nf, 1 + 2 * nterms), dtype=np.complex)
    Q[:, 0] = ws
    P = np.empty((Nf, 1 + nterms), dtype=np.complex)
    P[:, 0] = magws

    kwargs = dict(f0=f0, df=df, use_fft=True, N=Nf)

    Q[:, 1:] = np.array([
        trig_sum_nfft(times, inverr2, k, use_nfft, **kwargs)
        for k in range(1, 2 * nterms + 1)
    ]).T
    P[:, 1:] = np.array([
        trig_sum_nfft(times, magw, k, use_nfft, **kwargs)
        for k in range(1, nterms + 1)
    ]).T

    beta = np.empty((Nf, 2 * nterms + 1))
    alpha = np.empty((Nf, 2 * nterms + 1, 2 * nterms + 1))
    p = np.empty(Nf)
    soln = np.empty((Nf, 2 * nterms + 1))

    beta[:, ::2] = np.real(P)
    beta[:, 1::2] = np.imag(P[:, 1:])
    # Looks odd but when negative indices alpha set they are always zero
    # Possibly a saving here by only filling half the array and adding transpose
    for m in range(nterms + 1):
        for n in range(nterms + 1):
            alpha[:, 2 * m - 1, 2 * n - 1] = np.real(
                .5 * (Q[:, abs(m - n)] - Q[:, m + n]))
            alpha[:, 2 * m, 2 * n] = np.real(.5 *
                                             (Q[:, abs(m - n)] + Q[:, m + n]))
            alpha[:, 2 * m - 1, 2 * n] = np.imag(
                .5 * (np.sign(m - n) * Q[:, abs(m - n)] + Q[:, m + n]))
            alpha[:, 2 * m, 2 * n - 1] = np.imag(
                .5 * (np.sign(n - m) * Q[:, abs(m - n)] + Q[:, m + n]))

    ## To avoid singular matrices
    alpha += (np.identity(2 * nterms + npoly) * np.min(np.abs(alpha)) *
              1e-4)[np.newaxis, :, :]

    soln = np.linalg.solve(alpha, beta)
    p = np.sum(beta * soln, axis=1)
    p[p < 0.] = 0.
    soln[:, 0] += meanmag

    if normalization == 'psd':
        p *= 0.5
    elif normalization == 'standard':
        chi2_ref = np.sum(mag * mag * inverr2)
        p /= chi2_ref
    elif normalization == 'log':
        chi2_ref = np.sum(mag * mag * inverr2)
        p = -np.log(1 - p / chi2_ref)
    elif normalization == 'model':
        chi2_ref = np.sum(mag * mag * inverr2)
        p /= chi2_ref - p
    elif normalization == 'chi2':
        chi2_ref = np.sum(mag * mag * inverr2)
        p = chi2_ref - p
        p[p < 0.] = np.max(p)
    else:
        raise ValueError("normalization='{}' "
                         "not recognized".format(normalization))

    results = {}
    results['fourier_coeffs'] = soln
    results['frequency_grid'] = np.linspace(f0, f1, Nf)
    results['inv_covariance_matrix'] = alpha
    results['power'] = p

    return results

import time as Time
from itertools import product

def fourier_poly_chi2_fit(times,
                          mag,
                          err,
                          f0,
                          f1,
                          Nf,
                          nterms=1,
                          normalization='standard',
                          npoly=1,
                          use_nfft=True,
                          regularization=0.,
                          regularization_power=2.,
                          time_zeropoint_poly=2457000.,
                          regularize_by_trace=True):
    """
        Faster implementation of astropy fastchi2 lombscargle.
        Faster because the linalg.solve is vectorized
        
        Parameters
        ----------
        times, mag, err : array_like
            time, values and errors for datapoints
        f0, f1, Nf : (float, float, int)
            parameters describing the frequency grid, f = np.linspace(f0,f1,Nf)
        nterms : int
            number of Fourier terms to use (default 1)
        npoly: int
            number of polynomial terms to use (default 1)
        use_nfft: bool
            if True, use NFFT library. This puts limitations on frequency grid so 
            defaults to no NFFT if conditions not satisfied.
        normalization: string
            how to normalize power (see astropy.timeseries.LombScargle)
        regularization: float (default = 0.)
            regularization term sum_i (y-Mx)^2/sigma^2 + regularization n M^T M
        regularization_power: float (default = 2.)
            power of k to raise regularization term to
        time_zeropoint_poly: float (default = 2457000.)
            time shift to apply when evaluting polynomial terms
        regularize_by_trace: bool (default = True)
            regularization = regularization * sum(inverr^2) -- Vanderplas & Ivezic (2015)
    """

    df = (f1 - f0) / (Nf - 1)
    if df <= 0:
        raise ValueError("df must be positive")

    inverr2 = 1. / err**2
    ws = np.sum(inverr2)
    meanmag = np.dot(inverr2, mag) / ws
    mag = mag - meanmag
    magw = inverr2 * mag
    magws = np.sum(magw)

    # Scale regularization to number of datapoints
    if regularize_by_trace:
        regularization *= np.sum(inverr2)
    else:
        regularization *= len(inverr2)

    Q = np.empty((Nf, 1 + 2 * nterms), dtype=np.complex)
    Q[:, 0] = ws
    P = np.empty((Nf, 1 + nterms), dtype=np.complex)
    P[:, 0] = magws
    QT = np.empty((Nf, nterms, npoly - 1), dtype=np.complex)

    kwargs = dict(f0=f0, df=df, use_fft=True, N=Nf)

    Q[:, 1:] = np.array([
        trig_sum_nfft(times, inverr2, k, use_nfft, **kwargs)
        for k in range(1, 2 * nterms + 1)
    ]).T
    P[:, 1:] = np.array([
        trig_sum_nfft(times, magw, k, use_nfft, **kwargs)
        for k in range(1, nterms + 1)
    ]).T
    QT = np.array([[
        trig_sum_nfft(times,
                      inverr2 * np.power(times - time_zeropoint_poly, p), k,
                      use_nfft, **kwargs) for k in range(1, nterms + 1)
    ] for p in range(1, npoly)]).T

    Pl = np.array([
        np.sum(np.power(times - time_zeropoint_poly, k) * magw)
        for k in range(1, npoly)
    ])
    Ql = np.array([
        np.sum(np.power(times - time_zeropoint_poly, k) * inverr2)
        for k in range(1, 2 * npoly - 1)
    ])
    
    beta = np.empty((Nf, 2 * nterms + npoly))
    alpha = np.zeros((Nf, 2 * nterms + npoly, 2 * nterms + npoly))
    alpha_2 = np.zeros((Nf, 2 * nterms + npoly, 2 * nterms + npoly))
    
    p = np.empty(Nf)
    soln = np.empty((Nf, 2 * nterms + npoly))

    beta[:, :1 + 2 * nterms:2] = np.real(P)
    beta[:, 1:1 + 2 * nterms:2] = np.imag(P[:, 1:])
    beta[:, 1 + 2 * nterms:] = Pl
    
    # Looks odd but when negative indices they are always zero
    for m in range(nterms + 1):
        for n in range(nterms + 1):
            alpha[:, 2 * m - 1, 2 * n - 1] = np.real(
                .5 * (Q[:, abs(m - n)] - Q[:, m + n]))
            alpha[:, 2 * m, 2 * n] = np.real(.5 *
                                             (Q[:, abs(m - n)] + Q[:, m + n]))
            alpha[:, 2 * m - 1, 2 * n] = np.imag(
                .5 * (np.sign(m - n) * Q[:, abs(m - n)] + Q[:, m + n]))
            alpha[:, 2 * m, 2 * n - 1] = np.imag(
                .5 * (np.sign(n - m) * Q[:, abs(m - n)] + Q[:, m + n]))
            
    for m in range(1, npoly):
        for n in range(1, npoly):
            alpha[:, 2 * nterms + m, 2 * nterms + n] = Ql[m + n - 1]
        for n in range(1, nterms + 1):
            alpha[:, 2 * nterms + m, 2 * n - 1] = np.imag(QT[:, n - 1, m - 1])
            alpha[:, 2 * nterms + m, 2 * n] = np.real(QT[:, n - 1, m - 1])
            alpha[:, 2 * n - 1, 2 * nterms + m] = np.imag(QT[:, n - 1, m - 1])
            alpha[:, 2 * n, 2 * nterms + m] = np.real(QT[:, n - 1, m - 1])
        alpha[:, 2 * nterms + m, 0] = Ql[m - 1]
        alpha[:, 0, 2 * nterms + m] = Ql[m - 1]

    # Regularization term
    if regularization:
        for m in range(1, nterms + 1):
            alpha[:, 2 * m - 1, 2 * m -
                  1] += regularization * np.power(m, regularization_power)
            alpha[:, 2 * m, 2 *
                  m] += regularization * np.power(m, regularization_power)

    ## To avoid singular matrices
    alpha += (np.identity(2 * nterms + npoly) * np.min(np.abs(alpha)) *
              1e-4)[np.newaxis, :, :]

    soln = np.linalg.solve(alpha, beta)
    
    p = np.sum(beta * soln, axis=1)
    p[p < 0.] = 0.
    soln[:, 0] += meanmag

    if normalization == 'psd':
        p *= 0.5
    elif normalization == 'standard':
        chi2_ref = np.sum(mag * mag * inverr2)
        p /= chi2_ref
    elif normalization == 'log':
        chi2_ref = np.sum(mag * mag * inverr2)
        p = -np.log(1 - p / chi2_ref)
    elif normalization == 'model':
        chi2_ref = np.sum(mag * mag * inverr2)
        p /= chi2_ref - p
    elif normalization == 'chi2':
        chi2_ref = np.sum(mag * mag * inverr2)
        p = chi2_ref - p
    else:
        raise ValueError("normalization='{}' "
                         "not recognized".format(normalization))


#     p[p<0.]=0.

    results = {}
    results['fourier_coeffs_grid'] = soln
    results['frequency_grid'] = np.linspace(f0, f1, Nf)
    results['inv_covariance_matrix'] = alpha
    results['power'] = p
    results['chi2_ref'] = np.sum(mag * mag * inverr2)

    return results

def fourier_poly_chi2_fit_transpose(times,
                          mag,
                          err,
                          f0,
                          f1,
                          Nf,
                          nterms=1,
                          normalization='standard',
                          npoly=1,
                          use_nfft=True,
                          regularization=0.,
                          regularization_power=2.,
                          time_zeropoint_poly=2457000.,
                          regularize_by_trace=True):
    """
        Faster implementation of astropy fastchi2 lombscargle.
        Faster because the linalg.solve is vectorized
        
        Parameters
        ----------
        times, mag, err : array_like
            time, values and errors for datapoints
        f0, f1, Nf : (float, float, int)
            parameters describing the frequency grid, f = np.linspace(f0,f1,Nf)
        nterms : int
            number of Fourier terms to use (default 1)
        npoly: int
            number of polynomial terms to use (default 1)
        use_nfft: bool
            if True, use NFFT library. This puts limitations on frequency grid so 
            defaults to no NFFT if conditions not satisfied.
        normalization: string
            how to normalize power (see astropy.timeseries.LombScargle)
        regularization: float (default = 0.)
            regularization term sum_i (y-Mx)^2/sigma^2 + regularization n M^T M
        regularization_power: float (default = 2.)
            power of k to raise regularization term to
        time_zeropoint_poly: float (default = 2457000.)
            time shift to apply when evaluting polynomial terms
        regularize_by_trace: bool (default = True)
            regularization = regularization * sum(inverr^2) -- Vanderplas & Ivezic (2015)
    """

    df = (f1 - f0) / (Nf - 1)
    if df <= 0:
        raise ValueError("df must be positive")

    inverr2 = 1. / err**2
    ws = np.sum(inverr2)
    meanmag = np.dot(inverr2, mag) / ws
    mag = mag - meanmag
    magw = inverr2 * mag
    magws = np.sum(magw)

    # Scale regularization to number of datapoints
    if regularize_by_trace:
        regularization *= np.sum(inverr2)
    else:
        regularization *= len(inverr2)

    Q = np.empty((1 + 2 * nterms, Nf), dtype=np.complex)
    Q[0, :] = ws
    P = np.empty((1 + nterms, Nf), dtype=np.complex)
    P[0, :] = magws
    QT = np.empty((npoly - 1, nterms, Nf), dtype=np.complex)

    kwargs = dict(f0=f0, df=df, use_fft=True, N=Nf)
    
    Q[1:] = np.array([
        trig_sum_nfft(times, inverr2, k, use_nfft, **kwargs)
        for k in range(1, 2 * nterms + 1)
    ])
    
    P[1:] = np.array([
        trig_sum_nfft(times, magw, k, use_nfft, **kwargs)
        for k in range(1, nterms + 1)
    ])
    
    QT = np.array([[
        trig_sum_nfft(times,
                      inverr2 * np.power(times - time_zeropoint_poly, p), k,
                      use_nfft, **kwargs) for p in range(1, npoly) 
    ] for k in range(1, nterms + 1)])

    Pl = np.array([
        np.sum(np.power(times - time_zeropoint_poly, k) * magw)
        for k in range(1, npoly)
    ])
    
    Ql = np.array([
        np.sum(np.power(times - time_zeropoint_poly, k) * inverr2)
        for k in range(1, 2 * npoly - 1)
    ])
    
    beta = np.empty((2 * nterms + npoly, Nf))
    alpha = np.empty((2 * nterms + npoly, 2 * nterms + npoly, Nf))
    
    p = np.empty(Nf)
    soln = np.empty((Nf, 2 * nterms + npoly))

    beta[:1 + 2 * nterms:2] = np.real(P)
    beta[1:1 + 2 * nterms:2] = np.imag(P[1:])
    beta[1 + 2 * nterms:] = Pl[:,np.newaxis]

    # Looks odd but when negative indices they are always zero
    for n in range(nterms + 1):
        for m in range(nterms + 1):
            alpha[2 * n - 1, 2 * m - 1] = np.real(
                .5 * (Q[abs(m - n)] - Q[m + n]))
            alpha[2 * n, 2 * m] = np.real(.5 *
                                             (Q[abs(m - n)] + Q[m + n]))
            alpha[2 * n, 2 * m - 1] = np.imag(
                .5 * (np.sign(m - n) * Q[abs(m - n)] + Q[m + n]))
            alpha[2 * n - 1, 2 * m] = np.imag(
                .5 * (np.sign(n - m) * Q[abs(m - n)] + Q[m + n]))
    
    for n in range(1, npoly):
        for m in range(1, npoly):
            alpha[2 * nterms + n, 2 * nterms + m] = Ql[m + n - 1]
    for n in range(1, nterms + 1):
        for m in range(1, npoly):
            alpha[2 * n - 1, 2 * nterms + m] = np.imag(QT[n - 1, m - 1])
            alpha[2 * n, 2 * nterms + m] = np.real(QT[n - 1, m - 1])
            alpha[2 * nterms + m, 2 * n - 1] = np.imag(QT[n - 1, m - 1])
            alpha[2 * nterms + m, 2 * n] = np.real(QT[n - 1, m - 1])
    for m in range(1, npoly):
        alpha[0, 2 * nterms + m] = Ql[m - 1]
        alpha[2 * nterms + m, 0] = Ql[m - 1]

    # Regularization term
    if regularization:
        for m in range(1, nterms + 1):
            alpha[2 * m - 1, 2 * m - 1] \
                += regularization * np.power(m, regularization_power)
            alpha[2 * m, 2 * m] \
                += regularization * np.power(m, regularization_power)

    ## To avoid singular matrices
    alpha += (np.identity(2 * nterms + npoly) * np.min(np.abs(alpha)) *
              1e-4)[:, :, np.newaxis]
    
    alpha = alpha.T
    beta = beta.T

    soln = np.linalg.solve(alpha, beta)
    
    p = np.sum(beta * soln, axis=1)
    p[p < 0.] = 0.
    soln[:, 0] += meanmag

    if normalization == 'psd':
        p *= 0.5
    elif normalization == 'standard':
        chi2_ref = np.sum(mag * mag * inverr2)
        p /= chi2_ref
    elif normalization == 'log':
        chi2_ref = np.sum(mag * mag * inverr2)
        p = -np.log(1 - p / chi2_ref)
    elif normalization == 'model':
        chi2_ref = np.sum(mag * mag * inverr2)
        p /= chi2_ref - p
    elif normalization == 'chi2':
        chi2_ref = np.sum(mag * mag * inverr2)
        p = chi2_ref - p
    else:
        raise ValueError("normalization='{}' "
                         "not recognized".format(normalization))


    results = {}
    results['fourier_coeffs_grid'] = soln
    results['frequency_grid'] = np.linspace(f0, f1, Nf)
    results['inv_covariance_matrix'] = alpha
    results['power'] = p
    results['chi2_ref'] = np.sum(mag * mag * inverr2)

    return results



from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_probability as fap


def false_alarm_probability(power, frequencies, data,
                            normalization='standard'):
    norm = normalization
    if norm == 'chi2':
        norm = 'standard'
    fap_ = fap(power,
               np.nanmax(frequencies),
               data[find_time_field(data)].values,
               data['mag'].values,
               data['error'].values,
               normalization=normalization)
    return fap_


def lsq_fit_find_uncertainties(results, argc):
    Nf = len(results['frequency_grid'])
    f1 = np.max(results['frequency_grid'])
    f0 = np.min(results['frequency_grid'])
    # Find uncertainties
    if argc == 0:
        argc += 1
    if argc == Nf - 1:
        argc -= 1
    df = (f1 - f0) / (Nf - 1)
    results['lsq_freq_error'] = 1. / (.5 *
                                      (results['chi_squared_grid'][argc + 1] +
                                       results['chi_squared_grid'][argc - 1] -
                                       2 * results['chi_squared_grid'][argc]) /
                                      df**2)
    if results['lsq_freq_error'] < 0.:
        results['lsq_freq_error'] = 0.
    else:
        results['lsq_freq_error'] = np.sqrt(results['lsq_freq_error'])
#     results['lsq_period_original_error'] = results['lsq_freq_error'] * results[
#         'lsq_period_original']**2
    gradientA = (results['fourier_coeffs_grid'][argc + 1] -
                 results['fourier_coeffs_grid'][argc - 1]) / (2 * df)
    gradientA *= results['lsq_freq_error']
    results['fourier_coeffs_cov_best_period'] = results['fourier_coeffs_cov']
    if np.all(gradientA == gradientA):
        results['fourier_coeffs_cov'] += \
            gradientA[:,np.newaxis] * gradientA[np.newaxis,:]


#     results['fourier_coeffs_cov'] = .5*(results['fourier_coeffs_cov']+results['fourier_coeffs_cov'].T)
#     I=np.identity(np.shape(results['fourier_coeffs_cov'])[0])
#     results['fourier_coeffs_cov']+=1e-4*np.min(np.abs(results['fourier_coeffs_cov']))*I
    return results

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

from astropy.stats import circvar, circcorrcoef

def circcov(data):
    '''
        Covariance matrix for circular data
        The diagonals are variance and the off-diagonal terms are correlation coefficient * sqrt(var1*var2)
        I haven't seen this definition anywhere.
    '''
    result = np.diag(circvar(data, axis=0))
    
    for nn in range(1,np.shape(data)[1]):
        for jj in range(nn):
            result[nn][jj] = result[jj][nn] = circcorrcoef(data[:,jj], data[:,nn])*np.sqrt(result[jj][jj]*result[nn][nn])
            
    return result
    
    
def fourier_poly_chi2_fit_full(times,
                               mag,
                               err,
                               f0,
                               f1,
                               Nf,
                               nterms=1,
                               npoly=1,
                               use_nfft=True,
                               regularization=0.,
                               regularization_power=2.,
                               time_zeropoint_poly=2457000.,
                               keep_small=True,
                               regularize_by_trace=True, 
                               code_switch=True,
                               use_power_of_2=True):
    '''
        Wrapper of main algorithm to do all the post-processing
        
        if keep_small, remove arrays from output (e.g. chi_squared_grid at 
        range of periods).

        NFFT works way better if Nf is a power of 2. use_power_of_2 will automatically
        find the next power of 2 > Nf. If may be more convenient for speed reasons to 
        use the lower power of 2. 

        code_switch = True uses a slightly modified version  of the code that reorders
        the arrays leading to a 25% speedup. If it appears stable, change to default.
    '''
    fn = fourier_poly_chi2_fit
    if code_switch:
        fn = fourier_poly_chi2_fit_transpose

    if use_power_of_2:
        Nf = next_power_of_2(Nf)
    assert Nf % 2 == 0 ## Nf must be even
    
    results = fn(times,
                                    mag,
                                    err,
                                    f0,
                                    f1,
                                    Nf,
                                    nterms=nterms,
                                    normalization="chi2",
                                    npoly=npoly,
                                    use_nfft=use_nfft,
                                    regularization=regularization,
                                    regularization_power=regularization_power,
                                    time_zeropoint_poly=time_zeropoint_poly,
                                    regularize_by_trace=regularize_by_trace)
    
    
    results['chi_squared_grid'] = results.pop('power')

    # Fill with best result
    argc = np.argmin(results['chi_squared_grid'])

    results['lsq_period'] = 1. / results['frequency_grid'][argc]
    results['fourier_coeffs'] = results['fourier_coeffs_grid'][argc]
    results['fourier_coeffs_cov'] = np.linalg.inv(
        results['inv_covariance_matrix'][argc])
    results['lsq_nterms'] = nterms
    results['lsq_npoly'] = npoly
    results['lsq_regularization'] = regularization
    results['lsq_regularization_power'] = regularization_power
    results['lsq_time_zeropoint_poly'] = time_zeropoint_poly
    results['lsq_regularize_by_trace'] = regularize_by_trace
    results['lsq_chi_squared'] = results['chi_squared_grid'][argc]

    results = lsq_fit_find_uncertainties(results, argc)

    # Compute amplitudes & phases
    results['amplitudes'] = np.sqrt(
        results['fourier_coeffs'][1:2 * nterms + 1:2]**2 +
        results['fourier_coeffs'][2:2 * nterms + 1:2]**2)
    ## definition of phase as in Petersen 1986 -- a_n sin(n omega t) + b_n cos(n omega t) 
    ## = A_n cos(n omega t + phi_n) 
    ## = A_n cos(n omega t) cos(phi_n) - A_n sin(n omega t) sin(phi_n) 
    ## so tan(phi_n) = -a_n/b_n
    ## Then it is fine to use the phase difference phi_{k1} = phi_k - k phi_1
    results['phases'] = np.arctan2(
        -results['fourier_coeffs'][1:2 * nterms + 1:2],
        results['fourier_coeffs'][2:2 * nterms + 1:2])
    
    results['phases_diff'] = results['phases'][1:] - np.arange(2,len(results['phases'])+1)*results['phases'][0]
    results['phases_diff'] = results['phases_diff'] % (2.*np.pi)
    
    if np.all(np.isfinite(results['fourier_coeffs_cov'].flatten())):
        fg = np.random.multivariate_normal(results['fourier_coeffs'],
                                           results['fourier_coeffs_cov'],
                                           size=300)
        results['amplitudes_cov'] = np.cov(
            np.sqrt(fg[:, 1:2 * nterms + 1:2]**2 +
                    fg[:, 2:2 * nterms + 1:2]**2),
            rowvar=False)
        phases = np.arctan2(-fg[:, 1:2 * nterms + 1:2],
                            fg[:, 2:2 * nterms + 1:2])
        phases_diff = phases[:,1:] - np.arange(2, len(results['phases'])+1)[np.newaxis,:]*phases[:,0][:,np.newaxis]
        phases_diff = phases_diff % (2.*np.pi)
        results['phases_cov'] = circcov(phases)
        results['phases_diff_cov'] = circcov(phases_diff)
    else:
        results['amplitudes_cov'] = np.nan * np.ones(
            (len(results['amplitudes']), len(results['amplitudes'])))
        results['phases_cov'] = np.nan * np.ones(
            (len(results['phases']), len(results['phases'])))

#     results['lsq_period'] = results['lsq_period_original'] / (
#         1. + np.argmax(results['amplitudes']))
#     results['lsq_period_error'] = results['lsq_period_original_error'] / (
#         1. + np.argmax(results['amplitudes']))
    
    if np.argmax(results['amplitudes'])!=0.:
        double_result= fourier_poly_chi2_fit_full(times,
                                           mag,
                                           err,
                                           1./(results['lsq_period'] / (1. + np.argmax(results['amplitudes']))),
                                           200./(results['lsq_period'] / (1. + np.argmax(results['amplitudes']))),
                                           2,
                                           nterms=nterms,
                                           npoly=npoly,
                                           use_nfft=use_nfft,
                                           regularization=regularization,
                                           regularization_power=regularization_power,
                                           time_zeropoint_poly=time_zeropoint_poly,
                                           keep_small=keep_small,
                                           regularize_by_trace=regularize_by_trace, 
                                           code_switch=code_switch)
        dchiN=double_result['lsq_chi_squared']/len(times)
        chiN=results['lsq_chi_squared']/len(times)
        if dchiN-chiN<1.:
            return double_result
        
    results['ordering'] = 'new'

    if keep_small:
        for ii in [
                'chi_squared_grid', 'fourier_coeffs_grid',
                'inv_covariance_matrix', 'frequency_grid'
        ]:
            del results[ii]

    return results

def fourier_poly_chi2_fit_full_quick(lc, nterms=1, 
                                     npoly=1, NF=100000,
                                    minp=10, maxp=3000):
    return fourier_poly_chi2_fit_full(lc[find_time_field(lc)].values,
                                      lc['mag'].values,
                                      lc['error'].values,
                                      1. / maxp,
                                      1. / minp,
                                      NF,
                                      nterms=nterms,
                                      npoly=npoly)


def clip_data(data):
    mags = np.array(data['mag'])
    bot = np.percentile(mags, 1)
    top = np.percentile(mags, 99)
    
    data = data.drop(data[np.array(data['mag'])<bot].index)
    data = data.drop(data[np.array(data['mag'])>top].index)
    
    return data

def plot_lightcurve(data,
                    mag_label='K_s',
                    color='k',
                    period=1e50,
                    to_year=False,
                    clip=False):
    
    fld = find_time_field(data)
    data_use = data.copy()
    if clip:
        data = clip_data(data_use)
        
    if len(data)==0:
        plt.gca().invert_yaxis()
        return
    
    fld = find_time_field(data)
    if 'error' in data.keys():
        errors = data['error']
    else:
        errors = np.zeros_like(data['mag'])
    if to_year:
        times = jd_to_year(data[fld])
    else:
        times = data[fld]
    plt.errorbar(times % (2. * period),
                 data['mag'],
                 errors,
                 fmt='o',
                 ms=5,
                 color=color,
                 mec='k',
                 mew=0.3)

    plt.gca().invert_yaxis()
    return plt

def plot_lightcurve_ax(data, ax,
                    mag_label='K_s',
                    color='k',
                    period=1e50,
                    to_year=False, y=False, x=False, clip=False):
    
    fld = find_time_field(data)
    data_use = data.copy()
    if clip:
        data = clip_data(data_use)
    
    if len(data)==0:
        ax.gca().invert_yaxis()
        return
    
    
    if 'error' in data.keys():
        errors = data['error']
    else:
        errors = np.zeros_like(data['mag'])
    if to_year:
        times = jd_to_year(data[fld])
    else:
        times = data[fld]
    ax.errorbar(times % period,
                 data['mag'],
                 errors,
                 fmt='o',
                 ms=5,
                 color=color,
                 mec='k',
                 mew=0.3)
    ax.invert_yaxis()

    if y:
        ax.set_ylabel(r'$K_S$ $\mathrm{mag}$', family='serif',
               fontsize=15)
    if x:
        ax.set_xlabel(fld,  family='serif', fontsize=15)


def plot_lightcurve_stack(data, to_year=False):
    [
        plot_lightcurve(d,
                        mag_label='\mathrm{Magnitude}',
                        color=sns.color_palette(n_colors=10)[i],
                        to_year=to_year) for i, d in enumerate(data)
        if d is not None
    ]

def find_lag(times, period):
    """
    Find probabilistic metrics to determine if the time between observations of phase folded light curves constitute 'lagging'
    
    """
    
    # Find time diffs of ordered phase folded light curve
    times_fld = times%(period)
    times_fld_ord = np.sort(times_fld)
    times_fld_diff = np.diff(times_fld_ord)
    
    if len(times_fld_diff)==0:
        return {'time_lag_mean':np.nan, 'time_lag_median':np.nan, 'max_time_lag':np.nan}
    
    # Calculate summary statistics of difference array
    t_max = times_fld_diff.max()
    mean = np.mean(times_fld_diff)
    median = np.median(times_fld_diff)
    sd = np.std(times_fld_diff)
    
    # Dispersion of max
    mean_disp = abs(t_max-mean)/sd
    median_disp = abs(t_max-median)/sd
    
    return {'time_lag_mean':mean_disp, 'time_lag_median':median_disp, 'max_time_lag':t_max}

def power_stats(power):
    """
    Find dispersion of maximum power computed about mean and median
    
    """
    max_pow = power.max()
    mean = np.mean(power)
    median = np.median(power)
    sd = np.std(power)
    
    mean_disp = abs(max_pow-mean)/sd
    med_disp = abs(max_pow-median)/sd
    
    return {'pow_mean_disp':mean_disp, 'pow_med_disp':med_disp}

def lombscargle_stats(data, **ls_kwargs):
    """
    LombScargle analysis of lightcurve to extract certain summary statistics
    
    """
    # Find the time field
    fld = find_time_field(data)
    
    if 'error' in data.keys():
        errors = data['error']
    else:
        errors = None
        
    # Initialise model
    model = LombScargle(data[fld], 
                            data['mag'], errors,
                           normalization='standard')
        
    freq, power = model.autopower(**ls_kwargs)
    
    # Find max power and hence most likely period
    max_pow = power.max()
    max_pow_arg = np.argmax(power)
    periods = 1./freq   
    period = periods[max_pow_arg]
    
    # Find metric showing time lag
    time_lag = find_lag(data[fld], period)
    
    # Find power array stats
    pow_stats = power_stats(power)
    
    return {'ls_period':period, 'max_pow':max_pow,
            **pow_stats, **time_lag}


def plot_periodogram(data, ax,
                     plot_kwargs={}, ls_kwargs={},
                     window=False):
    (freq, power, fap) = lombscargle(data, window=window, FAP=True, **ls_kwargs)
    if window:
        ax.plot(1. / freq, 
                   power, **plot_kwargs)
    else:
        ax.plot(1. / freq, power, **plot_kwargs)
        teststr = 'FAP={}'.format(round(fap, 7))
    plt.xlabel('Period [day]')
    plt.ylabel('LS Power')
    
    return l


def plot_periodogram_ax(data, ax,
                     plot_kwargs={}, ls_kwargs={},
                     window=False):
    (freq, power, fap) = lombscargle(data, window=window, FAP=True, **ls_kwargs)
    if window:
        ax.plot(1. / freq, 
                   power, **plot_kwargs)
    else:
        ax.plot(1. / freq, power, **plot_kwargs)
        teststr = 'FAP={}'.format(round(fap, 7))
        

def plot_periodogram_stack(data, bands=None, with_leg=True):
    if bands is None:
        bands = [None] * len(data)
    [
        plot_periodogram(d,
                         plot_kwargs={
                             'color': sns.color_palette()[i],
                             'lw': 3,
                             'label': b
                         }) if d is not None and len(d) > 3 else None
        for i, (d, b) in enumerate(zip(data, bands))
    ]
    if with_leg:
        plt.legend(loc='lower left',
                   bbox_to_anchor=(0., 1.),
                   ncol=len(data) + 1)


def multiband_periodogram(lc, reg_band=0.01, **kwargs):

    # Optimizer complains if delta_T<max period
    maxperiod = 3000.
    mint = np.min(
        [np.min(lc_[find_time_field(lc_)]) for lc_ in lc if lc_ is not None])
    maxt = np.max(
        [np.max(lc_[find_time_field(lc_)]) for lc_ in lc if lc_ is not None])
    deltat = maxt - mint
    if maxperiod > deltat:
        maxperiod = deltat

    periods = np.linspace(10., maxperiod, 500)
    mls = periodic.LombScargleMultiband(fit_period=True,
                                        reg_band=reg_band,
                                        **kwargs)
    mls.optimizer.period_range = (10, maxperiod)
    tt = np.concatenate([l[find_time_field(l)] for l in lc if l is not None])
    yy = np.concatenate([l['mag'] for l in lc if l is not None])
    yye = np.concatenate([l['error'] for l in lc if l is not None])
    fltrs = np.concatenate(
        [np.ones(len(l)) * i for i, l in enumerate(lc) if l is not None])
    mls.fit(tt, yy, yye, fltrs)
    P_multi = mls.periodogram(periods)
    return periods, P_multi


def plot_multiband_periodogram(lc, **kwargs):
    periods, P_multi = multiband_periodogram(lc, **kwargs)
    plt.plot(periods, P_multi, lw=3, color='k', label='All', zorder=100)
    return periods, P_multi


## Fourier series fitting -- better to use cython implementation in fourier_chi2.pyx or
## above


def fit_fourier_slow(data, startp, endp, Kmax=5):
    ''' 
	Expects data to have columns HJD, mag, error
    Fits Fourier series
    '''
    chisq0 = 1e100
    for KK in np.arange(2, Kmax):
        for p in np.linspace(startp, endp, 100):
            Y = data['mag'].values
            Yerr = data['error'].values
            SINX = np.array([
                np.sin(2. * np.pi * k * data['HJD'].values / p)
                for k in np.arange(1, KK)
            ])
            COSX = np.array([
                np.cos(2. * np.pi * k * data['HJD'].values / p)
                for k in np.arange(1, KK)
            ])
            ONES = np.ones_like(Y)
            X = np.concatenate([ONES[np.newaxis, :], SINX, COSX])
            p0 = np.zeros(np.shape(X)[0])
            p0[0] = np.mean(Y)
            p0[1] = np.abs(np.diff(np.percentile(Y, [95., 5.])))
            p0[2] = np.abs(np.diff(np.percentile(Y, [95., 5.])))
            #print p0
            popt, pcov = curve_fit(lambda x, *P: np.dot(P, x),
                                   X,
                                   Y,
                                   p0=p0,
                                   sigma=Yerr)
            chisq = np.sum(
                (Y - np.dot(popt, X))**2 / Yerr**2) / (len(Y) - len(p0))
            if (chisq < chisq0):
                chisq0 = chisq
                optimum_params = popt
                optimum_params_err = pcov
                optimum_period = p
    return optimum_period, optimum_params, optimum_params_err, chisq0


# def retrieve_fourier(phase, fourier_components):
#     Kmax = (len(fourier_components) - 1) / 2 + 1
#     SINX = np.array([np.sin(k * phase) for k in np.arange(1, Kmax)])
#     COSX = np.array([np.cos(k * phase) for k in np.arange(1, Kmax)])
#     ONES = np.ones_like(phase)
#     X = np.concatenate([ONES[np.newaxis, :], SINX, COSX])
#     return np.dot(fourier_components, X)


def retrieve_fourier(phase, fourier_components):
    ''' Will not work with polynomial terms '''
    Kmax = (len(fourier_components) - 1) / 2 + 1
    SINX = np.array([np.sin(k * phase) for k in np.arange(1, Kmax)])
    COSX = np.array([np.cos(k * phase) for k in np.arange(1, Kmax)])
    ONES = np.ones_like(phase)
    SCX = np.empty((len(fourier_components) - 1, len(phase)))
    SCX[::2] = SINX
    SCX[1::2] = COSX
    X = np.concatenate([ONES[np.newaxis, :], SCX])
    return np.dot(fourier_components, X)


def retrieve_fourier_poly(times, results, with_var=False):
    Kmax = results['lsq_nterms'] + 1
    npoly = results['lsq_npoly']
    period = results['lsq_period']
    time_zeropoint_poly = results['lsq_time_zeropoint_poly']
    SINX = np.array(
        [np.sin(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    COSX = np.array(
        [np.cos(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    ONES = np.array(
        [pow(times - time_zeropoint_poly, k) for k in range(npoly)])
    SCX = np.empty(((Kmax - 1) * 2, len(times)))
    SCX[::2] = SINX
    SCX[1::2] = COSX
    if results['ordering'] == 'old':
        X = np.concatenate([ONES, SCX])
    else:
        X = np.concatenate([ONES[:1], SCX, ONES[1:]])
    if with_var:
        var = np.var(
            np.random.multivariate_normal(results['fourier_coeffs'],
                                          results['fourier_coeffs_cov'],
                                          size=100),
            axis=0)
        return np.dot(results['fourier_coeffs'], X), np.dot(var, X)
    return np.dot(results['fourier_coeffs'], X)


def plot_fourier(fourier_components):
    phase = np.linspace(0., 4 * np.pi, 1000)
    plt.plot(phase, retrieve_fourier(fourier_components), **plot_kwargs)


def run_lc_fit(data, with_plot=False, Kmax=10):
    Period = find_period(data)
    if Period == np.nan:
        return np.nan, np.ones(3) * np.nan, np.ones((3, 3)) * np.nan, np.nan
    Period, Fourier, FourierErrs, ChiSq = fit_fourier(data,
                                                      0.95 * Period,
                                                      1.05 * Period,
                                                      Kmax=Kmax)
    if with_plot:
        Phs = np.linspace(0., 4. * np.pi, 1000)
        FC = np.array([
            retrieve_fourier(Phs, fc) for fc in np.random.multivariate_normal(
                Fourier, FourierErrs, size=50)
        ])
        plt.fill_between(Phs,
                         np.percentile(FC, 16., axis=0),
                         np.percentile(FC, 84., axis=0),
                         color='gray',
                         alpha=0.7)
        plt.plot(Phs, np.median(FC, axis=0), color='k')
        data = wrap_df(data, 'HJD', period=Period)
        fltr = (data['error'] < 90.)
        plt.errorbar(data['HJD'][fltr] * 2. * np.pi / Period,
                     data['mag'][fltr],
                     yerr=data['error'][fltr],
                     fmt='o',
                     ms=5)
    return Period, Fourier, FourierErrs, ChiSq


def get_delta_log_likelihood(y, yerr, pred_mean, free_var=False):
    ''' Difference in log-likelihood between GP and pure Gaussian scatter '''
    
    mean = np.sum(y / yerr**2) / np.sum(1. / yerr**2)
    
    if free_var:
        varr = np.sum((y - mean)**2 / yerr**2) / np.sum(1. / yerr**2)
        
    else:
        varr = yerr**2
        
    loglikemean = -.5 * np.sum(
        np.log(2. * np.pi * varr) + (y - mean)**2 / varr)
    loglikemodel = -.5 * np.sum(
        np.log(2. * np.pi * yerr**2) + (y - pred_mean)**2 / yerr**2)
    
    return loglikemodel - loglikemean


def plot_lc_LSQ(mira_data,
                results,
                mag_label='K_s',
                past_stretch=1.1,
                future_stretch=2.,
                fold=1e30):
    
    if type(fold) is bool and fold==True:
        fold = 2*results['lsq_period']
        
    time_fld = find_time_field(mira_data)
    if 'error' in mira_data.keys():
        errors = mira_data['error']
    else:
        errors = np.zeros_like(mira_data['mag'].values)

    xcentre = np.median(mira_data[time_fld])
    color = "#ff7f0e"
    f, a = plt.subplots(2,
                        1,
                        figsize=[10., 5.],
                        sharex=True,
                        gridspec_kw={
                            'height_ratios': [4, 1],
                            'hspace': 0.
                        })
    plt.sca(a[0])
    plt.errorbar(mira_data[time_fld] % fold,
                 mira_data['mag'],
                 yerr=errors,
                 fmt='o', ms=6, mec='k', mfc='gray', 
                 mew=1, alpha=0.6, color='gray')
    plt.gca().invert_yaxis()
    xgrid = np.linspace(
        np.min(mira_data[time_fld] - xcentre) * past_stretch,
        np.max(mira_data[time_fld] - xcentre) * future_stretch, 5000) + xcentre
    
    if fold < 1e30:
        zpt = fold * (np.int(np.min(mira_data[time_fld]) / fold) + 1)
        xgrid = np.linspace(zpt + fold * 0.0001, zpt + fold * 0.9999, 5000)
    pred_mean, pred_var = retrieve_fourier_poly(xgrid, results, with_var=True)
    plt.plot(xgrid % fold, pred_mean, color=color)
    pred_std = np.sqrt(pred_var)
    plt.fill_between(xgrid % fold,
                     pred_mean + pred_std,
                     pred_mean - pred_std,
                     color=color,
                     alpha=0.3,
                     edgecolor="none")

    plt.sca(a[1])
    pred_mean = retrieve_fourier_poly(mira_data[time_fld], results)
    plt.errorbar(mira_data[time_fld] % fold,
                 mira_data['mag'] - pred_mean,
                 yerr=errors,
                 fmt='o', ms=6, mec='k', 
                 mfc='gray', mew=1, alpha=0.6, color='gray')
    plt.gca().invert_yaxis()
    plt.xlabel(time_fld)
    plt.ylabel('Residuals')
    plt.ylim(-0.3, 0.3)
    plt.annotate(
        r'$\chi^2=%0.3f$' %
        np.sum(1. / (len(mira_data['mag']) - len(results['fourier_coeffs'])) *
               (mira_data['mag'] - pred_mean)**2 / errors**2),
        xy=(0.99, 0.07),
        xycoords='axes fraction',
        ha='right',
        va='bottom',
        fontsize=20)
    plt.annotate(r'$\Delta\log\mathcal{L}=%0.3f$' %
                 get_delta_log_likelihood(mira_data['mag'], errors, pred_mean),
                 xy=(0.99, 0.93),
                 xycoords='axes fraction',
                 ha='right',
                 va='top',
                 fontsize=20)
    plt.sca(a[0])
    plt.ylabel(r'$%s/\,\mathrm{mag}$' % mag_label)
    plt.annotate('Least-squares solution',
                 xy=(0.99, 1.01),
                 xycoords='axes fraction',
                 ha='right',
                 va='bottom',
                 fontsize=20)
    plt.annotate(r'Period $=%0.5g$ days' % results['lsq_period'],
                 xy=(0.99, 0.99),
                 xycoords='axes fraction',
                 ha='right',
                 va='top',
                 fontsize=20)
    return f


def add_alias_lines():

    xx = np.linspace(0., 3000.)
    plt.plot(xx, xx, color='k')
    for ii in [1. / 4., 1. / 3., 1. / 2., 2., 3., 4.]:
        plt.plot(xx, ii * xx, color='k', alpha=0.2, ls='dashed')

    xxT, yy = xx.copy(), 1. / (1. / xx - 1. / 365)
    xxT, yy = xxT[yy > 0.], yy[yy > 0.]
    plt.plot(xxT, yy, color='r', alpha=0.2, ls='dashed')
    xxT, yy = xx.copy(), 1. / (1. / xx + 1. / 365)
    xxT, yy = xxT[yy > 0.], yy[yy > 0.]
    plt.plot(xxT, yy, color='r', alpha=0.2, ls='dashed')
    xxT, yy = xx.copy(), 1. / (1. / xx + 1. / 30.)
    xxT, yy = xxT[yy > 0.], yy[yy > 0.]
    plt.plot(xxT, yy, color='r', alpha=0.2, ls='dashed')
    xxT, yy = xx.copy(), 1. / (1. / xx - 1. / 30.)
    xxT, yy = xxT[yy > 0.], yy[yy > 0.]
    plt.plot(xxT, yy, color='r', alpha=0.2, ls='dashed')
    plt.annotate('1:4',
                 xy=(80 * 4., 80.),
                 fontsize=12.,
                 rotation=45.,
                 xycoords='data',
                 alpha=0.5)
    plt.annotate('1:3',
                 xy=(80 * 3., 80.),
                 fontsize=12.,
                 rotation=45.,
                 xycoords='data',
                 alpha=0.5)
    plt.annotate('1:2',
                 xy=(80 * 2., 80.),
                 fontsize=12.,
                 rotation=45.,
                 xycoords='data',
                 alpha=0.5)

    plt.annotate('4:1',
                 xy=(72, 72. * 4.),
                 fontsize=12.,
                 rotation=45.,
                 xycoords='data',
                 alpha=0.5)
    plt.annotate('3:1',
                 xy=(72, 72. * 3.),
                 fontsize=12.,
                 rotation=45.,
                 xycoords='data',
                 alpha=0.5)
    plt.annotate('2:1',
                 xy=(72, 72. * 2.),
                 fontsize=12.,
                 rotation=45.,
                 xycoords='data',
                 alpha=0.5)

    plt.annotate('year alias',
                 xy=(100., 1. / (1. / 100. + 1. / 365.) + 12.),
                 fontsize=12.,
                 rotation=35.,
                 xycoords='data',
                 alpha=0.5,
                 color='r')
    plt.annotate('year alias',
                 xy=(100., 1. / (1. / 100. - 1. / 365.) + 32.),
                 fontsize=12.,
                 rotation=52.,
                 xycoords='data',
                 alpha=0.5,
                 color='r')


def add_year_grid():
    plt.axhline(365., ls='dashed', zorder=-60, alpha=0.3)
    plt.axhline(365. / 2., ls='dashed', zorder=-60, alpha=0.3)
    plt.axhline(365. / 3., ls='dashed', zorder=-60, alpha=0.3)
    plt.axhline(365. / 4., ls='dashed', zorder=-60, alpha=0.3)
    plt.axvline(365., ls='dashed', zorder=-60, alpha=0.3)
    plt.axvline(365. / 2., ls='dashed', zorder=-60, alpha=0.3)
    plt.axvline(365. / 3., ls='dashed', zorder=-60, alpha=0.3)
    plt.axvline(365. / 4., ls='dashed', zorder=-60, alpha=0.3)
    plt.annotate(
        '1 year',
        xy=(980., 365.),
        xycoords='data',
        ha='right',
        va='bottom',
        fontsize=12.,
        color=sns.color_palette()[0],
    )
    plt.annotate('1 year',
                 xy=(365., 980.),
                 xycoords='data',
                 ha='right',
                 va='top',
                 fontsize=12.,
                 color=sns.color_palette()[0],
                 rotation=90)
    for ii in range(2, 5):
        plt.annotate(
            '1/%i year' % ii,
            xy=(980., 365. / ii),
            xycoords='data',
            ha='right',
            va='bottom',
            fontsize=12.,
            color=sns.color_palette()[0],
        )
        plt.annotate('1/%i year' % ii,
                     xy=(365. / ii, 980.),
                     xycoords='data',
                     ha='right',
                     va='top',
                     fontsize=12.,
                     color=sns.color_palette()[0],
                     rotation=90)


## String length calculation


def string_length(times, magnitudes, period):
    mm = (magnitudes - np.min(magnitudes)) * .5 / (np.max(magnitudes) -
                                                   np.min(magnitudes)) - .25
    phases = (times % period) / period
    asort = np.argsort(phases)
    phases, mm = phases[asort], mm[asort]
    ssum = np.sum(
        np.sqrt((mm[1:] - mm[:-1])**2 + (phases[1:] - phases[:-1])**2))
    ssum += np.sqrt((mm[0] - mm[-1])**2 + (phases[0] - phases[-1] + 1)**2)
    return ssum


def string_length_grid(times, magnitudes, minp, maxp, ngrid, log=True):
    period = np.linspace(minp, maxp, ngrid)
    if log:
        period = np.exp(np.linspace(np.log(minp), np.log(maxp), ngrid))
    mm = (magnitudes - np.min(magnitudes)) * .5 / (np.max(magnitudes) -
                                                   np.min(magnitudes)) - .25
    phases = (times % period[:, np.newaxis]) / period[:, np.newaxis]
    asort = np.argsort(phases, axis=1)
    phases, mm = phases[np.arange(phases.shape[0])[:, np.
                                                   newaxis], asort], mm[asort]
    ssum = np.sum(np.sqrt((mm[:, 1:] - mm[:, :-1])**2 +
                          (phases[:, 1:] - phases[:, :-1])**2),
                  axis=1)
    ssum += np.sqrt((mm[:, 0] - mm[:, -1])**2 +
                    (phases[:, 0] - phases[:, -1] + 1)**2)
    return period, ssum


def bin_lc(lc, number_of_nights=1):
    ''' Bin light curve by night'''
    if len(lc) < 1:
        return lc

    def weighted(x, cols, w="weights"):
        return pd.Series(np.average(x[cols], weights=x[w], axis=0), cols)

    time_col = find_time_field(lc)
    # Additional 0.5 for HJD or JD, MJD zero at noon, HJD at midnight
    lc['HJD_int'] = np.int64(lc[time_col] + 0.5 *
                             ((time_col == 'HJD') | (time_col == 'JD')))
    lc['HJD_int'] = lc['HJD_int'] // int(number_of_nights)
    lc['weights'] = 1. / lc['error'].values**2
    lcP = lc.groupby(lc['HJD_int'])
    lcS = lcP.sum()
    lcP = lcP.apply(weighted, set(lc.columns) - set(['HJD_int']))
    lcP = lcP.drop(['weights'], axis=1)
    lcP['error'] = 1. / np.sqrt(lcS['weights'])
    return lcP


def phase_coverage(lc, period):

    N = len(lc)
    t = lc[find_time_field(lc)]
    phase = (t % period) / period
    delta = 2 / N
    phase = np.sort(phase)
    deltaT = np.zeros(N)
    deltaT[1:-1] = .5 * (phase[2:] - phase[:-2])
    deltaT[0] = .5 * (phase[0] - phase[-1] + 1) + .5 * (phase[1] - phase[0])
    deltaT[-1] = .5 * (1 + phase[0] - phase[-1]) + .5 * (phase[-1] - phase[-2])
    deltaT[deltaT > delta] = delta

    return np.sum(deltaT)


def extra_lc_statistics(test):
    rslt = {}
    rslt['mag_diff']= {ii:np.nanstd(test['mag'][ii:].values-test['mag'][:-ii].values) for ii in [1,2,4,8,16,32,64,128]}
    rslt['ncross']= {ii:np.count_nonzero((test['mag'][1:]-np.nanpercentile(test['mag'],ii)).values
                                         *(test['mag'][:-1]-np.nanpercentile(test['mag'],ii)).values<0) 
                  for ii in [25,50,75]}
    rslt['structure']= {ii:np.nanmean([np.nanstd(test['mag'].values[(test['mjdobs'].values>=t)
                                                                    &(test['mjdobs'].values<t+ii)]) 
                                       for t in test['mjdobs'].values])
                  for ii in [1,2,4,8,16,32,64,128,256,512,1024]}
    rslt['time_above']= {ii: 
                  ## All timestep in range
                  np.nansum(
                  np.diff(test['mjdobs'].values)*(
                    (test['mag'][1:].values-np.nanpercentile(test['mag'],ii)>0.)
                    &(test['mag'][:-1].values-np.nanpercentile(test['mag'],ii)>0.))
                  )
                  ## Rising across limit
                 +np.nansum(np.diff(test['mjdobs'].values)
                            *(test['mag'][1:].values-np.nanpercentile(test['mag'],ii))/np.diff(test['mag'].values)
                            *((test['mag'][1:].values-np.nanpercentile(test['mag'],ii))
                              *(test['mag'][:-1].values-np.nanpercentile(test['mag'],ii))<0.)
                                *(test['mag'][:-1].values-np.nanpercentile(test['mag'],ii)>0.)) 
                  ## Falling across limit
                 +np.nansum(np.diff(test['mjdobs'].values)
                            *(np.nanpercentile(test['mag'],ii)-test['mag'][:-1].values)/np.diff(test['mag'].values)
                            *((test['mag'][:-1].values-np.nanpercentile(test['mag'],ii))
                              *(test['mag'][:-1].values-np.nanpercentile(test['mag'],ii))<0.)
                                *(test['mag'][1:].values-np.nanpercentile(test['mag'],ii)>0.)) 
                  for ii in [25,50,75]}
    rslt['std_range']= {'%i_%i'%(ii[0],ii[1]): np.nanstd(test['mjdobs'].values[(test['mag'].values>np.nanpercentile(test['mag'].values,ii[0]))&
                                                    (test['mag'].values<np.nanpercentile(test['mag'].values,ii[1]))])
                for ii in [[2,10],[10,20],[80,90],[90,98]]}
    rslt_combined = {}
    for ii in ['mag_diff','ncross','structure','time_above','std_range']:
        for kk in rslt[ii].keys():
            rslt_combined[ii+'_'+str(kk)]=rslt[ii][kk]
    
    return rslt_combined
