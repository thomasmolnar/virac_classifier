import numpy as np
import pandas as pd

from astropy.timeseries import LombScargle
try:
    from fourier_chi2 import find_period, fourier_poly_chi2_fit_cython
except:
    pass
from astropy.time import Time
from gatspy import periodic

from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
from .nfft_utils import complex_exponential_sum


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