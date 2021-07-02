import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from astropy.time import Time
from astropy.timeseries.periodograms.lombscargle.implementations.utils import trig_sum
from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_probability as fap
from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_level as fal
from astropy.stats import circvar, circcorrcoef
try:
    from .nfft_utils import complex_exponential_sum
except:
    pass

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
    data[fld] -= data[fld].to_numpy()[0] * phase_zero
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


def brute_trig_sum(t, y, freq_factor, freq_grid):
    '''
    Brute force trig sum with irregularly spaced frequency grid
    
    '''
    t, y = map(np.ravel, np.broadcast_arrays(t, y))
    
    f = freq_factor*np.array(freq_grid)
    C = np.dot(y, np.cos(2 * np.pi * f * t[:, np.newaxis]))
    S = np.dot(y, np.sin(2 * np.pi * f * t[:, np.newaxis]))
    
    return S, C
    
def trig_sum_nfft(t, y, freq_factor, use_nfft=False, irreg=False, **kwargs):
    '''
    Wrapper for trig sums based on conditions of computation
    Methods include Brute force (/w irregular frequency grid)/ FFT/ NFFT 
    
    '''
    if irreg:
        # Brute force trig sum for irregular frequency grid
        S, C =  brute_trig_sum(t, y, freq_factor, kwargs['freq_grid'])
        
        return C + S * 1j
    
    elif kwargs['df'] * freq_factor * (t.max() - t.min()) > 1 or not use_nfft:
        # lombscargle implementation of Fast Fourier Transform (if possible)
        S, C = trig_sum(t, y, freq_factor=freq_factor, **kwargs)
        
        return C + S * 1j
    
    else:
        return complex_exponential_sum(t * freq_factor, y, kwargs['f0'],
                                       kwargs['N'], kwargs['df'])

def construct_alpha_beta_matrices(times,
                                  mag_minus_mean,
                                  inverr2,
                                  freq_dict,
                                  nterms=1,
                                  npoly=1,
                                  use_nfft=False, 
                                  regularization=0.,
                                  regularization_power=2.,
                                  time_zeropoint_poly=2457000.,
                                  regularize_by_trace=True):
    
    ws = np.sum(inverr2)   
    magw = inverr2 * mag_minus_mean
    magws = np.sum(magw)

    if 'f0' in freq_dict.keys():
        f0, f1, Nf = freq_dict['f0'], freq_dict['f1'], freq_dict['Nf']
        freq_grid = np.linspace(f0, f1, Nf)
        df = (f1 - f0) / (Nf - 1)
        if df <= 0:
            raise ValueError("df must be positive")
        kwargs = dict(f0=f0, df=df, N=Nf, irreg=False)
    else:
        freq_grid = freq_dict['freq_grid']
        Nf = len(freq_grid)
        kwargs = dict(freq_grid=freq_grid, irreg=True)
    
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
    
    return alpha, beta, freq_grid

def construct_power_grid(mag, inverr2, soln, beta, normalization='standard'):
    
    p = np.sum(beta * soln, axis=1)
    p[p < 0.] = 0.

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

    return p

def build_output_dictionary(alpha, beta, soln, freq_grid, 
                            mag_minus_mean, inverr2, normalization='chi2',
                            nterms=1, npoly=1, use_nfft=False, regularization=0., regularization_power=2.,
                            time_zeropoint_poly=2457000., regularize_by_trace=True):
    results = {}
    results['fourier_coeffs_grid'] = soln
    results['frequency_grid'] = freq_grid
    results['inv_covariance_matrix'] = alpha
    results['chi_squared_grid'] = construct_power_grid(mag_minus_mean, inverr2, soln, beta, normalization=normalization)
    results['chi2_ref'] = np.sum(mag_minus_mean**2 * inverr2)
    results['lsq_nterms'] = nterms
    results['lsq_npoly'] = npoly
    results['lsq_regularization'] = regularization
    results['lsq_regularization_power'] = regularization_power
    results['lsq_time_zeropoint_poly'] = time_zeropoint_poly
    results['lsq_regularize_by_trace'] = regularize_by_trace
    
    return results
    

def fourier_poly_chi2_fit(times,
                          mag,
                          err,
                          freq_dict,
                          normalization='chi2',
                          nterms=1,
                          npoly=1,
                          use_nfft=False, 
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
        freq_dict : dictionary for frequency grid
            either {'f0':float, 'f1':float, 'Nf':int} so f = np.linspace(f0,f1,Nf)
            or {'freq_grid': array}
        normalization: string
            how to normalize power (see astropy.timeseries.LombScargle)
        nterms : int
            number of Fourier terms to use (default 1)
        npoly: int
            number of polynomial terms to use (default 1)
        use_nfft: bool
            if True, use NFFT library. This puts limitations on frequency grid so 
            defaults to no NFFT if conditions not satisfied.
        regularization: float (default = 0.)
            regularization term sum_i (y-Mx)^2/sigma^2 + regularization n M^T M
        regularization_power: float (default = 2.)
            power of k to raise regularization term to
        time_zeropoint_poly: float (default = 2457000.)
            time shift to apply when evaluting polynomial terms
        regularize_by_trace: bool (default = True)
            regularization = regularization * sum(inverr^2) -- Vanderplas & Ivezic (2015)
    """
    
    inverr2 = 1. / err**2
    ws = np.sum(inverr2)
    meanmag = np.dot(inverr2, mag) / ws
    mag_minus_mean = mag - meanmag
    
    alpha, beta, freq_grid = construct_alpha_beta_matrices(times, mag_minus_mean, inverr2, freq_dict,
                                                           nterms=nterms, npoly=npoly, use_nfft=use_nfft,
                                                           regularization=regularization, regularization_power=regularization_power,
                                                           time_zeropoint_poly=time_zeropoint_poly,regularize_by_trace=regularize_by_trace)
    
    soln = np.linalg.solve(alpha, beta)
    soln[:, 0] += meanmag
    
    return build_output_dictionary(alpha, beta, soln, freq_grid, mag_minus_mean, inverr2, normalization=normalization,
                                    nterms=nterms, npoly=npoly, use_nfft=use_nfft, regularization=regularization, regularization_power=regularization_power,
                                    time_zeropoint_poly=time_zeropoint_poly, regularize_by_trace=regularize_by_trace)

def fourier_poly_chi2_fit_test(times,
                          mag,
                          err,
                          freq_dict,
                          nterms_min=1,
                          nterms_max=10,
                          use_bic=False,
                          use_power_of_2=True,
                          check_multiples=True,
                          **kwargs):
    """
        Iterate over Fourier models with different nterms choosing best according to AIC
        
        Parameters
        ----------
        times, mag, err : array_like
            time, values and errors for datapoints
        freq_dict : dictionary for frequency grid
            either {'f0':float, 'f1':float, 'Nf':int} so f = np.linspace(f0,f1,Nf)
            or {'freq_grid': array}
        nterms_min : int
            minimum number of Fourier terms to use (default 1)
        nterms_min : int
            maximum number of Fourier terms to use (default 1)
        use_bic : bool
            Use Bayesian Information Criterion (instead of Akaike)
        use_power_of_2: bool
            if freq_dict is {f0, f1, Nf} format, round Nf to next power of 2
        check_multiples: bool
            check multiples of the best period if the maximum amplitude isn't in the 0th position
    """
    
    if 'Nf' in freq_dict.keys() and use_power_of_2:
        freq_dict['Nf'] = next_power_of_2(freq_dict['Nf'])
        assert freq_dict['Nf'] % 2 == 0 ## Nf must be even
    
    inverr2 = 1. / err**2
    ws = np.sum(inverr2)
    meanmag = np.dot(inverr2, mag) / ws
    mag_minus_mean = mag - meanmag
    
    npoly = 1
    if 'npoly' in kwargs:
        npoly = kwargs['npoly']
    
    alpha, beta, freq_grid = construct_alpha_beta_matrices(times, mag_minus_mean, inverr2, freq_dict,
                                                           nterms=nterms_max, **kwargs)
    
    nterms_use = np.arange(npoly + 2 * nterms_max)
    best_aic=1e300
    best_nterms=nterms_min
    results = {}
    
    for nterms in range(nterms_min,nterms_max+1)[::-1]:
        
        if nterms<nterms_max:
            for p in range(2):
                nterms_use = np.delete(nterms_use, -npoly)
        
        alpha_use = alpha[:,nterms_use[:,np.newaxis],nterms_use[np.newaxis,:]]
        beta_use = beta[:,nterms_use]
        
        soln = np.linalg.solve(alpha_use, beta_use)
        soln[:, 0] += meanmag
        
        results[nterms] = build_bestfit_output_dictionary(
                            build_output_dictionary(alpha_use, beta_use, soln, freq_grid, mag_minus_mean, inverr2, normalization="chi2",
                                    nterms=nterms, **kwargs),
                            freq_dict)

        aic = (.5 * results[nterms]['lsq_chi_squared'] + 2 * nterms) * 2
        if use_bic:
            aic = results[nterms]['lsq_chi_squared'] + np.log(len(mag)) * (2 * nterms)
        ## Second check stops the solutions being highly oscillatory in the gaps in the data
        if aic < best_aic and (~np.any(results[nterms]['amplitudes']>2.*np.diff(np.nanpercentile(mag,[5.,95.])))|(best_aic>1e299)):
            best_aic = aic
            best_nterms = nterms
    
    results = results[best_nterms]
    
    if np.argmax(results['amplitudes'])>0 and check_multiples:
        
        df = get_delta_freq(times)
        alt_result = results.copy()
        best_chi=1e30
        for ii in range(1,1 + np.argmax(results['amplitudes'])):
            if (1 + np.argmax(results['amplitudes'])) % ii != 0:
                continue
            double_freq_dict = {"freq_grid": 1./ii/(results['lsq_period'] / 
                                      (1. + np.argmax(results['amplitudes']))) + np.array([-df,0.,df])}
            double_result= fourier_poly_chi2_fit_full(times,mag,err, double_freq_dict,
                                                      nterms=best_nterms,
                                                       use_power_of_2=False,
                                                       check_multiples=False, 
                                                       **kwargs)

            if double_result['lsq_chi_squared']<results['lsq_chi_squared'] \
                and double_result['lsq_chi_squared']<best_chi:
                alt_result = double_result
                best_chi = double_result['lsq_chi_squared']
                
        if 'lsq_period_error' not in alt_result:
            alt_result['lsq_period_error'] = lsq_uncertainties_irreg(times, mag, err, 1./alt_result['lsq_period'], **kwargs)
                        
        return alt_result
    
    if 'lsq_period_error' not in results:
        results['lsq_period_error'] = lsq_uncertainties_irreg(times, mag, err, 1./results['lsq_period'], **kwargs)

    return results


def false_alarm_probability(power, max_freq, times, mag, error,
                            normalization='standard'):
    norm = normalization
    if norm == 'chi2':
        norm = 'standard'
    fap_ = fap(power,
               max_freq,
               times,
               mag,
               error,
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
        results['lsq_period_error'] = 0.
    else:
        lsq_freq_error = np.sqrt(results['lsq_freq_error'])
        results['lsq_period_error'] = lsq_freq_error/(results['frequency_grid'][argc]**2)

    gradientA = (results['fourier_coeffs_grid'][argc + 1] -
                 results['fourier_coeffs_grid'][argc - 1]) / (2 * df)
    gradientA *= results['lsq_freq_error']
    results['fourier_coeffs_cov_incl_period_error'] = np.copy(results['fourier_coeffs_cov'])
    if np.all(gradientA == gradientA):
        results['fourier_coeffs_cov_incl_period_error'] += \
            gradientA[:,np.newaxis] * gradientA[np.newaxis,:]

    return results

def get_delta_freq(times):
    baseline = times.max() - times.min()
    df = 0.2 / baseline
    return df

def lsq_uncertainties_irreg(times, mag, err, top_freq, **kwargs):
    '''
    Determine Least Squares period error based on curvature of chi2 surface
    around a given frequency top_freq
    '''

    # Period/Frequency error estimate based on curvature of chi2 surface 
    # -- for irregular grid no guarantee that grid is fine enough to well approximate chi2 around minimum
    df = get_delta_freq(times)
    freq_dict = dict(freq_grid=[top_freq-df, top_freq, top_freq+df])
        
    temp_results = fourier_poly_chi2_fit(times, mag, err, freq_dict, normalization="chi2", **kwargs)

    # Estimate error from chi2 curvature
    s_deriv = (temp_results['chi_squared_grid'][0] + temp_results['chi_squared_grid'][2] -
                2 * temp_results['chi_squared_grid'][1])
    
    top_freq = freq_dict['freq_grid'][1]
    
    if s_deriv!=0.:
        lsq_freq_error = 1. / (.5 * s_deriv / df**2)
    else:
        lsq_freq_error = 0.
        
    if lsq_freq_error < 0.:
        lsq_period_error = 0.
    else:
        lsq_freq_error = np.sqrt(lsq_freq_error)
        lsq_period_error = lsq_freq_error/(top_freq**2)
        
    return lsq_period_error


def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()


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

def build_bestfit_output_dictionary(results, freq_dict):
    
    # Fill with best result
    argc = np.argmin(results['chi_squared_grid'])
    results['lsq_period'] = 1. / results['frequency_grid'][argc]
    results['fourier_coeffs'] = results['fourier_coeffs_grid'][argc]
    results['fourier_coeffs_cov'] = np.linalg.inv(
        results['inv_covariance_matrix'][argc])
    results['lsq_chi_squared'] = results['chi_squared_grid'][argc]
    
    # Period/Frequency error estimate based on curvature of chi2 surface 
    # -- for irregular grid no guarantee that grid is fine enough to well approximate chi2 around minimum
    if 'Nf' in freq_dict.keys():
        results = lsq_fit_find_uncertainties(results, argc)
    
    # Compute amplitudes & phases
    nterms = results['lsq_nterms']
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
    
    ## Error at fixed period
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

    for ii in [
            'chi_squared_grid', 'fourier_coeffs_grid',
            'inv_covariance_matrix', 'frequency_grid'
    ]:
        del results[ii]
        
    return results
    
def fourier_poly_chi2_fit_full(times,
                               mag,
                               err,
                               freq_dict, 
                               check_multiples=True,
                               use_power_of_2=True,
                               return_period_error=True,
                               **kwargs
                              ):
    '''
        Wrapper of main algorithm to do all the post-processing
        
        freq_dict = {f0, f1, Nf} - (regular grid) 
                  = {freq_grid=[f_i]} - (irregular grid)

        NFFT works way better if Nf is a power of 2. use_power_of_2 will automatically
        find the next power of 2 > Nf. If may be more convenient for speed reasons to 
        use the lower power of 2. 
        
    '''
    
    if 'Nf' in freq_dict.keys() and use_power_of_2:
        freq_dict['Nf'] = next_power_of_2(freq_dict['Nf'])
        assert freq_dict['Nf'] % 2 == 0 ## Nf must be even
    
    results = fourier_poly_chi2_fit(times, mag, err, freq_dict, normalization="chi2", **kwargs)

    results = build_bestfit_output_dictionary(results, freq_dict)
    
    if np.argmax(results['amplitudes'])>0 and check_multiples:

        df = get_delta_freq(times)
        alt_result = results.copy()
        best_chi=1e30
        
        for ii in range(1,1 + np.argmax(results['amplitudes'])):
            
            if (1 + np.argmax(results['amplitudes'])) % ii != 0:
                continue
                
            double_freq_dict = {"freq_grid": 1./ii/(results['lsq_period'] / 
                                      (1. + np.argmax(results['amplitudes']))) + np.array([-df,0.,df])}
            
            double_result= fourier_poly_chi2_fit_full(times,mag,err, double_freq_dict,
                                                       use_power_of_2=False,
                                                       check_multiples=False, 
                                                       return_period_error=False,
                                                       **kwargs)

            if double_result['lsq_chi_squared']<results['lsq_chi_squared'] \
                and double_result['lsq_chi_squared']<best_chi:
                alt_result = double_result
                best_chi = double_result['lsq_chi_squared']
        
        results = alt_result
    
    if 'lsq_period_error' not in results and return_period_error:
        results['lsq_period_error'] = lsq_uncertainties_irreg(times, mag, err, 1./results['lsq_period'], **kwargs)
        
    return results

def fourier_poly_chi2_fit_nterms_iterations(times,
                                            mag,
                                            err,
                                            freq_dict,
                                            nterms_min,
                                            nterms_max,
                                            use_bic=False,
                                            **kwargs):
    
    best_aic = 1e300
    best_nterms = nterms_min
    results = {}
        
    for nterms in range(nterms_min, nterms_max+1):
        
        kwargs['nterms'] = nterms
        
        results[nterms] = fourier_poly_chi2_fit_full(times, mag, err, freq_dict, **kwargs)
        # Each Fourier term contributes 2 dof
        aic = (.5 * results[nterms]['lsq_chi_squared'] + 2 * nterms) * 2
        
        if use_bic:
            aic = results[nterms]['lsq_chi_squared'] + np.log(len(mag)) * (2 * nterms)
            
        ## Second check stops the solutions being highly oscillatory in the gaps in the data
        if aic < best_aic and (~np.any(results[nterms]['amplitudes']>2.*np.diff(np.nanpercentile(mag,[5.,95.])))|(best_aic>1e299)):
            best_aic = aic
            best_nterms = nterms
            
    return results[best_nterms]


def power_stats(power):
    """
    Find dispersion of maximum power computed about mean and median
    
    """
    max_pow = power.max()
    mean = np.mean(power)
    median = np.median(power)
    sd = np.std(power)
    
    mean_disp = abs(max_pow-mean)/sd
    
    return {'pow_mean_disp':mean_disp}

# # Sidereal and standard day aliases to be removed
# alias_periods = np.array([0.99726, 0.99999, 0.99726/2., 0.99999/2., 0.99726/3., 0.99999/3., 0.99999/4., 0.99726/4.])

# def get_topN_freq(freq, power, config, N=30, tol=1e-3):
#     """
#     Retrieve top N frequencies from LombScargle based on power ranking
    
#     """
#     arg_pows = np.argsort(power)
#     topN_freqs = freq[arg_pows][-N:][::-1]
#     topN_powrs = power[arg_pows][-N:][::-1]

#     period_tol = np.float64(config['period_tol'])
#     _ind = 0
#     while any(np.isclose(1./topN_freqs[_ind], alias_periods, rtol=0, atol=period_tol)):
#         _ind+=1
#         if(_ind==N):
#             break
#     if(_ind==N):
#         return dict(top_distinct_freqs=np.array([]), top_distinct_freq_power=np.array([]))
 
#     ls_period, max_pow = 1./topN_freqs[_ind], topN_powrs[_ind]
    
#     top_distinct_freqs = []
#     top_distinct_freq_power = []
#     while len(topN_freqs)>=1:
#         curr = topN_freqs[0]
#         if ~np.any(np.isclose(1./curr, alias_periods, rtol=0, atol=period_tol)):
#             top_distinct_freqs.append(curr)
#             top_distinct_freq_power.append(topN_powrs[0])
        
#         # Group frequencies within certain tolerance
#         group_bool = np.isclose(curr, topN_freqs, rtol=0., atol=tol)
#         topN_freqs = topN_freqs[~(group_bool)]
#         topN_powrs = topN_powrs[~(group_bool)]
        
#     return dict(ls_period=ls_period, max_pow=max_pow, top_distinct_freqs=np.array(top_distinct_freqs),
#                 top_distinct_freq_power=np.array(top_distinct_freq_power))

def is_window_function_peak(times, mags, errors, freqs, power):
    
    model = LombScargle(times,
                    np.ones_like(times),
                    errors,
                    center_data=False, fit_mean=False
                    )
    power_window = model.power(freqs)
    return power_window>power


def get_top_frequencies(times, mags, errors, freq, power, config, N=30):
    """
    Retrieve top N frequencies from LombScargle based on power ranking
    
    """
    peaks = find_peaks(power)[0]
    arg_pows = np.argsort(power[peaks])
    topN_freqs = freq[peaks][arg_pows][-N:][::-1]
    topN_powrs = power[peaks][arg_pows][-N:][::-1]
    
    is_wf = is_window_function_peak(times, mags, errors, 
                                    topN_freqs, 
                                    topN_powrs)
    
    topN_freqs, topN_powrs = topN_freqs[~is_wf], topN_powrs[~is_wf]
    
    if len(topN_freqs)==0:
        return dict(top_distinct_freqs=np.array([]), top_distinct_freq_power=np.array([]))
    
    return {'top_distinct_freqs':topN_freqs, 
            'max_pow':topN_powrs[0],
            'ls_period':1./topN_freqs[0]}

def lombscargle_stats(times, mags, errors, config, N=30, irreg=True, **ls_kwargs):
    """
    LombScargle analysis of lightcurve to extract certain summary statistics
    
    """
        
    # Initialise model
    model = LombScargle(times, mags, errors, normalization='standard')
        
    freq, power = model.autopower(**ls_kwargs)
    
    # Find power array stats
    pow_stats = power_stats(power)
    
    if irreg:
        
        freq_dict = get_top_frequencies(times, mags, errors, freq, power, config, N)
        Nstep = N
        
        while len(freq_dict['top_distinct_freqs'])==0:
            N+=Nstep
            if(N>1000):
                break
            freq_dict = get_top_frequencies(times, mags, errors, freq, power, config, N)
        
        out = {**freq_dict, **pow_stats}
    
    else:
        # Find max power and hence most likely period
        periods = 1./freq
        max_pow_arg = np.argmax(power)
        period, max_pow = periods[max_pow_arg], power[max_pow_arg]
        out = {'ls_period':period, 'max_pow':max_pow, **pow_stats}
    
    return out


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


def retrieve_fourier_poly(times, results, with_var=False, 
                          var_at_best_period=True, phase_min=None, 
                          zero_poly_terms=False):
    Kmax = results['lsq_nterms'] + 1
    npoly = results['lsq_npoly']
    period = results['lsq_period']
    time_zeropoint_poly = results['lsq_time_zeropoint_poly']
#     if phase_min==None:
#         time_zeropoint_poly = results['lsq_time_zeropoint_poly']
#     else: 
#         time_zeropoint_poly = phase_min
        
    SINX = np.array(
        [np.sin(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    COSX = np.array(
        [np.cos(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    ONES = np.array(
        [pow(times - time_zeropoint_poly, k) for k in range(npoly)])
    SCX = np.empty(((Kmax - 1) * 2, len(times)))
    SCX[::2] = SINX
    SCX[1::2] = COSX
    
    if zero_poly_terms:
        X = np.concatenate([ONES[:1], SCX])
    else:
        X = np.concatenate([ONES[:1], SCX, ONES[1:]])
    
    if with_var:
        cov = results['fourier_coeffs_cov'+'_incl_period_error'*(~var_at_best_period)]
        full_cov = np.dot(X.T, np.dot(cov[:len(X),:len(X)], X))
        return np.dot(results['fourier_coeffs'][:len(X)], X), np.diag(full_cov)
    
    return np.dot(results['fourier_coeffs'][:len(X)], X)


def retrieve_fourier_poly_firstderiv(times, results):
    Kmax = results['lsq_nterms'] + 1
    period = results['lsq_period']
    SINX = np.array(
        [k * (2.*np.pi/period) * np.cos(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    COSX = np.array(
        [-k * (2.*np.pi/period) * np.sin(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    SCX = np.empty(((Kmax - 1) * 2, len(times)))
    SCX[::2] = SINX
    SCX[1::2] = COSX
    return -np.dot(results['fourier_coeffs'][1:len(SCX)+1], SCX)

def retrieve_fourier_poly_secondderiv(times, results):
    Kmax = results['lsq_nterms'] + 1
    period = results['lsq_period']
    SINX = np.array(
        [k**2 * (2.*np.pi/period)**2 * np.sin(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    COSX = np.array(
        [k**2 * (2.*np.pi/period)**2 * np.cos(k * times / period * 2. * np.pi) for k in np.arange(1, Kmax)])
    SCX = np.empty(((Kmax - 1) * 2, len(times)))
    SCX[::2] = SINX
    SCX[1::2] = COSX
    return -np.dot(results['fourier_coeffs'][1:len(SCX)+1], SCX)

def integer_periods(t, results):
    return (t // results['lsq_period']) * results['lsq_period']

def find_maximum_fourier(results):
    phse = np.linspace(0.,1.,1000)
    full_phase_curve = retrieve_fourier_poly(results['lsq_period'] * phse + integer_periods(results['lsq_time_zeropoint_poly'], results),
                                             results, zero_poly_terms=True)
    return np.min(full_phase_curve)

def find_amplitude_fourier(results):
    phse = np.linspace(0.,1.,1000)
    full_phase_curve = retrieve_fourier_poly(results['lsq_period'] * phse + integer_periods(results['lsq_time_zeropoint_poly'], results),
                                             results, zero_poly_terms=True)
    return np.max(full_phase_curve)-np.min(full_phase_curve)

def find_phase_of_minimum(results):
    phse = np.linspace(0.,1.,1000)
    full_phase_curve = retrieve_fourier_poly(results['lsq_period'] * phse + integer_periods(results['lsq_time_zeropoint_poly'], results), 
                                             results, zero_poly_terms=True)
    return phse[np.argmax(full_phase_curve)]

def check_significant_second_minimum(results, min_phase, phase_range=[0.35,0.65], noise_thresh_factor=7., show_plot=False, return_min_location=False):
    '''
        Finds whether there is a minimum of depth > noise_thresh_factor * noise in the phase interval phase_range (normalized 0 to 1)
        Depth is determined by minimum separation from nearby maxima
        min_phase is the phase corresponding to absolute minimum of light curve (output from find_phase_minimum)
    '''
    
    # Find location of all minima in phase_range and all maxima
    phases = np.linspace(0.,1.,500)
    first_deriv = retrieve_fourier_poly_firstderiv(results['lsq_period'] * (phases + min_phase), results)
    turning_points = (first_deriv[1:]*first_deriv[:-1]<0)
    mid_phases = .5 * (phases[1:]+phases[:-1])
    sign_at_tp = np.sign(retrieve_fourier_poly_secondderiv(results['lsq_period'] * (mid_phases + min_phase), results))
    minima = np.argwhere((sign_at_tp<0)&turning_points&(mid_phases>phase_range[0])&(mid_phases<phase_range[1]))
    maxima = np.argwhere((sign_at_tp>0)&turning_points)
    
    # Find distance between minima and maxima so we can find the maxima that encompass minima
    n = len(first_deriv) - 1
    distance = ((minima[:,np.newaxis] - maxima[np.newaxis,:]) + n//2)%(n) - n//2
    distance_positive, distance_negative = distance.copy(), distance.copy()
    distance_positive[distance_positive<0]=n+1
    distance_negative[distance_negative>0]=-(n+1)

    fpoly, fpoly_var = retrieve_fourier_poly(results['lsq_period'] * (mid_phases + min_phase)
                                             + integer_periods(results['lsq_time_zeropoint_poly'], results), 
                                                       results, with_var=True, zero_poly_terms=True)
    
    min_distance = np.hstack([np.argsort(distance_positive, axis=1)[:,:1,0],
                                  np.argsort(-distance_negative, axis=1)[:,:1,0]])
    
    if show_plot:
        plt.plot(fpoly)
        plt.plot(fpoly-noise_thresh_factor * np.sqrt(fpoly_var))
        [plt.axvline(d) for d in minima]
        [plt.axvline(d,color='r') for d in maxima[min_distance.flatten()]]
    
    # Check if minima depth > noise_thresh * noise
    noise = np.sqrt(fpoly_var[minima])
    is_there_second_minimum = np.any((fpoly[minima] - np.nanmax(fpoly[maxima[min_distance]],axis=1)).flatten()>noise_thresh_factor*noise.flatten())

    # Return location of minimum if required
    if return_min_location:
        if is_there_second_minimum:
            min_loc = mid_phases[minima][np.argmax(fpoly[minima] - np.nanmean(fpoly[maxima[min_distance]],axis=1)).flatten()][0][0]
            return is_there_second_minimum, min_loc
        else:
            return is_there_second_minimum, None
    
    return is_there_second_minimum
                

def find_peak_ratio_model(results, min_phase, second_minimum, min_phase_2):
    '''
        Find the ratio of the minima depth (relative to 1st percentile of magnitude) using model.
        First checks if there are significant secondary minima. If not every other primary minimum is compared (which will always return 1)
        min_phase is the phase corresponding to absolute minimum of light curve (output from find_phase_minimum)
        second_minimum, min_phase_2 is the output of check_significant_second_minimum
    '''
    
    ## Two minima per period
    if second_minimum:
        middle_min=retrieve_fourier_poly(np.ones(1) * results['lsq_period'] * (min_phase_2 + min_phase)
                                        + integer_periods(results['lsq_time_zeropoint_poly'], results),results, 
                                         zero_poly_terms=True)[0]
        first_min=retrieve_fourier_poly(np.ones(1) * min_phase * results['lsq_period']
                                        + integer_periods(results['lsq_time_zeropoint_poly'], results),results, 
                                        zero_poly_terms=True)[0]
        maxx = find_maximum_fourier(results)
        if first_min>middle_min:
            first_min, middle_min = middle_min, first_min
        return (first_min-maxx)/(middle_min-maxx)
    ## One minimum per period
    else:
        return 1.

def _ivw(mag, errors, lc_min, fltr):
    return np.nansum(((mag-lc_min)/errors**2)[fltr]) / np.nansum((1./errors**2)[fltr])

def find_peak_ratio_data(times, mag, errors, results, min_phase, second_minimum, min_phase_2, min_bin_size=0.05, Ndatapoints=3):
    '''
        Find the ratio of the minima depth (relative to 1st percentile of magnitude) using data.
        First checks if there are significant secondary minima. If not every other primary minimum is compared.
        The depth of the peak is governed by data within +/-bin_size phase around minima. bin_size is broadened until Ndatapoints are encompassed
        min_phase is the phase corresponding to absolute minimum of light curve (output from find_phase_minimum)
        second_minimum, min_phase_2 is the output of check_significant_second_minimum
    '''
    ## Find location of absolute minimum phase and detect if significant minimum
    lc_min = np.nanpercentile(mag, 1.)
    min_time = min_phase * results['lsq_period']
    
    period = (1 + ~second_minimum) * results['lsq_period']
    
    if second_minimum:
        min_phase_2nd = min_phase_2
    else:
        min_phase_2nd = .5
    
    fltr1 = lambda phases, min_bin_size: (phases < min_bin_size)
    fltr2 = lambda phases, min_bin_size: (phases > min_phase_2nd) & (phases < min_phase_2nd + min_bin_size)
    
    phases = ((times-min_time) / period + min_bin_size) % 1
    while (np.count_nonzero(fltr1(phases, min_bin_size))<Ndatapoints) | (np.count_nonzero(fltr2(phases, min_bin_size))<Ndatapoints):
        min_bin_size+=0.01
        phases = ((times-min_time) / period + min_bin_size) % 1
        if(min_bin_size>min_phase_2nd):
            break
        
    peak1 = _ivw(mag, errors, lc_min, fltr1(phases, min_bin_size))
    peak2 = _ivw(mag, errors, lc_min, fltr2(phases, min_bin_size))
    
    ## check peak ratio <=1
    if peak1>peak2:
        peak2, peak1 = peak1, peak2
        
    return peak1/peak2
        


def get_delta_log_likelihood(y, yerr, pred_mean, free_var=False):
    ''' Difference in log-likelihood between Fourier model and pure Gaussian scatter '''
    
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
