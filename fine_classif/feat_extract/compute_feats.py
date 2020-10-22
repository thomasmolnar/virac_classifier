from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import astropy
import random

from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import LombScargle

from multiprocessing import Pool
from functools import partial

from fine_classif.lc_utils import lombscargle
from fine_classif.virac_lc import *
import extinction_map


def magarr_stats(data):
    """
    Returns simple dict of population statistics of timeseries photometric data
    
    """
    data_use = data.copy()
    mags = data_use.mag.values
    
    # Get common statistics with describe() functionality
    nobs, minmax, mean, var, skew, kurt = stats.describe(mags)
    sd = np.sqrt(var)
    amp = (minmax[1]-minmax[0])/2.
    
    # Comp range statisitics
    data_use.drop(data_use[data_use.mag.values > float(mean+sd)].index, inplace=True)
    data_use.drop(data_use[data_use.mag.values < float(mean-sd)].index, inplace=True)
    within = len(data_use.mag.values)
    beyondfrac = (len(mags)-within)/len(mags)
    median = np.median(mags)
    magdev = mags - median
    med_magdev = np.median(magdev)
    
    out = {'mean': mean, 'sd': sd, 'skew': skew, 'kurt': kurt, 'amplitude': amp,
           'beyondfrac': beyondfrac, 'med_magdev': med_magdev}

    return out


def time_span(data):
    """
    Returns the time span of observational epochs
    
    """
    data_use = data.copy()
    times = data_use.HJD.values
    max_t, min_t = np.max(times), np.min(times)
    span = max_t - min_t
    
    return span


def periodic_feats(data, nterms=4, npoly=1):
    """
    Returns periodic features (fourier components) of photometric lightcurves
    
    """
    times = data.HJD.values
    mags = data.mag.values
    err = data.error.values
    
    #dt = 0.1/t_range from classification paper
    span = round(time_span(data))
    sep = 0.1/span
    rang = math.ceil((20.-(1/span))/sep)
    
    #Need range to be even
    if rang % 2 == 0:
        pass
    else:
        rang+=1
        
    results = lc_utils.fourier_poly_chi2_fit_full(times=times,
                                         mag=mags,
                                         err=err,
                                         f0=1/span,
                                         f1=20.,
                                         Nf=rang,
                                         nterms=nterms,
                                         npoly=npoly,
                                         regularization=0.1,
                                         keep_small=True, 
                                         code_switch=True,
                                         use_power_of_2=True)
    
    # Calculate delta log likelihood between periodic and constant Gaussian scatter models
    pred_mean = lc_utils.retrieve_fourier_poly(times=times, results=results)
    delta_loglik = lc_utils.get_delta_log_likelihood(mags, err, pred_mean=pred_mean)
    
    # Extract relevant Fourier terms
    amps = np.array(results['amplitudes'])
    phases = np.array(results['phases'])
    per = float(results['lsq_period'])
    
    return {'lsq_period':per, 'amp_0':amps[0], 'amp_1':amps[1], 'amp_2':amps[2],
            'amp_3':amps[3], 'phi_0':phases[0], 'phi_1':phases[1], 'phi_2':phases[2],
            'phi_3':phases[3], 'delta_loglik':delta_loglik}


def periodic_feats_force(data, period, nterms=4, npoly=1):
    """
    Returns periodic features (fourier components) of photometric lightcurves
    
    """
    times = data.HJD.values
    mags = data.mag.values
    err = data.error.values
    
    # Force frequency grid to include LombScargle analysis period and integer multiples 
    f0 = 1./(period*3)
    f1 = 1./(period*0.25)
        
    results = lc_utils.fourier_poly_chi2_fit_full(times=times,
                                         mag=mags,
                                         err=err,
                                         f0=f0,
                                         f1=f1,
                                         Nf=5,
                                         nterms=nterms,
                                         npoly=npoly,
                                         regularization=0.1,
                                         keep_small=True, 
                                         code_switch=True,
                                         use_power_of_2=True)
    
    # Calculate delta log likelihood between periodic and constant Gaussian scatter models
    pred_mean = lc_utils.retrieve_fourier_poly(times=times, results=results)
    delta_loglik = lc_utils.get_delta_log_likelihood(mags, err, pred_mean=pred_mean)
    
    # Extract relevant Fourier terms
    amps = np.array(results['amplitudes'])
    phases = np.array(results['phases'])
    per = float(results['lsq_period'])
    
    return {'lsq_period':per, 'amp_0':amps[0], 'amp_1':amps[1], 'amp_2':amps[2],
            'amp_3':amps[3], 'phi_0':phases[0], 'phi_1':phases[1], 'phi_2':phases[2],
            'phi_3':phases[3], 'delta_loglik':delta_loglik}


def correct_to_HJD(data, ra, dec):
    """
    Convert from MJD to HJD based on skycoords
    
    """
    coordinate = SkyCoord(ra * u.deg,
                          dec * u.deg,
                          frame='icrs')
    times = time.Time(data['mjdobs'],
                      format='mjd',
                      scale='utc',
                      location=paranal)
    data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
        coordinate).to(u.day).value

def source_feat_extract(data, ls_kwargs={}, config):
    """
    Wrapper to extract all features for a given
    source light curve (panda format).
   
    input: Time Series light curve data
    
    returns: dict of features
    
    """
    ra, dec, lc = data[0],data[1],data[2]
    
    # Correct MJD to HJD
    correct_to_HJD(lc, ra, dec)
    
    # Need to inlcude quality cuts for amb_match, ast_res_chisq, chi
    # Pre-process light curve data with quality cut and 3 sigma conservative cut 
    lc = sigclipper(quality_cut_phot(lightcurve, config['amb_correction']),
                             thresh=3.)
    
    # Extract non-periodic and extra statistics 
    nonper_feats = magarr_stats(lc)
    
    # Exract light-curve summary periodic statistics from Fourier analysis
    per_dict = lc_utils.lombscargle(lc, **ls_kwargs)
    per_feats = periodic_feats_force(lc, period=per_dict['ls_period'], nterms=4, npoly=1)
    
    features = {**per_feats, **per_dict, **nonper_feats}
    
    return features



