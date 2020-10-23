from scipy import stats
import numpy as np
import pandas as pd
import math

from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord

from multiprocessing import Pool
from functools import partial

import lc_utils

def magarr_stats(data):
    """
    Returns simple dict of population statistics of timeseries photometric data
    
    """
    mags = data.mag.values
    
    # Get common statistics
    nobs, minmax, mean, var, skew, kurt = stats.describe(mags)
    sd = np.sqrt(var)
    amp = (minmax[1]-minmax[0])/2.
    
    # Comp range statisitics
    beyond = data[(mags < float(mean+sd))].reset_index(drop=True)
    beyond = beyond[(beyond.mag.values > float(mean-sd))].reset_index(drop=True)
    within = len(beyond)
    beyondfrac = (len(mags)-within)/len(mags)
    median = np.median(mags)
    magdev = mags - median
    med_magdev = np.median(magdev)
    
    out = {'mean': mean, 'sd': sd, 'skew': skew, 'kurt': kurt, 'amplitude': amp,
           'beyondfrac': beyondfrac, 'med_magdev': med_magdev}

    return out


def time_span(times):
    """
    Returns the time span of observational epochs
    
    """
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
    span = round(time_span(times))
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
    with forced frequency input grid (determined from LombScargle)
    
    """
    times = data.HJD.values
    mags = data.mag.values
    err = data.error.values
    
    # Forced frequency grid 
    f0 = 1./(2*period)
    f1 = 1./(period)
    Nf = 2
        
    results = lc_utils.fourier_poly_chi2_fit_full(times=times,
                                         mag=mags,
                                         err=err,
                                         f0=f0,
                                         f1=f1,
                                         Nf=Nf,
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
    
    times = time.Time(data['mjdobs'][0],
                      format='mjd',
                      scale='utc',
                      location=paranal)
    
    data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
        coordinate).to(u.day).value

def quality_cut(data, chicut=5., ast_cut=11.829, amb_corr=True):
    """
    Photometry/Astrometric quality cuts
    
    chi < 5. -- require high quality detection
    ast_res_chisq < 11.829 -- residual to ast fit (approx. < 3sigma)
    ambiguous_match = False -- require detection is unambiguously associated with this source
    
    """
    maglimit = 13.2 ## BASED ON EXPERIMENTS WITH MATSUNAGA
    
    data = data[~((data['chi'] > dpchicut) &
                (data['mag'] < maglimit))].reset_index(drop=True)
    data = data[~(data['ast_res_chisq']>ast_cut)].reset_index(drop=True)
    
    if amb_corr:
        data = data[~(data['ambiguous_match'])].reset_index(drop=True)
    
    return data

def sigclipper(data, sig_thresh=4.):
    """
    Clip light curve based on sigma distance from mean 
    
    """
    if len(data) <= 1:
        return data
    
    stdd = .5 * np.diff(np.nanpercentile(data['mag'].values, [16, 84]))
    midd = np.nanmedian(data['mag'].values)
    
    return data[np.abs(data['mag'].values - midd) / stdd < sig_thresh].reset_index(
        drop=True)
    
def source_feat_extract(data, ls_kwargs={}, config):
    """
    Wrapper to extract all features for a given
    source light curve (panda format).
   
    input: Time Series light curve data
    
    returns: dict of features
    
    columns of light curves:
    ['sourceid' 'mjdobs' 'mag' 'error' 'ambiguous_match' 'ast_res_chisq' 'chi']
    """
    # Get Source info
    ra, dec, lc = data[0],data[1],data[2]
    sourceid = lc['sourceid'].values[0]
    
    # Correct MJD to HJD
    correct_to_HJD(lc, ra, dec)
    
    # Need to inlcude quality cuts for amb_match, ast_res_chisq, chi
    # Pre-process light curve data with quality cut and 3 sigma conservative cut 
    lc_clean = sigclipper(quality_cut_phot(lc, config['chi_cut'],config['ast_cut'],
                                           config['amb_correction']), config['sig_thresh'])
    
    # Length check post quality cuts
    if len(lc_clean)<=1:
        return {'sourceid':sourceid, 'small':True}
    
    # Extract non-periodic statistics 
    nonper_feats = magarr_stats(lc_clean)
    
    # Exract light-curve summary periodic statistics from Fourier analysis
    per_dict = lc_utils.lombscargle_stats(lc_clean, **ls_kwargs)
    
    if config['force_method']:
        per_feats = periodic_feats_force(lc_clean, period=per_dict['ls_period'],
                                     nterms=config['nterms'], npoly=config['npoly'])
    else:
        per_feats = periodic_feats(lc_clean, period=per_dict['ls_period'],
                                     nterms=config['nterms'], npoly=config['npoly'])
    
    features = {'sourceid':sourceid, **per_feats, **per_dict, **nonper_feats}
    
    return features



