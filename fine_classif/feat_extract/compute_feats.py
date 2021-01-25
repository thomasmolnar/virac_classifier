from scipy import stats
import numpy as np
import pandas as pd
import math

import time as tt

from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord

from multiprocessing import Pool
from functools import partial

from .lc_utils import *
from astropy.timeseries.periodograms.lombscargle._statistics import false_alarm_probability as fap

from astropy.utils.iers import IERS_A_URL, IERS_B_URL, conf
from astropy.utils.data import download_file
conf.auto_download = True
download_file(IERS_A_URL, cache="update")
download_file(IERS_B_URL, cache="update")
conf.auto_download = False

# Earth coord global
paranal = coord.EarthLocation(1946404.3,-5467644.3,-2642728.2, unit='m') #.of_site('paranal')

def magarr_stats(mags):
    """
    Returns simple dict of population statistics of timeseries photometric data
    
    """
    # Get common statistics
    nobs, minmax, mean, var, skew, kurt = stats.describe(mags)
    sd = np.sqrt(var)
    amp = (minmax[1]-minmax[0])/2.
    
    # Comp range statisitics
    beyond = mags[(mags < float(mean+sd))]
    beyond = beyond[(beyond > float(mean-sd))]
    beyondfrac = (mags.size-beyond.size)/mags.size
    median = np.median(mags)
    magdev = mags - median
    med_magdev = np.median(magdev)
    
    out = {'mean': mean, 'sd': sd, 'skew': skew, 'kurt': kurt, 'amplitude': amp,
           'beyondfrac': beyondfrac, 'med_magdev': med_magdev}

    return out

def optimum_regularization(mags, errors):
    '''
    Appropriate for use with regularize_by_trace = True and regularization_power=2
    '''
    
    p5_p95 = np.diff(np.nanpercentile(mags,[5.,95.]))[0]
    noise = np.nanmean(errors)
    
    N = len(mags)
    
    return np.min([0.25, 2./(p5_p95/noise)**3/(N/100)])

def compute_peak_properties(times, mags, errors, results, nterms_min):
    
    min_phase = find_phase_of_minimum(results)

    second_minimum, min_phase_2 = check_significant_second_minimum(results, min_phase, return_min_location=True)
    
    peak_ratio_model = find_peak_ratio_model(results, min_phase, second_minimum, min_phase_2)
    peak_ratio_data = find_peak_ratio_data(times, mags, errors, results, min_phase, second_minimum, min_phase_2)
    
    # If no significant second minimum detected we also compute coefficients for the Fourier series at double the period
    if ~second_minimum:
        results_d = fourier_poly_chi2_fit_full(
                                         times=times,
                                         mag=mags,
                                         err=errors,
                                         freq_dict={'freq_grid':np.array([.5/results['lsq_period']])},
                                         nterms=results['lsq_nterms'],
                                         npoly=results['lsq_npoly'],
                                         regularization=results['lsq_regularization'],
                                         time_zeropoint_poly=results['lsq_time_zeropoint_poly'],
                                         regularize_by_trace = results['lsq_regularize_by_trace'],
                                         check_multiples=False, 
                                         use_power_of_2=False,
                                         return_period_error=False)
        amp_double, phase_double = results_d['amplitudes'], results_d['phases']
    else:
        amp_double, phase_double = results['amplitudes'], results['phases']
        
    amp_double_dict = {'amp_double_%i'%i:a for i,a in enumerate(amp_double[:nterms_min])}
    phase_double_dict = {'phi_double_%i'%i:a for i,a in enumerate(phase_double[:nterms_min])}
    
    return {'peak_ratio_model':peak_ratio_model, 'peak_ratio_data':peak_ratio_data, 
            'significant_second_minimum':second_minimum, 
            **amp_double_dict, **phase_double_dict}

def periodic_feats(times, mags, errors, nterms_min=4, nterms_max=10, npoly=1, max_freq=20.):
    """
    Returns periodic features (fourier components) of photometric lightcurves
    
    """
    # df = 0.2/baseline -- same as LombScargle/timeseries literature
    baseline = times.max() - times.min()
    df = 0.2 / baseline
    
    f0 = 0.5 * df
    Nf = 1 + int(np.round((max_freq - f0) / df))
    
    # Require range to be even
    if Nf % 2 == 0:
        pass
    else:
        Nf += 1
    
    results = fourier_poly_chi2_fit_nterms_iterations(
                                         times=times,
                                         mag=mags,
                                         err=errors,
                                         freq_dict = {'f0':f0, 'f1':f1, 'Nf':Nf},
                                         nterms_min=nterms_min,
                                         nterms_max=nterms_max,
                                         npoly=npoly,
                                         regularization=optimum_regularization(mags, errors),
                                         use_power_of_2=True)
    
    # Calculate delta log likelihood between periodic and constant Gaussian scatter models
    pred_mean = retrieve_fourier_poly(times=times, results=results)
    delta_loglik = get_delta_log_likelihood(mags, errors, pred_mean=pred_mean)
    peak_output = compute_peak_properties(times, mags, errors, results)
    
    # Extract relevant Fourier terms
    amps = {'amp_%i'%i:a for i,a in enumerate(results['amplitudes'])[:nterms_min]}
    phases = {'phi_%i'%i:a for i,a in enumerate(results['phases'])[:nterms_min]}
    per = float(results['lsq_period'])
    
    peak_output = compute_peak_properties(times, mags, errors, results, nterms)
    
    lsq_power = 1 - results['lsq_chi_squared'] / results['chi2_ref']
    
    return {'lsq_period':per, 'delta_loglik':delta_loglik, 'lsq_power':lsq_power,
            'lsq_nterms':results['lsq_nterms'],
            **amps, **phases, **peak_output}


def periodic_feats_force(times, mags, errors, freq_dict,
                         nterms_min=4, nterms_max=10, npoly=1,
                         fn=fourier_poly_chi2_fit_test):
    """
    Returns periodic features (fourier components) of photometric lightcurves
    with forced frequency input grid (determined from LombScargle).
    
    """
    results = fn(
                 times=times,
                 mag=mags,
                 err=errors,
                 freq_dict=freq_dict,
                 nterms_min=nterms_min,
                 nterms_max=nterms_max,
                 npoly=npoly,
                 regularization=optimum_regularization(mags, errors),
                 use_power_of_2=True,
                 use_bic=False)
    
    # Calculate delta log likelihood between periodic and constant Gaussian scatter models
    pred_mean = retrieve_fourier_poly(times=times, results=results)
    delta_loglik = get_delta_log_likelihood(mags, errors, pred_mean=pred_mean)
    peak_output = compute_peak_properties(times, mags, errors, results, nterms_min)
    
    # Extract relevant Fourier terms
    amps = {'amp_%i'%i:a for i,a in enumerate(results['amplitudes'][:nterms_min])}
    phases = {'phi_%i'%i:a for i,a in enumerate(results['phases'][:nterms_min])}
    per = float(results['lsq_period'])
    per_error = results['lsq_period_error']
    
    peak_output = compute_peak_properties(times, mags, errors, results, nterms_min)
    
    lsq_power = 1 - results['lsq_chi_squared'] / results['chi2_ref']
    
    return {'lsq_period':per, 'lsq_period_error':per_error, 
            'delta_loglik':delta_loglik, 'lsq_power': lsq_power,
            'lsq_nterms':results['lsq_nterms'],
            **amps, **phases, **peak_output}

def find_lag(times, period):
    """
    Find probabilistic metrics to determine if the time between observations of phase folded light curves constitute 'lagging'
    
    """
    # Find time diffs of ordered phase folded light curve
    times_fld = times%(period)
    times_fld_ord = np.sort(times_fld)
    times_fld_diff = np.diff(times_fld_ord)
    
    if times_fld_diff.size==0:
        return {'error':True}
    
    # Calculate summary statistics of difference array
    t_max = np.nanmax(times_fld_diff)
    mean = np.nanmean(times_fld_diff)
    sd = np.nanstd(times_fld_diff)
    
    # Dispersion of max
    mean_disp = np.abs(t_max-mean)/sd
    
    return {'time_lag_mean':mean_disp, 'max_time_lag':t_max, 'error':False}


def correct_to_HJD(data, ra, dec):
    """
    Convert from MJD to HJD based on skycoords
    
    """
    coordinate = SkyCoord(ra * u.deg,
                          dec * u.deg,
                          frame='icrs')
    
    times = time.Time(data['mjdobs'].values[0],
                      format='mjd',
                      scale='utc',
                      location=paranal)
    
    data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
        coordinate).to(u.day).value

def quality_cut(data, chicut=5., ast_cut=11.829, amb_corr=1):
    """
    Photometry/Astrometric quality cuts
    
    chi < 5. -- require high quality detection
    ast_res_chisq < 11.829 -- residual to ast fit (approx. < 3sigma)
    ambiguous_match = False -- require detection is unambiguously associated with this source
    
    """
    maglimit = 13.2 ## BASED ON EXPERIMENTS WITH MATSUNAGA
    
    data = data[~((data['chi'] > chicut) &
                (data['mag'] < maglimit))].reset_index(drop=True)
    data = data[~(data['ast_res_chisq']>ast_cut)].reset_index(drop=True)
    data = data[(data['emag']>0)].reset_index(drop=True)
    
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


def source_feat_extract(lc, config, ls_kwargs={}, method_kwargs={}):
    """
    Wrapper to extract all features for a given
    source light curve (panda format).
   
    inputs: 
    
    - data = [ra, dec, lc_df] Time Series light curve data and position for each source
    
    - config: configuration of global variables

    - ls_kwargs: arguments to be passed for frequency grid spacing in LombScargle
    
    - method_kwargs = {irreg=(False), use_fft=(False), use_nfft=(False)}, arguments defining method of computation
    
    returns: dict of features
    
    columns of light curves:
    ['sourceid' 'mjdobs' 'mag' 'error' 'ambiguous_match' 'ast_res_chisq' 'chi']
    
    """
    tot_s = tt.time()
    pp_s = tt.time()
    # Split source info
    ra, dec = lc['ra'].values[0], lc['dec'].values[0]
    sourceid = lc['sourceid'].values[0]
    
    # Correct MJD to HJD
    correct_to_HJD(lc, ra, dec)

    # Pre-process light curve data with quality cuts and 3 sigma conservative cut 
    chi_cut, ast_cut = float(config['chi_cut']), float(config['ast_cut'])
    amb_corr = int(config['amb_correction'])
    sig_thresh = float(config['sig_thresh'])
    lc_clean = sigclipper(quality_cut(lc.dropna(subset=['mag', 'emag']), chi_cut, ast_cut, amb_corr), sig_thresh)
    
    # Length check post quality cuts
    if len(lc_clean)<=int(config['n_detection_threshold']):
        return {'sourceid':sourceid, 'error':True}
    
    pp_time = tt.time()-pp_s
    
    # Timeseries data
    nterms_min, nterms_max = int(config['nterms_min']), int(config['nterms_max'])
    npoly = int(config['npoly'])
    times = np.array(lc_clean.HJD.values)
    mags = np.array(lc_clean.mag.values)
    errors = np.array(lc_clean.emag.values)
                 
    # Extract non-periodic statistics
    np_s = tt.time()
    nonper_feats = magarr_stats(mags)
    np_time = tt.time()-np_s
                 
    # Division into forced frequency grid input or not for lsq comp.
    if int(config['force_method']):
       # Division into irregular grid input for LSQ computation 
        if method_kwargs['irreg']:
            ls_s = tt.time()
            per_dict = lombscargle_stats(times, mags, errors, **ls_kwargs)
            # Top N LS frequency grid and half multiple
            freqs = np.array(per_dict['top_distinct_freqs'])
            freq_dict = dict(freq_grid=
                             np.concatenate((.5*freqs,freqs))) 
            ls_time = tt.time()-ls_s
            
            lsq_s = tt.time()
            per_feats = periodic_feats_force(times, mags, errors, freq_dict=freq_dict,
                                             nterms_min=nterms_min, nterms_max=nterms_max,
                                             npoly=npoly)
            lsq_time = tt.time()-lsq_s
        else:
            per_dict = lombscargle_stats(times, mags, errors, irreg=False, **ls_kwargs)
            
            period = per_dict['ls_period']
            freq_dict = dict(f0=1./(2*period), f1=1./period, Nf=2) # Forced frequency grid
            per_feats = periodic_feats_force(times, mags, errors, freq_dict=freq_dict,
                                             nterms_min=nterms_min, nterms_max=nterms_max,
                                             npoly=npoly)
            
        # compute false-alarm probability
        fap_ = fap(per_feats['lsq_power'],ls_kwargs['maximum_frequency'], times, mags, errors)
        if fap_ > 0.:
            per_feats['log10_fap'] = np.log10(fap_)
        else:
            per_feats['log10_fap'] = -323.
        
        lag_s = tt.time()
        lag_feats = find_lag(times, period=per_dict['ls_period'])
        lag_time = tt.time()-lag_s
        
        times = dict(pp_time=pp_time, np_time=np_time, ls_time=ls_time, lsq_time=lsq_time, lag_time=lag_time)
        features = {'sourceid':sourceid, **per_feats, **per_dict, **nonper_feats, **lag_feats, **times}
        
    else:
        per_feats = periodic_feats(times, mags, errors, nterms_min, nterms_max, npoly)
        
        # compute false-alarm probability
        fap_ = fap(per_feats['lsq_power'],ls_kwargs['maximum_frequency'], times, mags, errors)
        if fap_ > 0.:
            per_feats['log10_fap'] = np.log10(fap_)
        else:
            per_feats['log10_fap'] = -323.
        
        lag_feats = find_lag(times, period=per_feats['lsq_period'])
        
        features = {'sourceid':sourceid, 
                    **per_feats, **nonper_feats, **lag_feats}
    
    #print('%s feats loaded in %s s'%(sourceid, tt.time()-tot_s))
    
    return features
