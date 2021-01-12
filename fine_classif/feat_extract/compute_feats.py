from scipy import stats
import numpy as np
import pandas as pd
import math

from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord

from multiprocessing import Pool
from functools import partial

from .lc_utils import *

from astropy.utils.iers import IERS_A_URL, IERS_B_URL
from astropy.utils.data import download_file
download_file(IERS_A_URL, cache=True)
download_file(IERS_B_URL, cache=True)

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

def compute_peak_properties(times, mags, errors, results):
    
    min_phase = find_phase_of_minimum(results)
    
    second_minimum, min_phase_2 = check_significant_second_minimum(results, return_min_location=True)
    
    peak_ratio_model = find_peak_ratio_model(results, min_phase, second_minimum, min_phase_2)
    peak_ratio_data = find_peak_ratio_data(times, mags, errors, results, min_phase, second_minimum, min_phase_2)
    
    # If no significant second minimum detected we also compute coefficients for the Fourier series at double the period
    if ~second_minimum:
        results_d = fourier_poly_chi2_fit_full(
                                         times=times,
                                         mag=mags,
                                         err=errors,
                                         f0=.5/results['lsq_period'],
                                         f1=.50001/results['lsq_period'],
                                         Nf=2,
                                         nterms=results['lsq_nterms'],
                                         npoly=results['lsq_npoly'],
                                         regularization=results['lsq_regularization'])
        amp_double, phase_double = results_d['amplitudes'], results_d['phases']
    else:
        amp_double, phase_double = results['amplitudes'], results['phases']
        
    return {'peak_ratio_model':peak_ratio_model, 'peak_ratio_data':peak_ratio_model, 
            'significant_second_minimum':second_minimum, 
            'amp_double_0':amp_double[0], 'amp_double_1':amp_double[1], 
            'amp_double_2':amp_double[2],'amp_double_3':amps[3], 
            'phi_double_0':phase_double[0], 'phi_double_1':phase_double[1], 
            'phi_double_2':phase_double[2], 'phi_double_3':phase_double[3]}
        

def periodic_feats(times, mags, errors, nterms=4, nterms_max=10, npoly=1, max_freq=20.):
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
    
    #print("LSQ method: max_freq={}, min_freq={} -- with Nf={}".format(max_freq, f0, Nf))             
    
    results = fourier_poly_chi2_fit_full_nterms_iterations(
                                         times=times,
                                         mag=mags,
                                         err=errors,
                                         f0=f0,
                                         f1=max_freq,
                                         Nf=Nf,
                                         nterms_min=nterms,
                                         nterms_max=nterms_max,
                                         npoly=npoly,
                                         regularization=0.1,
                                         keep_small=True, 
                                         code_switch=True,
                                         use_power_of_2=True, force=False)
    
    # Calculate delta log likelihood between periodic and constant Gaussian scatter models
    pred_mean = retrieve_fourier_poly(times=times, results=results)
    delta_loglik = get_delta_log_likelihood(mags, errors, pred_mean=pred_mean)
    
    # Extract relevant Fourier terms
    amps = np.array(results['amplitudes'])
    phases = np.array(results['phases'])
    per = float(results['lsq_period'])
    
    # Dispersion of min chi2
    disp_min_chi2 = results['lsq_chi2_min_disp']
    
    peak_output = compute_peak_properties(times, mags, errors, results)
    
    return {'lsq_period':per, 'amp_0':amps[0], 'amp_1':amps[1], 'amp_2':amps[2],
            'amp_3':amps[3], 'phi_0':phases[0], 'phi_1':phases[1], 'phi_2':phases[2],
            'phi_3':phases[3], 'delta_loglik':delta_loglik, 'disp_min_chi2':disp_min_chi2,
            **peak_output}


def periodic_feats_force(times, mags, errors, freq_dict,
                         nterms=4, nterms_max=10, npoly=1,method_kwargs={}):
    """
    Returns periodic features (fourier components) of photometric lightcurves
    with forced frequency input grid (determined from LombScargle).
    
    """
    results = fourier_poly_chi2_fit_nterms_iterations(
                                         times=times,
                                         mag=mags,
                                         err=errors,
                                         freq_dict=freq_dict,
                                         nterms_min=nterms,
                                         nterms_max=nterms_max,
                                         npoly=npoly,
                                         regularization=0.1,
                                         keep_small=True, 
                                         code_switch=True,
                                         use_power_of_2=True,
                                         force=True,
                                         **method_kwargs)
        
    # Calculate delta log likelihood between periodic and constant Gaussian scatter models
    pred_mean = retrieve_fourier_poly(times=times, results=results)
    delta_loglik = get_delta_log_likelihood(mags, errors, pred_mean=pred_mean)
    
    # Extract relevant Fourier terms
    amps = np.array(results['amplitudes'])
    phases = np.array(results['phases'])
    per = float(results['lsq_period'])
    per_error = results['lsq_period_error']
    
    peak_output = compute_peak_properties(times, mags, errors, results)
    
    return {'lsq_period':per, 'lsq_period_error':per_error, 'amp_0':amps[0], 'amp_1':amps[1], 'amp_2':amps[2],
            'amp_3':amps[3], 'phi_0':phases[0], 'phi_1':phases[1], 'phi_2':phases[2],
            'phi_3':phases[3], 'delta_loglik':delta_loglik,
            **peak_output}

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
    median = np.nanmedian(times_fld_diff)
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
    
    # Timeseries data
    nterms, npoly = int(config['nterms']), int(config['npoly'])
    times = np.array(lc_clean.HJD.values)
    mags = np.array(lc_clean.mag.values)
    errors = np.array(lc_clean.emag.values)
                 
    # Extract non-periodic statistics 
    nonper_feats = magarr_stats(mags)    
                 
    # Division into forced frequency grid input or not for lsq comp.
    if int(config['force_method']):
       # Division into irregular grid input or not for lsq comp.
        if method_kwargs['irreg']:
            per_dict = lombscargle_stats(times, mags, errors, **ls_kwargs)
            # Top N LS frequency grid and half multiple
            freq_dict = dict(freq_grid=
                             np.concatenate([.5*per_dict['top_distinct_freqs'],
                                             per_dict['top_distinct_freqs']])) 
            per_feats = periodic_feats_force(times, mags, errors, freq_dict=freq_dict,
                                             nterms=nterms,npoly=npoly, method_kwargs=method_kwargs)
        else:
            per_dict = lombscargle_stats(times, mags, errors, irreg=False, **ls_kwargs)
            
            period = per_dict['ls_period']
            freq_dict = dict(f0=1./(2*period), f1=1./period, Nf=2) # Forced frequency grid
            per_feats = periodic_feats_force(times, mags, errors, freq_dict=freq_dict,
                                             nterms=nterms,npoly=npoly, method_kwargs=method_kwargs)
        
        lag_feats = find_lag(times, period=per_dict['ls_period'])
        features = {'sourceid':sourceid, **per_feats, **per_dict, **nonper_feats, **lag_feats}
        
    else:
        per_feats = periodic_feats(times, mags, errors, nterms, npoly)
        lag_feats = find_lag(times, period=per_feats['lsq_period'])
        
        features = {'sourceid':sourceid, 
                    **per_feats, **nonper_feats, **lag_feats}

    return features
