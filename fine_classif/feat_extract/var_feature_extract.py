from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import astropy
import random

import time

from astropy.timeseries import BoxLeastSquares
from astropy.timeseries import LombScargle

from multiprocessing import Pool
from functools import partial

import lc_utils
import virac_lc
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


def periodic_feats(data, nterms=4, npoly=1, full=False):
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

    if full:
        return results
    
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


def periodic_feats_force(data, period, nterms=4, npoly=1, full=False):
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
    
    if full:
        return results
    
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


def source_feat_extract_id(source_id, ls_kwargs={}, force=True):
    """
    Wrapper to extract all periodic/non-periodic/colour features for a given source
    
    """
    # Offset multiple cores to not overload wsbd server
    lc = None
    while lc is None:
        try:
            lc = virac_lc.load_quality_virac_lc(source_id, amb_corr=False)
        except:
            pass
        time.sleep(1)

    # Control size of light curve data
    if lc['mjdobs'].size == 0 or lc['mag'].size == 0:
        return {'error':True}
    
    # Extract non-periodic and extra statistics 
    nonper_feats = magarr_stats(lc)
    
    # Exract light-curve summary periodic statistics from Fourier analysis
    if method == 'force':
        per_dict = lc_utils.lombscargle(lc, **ls_kwargs)
        per_feats = periodic_feats_force(lc, period=per_dict['ls_period'], nterms=4, npoly=1)
    
    elif method == 'standard':
        per_dict = lc_utils.lombscargle_period(lc, **ls_kwargs)
        per_feats = periodic_feats(lc, nterms=4, npoly=1)
          
    features = {**per_feats, **per_dict, **nonper_feats}
    
    return features


def source_feat_extract_lc(lightcurve, method='standard', ls_kwargs={}, amb_corr=False, ecl=False):
    """
    Wrapper to extract all periodic/non-periodic features for a given
    panda entry of the source light curve (from batch-wise loading).
    
    method - 'force' or 'standard', specifies the method of periodic feature computation
    ls_kwargs - arguments to specify LombScargle periodogram frequency grid in 'force' method
    
    returns:
    features - dict of features
    
    """
    
    # Pre-process light curve data with quality cut and 3 sigma conservative cut 
    lc = virac_lc.sigclipper(virac_lc.quality_cut_phot(lightcurve, amb_corr),
                             thresh=3.)
    
    # Control size of light curve data
    if lc['mjdobs'].size == 0 or lc['mag'].size == 0:
        return {'error':True}
    
    # Control for possible ordering error (as bug kept popping up for rel time series in Astropy)
    if all(np.diff(lc[lc_utils.find_time_field(lc)])>0):
        pass
    else:
        raise ValueError("Light curve time array not ordered chronologically.")
    
    # Extract non-periodic and extra statistics 
    nonper_feats = magarr_stats(lc)
    
    # Exract light-curve summary periodic statistics from Fourier analysis
    if method == 'force':
        per_dict = lc_utils.lombscargle(lc, ecl=ecl, **ls_kwargs)
        per_feats = periodic_feats_force(lc, period=per_dict['ls_period'], nterms=4, npoly=1)
    
    elif method == 'standard':
        per_dict = lc_utils.lombscargle_period(lc, ecl=ecl, **ls_kwargs)
        per_feats = periodic_feats(lc, nterms=4, npoly=1)
          
    features = {**per_feats, **per_dict, **nonper_feats}
    
    return features


def combine_pool_lcs(lightcurves, classname, ls_kwargs, method='standard', ncores=20, ecl=False):
    """
    Mulitprocessing split to extract summary statistics from list of light curve entries
    ----
    lightcurves - list of panda dataframes for each light curve
    classname - output name of stellar class
    method - 'force' or 'standard', specifies the method of periodic feature computation
    ls_kwargs - arguments to specify LombScargle periodogram frequency grid in 'force' method
    ncores - number of cores used in multi-processor split
    
    returns:
    df - panda dataframe (n_sources x n_feats) of source features
    
    """
    # Commence multi-processor split
    pool = Pool(ncores)
    features = pool.map(partial(source_feat_extract_lc, ls_kwargs=ls_kwargs,
                                method=method, ecl=ecl), lightcurves)

    pool.close()
    pool.join()
    
    df = pd.DataFrame.from_dict(features)
    
    df['sourceid'] = pd.Series([i.sourceid.values[0] for i in lightcurves])
    df['classname'] = pd.Series(len(df)*[classname])   
    
    return df


def combine_pool_ids(sourceids, classname, ls_kwargs, method='standard', ncores=20):
    """
    Mulitprocessing split to extract summary statistics from list of sourceids
    ----
    inputs:
    
    lightcurves - list of panda dataframes for each light curve
    classname - output name of stellar class
    method - 'force' or 'standard', specifies the method of periodic feature computation
    ls_kwargs - arguments to specify LombScargle periodogram frequency grid in 'force' method
    ncores - number of cores used in multi-processor split
    
    returns:
    
    df - panda dataframe (n_sources x n_feats) of source features
    
    """    
    # Commence multi-processor split
    pool = Pool(ncores)
    features = pool.map(partial(source_feat_extract_id, ls_kwargs=ls_kwargs,
                                method=method), lightcurves)
    pool.close()
    pool.join()
    
    df = pd.DataFrame.from_dict(features)
    
    df['sourceid'] = pd.Series(sourceids)
    df['classname'] = pd.Series(len(df)*[classname])
    
    return df


def calc_excess_colour(glon, glat, jk=False, hk=False):
    """
    Returns colour excess of J-K or H-K colour
    
    """
    if jk:
        extM = extinction_map.extinction_map_healpix(version='v2_NEW_JK_PUBLIC')
    elif hk:
        extM = extinction_map.extinction_map_healpix(version='v2_NEW_HK_PUBLIC')
        
    col_excess = extM.query(glon,glat)
    return col_excess


def add_catalogue_info(df_feats, cm_df_name, ogle=True, colour=True):
    """
    Add features extracted from crossmatch between OGLE and VVV catalogues to extracted features
    
    
    """
    df_cm = pd.read_pickle('{}.pkl'.format(cm_df_name))
    
    df_feats = df_feats.drop_duplicates(subset=['sourceid']).dropna(subset=['sourceid'])
    
    df_match = df_feats.merge(right=df_cm, how='inner', on='sourceid').dropna(subset=['sourceid'])
    
    if colour:
        #Adding extinction corrected colour information
        glon = df_match.l.values
        glat = df_match.b.values
        jmag = df_match.jmag.values
        hmag = df_match.hmag.values
        kmag = df_match.kmag.values
        
        jk_col_excess = calc_excess_colour(glon, glat, jk=True)
        hk_col_excess = calc_excess_colour(glon, glat, hk=True)
        
        df_match['JK_col'] = pd.Series(np.array((jmag - kmag) - jk_col_excess))
        df_match['HK_col'] = pd.Series(np.array((hmag - kmag) - hk_col_excess))
        
    return df_match


def find_princ(values):
    corr_phi = []
    for i in values:
        if i>0:
            corr = float(i)%(2*np.pi)
        elif i<0:
            corr = -float(abs(i))%(2*np.pi)
        else:
            corr = float(i)
        corr_phi.append(corr)
        
    return np.array(corr_phi)

def construct_final_df(df_feats):
    """
    Finalise feature dataframe + include phases and amplitude unbiased features
    
    """
    df_use = df_feats.copy()
    
    # Taking amplitude ratios and phases differences after finding principal phases
    corr_phi0 = find_princ(df_use['phi_0'])
    corr_phi1 = find_princ(df_use['phi_1'])
    corr_phi2 = find_princ(df_use['phi_2'])
    corr_phi3 = find_princ(df_use['phi_3'])
    # phase differences based on def phi_{ij} = phi_i - i phi_j
    df_use['phi0_phi1'] = pd.Series(corr_phi0-corr_phi1)
    df_use['phi0_phi2'] = pd.Series(corr_phi0-corr_phi2)
    df_use['phi0_phi3'] = pd.Series(corr_phi0-corr_phi3)
    df_use['phi1_phi2'] = pd.Series(corr_phi1-2*corr_phi2)
    df_use['phi1_phi3'] = pd.Series(corr_phi1-2*corr_phi3)
    df_use['phi2_phi3'] = pd.Series(corr_phi2-3*corr_phi3) 
    df_use['a0_a1'] = pd.Series(np.array(df_use['amp_0'])/np.array(df_use['amp_1']))
    df_use['a0_a2'] = pd.Series(np.array(df_use['amp_0'])/np.array(df_use['amp_2']))
    df_use['a0_a3'] = pd.Series(np.array(df_use['amp_0'])/np.array(df_use['amp_3']))
    df_use['a1_a2'] = pd.Series(np.array(df_use['amp_1'])/np.array(df_use['amp_2']))
    df_use['a1_a3'] = pd.Series(np.array(df_use['amp_1'])/np.array(df_use['amp_3']))
    df_use['a2_a3'] = pd.Series(np.array(df_use['amp_2'])/np.array(df_use['amp_3']))

    # Drop nan entries for relevant features to be used
    data_cols = ['a0_a1', 'a0_a2', 'a0_a3', 'a1_a2', 'a1_a3', 'a2_a3',
                 'phi0_phi1','phi0_phi2','phi0_phi3', 'phi1_phi2', 'phi1_phi3',
                 'phi2_phi3','sd','skew','kurtosis', 'amplitude', 'beyondfrac',
                 'med_magdev','delta_loglik','lsq_period','stetson_i',
                 'mags_q100mq0','mags_q99mq1','mags_q95mq5',
                 'mags_q90mq10','mags_q75mq25', 'JK_col', 'HK_col', 'ls_period',
                 'max_pow', 'pow_mean_disp', 'pow_med_disp',
                 'time_lag_mean','time_lag_median', 'max_time_lag']

    df_use = df_use.dropna(subset=data_cols)
    
    return df_use