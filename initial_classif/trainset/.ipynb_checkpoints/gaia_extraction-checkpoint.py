import numpy as np
import pandas as pd
from scipy import stats

from sqlutilpy import *

from virac_classifier.wsdb_utils.wsdb_cred import wsdb_kwargs

### Code for the automatic extraction of constant stellar sources, based on population statistics of a predetermined variability measure in Gaia.
### WSDB Gaia DR2 (gaia_dr2.gaia_source) table specifications are used.


def cm_virac_to_gaia(data, **wsdb_kwargs):
    """
    Crossmatch of tile sources to Gaia
    ---
    input: data = (RAJ2000, DECJ2000, sourceid)
    
    return: Gaia DR2 photometry dataframe
    
    """
    ra, dec, star = data['ra'], data['dec'], data['sourceid']
    
    decaps = pd.DataFrame(sqlutil.local_join("""
                select * from mytable as m
                left join lateral (select *, q3c_dist(m.ra_virac, m.dec_virac, s.ra, s.dec)
                from gaia_dr2.gaia_source as s 
                where q3c_join(m.ra_virac, m.dec_virac, s.ra, s.dec,{0}/3600)
                order by q3c_dist(m.ra_virac, m.dec_virac, s.ra, s.dec)
                asc limit 1)
                as tt on  true  order by xid """.format(cm_radius),
                'mytable',
                (ra,dec,star,np.arange(len(dec))),('ra_virac','dec_virac','virac_id',
                                                   'xid'),**wsdb_kwargs))
    
    
    gaia_cols = ['ra_virac','dec_virac','virac_id','source_id','ra', 'dec',
                 'l', 'b', 'phot_g_n_obs','phot_g_mean_flux', 'phot_g_mean_flux_error',
                 'phot_g_mean_mag']
    
    # Remove unsuccessful matches
    decaps = decaps[gaia_cols].copy()
    decaps = decaps.drop(decaps[decaps['virac_id']<0].index)
    
    return decaps
    

def binned_stats(x, values, nbins):
    """
    Calculate binned statistics 
    """
    meds, bin_edges, bin_number = stats.binned_statistic(x=x, values=values,
                                                         statistic='median', bins=nbins)
    med_sds, bin_edges_, bin_number_ = stats.binned_statistic(x=x, values=values,
                                                              statistic='std', bins=nbins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    
    return meds, med_sds, bin_centers, bin_edges


def add_gvar_amp(df):
    """
    Compute Gaia variaibility amplitude from Belokurov 2017
    amp = log(sqr(Nobs)flux_error/flux) all in G
    
    """
    mean_flux = np.array(df["phot_g_mean_flux"])
    mean_flux_error = np.array(df["phot_g_mean_flux_error"])
    n_obs = np.array(df["phot_g_n_obs"])
    
    # Var computation
    g_amp = np.log10(np.sqrt(n_obs)*(mean_flux_error/mean_flux))
    
    df['g_amp'] = pd.Series(amp, dtype='float64')
    
    return df


def gen_binned_df(df, nbins=50):
    """
    Returns initial df with new columns showing Median and S.D. g_amp of
    the bin in which the source is found, as well as the G magnitude bin center. 
    
    """
    
    # Add variability index
    df = add_gvar_amp(df)
    
    g_mag = np.array(df['phot_g_mean_mag'])
    g_amp = np.array(df['g_amp'])
    
    # Find binned statistics
    meds, med_sds, bin_centers, bin_edges = binned_stats(x=mag, values=g_amp, nbins=nbins)
    
    # Add statistic columns
    df['binmed_g_amp'] = pd.cut(g_mag, bins=bin_edges, labels=meds, include_lowest=True)
    df['binsd_g_amp'] = pd.cut(g_mag, bins=bin_edges, labels=meds, include_lowest=True)
    df['bincent_mag'] = pd.cut(g_mag, bins=bin_edges, labels=meds, include_lowest=True)
    
    return df

