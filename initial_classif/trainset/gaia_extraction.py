import numpy as np
import pandas as pd
from scipy import stats
from sqlutilpy import *
from interface_utils.add_stats import pct_diff

### Code for the automatic extraction of constant stellar sources, based on population statistics of a predetermined variability measure in Gaia.
### WSDB Gaia DR2 (gaia_dr2.gaia_source) table specifications are used.

def grab_virac_gaia_with_stats(l,b,sizel,sizeb,config):
    
    sizel /= 60.
    sizeb /= 60.
    poly_string = "t.l>%0.3f and t.l<%0.3f and t.b>%0.3f and t.b<%0.3f"\
                    %(l-.5*sizel,l+.5*sizel,b-.5*sizeb,b+.5*sizeb)

    if (l - .5 * sizel < 0.):
        poly_string = "(t.l>%0.3f or t.l<%0.3f) and t.b>%0.3f and t.b<%0.3f"\
                        %(l-.5*sizel+360.,l+.5*sizel,b-.5*sizeb,b+.5*sizeb)
    if (l + .5 * sizel > 360.):
        poly_string = "(t.l>%0.3f or t.l<%0.3f) and t.b>%0.3f and t.b<%0.3f"\
                        %(l-.5*sizel,l+.5*sizel-360.,b-.5*sizeb,b+.5*sizeb)
        
    data = pd.DataFrame(sqlutil.get("""
            select t.*, s.*, x.sep_arcsec,
            g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs
            from leigh_smith.virac2 as t
            left join leigh_smith.virac2_x_gdr2 as x on x.virac2_id=t.sourceid
            left join leigh_smith.virac2_var_indices_tmp as s on s.sourceid=t.sourceid
            left join gaia_dr2.gaia_source as g on g.source_id=x.gdr2_id
            where %s and duplicate=0 and astfit_params=5 and x.sep_arcsec<0.4 
            and ks_n_detections>%i and ks_ivw_mean_mag>%0.4f and ks_ivw_mean_mag<%0.4f"""%(
		poly_string,config['n_detection_threshold'],config['lower_k'],config['upper_k']), 
                                    config.wsdb_kwargs))
    
    data = pct_diff(data)
    
    data = data.sort_values(by='sep_arcsec').drop_duplicates(subset=['sourceid']).reset_index(drop=True)
    
    return data

def add_gvar_amp(df):
    """
    Compute Gaia variaibility amplitude from Belokurov 2017
    amp = log(sqr(Nobs)flux_error/flux) all in G
    
    """
    mean_flux_over_error = df["phot_g_mean_flux_over_error"].values
    n_obs = df["phot_g_n_obs"].values
    
    # Var computation
    g_amp = np.log10(np.sqrt(n_obs)/mean_flux_over_error)
    
    df['g_amp'] = pd.Series(g_amp, dtype='float64')
    
    return df


def running_stat(xvals,yvals,nbins=15,percentiles=[0.5,99.5],stat=np.nanstd, equal_counts=False, weights=None):
    '''
     Plots a running statistic between the 0.5th and 99.5th percentile -- bins equally spaced unless
     equal_counts=True, then bins contain ~same number of stars
    '''
    rangex = np.nanpercentile(xvals,[percentiles[0],percentiles[1]])
    bins = np.linspace(rangex[0],rangex[1],nbins)
    if equal_counts:
        bins = np.nanpercentile(xvals,np.linspace(percentiles[0],percentiles[1],nbins))
        
    bc = .5*(bins[1:]+bins[:-1])
    #sd,m,su = np.nan*np.ones(np.shape(bc)), np.nan*np.ones(np.shape(bc)), np.nan*np.ones(np.shape(bc))
    m = np.nan*np.ones(np.shape(bc))
    cnts = np.zeros_like(m)
    
    for II,(bd,bu) in enumerate(zip(bins[:-1],bins[1:])):
        if(len(yvals[(xvals>bd)&(xvals<bu)])==0):
            continue
        if weights is not None:
            cnts[II] = np.sum(weights[(xvals>bd)&(xvals<bu)])
            try:
                m[II]=stat(yvals[(xvals>bd)&(xvals<bu)],weights=weights[(xvals>bd)&(xvals<bu)])
            except:
                raise TypeError("Weights not possible for stat")
        else:
            cnts[II] = len(yvals[(xvals>bd)&(xvals<bu)])
            m[II]=stat(yvals[(xvals>bd)&(xvals<bu)])
            
    return bc, m, cnts


def gen_binned_df(df, pct=50., nbins=50, equal_counts=True):
    """
    Returns initial df with new columns showing percentile pct g_amp
    
    """
    
    # Add variability index
    df = add_gvar_amp(df)
    
    g_mag = np.array(df['phot_g_mean_mag'])
    g_amp = np.array(df['g_amp'])
    
    # Find binned statistics
    bin_centers, pct_cut, cnts = running_stat(g_mag, g_amp, nbins=nbins, 
                                        stat=lambda x: np.nanpercentile(x, pct),
                                        equal_counts=equal_counts)
    
    # Add statistic columns
    df['binpct_g_amp'] = np.interp(df['phot_g_mean_mag'], bin_centers, pct_cut)
    
    return df


def generate_gaia_training_set(l,b,sizel,sizeb,percentile,config):
    
    df = grab_virac_gaia_with_stats(l, b, sizel, sizeb, config)
    df = gen_binned_df(df, pct=percentile, nbins=len(df)//100, equal_counts=True)
    df = df[df['g_amp']<df['binpct_g_amp']].reset_index(drop=True)
    
    return df
    
    
    
