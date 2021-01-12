import numpy as np
import pandas as pd
from scipy import stats
from sqlutilpy import *
from interface_utils.add_stats import pct_diff, main_string, var_string, phot_string, error_ratios

### Code for the automatic extraction of constant stellar sources, based on population statistics of a predetermined variability measure in Gaia.
### WSDB Gaia DR2 (gaia_*.gaia_source) table specifications are used.

gaia_version = 'gaia_edr3' #'gaia_dr2'

def grab_virac_gaia_random_sample(random_index_max,config):
    
    max_cross_match_distance = 2.
    
    if int(config['test']):
        test_string = "healpix_ang2ipix_ring(512,t.ra,t.dec)=2318830 and t.l<0.897411 and t.l>0.677411 and t.b<0.064603 and t.b>-0.164603"
        
        data = pd.DataFrame(sqlutil.get("""

                with t as (select {0} from leigh_smith.virac2 as t where 
                            duplicate=0 and astfit_params=5 
                            and ks_n_detections>{2} and ks_ivw_mean_mag>{3} and ks_ivw_mean_mag<{4} and {5}),
                     g as (select t.ra, t.dec, t.ref_epoch, t.pmra, t.pmdec, t.phot_g_mean_flux_over_error,
                                    t.phot_g_mean_mag, t.phot_g_n_obs from {7}.gaia_source as t where {5})
                                    
                select t.*, {1}, {6}, x.sep_arcsec,
                x.phot_g_mean_flux_over_error, x.phot_g_mean_mag, x.phot_g_n_obs from t
                inner join leigh_smith.virac2_photstats as y on t.sourceid=y.sourceid
                inner join leigh_smith.virac2_var_indices as s on s.sourceid=t.sourceid
                
                left join lateral (
                    select g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs, 
                            q3c_dist_pm(t.ra, t.dec, g.pmra, g.pmdec, 1, 2014., g.ra, g.dec, g.ref_epoch)*3600. as sep_arcsec 
                    from g 
                    where q3c_join(t.ra, t.dec, g.ra, g.dec, {8}/3600.)
                    order by q3c_dist_pm(t.ra, t.dec, g.pmra, g.pmdec, 1, 2014., g.ra, g.dec, g.ref_epoch) asc limit 1) 
                    as x on true
                where x.sep_arcsec<0.4 and x.phot_g_mean_mag<30;""".format(
                main_string,
                var_string,
                np.int64(config['n_detection_threshold']),
                np.float64(config['lower_k']),
                np.float64(config['upper_k']),
                test_string, 
                phot_string,
                gaia_version, 
                max_cross_match_distance), 
            **config.wsdb_kwargs))
    else:
        test_string = "random_index<%i" % random_index_max
        
        data = pd.DataFrame(sqlutil.get("""
                with g as (
                        with p as (
                        with q as (
                                    select source_id, phot_g_mean_flux_over_error, phot_g_mean_mag, phot_g_n_obs, 
                                    ra, dec, ref_epoch, pmra, pmdec, l, b from {7}.gaia_source where {5}
                                )
                            select * from q where phot_g_mean_mag<30 and b<6. and b>-11 and (l<12. or l>290.)
                        )
                        select p.source_id as gaia_sourceid, phot_g_mean_flux_over_error, phot_g_mean_mag, phot_g_n_obs, sep_arcsec, virac2_id from p
                        left join lateral
                            (select t.sourceid as virac2_id, duplicate, astfit_params, ks_n_detections, ks_ivw_mean_mag,
                            q3c_dist_pm(t.ra, t.dec, p.pmra, p.pmdec, 1, 2014., p.ra, p.dec, p.ref_epoch)*3600. as sep_arcsec 
                                from leigh_smith.virac2 as t
                                where q3c_join(p.ra, p.dec, t.ra, t.dec, {8}/3600.)
                                order by q3c_dist_pm(p.ra, p.dec, p.pmra, p.pmdec, 1, p.ref_epoch, t.ra, t.dec, 2014.) asc limit 1) 
                                as x on true where x.sep_arcsec<0.4
                                and duplicate=0 and astfit_params=5 
                            and ks_n_detections>{2} and ks_ivw_mean_mag>{3} and ks_ivw_mean_mag<{4}
                    )
                select t.*, {1}, {6},  
                g.gaia_sourceid, g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs, g.sep_arcsec from g
                inner join leigh_smith.virac2 as t on t.sourceid=g.virac2_id
                inner join leigh_smith.virac2_photstats as y on y.sourceid=g.virac2_id
                inner join leigh_smith.virac2_var_indices as s on s.sourceid=g.virac2_id;""".format(
                main_string,
                var_string,
                np.int64(config['n_detection_threshold']),
                np.float64(config['lower_k']),
                np.float64(config['upper_k']),
                test_string,
                phot_string,
                gaia_version,
                max_cross_match_distance), 
            **config.wsdb_kwargs))
    
    data = data[data['ks_b_ivw_mean_mag']>0.].reset_index(drop=True)
    
    data = pct_diff(data)
    data = error_ratios(data)
    
    data = data.sort_values(by='sep_arcsec').drop_duplicates(subset=['sourceid']).reset_index(drop=True)
    
    return data


def grab_virac_gaia_region_with_stats(l,b,sizel,sizeb,config):
    
    l = l + 360.*(l<0.)
    
    max_cross_match_distance=1.
    
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
        
    virac2_gedr3_xmatch = """
        with g as (
                    select ra, dec,pmra,pmdec, source_id as gaia_id 
                            from gaia_edr3.gaia_source as t
                            where 
                            ((t.l>350 or t.l<10.8) and t.b>-10.3 and t.b<5.2)
                            or (t.l<350. and t.l>360-65.6428571434 and t.b>-1.1057142857*2 and t.b<1.1057142857*2)
                    )
                select sourceid, gaia_id, t.sep_arcsec from g
                left join lateral (
                                select sourceid, 
                                q3c_dist_pm(g.ra, g.dec, g.pmra, g.pmdec, 1, 2016., t.ra, t.dec, 2014.)*3600. as sep_arcsec
                                from leigh_smith.virac2 as t
                                where q3c_join(g.ra, g.dec, t.ra, t.dec, 2.0/3600.)
                                order by q3c_dist_pm(g.ra, g.dec, g.pmra, g.pmdec, 1, 2016., t.ra, t.dec, 2014.) asc limit 1
                                ) 
                as t on true where sourceid>0 and sep_arcsec<0.4;
    """
    full_table = """
                select x.*, g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs,
                          t.ra,t.dec,t.l,t.b,t.ks_n_detections,
                          s.ks_stdev,s.ks_mad,s.ks_kurtosis,s.ks_skew,s.ks_eta,s.ks_stetson_i,s.ks_stetson_j,
                          s.ks_stetson_k,s.ks_p100,s.ks_p0,s.ks_p99,s.ks_p1,s.ks_p95,s.ks_p5,s.ks_p84,s.ks_p16,
                          s.ks_p75,s.ks_p25, y.ks_b_ivw_err_mag,y.j_b_ivw_mean_mag,y.h_b_ivw_mean_mag,
                          y.ks_b_ivw_mean_mag,y.ks_n_b_phot
                from jason_sanders.virac2_gedr3_xmatch as x
                inner join gaia_edr3.gaia_source as g on g.source_id=x.gaia_id
                inner join leigh_smith.virac2 as t on t.sourceid=x.sourceid
                inner join leigh_smith.virac2_photstats as y on y.sourceid=x.sourceid
                inner join leigh_smith.virac2_var_indices as s on s.sourceid=x.sourceid
                
                where phot_g_mean_mag<30 and t.duplicate=0 and t.astfit_params=5 and t.ks_n_detections>20
                and t.ks_ivw_mean_mag>11.5 and t.ks_ivw_mean_mag<17.0
    """
#     print("""
#             with q as (
#             with g as (
#                 with g as (
#                     select phot_g_mean_flux_over_error, phot_g_mean_mag, phot_g_n_obs, source_id as gaia_id 
#                             from gaia_edr3.gaia_source as t
#                             where {5}
#                     )
#                 select g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs, g.gaia_id, x.sourceid, x.sep_arcsec from g
#                 inner join jason_sanders.virac2_gedr3_xmatch as x on g.gaia_id=x.gaia_id
#                 where phot_g_mean_mag<30 and x.sep_arcsec<0.4 
#                 )
#                 select g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs, g.gaia_id, g.sep_arcsec, {0} from g
#                 inner join leigh_smith.virac2 as t on g.sourceid=t.sourceid
#                 and t.duplicate=0 and t.astfit_params=5 and t.ks_n_detections>{2}
#                 and ks_ivw_mean_mag>{3} and ks_ivw_mean_mag<{4}
#                 )
#             select q.*, {1}, {6} from q
#                 inner join leigh_smith.virac2_photstats as y on y.sourceid=q.sourceid
#                 inner join leigh_smith.virac2_var_indices as s on s.sourceid=q.sourceid
#                 ;""".format(
#             main_string,
#             var_string,
#             np.int64(config['n_detection_threshold']),
#             np.float64(config['lower_k']),
#             np.float64(config['upper_k']),
#             poly_string,
#             phot_string,
#             gaia_version,
#             max_cross_match_distance))
#     data = pd.DataFrame(sqlutil.get("""
#             with q as (
#             with g as (
#                 with g as (
#                     select phot_g_mean_flux_over_error, phot_g_mean_mag, phot_g_n_obs, source_id as gaia_id 
#                             from gaia_edr3.gaia_source as t
#                             where {5}
#                     )
#                 select g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs, g.gaia_id, x.sourceid, x.sep_arcsec from g
#                 inner join jason_sanders.virac2_gedr3_xmatch as x on g.gaia_id=x.gaia_id
#                 where phot_g_mean_mag<30 and x.sep_arcsec<0.4 
#                 )
#                 select g.phot_g_mean_flux_over_error, g.phot_g_mean_mag, g.phot_g_n_obs, g.gaia_id, g.sep_arcsec, {0} from g
#                 inner join leigh_smith.virac2 as t on g.sourceid=t.sourceid
#                 and t.duplicate=0 and t.astfit_params=5 and t.ks_n_detections>{2}
#                 and ks_ivw_mean_mag>{3} and ks_ivw_mean_mag<{4}
#                 )
#             select q.*, {1}, {6} from q
#                 inner join leigh_smith.virac2_photstats as y on y.sourceid=q.sourceid
#                 inner join leigh_smith.virac2_var_indices as s on s.sourceid=q.sourceid
#                 ;""".format(
#             main_string,
#             var_string,
#             np.int64(config['n_detection_threshold']),
#             np.float64(config['lower_k']),
#             np.float64(config['upper_k']),
#             poly_string,
#             phot_string,
#             gaia_version,
#             max_cross_match_distance), 
#         **config.wsdb_kwargs))
    
    data = pd.DataFrame(sqlutil.get('''
        select * from jason_sanders.virac2_gedr3_xmatch_with_stats as t where {0}
    '''.format(poly_string), **config.wsdb_kwargs))

    data = data[data['ks_b_ivw_mean_mag']>0.].reset_index(drop=True)
    data = data.sort_values(by='sep_arcsec').reset_index(drop=True)
    data = data[~data['sourceid'].duplicated()].reset_index(drop=True)
    
    data = pct_diff(data)
    data = error_ratios(data)
    
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


def downsample_training_set(data, number):
    weight_b, bins = np.histogram(data['ks_b_ivw_mean_mag'].values, 
                                 range=np.nanpercentile(data['ks_b_ivw_mean_mag'].values,[1.,99.]))
    bc = .5*(bins[1:]+bins[:-1])
    weights = np.interp(data['ks_b_ivw_mean_mag'].values, bc, weight_b)
    weights[weights<1.]=1.
    return data.iloc[np.random.choice(np.arange(len(data)), number, replace=False,
                                      p=1./weights/np.nansum(1./weights))].reset_index(drop=True)


def generate_gaia_training_set(l,b,sizel,sizeb,percentile,size,config):
    
    df = grab_virac_gaia_region_with_stats(l, b, sizel, sizeb, config)
    df = gen_binned_df(df, pct=percentile, nbins=len(df)//100, equal_counts=True)
    df = df[df['g_amp']<df['binpct_g_amp']].reset_index(drop=True)
    
    if size<len(df):
        df = downsample_training_set(df, size)

    return df
    
    
    
def generate_gaia_training_set_random(size, config, percentile, random_index = 600000):    
    
    df = grab_virac_gaia_random_sample(random_index, config)
    df = gen_binned_df(df, pct=percentile, nbins=len(df)//100, equal_counts=True)
    df = df[df['g_amp']<df['binpct_g_amp']].reset_index(drop=True)
    
    if size<len(df):
        df = downsample_training_set(df, size)

    return df
