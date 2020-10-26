import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial

from .compute_feats import source_feat_extract
from .extinction_map import extinction_map_healpix

def calc_excess_colour(glon, glat, config, jk=False, hk=False):
    """
    Returns colour excess of J-K or H-K colour from extinction map
    
    """
    #Path to extinction map
    pathtofile = str(config['path_extinction_map'])
    
    if jk:
        extM = extinction_map_healpix(pathtofile, version='v2_NEW_JK_PUBLIC')
    elif hk:
        extM = extinction_map_healpix(pathtofile, version='v2_NEW_HK_PUBLIC')
        
    col_excess = extM.query(glon,glat)
    
    return col_excess


def add_colour_info(df, config):
    """
    Add reddening free J-K and H-K colour features 
    
    """
    
    glon = df.l.values
    glat = df.b.values
    ## Need to check columns names for filter phots
#     jmag = df.j_ivw_mean_mag .values
#     hmag = df.h_ivw_mean_mag.values
#     kmag = df.ks_ivw_mean_mag.values
    jmag = df.j_b_ivw_mean_mag .values
    hmag = df.h_b_ivw_mean_mag.values
    kmag = df.ks_b_ivw_mean_mag.values
        
    jk_col_excess = calc_excess_colour(glon, glat, config, jk=True)
    hk_col_excess = calc_excess_colour(glon, glat, config, hk=True)
        
    df['JK_col'] = pd.Series(np.array((jmag - kmag) - jk_col_excess))
    df['HK_col'] = pd.Series(np.array((hmag - kmag) - hk_col_excess))


def find_princ(angles):
    """
    Find principal angles for phases in 0<=theta<=2*pi range
    
    """
    corr_angs = np.remainder(np.array(angles), 2*np.pi)
        
    return corr_angs


def construct_final_df(df_use):
    """
    Finalise feature dataframe + include phases and amplitude unbiased features
    
    """
    
    # Taking amplitude ratios and phases differences after finding principal phases
    phi0 = find_princ(df_use['phi_0'].values)
    phi1 = find_princ(df_use['phi_1'].values)
    phi2 = find_princ(df_use['phi_2'].values)
    phi3 = find_princ(df_use['phi_3'].values)
    
    # phase differences based on def phi_{ij} = phi_i - i phi_j
    df_use['phi0_phi1'] = phi0-phi1
    df_use['phi0_phi2'] = phi0-phi2
    df_use['phi0_phi3'] = phi0-phi3
    df_use['phi1_phi2'] = phi1-2*phi2
    df_use['phi1_phi3'] = phi1-2*phi3
    df_use['phi2_phi3'] = phi2-3*phi3
    df_use['a0_a1'] = np.array(df_use['amp_0'])/np.array(df_use['amp_1'])
    df_use['a0_a2'] = np.array(df_use['amp_0'])/np.array(df_use['amp_2'])
    df_use['a0_a3'] = np.array(df_use['amp_0'])/np.array(df_use['amp_3'])
    df_use['a1_a2'] = np.array(df_use['amp_1'])/np.array(df_use['amp_2'])
    df_use['a1_a3'] = np.array(df_use['amp_1'])/np.array(df_use['amp_3'])
    df_use['a2_a3'] = np.array(df_use['amp_2'])/np.array(df_use['amp_3'])


def finalise_feats(features_df, input_df, config):
    """
    Combine periodic features with catalogue features
    Add colour information from extinction map
    --- can be done locally for variable sources but will
    have to be done remotely when classifying tiles on-the-fly 
    
    """
    
    print("merging feature dfs..")
    # Merge with input stats df
    df_match = features_df.merge(right=input_df, how='inner', on='sourceid').dropna(subset=['sourceid'])
    print("merged --- loading colour info..")
    # Add reddening free colour info
    add_colour_info(df_match, config)
    print("loaded colour info --- finalising..")
    # Finalise features
    construct_final_df(df_match)
    
    return df_match
    

def extract_per_feats(lc_dfs, input_df, ls_kwargs, config):
    """
    Wrapper for periodic feature extraction based on list of light curves
    in panda format
    
    """
    
    print("computing features..")
    p = Pool(int(config['var_cores']))
    features = p.map(partial(source_feat_extract, ls_kwargs=ls_kwargs, config=config),
          lc_dfs)
    p.close()
    p.join()
    print("computed features.")
    
    feature_df = pd.DataFrame.from_dict(features)
    
    final_feature_df = finalise_feats(feature_df, input_df, config)
    print("finalised.")
    
    return final_feature_df
    
