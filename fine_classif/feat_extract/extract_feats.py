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
        extM = extinction_map_healpix(pathtofile, version='v2_NEW_JK_PHOTSTATS_FULL')
    elif hk:
        extM = extinction_map_healpix(pathtofile, version='v2_NEW_HK_PHOTSTATS_FULL')
        
    col_excess = extM.query(glon,glat)
    
    return col_excess


def add_colour_info(df, config):
    """
    Add reddening free J-K and H-K colour features 
    
    """
    
    glon = df.l.to_numpy()
    glat = df.b.to_numpy()
    ## Need to check columns names for filter phots
    jmag = df.j_b_ivw_mean_mag.to_numpy()
    hmag = df.h_b_ivw_mean_mag.to_numpy()
    kmag = df.ks_b_ivw_mean_mag.to_numpy()
        
    jk_col_excess = calc_excess_colour(glon, glat, config, jk=True)
    hk_col_excess = calc_excess_colour(glon, glat, config, hk=True)
        
    df['JK_col'] = pd.Series(np.array((jmag - kmag) - jk_col_excess))
    df['HK_col'] = pd.Series(np.array((hmag - kmag) - hk_col_excess))


def find_princ(angles):
    """
    Find principal angles for phases in 0<=theta<=2*pi range
    
    """

    corr_angs = np.copy(angles)
 
    corr_angs[np.isfinite(angles)] = np.remainder(angles[np.isfinite(angles)], 2*np.pi)
        
    return corr_angs


def construct_final_df(df_use):
    """
    Finalise feature dataframe + include phases and amplitude unbiased features
    
    """
    # phase differences based on def phi_{ij} = phi_i - i phi_j
    pairs = [[1,0],[2,0],[3,0],[2,1],[3,1],[3,2]]
    for i, j in pairs:
        df_use['phi%i_phi%i'%(i,j)] = find_princ((j+1)*df_use['phi_%i'%i].to_numpy()
                                                 -(i+1)*df_use['phi_%i'%j].to_numpy())
        df_use['phi%i_phi%i_double'%(i,j)] = find_princ((j+1)*df_use['phi_double_%i'%i].to_numpy()
                                                        -(i+1)*df_use['phi_double_%i'%j].to_numpy())
        
        df_use['a%i_a%i'%(j,i)] = df_use['amp_%i'%j].to_numpy()/df_use['amp_%i'%i].to_numpy()
        df_use['a%i_a%i_double'%(j,i)] = df_use['amp_double_%i'%j].to_numpy()/df_use['amp_double_%i'%i].to_numpy()

def finalise_feats(features_df, input_df, config):
    """
    Combine periodic features with catalogue features
    Add colour information from extinction map
    --- can be done locally for variable sources but will
    have to be done remotely when classifying tiles on-the-fly 
    
    """
    
    # Merge with input stats df
    df_match = features_df.merge(right=input_df, how='inner', on='sourceid', suffixes=('', 'old_trainset')).dropna(subset=['sourceid'])
    
    # Add reddening free colour info
    add_colour_info(df_match, config)
    
    # Finalise features
    if 'phi_0' in df_match.columns:
        construct_final_df(df_match)
    
    return df_match
    

def extract_per_feats(lc_dfs, input_df, ls_kwargs, method_kwargs,
                      config, serial=True):
    """
    Wrapper for periodic feature extraction based on list of light curves
    in panda format
    
    """
    
    if serial:
        features = [source_feat_extract(data, ls_kwargs=ls_kwargs,
                    method_kwargs=method_kwargs, config=config) 
                    for data in zip(input_df['ra'].to_numpy(), input_df['dec'].to_numpy(), lc_dfs)]
    else:
        with Pool(int(config['var_cores'])) as p:
            features = p.map(partial(source_feat_extract, ls_kwargs=ls_kwargs,
                                     method_kwargs=method_kwargs, config=config), 
                                 zip(input_df['ra'].to_numpy(), input_df['dec'].to_numpy(), lc_dfs))
    
    feature_df = pd.DataFrame.from_dict(features)

    final_feature_df = finalise_feats(feature_df, input_df, config)
    
    return final_feature_df
