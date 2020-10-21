import pandas as pd
import numpy as np

from compute_feats import comp_lc_feats

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


def extract_per_feats(lc, ls_kwargs, config):
    """
    Wrapper for periodic feature extraction based on list of light curves
    in panda format
    
    """
    
    p = Pool(config['var_cores'])
    features = p.map(partial(source_feat_extract, ls_kwargs, config),
          lc)
    p.close()
    p.join()
    
    #Finalise construction of feature dataframe - To be finished
    #Adds combined phase and amplitude features
    features = cosntruct_feats(pd.DataFrame.from_dict(features))
    
    #Add colour information - To be changed 
    colour_features = var_feature_extract.add_catalogue_info(cand_feats,
                                                              df_cm_name)
    
    
    return final_features
    
