import numpy as np
import pandas as pd
from scipy import stats
import pickle


def load_all_variable_stars(config):
    
    with open(config['variable_dir']+'var_trainset_virac2.pkl', 'rb') as f:
        dsets = pickle.load(f)
    
    dsets = cm_virac_stats_table(dsets, config)

    ## Now filter
    dsets = dsets[(dsets['ks_n_detections']>config['n_detection_threshold'])&
                  (dsets['ks_ivw_mean_mag']>config['lower_k'])&
                  (dsets['ks_ivw_mean_mag']<config['upper_k'])].reset_index(drop=True)
    
    return dsets
