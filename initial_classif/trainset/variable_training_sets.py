import numpy as np
import pandas as pd
from scipy import stats
import pickle
from interface_utils.add_stats import cm_virac_stats_table

def load_all_variable_stars(config, test=False):
    
    with open(config['variable_dir']+'var_trainset_virac2.pkl', 'rb') as f:
        dsets = pickle.load(f)
    
    dsets = dsets.sort_values(by='virac2_id').reset_index(drop=True)
    
    if test:
        dsets = dsets.sample(n=500).reset_index(drop=True)
    
    dsets = cm_virac_stats_table(dsets, config)

    ## Now filter
    dsets = dsets[(dsets['ks_n_detections']>np.int64(config['n_detection_threshold']))&
                  (dsets['ks_ivw_mean_mag']>np.float64(config['lower_k']))&
                  (dsets['ks_ivw_mean_mag']<np.float64(config['upper_k']))].reset_index(drop=True)
    
    return dsets
