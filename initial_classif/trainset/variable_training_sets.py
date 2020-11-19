import numpy as np
import pandas as pd
from scipy import stats
import pickle
from sqlutilpy import *
from interface_utils.add_stats import main_string, var_string, pct_diff, phot_string, error_ratios

def load_all_variable_stars(config):
    
    test_string = ''
    if int(config['test']):
        config.wsdb_kwargs['preamb']+='select setseed(0.5);'
        test_string = 'order by random() limit 20000'
    
#     dsets = pd.DataFrame(
#             sqlutil.get("""select {0}, v.cat_period, v.var_class, {3}, 
#                             {1} from jason_sanders.variable_training_set_virac2 as v
#                             inner join leigh_smith.virac2 as t on t.sourceid=v.virac_id
#                             inner join leigh_smith.virac2_photstats as y on y.sourceid=v.virac_id
#                             inner join leigh_smith.virac2_var_indices as s on s.sourceid=v.virac_id
#                             {2};""".format(main_string,var_string, test_string, phot_string),
#                         **config.wsdb_kwargs))

#     ## Now filter
#     dsets = dsets[(dsets['ks_n_detections']>np.int64(config['n_detection_threshold']))&
#                   (dsets['ks_b_ivw_mean_mag']>np.float64(config['lower_k']))&
#                   (dsets['ks_b_ivw_mean_mag']<np.float64(config['upper_k']))].reset_index(drop=True)
    
#     This table already has been cut on magnitude and minimum epochs
    dsets = pd.DataFrame(
        sqlutil.get("""select * from jason_sanders.variable_training_set_virac2_stats {0};""".format(test_string),
                        **config.wsdb_kwargs))
    dsets = dsets[dsets['ks_b_ivw_mean_mag']>0.].reset_index(drop=True)
    
    dsets = pct_diff(dsets)
    dsets = error_ratios(dsets)
    
    return dsets
