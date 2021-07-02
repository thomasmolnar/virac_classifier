import numpy as np
import pandas as pd
from scipy import stats
import pickle
from sqlutilpy import *
from interface_utils.add_stats import main_string, var_string, preprocess_data, phot_string, error_ratios

def generate_table_query(config):

    table = """create table variable_training_set_virac2_stats as (
	     select {0}, v.cat_period, v.var_class, {3}, 
	     {1} from jason_sanders.variable_training_set_virac2 as v
	     inner join leigh_smith.virac2 as t on t.sourceid=v.virac_id
	     inner join leigh_smith.virac2_photstats as y on y.sourceid=v.virac_id
	     inner join leigh_smith.virac2_var_indices as s on s.sourceid=v.virac_id
	     );""".format(main_string,var_string, test_string, phot_string)
   
    return table

def load_all_variable_stars(config):
    
    test_string = ''
    if int(config['test']):
        config.wsdb_kwargs['preamb']+='select setseed(0.5);'
        test_string = 'order by random() limit 20000'
    
    dsets = pd.DataFrame(
        sqlutil.get("""select * from jason_sanders.variable_training_set_virac2_stats 
                       where ks_n_detections>{1} and ks_b_ivw_mean_mag>{2} and ks_b_ivw_mean_mag<{3} 
                       {0};""".format(test_string, np.int64(config['n_detection_threshold']),
                                      np.float64(config['lower_k']), np.float64(config['upper_k'])),
                        **config.wsdb_kwargs))
    
    dsets = preprocess_data(dsets)

    # Remove BCEP
    dsets = dsets[~(dsets['sourceid']==9855850009752)].reset_index(drop=True)

    return dsets


def load_mira_sample(config, serial=False):
    
    variable_output_dir = str(config['variable_output_dir'])
    
    mira_table = pd.read_csv('mira_sample.csv')
    
    mira_ids = ','.join(np.str(s) for s in mira_table['virac_id'].values)
    
    dsets = pd.DataFrame(
            sqlutil.get("""with t as (
                            select {0} from 
                            leigh_smith.virac2 as t where t.sourceid in ({3}))
                            select {0}, {2}, {1} from t
                            inner join leigh_smith.virac2_photstats as y on y.sourceid=t.sourceid
                            inner join leigh_smith.virac2_var_indices as s on s.sourceid=t.sourceid;
                            """.format(main_string, var_string, phot_string, mira_ids),
                        **config.wsdb_kwargs))

    ## Now filter
    dsets = dsets[(dsets['ks_n_detections']>np.int64(config['n_detection_threshold']))&
                  (dsets['ks_b_ivw_mean_mag']>np.float64(config['lower_k']))&
                  (dsets['ks_b_ivw_mean_mag']<np.float64(config['upper_k']))].reset_index(drop=True)
    
    dsets = preprocess_data(dsets)

    dsets['var_class'] = 'MIRA'
    dsets = pd.merge(dsets, mira_table, left_on='sourceid', right_on='virac_id')
    dsets = dsets.rename(columns = {'period':'cat_period'})

    return dsets

