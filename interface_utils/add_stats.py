import numpy as np
import pandas as pd
from scipy import stats

from sqlutilpy import *
    
main_table_cols = ['sourceid','ra','dec','l','b','ks_n_detections']
main_string = 't.'+',t.'.join(main_table_cols)

var_indices = ["ks_stdev","ks_mad","ks_kurtosis","ks_skew",
               "ks_eta","ks_n_epochs",
               "ks_stetson_i","ks_stetson_j","ks_stetson_k",
               "ks_p100","ks_p0","ks_p99","ks_p1","ks_p95","ks_p5",
               "ks_p84","ks_p16","ks_p75","ks_p25"]
var_string = 's.'+',s.'.join(var_indices)

phot_stats = ["ks_b_ivw_err_mag", "j_b_ivw_mean_mag", "h_b_ivw_mean_mag", 
              "ks_b_ivw_mean_mag", "ks_n_b_phot"]
phot_string = 'y.'+',y.'.join(phot_stats)

def pct_diff(dataV):
    
    for p in [[75,25],[84,16],[95,5],[99,1],[100,0]]:
        dataV['ks_p%i_p%i' % (p[0], p[1])] = dataV['ks_p%i' % p[0]] - dataV['ks_p%i' % p[1]]
        
    return dataV

def error_ratios(dataV):
    
    dataV['ks_mean_error'] = dataV['ks_b_ivw_err_mag'] * np.sqrt(dataV['ks_n_b_phot'])
    
    for p in ["ks_stdev", "ks_mad"]:
        dataV[p+'_over_error'] = dataV[p] / dataV['ks_mean_error']
    for p in [[75,25],[84,16],[95,5],[99,1],[100,0]]:
        dataV['ks_p%i_p%i' % (p[0], p[1]) + '_over_error'] = dataV['ks_p%i_p%i' % (p[0], p[1])]/dataV['ks_mean_error']
        
    return dataV

def fix_stetson_J(dataV):
    dataV['ks_stetson_j'] /= (dataV['ks_n_epochs']-1)
    return dataV

def preprocess_data(dataV):
    dataV = pct_diff(dataV)
    dataV = error_ratios(dataV)
    dataV = fix_stetson_J(dataV)
    return dataV

# def cm_virac_stats_table(data, config):
#     """
#     Crossmatch of VIRAC ids with the variability indices (and VIRAC2 table)
#     ---
#     input: data = (sourceid)
    
#     return: VIRAC2 variability indices
    
#     """
    
#     dataV = pd.DataFrame(sqlutil.local_join("""
#                 select {0}, y.j_b_ivw_mean_mag, y.h_b_ivw_mean_mag, y.ks_b_ivw_mean_mag, {1} from mytable as m
#                 inner join leigh_smith.virac2 as t on t.sourceid=m.sourceid
#                 inner join leigh_smith.virac2_photstats as y on t.sourceid=y.sourceid
#                 inner join leigh_smith.virac2_var_indices as s on t.sourceid=s.sourceid order by m.xid""".format(
#                 main_string,var_string),
#                 'mytable',(data['virac2_id'].values,np.arange(len(data))),('sourceid','xid'),**config.wsdb_kwargs))
    
#     dataV = pct_diff(dataV)
    
#     return pd.merge(data, dataV, left_on='virac2_id', right_on='sourceid', how='right')
