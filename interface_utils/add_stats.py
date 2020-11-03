import numpy as np
import pandas as pd
from scipy import stats

from sqlutilpy import *

def cm_virac(data, config):
    """
    Crossmatch of VIRAC ids with the VIRAC2 tables
    ---
    input: data = (sourceid)
    
    return: VIRAC2 data
    
    """
    
    data = pd.DataFrame(sqlutil.local_join("""
                select l.*, y.j_b_ivw_mean_mag, y.h_b_ivw_mean_mag, y.ks_b_ivw_mean_mag from mytable as m
                inner join leigh_smith.virac2 as l on l.sourceid=m.sourceid
                inner join leigh_smith.virac2_photstats as y on t.sourceid=y.sourceid""",
                'mytable',(data['virac2_id'].values,),('sourceid',),**config.wsdb_kwargs))
    
    return data

def pct_diff(dataV):
    
    for p in [[75,25],[84,16],[95,5],[99,1],[100,0]]:
        dataV['ks_p%i_p%i' % (p[0], p[1])] = dataV['ks_p%i' % p[0]] - dataV['ks_p%i' % p[1]]
        
    return dataV

def cm_virac_stats_table(data, config):
    """
    Crossmatch of VIRAC ids with the variability indices (and VIRAC2 table)
    ---
    input: data = (sourceid)
    
    return: VIRAC2 variability indices
    
    """
    
    dataV = pd.DataFrame(sqlutil.local_join("""
                select t.*, y.j_b_ivw_mean_mag, y.h_b_ivw_mean_mag, y.ks_b_ivw_mean_mag, l.* from mytable as m
                inner join leigh_smith.virac2 as t on t.sourceid=m.sourceid
                inner join leigh_smith.virac2_photstats as y on t.sourceid=y.sourceid
                inner join leigh_smith.virac2_var_indices as l on l.sourceid=m.sourceid""",
                'mytable',(data['virac2_id'].values,),('sourceid',),**config.wsdb_kwargs))
    
    dataV = pct_diff(dataV)
    
    return pd.merge(data, dataV, left_on='virac2_id', right_on='sourceid', how='right')
