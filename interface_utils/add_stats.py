from config import *
import numpy as np
import pandas as pd
from scipy import stats

from sqlutilpy import *

from wsdb_utils.wsdb_cred import wsdb_kwargs

def cm_virac(data, **wsdb_kwargs):
    """
    Crossmatch of VIRAC ids with the VIRAC2 tables
    ---
    input: data = (sourceid)
    
    return: VIRAC2 data
    
    """
    
    data = pd.DataFrame(sqlutil.local_join("""
                select * from mytable as m
                left join leigh_smith.virac2 as l on l.sourceid=m.sourceid""",
                'mytable',(data['virac2_id'],),('sourceid',),
                                           password=config['password'],**wsdb_kwargs))
    
    
    
    return data

def cm_virac_stats_table(data, **wsdb_kwargs):
    """
    Crossmatch of VIRAC ids with the variability indices
    ---
    input: data = (sourceid)
    
    return: VIRAC2 variability indices
    
    """
    
    data = pd.DataFrame(sqlutil.local_join("""
                select * from mytable as m
                left join leigh_smith.virac2_var_indices_tmp as l on l.sourceid=m.sourceid""",
                'mytable',(data['virac2_id'],),('sourceid',),
                                           password=config['password'],**wsdb_kwargs))
    
    return data