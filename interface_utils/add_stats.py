import numpy as np
import pandas as pd
from scipy import stats

from sqlutilpy import *

from virac_classifier.wsdb_utils.wsdb_cred import wsdb_kwargs

def cm_virac_stats_table(data, **wsdb_kwargs):
    """
    Crossmatch of VIRAC ids with the variability indices
    ---
    input: data = (sourceid)
    
    return: VIRAC2 variability indices
    
    """
    
    data = pd.DataFrame(sqlutil.local_join("""
                select t.* from mytable as m
                left join leigh_smith.virac2_var_indices_tmp as l on t.sourceid=l.sourceid""",
                'mytable',data['sourceid'],('virac_id'),**wsdb_kwargs))
    
    return data