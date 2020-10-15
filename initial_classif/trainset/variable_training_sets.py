import numpy as np
import pandas as pd
from scipy import stats

from sqlutilpy import *

from virac_classifier.wsdb_utils.wsdb_cred import wsdb_kwargs
from virac_classifier.interface_utils.add_stats import cm_virac_stats_table


def load_rr_lyrae():
    data = cm_virac_stats_table(data)
    data['class'] = 'RRab'
    
def load_eclipsing_binaries():
    data = cm_virac_stats_table(data)
    data['class'] = 'EB'

def load_all_variable_stars():
    dsets = [load_eclipsing_binaries, load_rr_lyrae]
    return pd.concat([r for r in dsets],axis=0)