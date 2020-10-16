from config import *
import numpy as np
import pandas as pd
from scipy import stats
import pickle

from sqlutilpy import *

from wsdb_utils.wsdb_cred import wsdb_kwargs
from interface_utils.add_stats import cm_virac_stats_table


def load_all_variable_stars():
    
    with open(config['variable_dir']+'var_trainset_virac2.pkl', 'rb') as f:
        dsets = pickle.load(f)
    
    dsets = cm_virac_stats_table(dsets)
    
    return dsets