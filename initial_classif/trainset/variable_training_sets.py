from config import *
import numpy as np
import pandas as pd
from scipy import stats
import pickle
from sqlutilpy import *
from interface_utils.add_stats import cm_virac_stats_table


def load_all_variable_stars(**wsdb_kwargs):
    
    with open(config['variable_dir']+'var_trainset_virac2.pkl', 'rb') as f:
        dsets = pickle.load(f)
    
    dsets = cm_virac_stats_table(dsets, **wsdb_kwargs)
    
    return dsets