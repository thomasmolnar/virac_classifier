import astropy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import virac_lc
import lc_utils
import lc_utils_fix
import var_feature_extract

from multiprocessing import Pool
from functools import partial

ext = '/home/tam68/astro_project/data/var_classes/'

def comb_wfull(df_in):
    query = """
    select * from leigh_smith.virac_pm2_jhk
    where sourceid in {}""".format(tuple(df_in['sourceid']))
    df = pd.DataFrame(sqlutil.get(query, **wsdb_kwargs))
    df_out = df_in.merge(df, on=['sourceid'], suffixes=('pred', 'full'))
    
    return df_out


def freq_range(df, name):
    """
    Determine frequency range over which LS analysis is run
    (take longest period sources as maximum for unclassified sourcesb 
    -- i.e. Non-contact eclipsing binaries)
    """
    
    # Find period column entry in crossmatch df
    if 'varcat_period' in list(df.columns.values):
        varcat_per = list(df['{}'.format('varcat_period')])
    elif 'Per' in list(df.columns.values):
        varcat_per = list(df['{}'.format('Per')])
    elif 'Porb' in list(df.columns.values):
        varcat_per = list(df['{}'.format('Porb')])
    else:
        raise ValueError('No period column in crossmatched catalogue')
   
    # Determine range of varcat periods
    max_per = max(varcat_per)+1.
    ls_kwargs = {'maximum_frequency': 20., 'minimum_frequency':1./max_per}
    print("{} - - {}".format(ls_kwargs, name))
    
    return ls_kwargs


def extract_cand_feats(df_cm_name, classname, ncores=15, const=False):
    """
    Wrapper function for feature extraction from sourceid list
    
    """
    
    # Use set sourceid extract
    if df_cm_name == '{}/bin_nc_class'.format(ext):
        cand = pd.read_pickle('{}.pkl'.format(df_cm_name)).sample(frac=0.45, random_state=1)
    else:
        cand = pd.read_pickle('{}.pkl'.format(df_cm_name)).sample(frac=0.45, random_state=1)
    
    # Perform preliminary cuts on number of epochs and magnitude
    cand = cand.loc[cand['kepochs']>=30].copy()
    cand = cand.loc[cand['kmag']<=17].copy()
    cand = cand.loc[cand['kmag']>=11.5].copy()
    ids = cand['sourceid']
    
    # Find frequency grid based on OGLE period ranges
    if const:
        #Set frequency range to cover all nc periods (as adequately large range)
        nc = pd.read_pickle('{}/bin_nc_class.pkl'.format(ext))
        ls_kwargs = freq_range(nc, 'NC (for const sources)')
    else:
        ls_kwargs  = freq_range(cand, classname)

    print('Total {} sources (this batch) = {}'.format(classname, len(ids)))

    print("Loading {} light curves..".format(classname))
    lcs_inp = virac_lc.load_virac_lc_idlist(ids)
    lc_fld = lc_utils_fix.find_time_field(lcs_inp[0])
    lcs = []
    for i in lcs_inp:
        if len(i[lc_fld])>2:
            lcs.append(i)
        else:
            pass
    print("Loaded {} light curves.".format(len(lcs)))

    cand_feats = var_feature_extract.combine_pool_lcs(lcs, classname,
                                                     ls_kwargs=ls_kwargs,
                                                     ncores=ncores,method='force')
    print("Features loaded.")

    print("Adding colour information..")
    cand_feats_colour = var_feature_extract.add_catalogue_info(cand_feats,
                                                              df_cm_name)
    print("Added colour information.")

    print("Finalising dataframe..")
    cand_feats_final = var_feature_extract.construct_final_df(cand_feats_colour)
    print("All features extracted and processed.")

    cand_feats_final.to_pickle("{}_feats.pkl".format(classname))

