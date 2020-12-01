from config import *
import numpy as np
import pandas as pd
import pickle
import sqlutilpy as sqlutil
from interface_utils.light_curve_loader import lightcurve_loader
from initial_classif.trainset.variable_training_sets import load_all_variable_stars
from initial_classif.trainset.gaia_extraction import generate_gaia_training_set_random
from fine_classif.classifier.classifier import variable_classification
from fine_classif.feat_extract.extract_feats import extract_per_feats


def get_periodic_features_var(data, config, serial=True):
    """
    Periodic feature extracter - to be used for variable classification
    """
    print("Loading Timeseries data...")
    lcs = pd.DataFrame(sqlutil.get('''select sourceid, 
                                unnest(mjdobs) as mjdobs,
                                unnest(mag) as mag,
                                unnest(emag) as emag,
                                unnest(chi) as chi,
                                unnest(ast_res_chisq) as ast_res_chisq,
                                unnest(ambiguous_match) as ambiguous_match  
                                from leigh_smith.virac2_ts_tmolnar_train''',
                     **config.wsdb_kwargs))
    print("---Timeseries data loaded.")
    
    lcs = lcs.sort_values(by=['sourceid', 'mjdobs']).reset_index(drop=True)
    uniq_ids, indices, inv_ids = np.unique(lcs['sourceid'], return_index=True, return_inverse=True)
    indices = indices[1:]
    
    data = data.sort_values(by='sourceid').reset_index(drop=True)
    
    # Add sky position of sources to be passed into periodic computation
    assert len(data)==len(uniq_ids)
    
    assert all(np.diff(data['sourceid'].values, uniq_ids)==0)

    ras_full = data['ra'].values[inv_ids]
    decs_full = data['dec'].values[inv_ids]
    
    ts_dict = {c: np.split(lcs[c], indices) for c in list(lcs.keys())}
    ra_dict = dict(ra=np.split(ras_full, indices))
    dec_dict = dict(dec=np.split(decs_full, indices))

    datadict = {**ts_dict, **ra_dict, **dec_dict}
    lightcurves = [
        pd.DataFrame(dict(list(zip(datadict, t))))
        for t in zip(*list(datadict.values()))
    ]
    
    # Universal frequency grid conditions 
    ls_kwargs = dict(maximum_frequency=np.float64(config['ls_max_freq']),
                     minimum_frequency=np.float64(config['ls_min_freq']))
    method_kwargs = dict(irreg=False)
    
    variable_output_dir = str(config['variable_output_dir'])
    
    #Extract features
    step = len(lightcurves)//10
    curr_start, curr_end = 0, step
    batch_numb = 0
    total_features = pd.DataFrame()
    while curr_end < len(lightcurves):
        batch_features = extract_per_feats(lightcurves[curr_start:curr_end],
                                           data, ls_kwargs, method_kwargs,
                                           config, serial=serial)
        batch_features.to_pickle(variable_output_dir+'variable_features_batch{}.pkl'.format(batch_numb))
        total_features = pd.concat([total_features, batch_features])
        batch_numb+=1
        curr_start+=step
        curr_end+=step
    
    total_features.to_pickle(variable_output_dir+'variable_features_total.pkl')
    
    return features


save_cols = ['sourceid','ra','dec','l','b','ks_b_ivw_mean_mag',
             'amp_0', 'amp_1', 'amp_2', 'amp_3', 
             'amplitude', 'beyondfrac', 'delta_loglik', 
             'ls_period', 'lsq_period',
             'max_pow', 'max_time_lag', 'pow_mean_disp', 'time_lag_mean',
             'phi_0','phi_1','phi_2','phi_3','JK_col','HK_col',
             'class','prob']

save_cols_types = dict(zip(['amp_0', 'amp_1', 'amp_2', 'amp_3', 
                 'amplitude', 'beyondfrac', 'delta_loglik', 
                 'ls_period', 'ls_period_error', 'lsq_period',
                 'max_pow', 'max_time_lag', 'pow_mean_disp', 'time_lag_mean',
                 'phi_0','phi_1','phi_2','phi_3','JK_col','HK_col','prob'],[np.float32]*20))

def generate_secondstage_training(config):
    variable_stars = load_all_variable_stars(config)
#     constant_data = generate_gaia_training_set_random(len(variable_stars)//10, config,
#                                                       np.float64(config['gaia_percentile']),
#                                                       600000)
#     constant_data['var_class']='CONST'
    
#     trainset = pd.concat([variable_stars, constant_data], axis=0, sort=False).reset_index(drop=True)

    trainset = trainset[~trainset['sourceid'].duplicated()].reset_index(drop=True)
    
    return trainset

if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    print("Loading trainset...")
    trainset = generate_secondstage_training(config)
    print("---Trainset loaded - {} stars".format(len(trainset)))
    print("Loading periodic features...")
    features = get_periodic_features_var(trainset, config, serial=False)
    
#     features= features[~features['error']].reset_index(drop=True)
    
#     classifier = variable_classification(features, config)
        
#     classifier.training_set.astype(save_cols_types).to_pickle(
#         config['variable_output_dir'] + 'results%s.pkl'%(''+'_test'*int(config['test'])))
#     del classifier.training_set
    
#     with open(config['variable_output_dir'] 
#               + 'variable%s.pkl'%(''+'_test'*int(config['test'])), 'wb') as f:
#         pickle.dump(classifier, f)
