import numpy as np

from config import *
from interface_utils.light_curve_loader import split_lcs
from initial_classif.trainset.variable_training_sets import load_all_variable_stars
from fine_classif.feat_extract.extract_feats import extract_per_feats
#from initial_classif import variable_classification

def get_periodic_features(data, config, trainset=False):
    """
    Periodic feature extracter - to be used for variable classification
    
    """
    
    print("loading lightcurves..")
    # Load variable ligth curves in pd format
    lc = split_lcs(data)
    print("loaded {} light curves".format(len(lc)))
    
    #general LombScargle frequency grid conditions 
    if trainset:
        ls_kwargs = {'maximum_frequency': np.float64(config['ls_max_freq']),
                     'minimum_frequency':1./(1.5*max(data['varcat_period'].values))}
    else:
        ls_kwargs = {'maximum_frequency': np.float64(config['ls_max_freq']),
                     'minimum_frequency':np.float64(config['ls_min_freq'])}
        
    print("loading features..")
    #Extract features
    features = extract_per_feats(lc, data, ls_kwargs, config)
    
    ### Now find the periodic features from light curves
    ### Will need to reorder output
    ### Add a serial/parallel version -- can run parallelfor full variable set and serial for each tile
    
    return features
    

if __name__=="__main__":
    
    import h5py
    import pandas as pd
    
    config = configuration()
    config.request_password()
    
    file_path = '/data/jls/virac/'

    with h5py.File(file_path+'n512_2318830.hdf5', 'r') as f:
        randints = np.sort(np.random.randint(0,55000,20))
        s=f['sourceList']['sourceid'][:][randints]
        ra=f['sourceList']['ra'][:][randints]
        dec=f['sourceList']['dec'][:][randints]

    data = pd.DataFrame()
    data['sourceid'] = s
    data['ra'] = ra
    data['dec'] = dec
    
    test_feats = get_periodic_features(data, config)
    
    test_feats.to_pickle('test_var_out.pkl')
    
#     variable_stars = load_all_variable_stars(config)
#     constant_data = load_constant_data(len(variable_stars), config)
#     constant_data['class']='CONST'
#     trainset = pd.concat([variable_stars, constant_data], axis=0).reset_index(drop=True)
#     variable_stars = get_periodic_features(trainset, config)
#     classifier = variable_classification(trainset)
    
#     with open(config['variable_output_dir'] + 'variable%s.pkl'%(index,''+'_test'*bool(config['test'])), 'wb') as f:
#         pickle.dump(classifier, f)
