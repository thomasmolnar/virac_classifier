from config import *
import numpy as np
import pandas as pd
from interface_utils.light_curve_loader import lightcurve_loader
from initial_classif.trainset.variable_training_sets import load_all_variable_stars
from initial_classif.trainset.gaia_extraction import generate_gaia_training_set_random
from fine_classif.feat_extract.extract_feats import extract_per_feats


def get_periodic_features(data, lightcurve_loader, config):
    
    # Load variable light curves in pd format
    lc = lightcurve_loader.split_lcs(data)
    
    #general LombScargle frequency grid conditions 
    ls_kwargs = {'maximum_frequency': config['ls_max_freq'],
                 'minimum_frequency':1./(1.5*max(data['varcat_period'].values))}
    
    #Extract features
    features = extract_per_feats(lc, ls_kwargs, config)
    
    ### Now find the periodic features from light curves
    ### Will need to reorder output
    ### Add a serial/parallel version -- can run parallel for full variable set and serial for each tile
    
    return features
    

if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    variable_stars = load_all_variable_stars(config, test=bool(config['test']))
    constant_data = generate_gaia_training_set_random(len(variable_stars)//10, config, np.float64(config['gaia_percentile']))
    constant_data['class']='CONST'
    trainset = pd.concat([variable_stars, constant_data], axis=0, sort=False).reset_index(drop=True)
    
    lightcurve_loader = lightcurve_loader()
    
    variable_stars = get_periodic_features(trainset, lightcurve_folder, config)
    classifier = variable_classification(trainset)
    
    with open(config['variable_output_dir'] + 'variable%s.pkl'%(index,''+'_test'*bool(config['test'])), 'wb') as f:
        pickle.dump(classifier, f)
