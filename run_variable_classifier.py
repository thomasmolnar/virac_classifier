from config import *
import numpy as np
import pandas as pd
import pickle
from interface_utils.light_curve_loader import lightcurve_loader
from initial_classif.trainset.variable_training_sets import load_all_variable_stars
from initial_classif.trainset.gaia_extraction import generate_gaia_training_set_random
from fine_classif.classifier.classifier import variable_classification
from fine_classif.feat_extract.extract_feats import extract_per_feats


def get_periodic_features(data, lightcurve_loader, config, trainset=False):
    """
    Periodic feature extracter - to be used for variable classification
    """
    
    print("loading lightcurves..")
    # Load variable light curves in pd format
    lc = lightcurve_loader.split_lcs(data)
    print("loaded {} light curves".format(len(lc)))
    
    #general LombScargle frequency grid conditions 
    if trainset:
        ls_kwargs = {'maximum_frequency': np.float64(config['ls_max_freq']),
                     'minimum_frequency':1./(1.5*np.nanmax(data['varcat_period'].values))}
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
    
    config = configuration()
    config.request_password()
    
    variable_stars = load_all_variable_stars(config, test=bool(config['test']))
    constant_data = generate_gaia_training_set_random(len(variable_stars)//10, config,
                                                      np.float64(config['gaia_percentile']),
                                                      20000)
    constant_data['class']='CONST'
    constant_data['varcat_period'] = 1./np.float64(config['ls_min_freq'])/1.5
    
    if bool(config['test']):
        trainset = constant_data.copy()
    else:
        trainset = pd.concat([variable_stars, constant_data], axis=0, sort=False).reset_index(drop=True)
    
    lightcurve_loader = lightcurve_loader()
    
    features = get_periodic_features(trainset, lightcurve_loader, config, trainset=True)
    features.to_csv('tmp.csv')
    
    classifier = variable_classification(features)
    
    with open(config['variable_output_dir'] 
              + 'variable%s.pkl'%(''+'_test'*bool(config['test'])), 'wb') as f:
        pickle.dump(classifier, f)
