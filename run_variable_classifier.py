from config import *
from interface_utils.light_curve_loader import split_lcs
from initial_classif.trainset.variable_training_set import load_all_variable_stars
from fine_classif.feat_extract.extract_feats import extract_per_feats
from initial_classif import variable_classification

def get_periodic_features(data, config):
    
    # Load variable ligth curves in pd format
    lc = split_lcs(data)
    
    #LombScargle frequency grid conditions 
    ls_kwargs = {'maximum_frequency': config['ls_max_freq'],
                 'minimum_frequency':1./max(data['varcat_period'].values)}
    
    #Extract features
    features = extract_per_feats(lc, ls_kwargs, config)
    
    ### Now find the periodic features from light curves
    ### Will need to reorder output
    ### Add a serial/parallel version -- can run parallel for full variable set and serial for each tile
    
    return features
    

if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    variable_stars = load_all_variable_stars(config)
    constant_data = load_constant_data(len(variable_stars), config)
    constant_data['class']='CONST'
    trainset = pd.concat([variable_stars, constant_data], axis=0).reset_index(drop=True)
    variable_stars = get_periodic_features(trainset, config)
    classifier = variable_classification(trainset)
    
    with open(config['variable_output_dir'] + 'variable%s.pkl'%(index,''+'_test'*bool(config['test'])), 'wb') as f:
        pickle.dump(classifier, f)
