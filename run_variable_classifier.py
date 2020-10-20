from config import *
from interface_utils.light_curve_loader import lightcurve_loader
from trainset.variable_training_set import load_all_variable_stars
from initial_classif import variable_classification

def get_periodic_features(data):
    
    ll = lightcurve_loader()
    
    lc = ll(data['sourceid'].values)
    
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
    variable_stars = get_periodic_features(trainset)
    classifier = variable_classification(trainset)
    
    with open(config['variable_output_dir'] + 'variable%s.pkl'%(index,''+'_test'*bool(config['test'])), 'wb') as f:
        pickle.dump(classifier, f)
