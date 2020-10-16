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
    
    request_password()
    
    variable_stars = load_all_variable_stars()
    variable_stars = get_periodic_features(variable_stars)
    classfier = variable_classification(variable_stars)
    
    with open(config['variable_output_dir'] + 'variable%s.pkl'%(index,''+'_test'*config['test']), 'wb') as f:
        pickle.dump(classifier.model, f)