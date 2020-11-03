from config import *
import pandas as pd
import pickle
import numpy as np
import sqlutilpy as sqlutil
from initial_classif.classifier.classifier import binary_classification, feat_clip
from fine_classif.classifier.classifier import variable_classification
from interface_utils.add_stats import pct_diff
from run_variable_classifier import get_periodic_features
from interface_utils.light_curve_loader import lightcurve_loader


def grab_virac_with_stats(l,b,sizel,sizeb,config):
    
    sizel /= 60.
    sizeb /= 60.
    poly_string = "t.l>%0.3f and t.l<%0.3f and t.b>%0.3f and t.b<%0.3f"\
                    %(l-.5*sizel,l+.5*sizel,b-.5*sizeb,b+.5*sizeb)

    if (l - .5 * sizel < 0.):
        poly_string = "(t.l>%0.3f or t.l<%0.3f) and t.b>%0.3f and t.b<%0.3f"\
                        %(l-.5*sizel+360.,l+.5*sizel,b-.5*sizeb,b+.5*sizeb)
    if (l + .5 * sizel > 360.):
        poly_string = "(t.l>%0.3f or t.l<%0.3f) and t.b>%0.3f and t.b<%0.3f"\
                        %(l-.5*sizel,l+.5*sizel-360.,b-.5*sizeb,b+.5*sizeb)
        
    data = pd.DataFrame(sqlutil.get("""
            select t.*, s.*, y.j_b_ivw_mean_mag, y.h_b_ivw_mean_mag, y.ks_b_ivw_mean_mag
            from leigh_smith.virac2 as t
            inner join leigh_smith.virac2_photstats as y on y.sourceid=t.sourceid
            inner join leigh_smith.virac2_var_indices as s on s.sourceid=t.sourceid
            where %s and duplicate=0 and astfit_params=5
            and ks_n_detections>%i and ks_ivw_mean_mag>%0.4f and ks_ivw_mean_mag<%0.4f"""%(
		poly_string,np.int64(config['n_detection_threshold']),
        np.float64(config['lower_k']),np.float64(config['upper_k'])), 
                                    **config.wsdb_kwargs))
    
    data = pct_diff(data)
    
    return data


def classify_region(grid, variable_classifier, lightcurve_loader, config, index):
    
    with open(config['binary_output_dir'] + 'binary_%i%s.pkl'%(index,''+'_test'*bool(config['test'])), 'rb') as f:
        binary_classifier = pickle.load(f)
    
    input_data = grab_virac_with_stats(grid['l'].values[index], grid['b'].values[index], 
                                       np.float64(config['sizel'])*60., np.float64(config['sizeb'])*60., 
                                       config)
    binary_input = feat_clip(input_data, binary_classifier.data_cols, [], impute=True, return_full=True)
    classes = binary_classifier.model.predict(binary_input[binary_classifier.data_cols])
    probability = binary_classifier.model.predict_proba(binary_input[binary_classifier.data_cols])
    var_index = np.argwhere(binary_classifier.model.classes_=='VAR')[0][0]
    
    variable_candidates = binary_input[(classes=='VAR')&
                                       (probability[:,var_index]>np.float64(config['probability_thresh']))].reset_index(drop=True)
    print('%i/%i variable candidates' % (len(variable_candidates), len(input_data)))
    variable_candidates = get_periodic_features(variable_candidates, lightcurve_loader, config)
    
    variable_input = feat_clip(variable_candidates, variable_classifier.data_cols, [], impute=True, return_full=True)
    variable_classes = variable_classifier.model.predict(variable_input[variable_classifier.data_cols])
    
    results = pd.DataFrame({'class':variable_classes, **variable_input})
    results.to_pickle(config['results_dir'] + 'results_%i%s.pkl'%(index,''+'_test'*bool(config['test'])))
    
    
if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    if bool(config['test']):
        config['sizel']=0.01
        config['sizeb']=0.01
    
    grid = pd.read_pickle(config['binary_output_dir'] + 'grid%s.pkl'%(''+'_test'*bool(config['test'])))
    
    with open(config['variable_output_dir'] + 'variable%s.pkl'%(''+'_test'*bool(config['test'])), 'rb') as f:
        variable_classifier = pickle.load(f)
    
    light_curve_loader = lightcurve_loader()
    
    for ii in np.arange(len(grid)):
        classify_region(grid, variable_classifier, light_curve_loader, config, ii)
