from config import *
import sys
import pandas as pd
import time
import pickle
import warnings
import numpy as np
from functools import partial
from multiprocessing import Pool

from initial_classif.classifier.classifier import binary_classification
from fine_classif.classifier.classifier import variable_classification
from fine_classif.feat_extract.extract_feats import extract_per_feats
from interface_utils.add_stats import pct_diff, main_string, var_string, phot_string, error_ratios
from run_variable_classifier import save_cols_types
from interface_utils.light_curve_loader import lightcurve_loader

def wrap(l):
    return l-360.*(l>180.)


def get_periodic_features(data, lightcurve_loader, config, serial=True):
    """
    Periodic feature extracter - to be used for variable classification
    """
    
    # Load variable light curves in pd format
    lc, data = lightcurve_loader.split_lcs(data)
   
    # Universal frequency grid conditions 
    ls_kwargs = {'maximum_frequency': np.float64(config['ls_max_freq'])}
    method_kwargs = dict(irreg=True)
        
    #Extract features
    features = extract_per_feats(lc, data, ls_kwargs, method_kwargs, config, serial=serial)
    
    features['max_phase_lag'] = features['max_time_lag']/features['lsq_period']
    del features['max_time_lag']
    
    return features

def find_cells(input_data, grid):
        
    ## A hack so if things fall outside the footprint it doesn't break
    input_data['ltmp']=input_data['l'].copy()
    input_data['btmp']=input_data['b'].copy()
    maxlgridbulge = np.max(grid['l']+.5*grid['sizel'])
    minlgridbulge = np.min((grid['l']-.5*grid['sizel'])[(grid['l']>180.)&(grid['b']>2.5)])
    input_data.loc[(np.abs(input_data['b'])>2.5)&(input_data['l']>maxlgridbulge),'ltmp']=maxlgridbulge
    input_data.loc[(np.abs(input_data['b'])>2.5)&(input_data['l']<minlgridbulge),'ltmp']=minlgridbulge
    
    maxbgridbulge = np.max(grid['b']+.5*grid['sizeb'])
    minbgridbulge = np.min(grid['b']-.5*grid['sizeb'])
    input_data.loc[input_data['b']>maxbgridbulge,'btmp']=maxbgridbulge
    input_data.loc[input_data['b']<minbgridbulge,'btmp']=minbgridbulge
                           
    maxbgriddisc = np.max((grid['b']+.5*grid['sizeb'])[grid['l']<340.])
    minlgriddisc = np.min((grid['l']-.5*grid['sizel'])[grid['l']<340.])
    input_data.loc[(input_data['l']<350.)&(input_data['b']>maxbgriddisc),'btmp']=maxbgriddisc
    input_data.loc[(input_data['l']<350.)&(input_data['b']<-maxbgriddisc),'btmp']=-maxbgriddisc
    input_data.loc[(input_data['l']<minlgriddisc),'ltmp'] = minlgriddisc
    
    cells = np.abs(wrap(input_data['ltmp'].to_numpy()[:,np.newaxis]-grid['l'].to_numpy()[np.newaxis,:]))<=.5*np.float64(config['sizel'])
    cells &= np.abs(input_data['btmp'].to_numpy()[:,np.newaxis]-grid['b'].to_numpy()[np.newaxis,:])<=.5*np.float64(config['sizeb'])
    del input_data['ltmp']
    del input_data['btmp']
    
    return np.argwhere(cells)[:,1]

save_cols = [
    'sourceid', 'class', 'prob','n_epochs',
    'amp_0', 'amp_1', 'amp_2', 'amp_3', 
    'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
    'amplitude', 'beyondfrac', 'delta_loglik', 'log10_fap',
    'lsq_period', 'lsq_period_error', 'lsq_nterms', 'max_pow', 'max_phase_lag', 'pow_mean_disp', 'time_lag_mean',
    'significant_second_minimum',
    'phi_0','phi_1','phi_2','phi_3',
    'phi_double_0','phi_double_1','phi_double_2','phi_double_3',
    'peak_ratio_model', 'peak_ratio_data',
    'JK_col','HK_col', 'prob_1st_stage',
    'Z_scale','Z_model','Y_scale','Y_model','J_scale','J_model','H_scale','H_model'
]

col32_save = [
    'amp_0', 'amp_1', 'amp_2', 'amp_3', 
    'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
    'amplitude', 'beyondfrac', 'delta_loglik',  'log10_fap',
    'lsq_period_error', 'max_pow', 'max_phase_lag', 'pow_mean_disp', 'time_lag_mean',
    'phi_0','phi_1','phi_2','phi_3',
    'phi_double_0','phi_double_1','phi_double_2','phi_double_3',           
    'peak_ratio_model', 'peak_ratio_data',
    'JK_col','HK_col','prob', 'prob_1st_stage',
    'Z_scale','Z_model','Y_scale','Y_model','J_scale','J_model','H_scale','H_model'
]

save_cols_types = dict(zip(col32_save,[np.float32]*len(col32_save)))

def classify_region(grid, variable_classifier, lightcurve_loader, 
                    config, hpx_table_index):
    
    initial_time = time.time()
    
    hfltr = lightcurve_loader.healpix_grid['index']==hpx_table_index
    
    input_data = lightcurve_loader.get_data_table_per_file(
                    lightcurve_loader.healpix_grid['hpx'].to_numpy()[hfltr][0], 
                    lightcurve_loader.healpix_grid['nside'].to_numpy()[hfltr][0], 
                    config)
    
    #if int(config['test']):
    #    input_data = input_data.sample(100, random_state=42)
    
    cell = find_cells(input_data, grid)
    
    def run_cell(index):
        with warnings.catch_warnings():
            print('Suppressing sklearn pickle warnings for binary classifier') 
            warnings.simplefilter("ignore")
            with open(config['binary_output_dir'] + 'binary_%i.pkl'%index, 'rb') as f:
                binary_classifier = pickle.load(f)
                if ~hasattr(binary_classifier, 'periodic_features'):
                    binary_classifier.periodic_features=[]
                if ~hasattr(binary_classifier, 'no_upper_features'):
                    binary_classifier.no_upper_features=[]
        clss = binary_classifier.predict(input_data[cell==index].reset_index(drop=True))
        return clss
    
    binary_output = pd.concat([run_cell(index) for index in np.unique(cell)], axis=0).sort_values(by='sourceid')
    
    variable_candidates = binary_output[(binary_output['class']=='VAR')&
                                     (np.float64(binary_output['prob'])>np.float64(config['probability_thresh']))].reset_index(drop=True)
    
    print('Healpix {0}: {1}/{2} variable candidates'.format(hpx_table_index, len(variable_candidates), len(binary_output)))
    
    if(len(variable_candidates)==0):
        return
    
    variable_candidates = variable_candidates.rename(columns={"prob": "prob_1st_stage"})
    
    variable_candidates = get_periodic_features(variable_candidates, lightcurve_loader, config)
    variable_candidates['log10_fap'] = variable_candidates['log10_fap_ls']
    del variable_candidates['log10_fap_ls']
    variable_candidates = variable_candidates[~variable_candidates['error']].reset_index(drop=True)

    variable_output = variable_classifier.predict(variable_candidates)
    
    variable_output[save_cols].astype(save_cols_types).to_pickle(
        config['results_dir'] + 'results_%i%s.pkl'%(hpx_table_index,''+'_test'*int(config['test'])))
    
    final_time = time.time()
    
    output_ = ''
    for ii in list(set(np.unique(variable_output['class']))-set(['CONST'])):
        output_+='{0}:{1},'.format(ii, np.count_nonzero((variable_output['class']==ii)&
                                                        (variable_output['log10_fap']<np.float64(config['log10_fap']))&
                                                        (variable_output['prob']>0.8)))
    output_ = output_[:-1]
    
    print('Healpix {0}: run in {1}s: {2}'.format(hpx_table_index, final_time-initial_time, output_))
    
    return
    
if __name__=="__main__":
    
    config = configuration()
    
    grid = pd.read_pickle(config['binary_output_dir'] + 'grid.pkl')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print('Suppressing XGB & sklearn pickle warnings for variable classifier') 
        with open(config['variable_output_dir'] + 'variable_classifier.pkl', 'rb') as f:
            variable_classifier = pickle.load(f)

    light_curve_loader = lightcurve_loader(config['healpix_files_dir'])
    indices = light_curve_loader.healpix_grid['index'].to_numpy()
    
    if int(config['test']):
        chunked_indices_subset = [7380776]
    else:
        assert len(sys.argv)==3
        chunk_index = int(sys.argv[1])
        n_chunks = int(sys.argv[2])
        chunked_indices = np.array_split(indices, n_chunks)
        chunked_indices_subset = chunked_indices[chunk_index]
    
    with Pool(np.int64(config['var_cores'])) as pool:
        pool.map(partial(classify_region, grid, variable_classifier, 
                         light_curve_loader, config), 
                 chunked_indices_subset)
