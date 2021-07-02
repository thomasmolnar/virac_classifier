import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_MAX_THREADS"]="1"
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
from config import *
import sys
import logging
import multiprocessing_logging
multiprocessing_logging.install_mp_handler()
import pandas as pd
import time
import pickle
import warnings
import numpy as np
from functools import partial
from multiprocessing import Pool
from datetime import datetime, timedelta

from initial_classif.classifier.classifier import binary_classification
from fine_classif.classifier.classifier import variable_classification
from fine_classif.feat_extract.extract_feats import extract_per_feats
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
    ls_kwargs = {'maximum_frequency': np.float64(config['ls_max_freq']),
                 'minimum_frequency': np.float64(config['ls_min_freq'])}
    method_kwargs = dict(irreg=True)
        
    #Extract features
    features = extract_per_feats(lc, data, ls_kwargs, method_kwargs, config, serial=serial)
    
    return features

def find_cells(input_data, grid, config):
        
    ## A hack so if things fall outside the footprint it doesn't break
    input_data['ltmp']=input_data['l'].copy()
    input_data['btmp']=input_data['b'].copy()
    
    maxbgriddisc = np.max((grid['b']+.5*grid['sizeb'])[grid['l']<-20.])
    maxbgriddisc_f = np.max((grid['b']+grid['sizeb'])[grid['l']<-20.])
    maxbgridbulge = np.max(grid['b']+.5*grid['sizeb'])
    minbgridbulge = np.min(grid['b']-.5*grid['sizeb'])
    maxlgridbulge = np.max(grid['l']+.5*grid['sizel'])
    minlgridbulge = np.min((grid['l']-.5*grid['sizel'])[(grid['l']<0.)&(grid['b']>maxbgriddisc)])
    minlgridbulge_f = np.min((grid['l']-grid['sizel'])[(grid['l']<0.)&(grid['b']>maxbgriddisc)])
    minlgriddisc = np.min((grid['l']-.5*grid['sizel'])[grid['l']<-20.])
    
    input_data.loc[(wrap(input_data['l'])>=maxlgridbulge),'ltmp']=maxlgridbulge-1e-9
    input_data.loc[(wrap(input_data['l'])<minlgriddisc),'ltmp'] = 360.+minlgriddisc+1e-9
    
    input_data.loc[input_data['b']>=maxbgridbulge,'btmp']=maxbgridbulge-1e-9
    input_data.loc[input_data['b']<minbgridbulge,'btmp']=minbgridbulge+1e-9
    input_data.loc[(wrap(input_data['l'])<minlgridbulge_f)&(input_data['b']<-maxbgriddisc),'btmp']=-maxbgriddisc+1e-9
    input_data.loc[(wrap(input_data['l'])<minlgridbulge_f)&(input_data['b']>maxbgriddisc),'btmp']=maxbgriddisc-1e-9
    input_data.loc[(wrap(input_data['l'])<minlgridbulge)&(np.abs(input_data['btmp'])>maxbgriddisc),'ltmp']=360.+minlgridbulge+1e-9
    
    cells =  wrap(input_data['ltmp'].to_numpy()[:,np.newaxis])>=(grid['l']-.5*grid['sizel']).to_numpy()[np.newaxis,:]
    cells &= wrap(input_data['ltmp'].to_numpy()[:,np.newaxis])<(grid['l']+.5*grid['sizel']).to_numpy()[np.newaxis,:]
    cells &= input_data['btmp'].to_numpy()[:,np.newaxis]>=(grid['b']-.5*grid['sizeb']).to_numpy()[np.newaxis,:]
    cells &= input_data['btmp'].to_numpy()[:,np.newaxis]<(grid['b']+.5*grid['sizeb']).to_numpy()[np.newaxis,:]
    del input_data['ltmp']
    del input_data['btmp']
    
    cell_index=np.argwhere(cells)[:,1]
    assert len(cell_index)==len(input_data)
    return cell_index

save_cols = [
    'sourceid', 'class', 'prob','n_epochs',
    'amp_0', 'amp_1', 'amp_2', 'amp_3', 
    'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
    'amplitude', 'model_amplitude', 'beyondfrac', 'delta_loglik', 'log10_fap',
    'lsq_period', 'lsq_period_error', 'lsq_nterms', 'max_pow', 'max_phase_lag', 'pow_mean_disp', 'phase_lag_mean',
    'significant_second_minimum',
    'phi_0','phi_1','phi_2','phi_3',
    'phi_double_0','phi_double_1','phi_double_2','phi_double_3',
    'peak_ratio_model', 'peak_ratio_data',
    'model_amplitude',
    'JK_col','HK_col', 'prob_1st_stage',
    'Z_scale','Z_model','Y_scale','Y_model','J_scale','J_model','H_scale','H_model'
]

col32_save = [
    'amp_0', 'amp_1', 'amp_2', 'amp_3', 
    'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
    'amplitude', 'model_amplitude', 'beyondfrac', 'delta_loglik',  'log10_fap',
    'lsq_period_error', 'max_pow', 'max_phase_lag', 'pow_mean_disp', 'phase_lag_mean',
    'phi_0','phi_1','phi_2','phi_3',
    'phi_double_0','phi_double_1','phi_double_2','phi_double_3',           
    'peak_ratio_model', 'peak_ratio_data',
    'model_amplitude',
    'JK_col','HK_col','prob', 'prob_1st_stage',
    'Z_scale','Z_model','Y_scale','Y_model','J_scale','J_model','H_scale','H_model'
]

save_cols_types = dict(zip(col32_save,[np.float32]*len(col32_save)))

def classify_region(grid, variable_classifier, lightcurve_loader, 
                    config, hpx_table_index):
   
    output_file = config['results_dir'] + 'results_%i%s.csv.tar.gz'%(hpx_table_index,''+'_test'*int(config['test']))

    if os.path.isfile(output_file) and int(config['overwrite'])==0:
        logging.info('Healpix {0}: already exists, skipping'.format(hpx_table_index))
        return
    
    initial_time = time.time()
 
    logging.info('Healpix {0}: started, generating input data from lightcurve files'.format(hpx_table_index))
   
    hfltr = lightcurve_loader.healpix_grid['index']==hpx_table_index
    
    input_data = lightcurve_loader.get_data_table_per_file(
                    lightcurve_loader.healpix_grid['hpx'].to_numpy()[hfltr][0], 
                    lightcurve_loader.healpix_grid['nside'].to_numpy()[hfltr][0], 
                    config)
    
    if(len(input_data)==0):
        return 

    #if int(config['test']):
    #    input_data = input_data.sample(100, random_state=42)
    
    logging.info('Healpix {0}: Running binary classifier for {1} lightcurves. Predicted finish time: {2}'.format(
                 hpx_table_index, len(input_data), datetime.now()+timedelta(seconds=0.6*0.2*len(input_data)))) 
    
    cell = find_cells(input_data, grid, config)
    
    def run_cell(index):
        with warnings.catch_warnings():
            logging.warn('Healpix {0}: Suppressing sklearn pickle warnings for binary classifier'.format(hpx_table_index)) 
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
    
    logging.info('Healpix {0}: {1}/{2} variable candidates, predicted finish time {3}'.format(
        hpx_table_index, len(variable_candidates), len(binary_output), 
        datetime.now()+timedelta(seconds=0.6*len(variable_candidates)))) 
    
    if(len(variable_candidates)==0):
        return
    
    variable_candidates = variable_candidates.rename(columns={"prob": "prob_1st_stage"})
    
    variable_candidates = get_periodic_features(variable_candidates, lightcurve_loader, config)
    
    variable_candidates = variable_candidates[~variable_candidates['error']].reset_index(drop=True)

    if(len(variable_candidates)==0):
        return
    
    variable_candidates['log10_fap'] = variable_candidates['log10_fap_ls']
    del variable_candidates['log10_fap_ls']
   
    variable_output = {}

    for fap_cut in variable_classifier: 
        variable_output[fap_cut] = variable_classifier[fap_cut].predict(variable_candidates)
    for s in ['class', 'prob', 'prob_var']:
        variable_output[-10][s+'_nofap'] = variable_output[0][s]

    variable_output[-10][save_cols].astype(save_cols_types).to_csv(output_file, index=False)
    
    final_time = time.time()
    
    output_ = ''
    for ii in list(set(np.unique(variable_output[-10]['class']))-set(['CONST'])):
        output_+='{0}:{1},'.format(ii, np.count_nonzero((variable_output[-10]['class']==ii)&
                                                        (variable_output[-10]['log10_fap']<np.float64(config['log10_fap']))&
                                                        (variable_output[-10]['prob']>0.8)))
    output_ = output_[:-1]
    
    logging.info('Healpix {0}: finished, run in {1}s: {2}'.format(hpx_table_index, final_time-initial_time, output_))
    
    return
    
if __name__=="__main__":
    
    config = configuration()
    
    light_curve_loader = lightcurve_loader(config['healpix_files_dir'])
    indices = light_curve_loader.healpix_grid['index'].to_numpy().copy()
    np.random.seed(42)
    np.random.shuffle(indices)
  
    if int(config['test']):
        chunked_indices_subset = [7380776] * int(config['var_cores'])
        chunk_index=0
    else:
        assert len(sys.argv)==3
        chunk_index = int(sys.argv[1])
        n_chunks = int(sys.argv[2])
        chunked_indices = np.array_split(indices, n_chunks)
        chunked_indices_subset = chunked_indices[chunk_index]
    
    logging.basicConfig(filename=config['results_dir']+'log_{0}{1}.log'.format(chunk_index,''+'_test'*int(config['test'])), 
                        filemode='w', level=logging.INFO, format='%(asctime)s %(message)s')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.warn('Suppressing XGB & sklearn pickle warnings for variable classifier') 
        variable_classifier = {}
        for fap_cut in [-10, 0]:
            with open(config['variable_output_dir'] + 'variable_classifier_%i.pkl' % fap_cut, 'rb') as f:
                variable_classifier[fap_cut] = pickle.load(f)
    grid = pd.read_pickle(config['binary_output_dir'] + 'grid.pkl')
    
    with Pool(np.int64(config['var_cores'])) as pool:
        pool.map(partial(classify_region, grid, variable_classifier, 
                         light_curve_loader, config), 
                 chunked_indices_subset)
