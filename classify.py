from config import *
import pandas as pd
import pickle
import numpy as np
import sqlutilpy as sqlutil
from functools import partial
from multiprocessing import Pool
from initial_classif.classifier.classifier import binary_classification
from fine_classif.classifier.classifier import variable_classification
from interface_utils.add_stats import pct_diff, main_string, var_string, phot_string, error_ratios
from run_variable_classifier import get_periodic_features, save_cols, save_cols_types
from interface_utils.light_curve_loader import lightcurve_loader

def wrap(l):
    return l-360.*(l>180.)

def find_cells(input_data, grid):
        
    ## A hack so if things fall outside the footprint it doesn't break
    input_data['ltmp']=input_data['l'].copy()
    input_data['btmp']=input_data['b'].copy()
    maxlgridbulge = np.max(grid['l']+.5*grid['sizel'])
    minlgridbulge = np.min((grid['l']-.5*grid['sizel'])[(grid['l']>180.)&(grid['b']>2.5)])
    input_data.loc[(np.abs(input_data['b'])>2.5)&(input_data['l']>maxlgridbulge),'ltmp']=maxlgridbulge
    input_data.loc[(np.abs(input_data['b'])>2.5)&(input_data['l']<minlgridbulge),'btmp']=minlgridbulge
    
    maxbgridbulge = np.max(grid['b']+.5*grid['sizeb'])
    minbgridbulge = np.min(grid['b']-.5*grid['sizeb'])
    input_data.loc[input_data['b']>maxbgridbulge,'btmp']=maxbgridbulge
    input_data.loc[input_data['b']<minbgridbulge,'btmp']=minbgridbulge
                           
    maxbgriddisc = np.max((grid['b']+.5*grid['sizeb'])[grid['l']<340.])
    minlgriddisc = np.min((grid['l']-.5*grid['sizel'])[grid['l']<340.])
    input_data.loc[(input_data['l']<350.)&(input_data['b']>maxbgriddisc),'btmp']=maxbgriddisc
    input_data.loc[(input_data['l']<350.)&(input_data['b']<-maxbgriddisc),'btmp']=-maxbgriddisc
    input_data.loc[(input_data['l']<minlgriddisc),'ltmp'] = minlgriddisc
    
    cells = np.abs(wrap(input_data['ltmp'][:,np.newaxis]-grid['l'][np.newaxis,:]))<=.5*np.float64(config['sizel'])
    cells &= np.abs(input_data['btmp'][:,np.newaxis]-grid['b'][np.newaxis,:])<=.5*np.float64(config['sizeb'])
    del input_data['ltmp']
    del input_data['btmp']
    
    return np.argwhere(cells)[:,1]

def classify_region(grid, variable_classifier, lightcurve_loader, 
                    config, hpx_table_index):
    
    hfltr = lightcurve_loader.healpix_grid['index']==hpx_table_index
    
    input_data = lightcurve_loader.get_data_table_per_file(
                    lightcurve_loader.healpix_grid['hpx'].values[hfltr][0], 
                    lightcurve_loader.healpix_grid['nside'].values[hfltr][0], 
                    config)
    
    if int(config['test']):
        input_data = input_data.sample(10)
    
    cell = find_cells(input_data, grid)
    
    def run_cell(index):
        with open(config['binary_output_dir'] + 'binary_%i%s.pkl'%(index,''+'_test'*int(config['test'])), 'rb') as f:
            binary_classifier = pickle.load(f)
        clss = binary_classifier.predict(input_data[cell==index].reset_index(drop=True))
        return clss
    
    binary_output = pd.concat([run_cell(index) for index in np.unique(cell)], axis=0).sort_values(by='sourceid')
    
    variable_candidates = binary_output[(binary_output['class']=='VAR')&
                                     (binary_output['prob']>np.float64(config['probability_thresh']))].reset_index(drop=True)
    print('%i/%i variable candidates' % (len(variable_candidates), len(binary_output)))
    
    variable_candidates = get_periodic_features(variable_candidates, lightcurve_loader, config)
    variable_candidates = variable_candidates[~variable_candidates['error']].reset_index(drop=True)
    
    variable_output = variable_classifier.predict(variable_candidates)
    
    variable_output[save_cols].astype(save_cols_types).to_pickle(
        config['results_dir'] + 'results_%i%s.pkl'%(hpx_table_index,''+'_test'*int(config['test'])))
    
    
if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    grid = pd.read_pickle(config['binary_output_dir'] + 'grid%s.pkl'%(''+'_test'*int(config['test'])))
    
    with open(config['variable_output_dir'] + 'variable%s.pkl'%(''+'_test'*int(config['test'])), 'rb') as f:
        variable_classifier = pickle.load(f)
    
    light_curve_loader = lightcurve_loader()
    indices = light_curve_loader.healpix_grid['index'].values
    
    if int(config['test']):
        indices = [7380776]
    
    with Pool(np.int64(config['var_cores'])) as pool:
        pool.map(partial(classify_region, grid, variable_classifier, light_curve_loader, config), 
                     indices)
