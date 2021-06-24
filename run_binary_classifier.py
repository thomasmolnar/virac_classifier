from config import *
import numpy as np
import pandas as pd
import os
import pickle
from multiprocessing import Pool, BoundedSemaphore
from functools import partial
from initial_classif.trainset.gaia_extraction import generate_gaia_training_set
from initial_classif.trainset.variable_training_sets import load_all_variable_stars, load_mira_sample
from initial_classif.classifier.classifier import binary_classification

lock = BoundedSemaphore(4)

def train_classification_region(grid, variable_stars, config, index):
    
    output_file = config['binary_output_dir'] + 'binary_%i%s.pkl'%(index,''+'_test'*int(config['test']))
    output_file_training_set = config['binary_output_dir'] + \
        'binary_training_set_%i%s.pkl'%(index,''+'_test'*int(config['test']))
    if os.path.isfile(output_file) and int(config['overwrite'])==0:
        return
    
    print('Generating Gaia training set for (%s, %s)'%(grid['l'].values[index], grid['b'].values[index]))
    lock.acquire()
    gaia = generate_gaia_training_set(grid['l'].values[index], grid['b'].values[index], 
                                      grid['sizel'].values[index] * 60., grid['sizeb'].values[index] * 60., 
                                      np.float64(config['gaia_percentile']),
                                      len(variable_stars),
                                      config)
    lock.release()
    gaia['detailed_var_class']='CONST'
    gaia['var_class']='CONST'
    
#     variable_stars = variable_stars[~((variable_stars['detailed_var_class']=='OSARG')|
#                                       (variable_stars['detailed_var_class']=='DSCT'))].reset_index(drop=True)
    
    full_data = pd.concat([variable_stars, gaia], axis=0, sort=False)
    
    print('Running classifier for (%s, %s): %i stars'%(grid['l'].values[index], grid['b'].values[index],len(full_data)))
    
    classifier = binary_classification(full_data)
    
    with open(output_file_training_set, 'wb') as f:
        pickle.dump(classifier, f)
        
    del classifier.training_set
    with open(output_file, 'wb') as f:
        pickle.dump(classifier, f)
        
def make_grid(lstart,lend,bstart,bend,sizel,sizeb):
    
    l_arr, b_arr = np.arange(lstart, lend, sizel), np.arange(bstart, bend, sizeb)
    l_arr, b_arr = .5*(l_arr[1:]+l_arr[:-1]), .5*(b_arr[1:]+b_arr[:-1])
    L,B = np.meshgrid(l_arr, b_arr)
    
    grid = np.vstack([L.flatten(), B.flatten(), 
                      np.ones_like(B.flatten())*sizel, 
                      np.ones_like(B.flatten())*sizeb]).T
    return pd.DataFrame({'index':np.arange(len(grid)), 
                  'l':grid[:,0],'b':grid[:,1],
                  'sizel':grid[:,2],'sizeb':grid[:,3]})
        
def run_loop(lstart, lend, bstart, bend, 
             lstart_disc, lend_disc, bstart_disc, bend_disc, 
             variable_stars, config):
    
    sizel, sizeb = np.float64(config['sizel']), np.float64(config['sizeb'])
    bulge_grid = make_grid(lstart, lend, bstart, bend, sizel, sizeb)
    disc_grid = make_grid(lstart_disc, lend_disc, bstart_disc, bend_disc, sizel, sizeb)
    
    grid = pd.concat([bulge_grid, disc_grid], axis=0).reset_index(drop=True)
    grid['index'] = np.arange(len(grid))
    
    grid.to_pickle(
        config['binary_output_dir'] + 'grid%s.pkl'%(''+'_test'*int(config['test']))
    )
    
    with Pool(np.int64(config['var_cores'])) as p:
        p.map(partial(train_classification_region, grid, variable_stars, config),
              np.arange(len(grid)))
        
    
if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    print('Loading variable stars...') 
    variable_stars = load_all_variable_stars(config)
    mira = load_mira_sample(config)
    variable_stars = pd.concat([variable_stars, mira], axis=0, sort=False).reset_index(drop=True)
    variable_stars['detailed_var_class']=variable_stars['var_class'].copy()
    variable_stars['var_class']='VAR'
    
    if int(config['test']):
        config['sizel']=0.4
        config['sizeb']=0.8
        l, b = 0.787411, -0.054603
        run_loop(l - .5 * np.float64(config['sizel']), 
                 l + 1.01 * .5 * np.float64(config['sizel']), 
                 b - .5 * np.float64(config['sizeb']),
                 b + 1.01 * .5 * np.float64(config['sizeb']), 
                 0.,0.,0.,0.,
                 variable_stars, config)
    else:
        run_loop(-10,10.8,-10.3,5.2, 
                 -65.6428571434, -9.9,
                 -1.1057142857*2, 2.3,
                 variable_stars, config)
