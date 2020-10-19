from config import *
import numpy as np
import pandas as pd
import pickle
from multiprocessing import Pool
from functools import partial
from initial_classif.trainset.gaia_extraction import generate_gaia_training_set
from initial_classif.trainset.variable_training_sets import load_all_variable_stars
from initial_classif.classifier.classifier import binary_classification


def train_classification_region(grid, sizel, sizeb, variable_stars, config, index):
    
    l,b = grid[index]
    print('Generating Gaia training set for (%s, %s)'%(l,b))
    gaia = generate_gaia_training_set(l, b, sizel * 60., sizeb * 60., 
                                      np.float64(config['gaia_percentile']),
                                      **config.wsdb_kwargs)
    gaia['class']='CONST'
    
    full_data = pd.concat([variable_stars, gaia], axis=0, sort=False)
    
    print('Running classifier for (%s, %s): %i stars'%(l,b,len(full_data)))
    classifier = binary_classification(full_data[:20])
    
    with open(config['binary_output_dir'] + 'binary_%i%s.pkl'%(index,''+'_test'*bool(config['test'])), 'wb') as f:
        pickle.dump(classifier, f)

        
def run_loop(lstart, lend, bstart, bend, variable_stars, config):
    
    sizel, sizeb = np.float64(config['sizel']), np.float64(config['sizeb'])
    
    l_arr, b_arr = np.arange(lstart, lend, sizel), np.arange(bstart, bend, sizeb)
    l_arr, b_arr = .5*(l_arr[1:]+l_arr[:-1]), .5*(b_arr[1:]+b_arr[:-1])
    
    L,B = np.meshgrid(l_arr, b_arr)
    grid = np.vstack([L.flatten(), B.flatten()]).T
    
    pd.DataFrame({'index':np.arange(len(grid)), 'l':L.flatten(),'b':B.flatten()}).to_pickle(
        config['binary_output_dir'] + 'grid%s.pkl'%(''+'_test'*bool(config['test']))
    )
    
    p = Pool(32)
    p.map(partial(train_classification_region, grid, sizel, sizeb, variable_stars, config),
          np.arange(len(grid)))
    p.close()
    p.join()
        
    
if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    print('Loading variable stars...') 
    variable_stars = load_all_variable_stars(config, **config.wsdb_kwargs)
    variable_stars['class']='VAR'
    
    if bool(config['test']):
        config['sizel']=0.09
        config['sizeb']=0.09
        l, b = 1.275,-0.385
        run_loop(l - .5 * np.float64(config['sizel']), 
                 l + 1.01 * .5 * np.float64(config['sizel']), 
                 b - .5 * np.float64(config['sizeb']),
                 b + 1.01 * .5 * np.float64(config['sizeb']), 
                 variable_stars, config)
    else:
        run_loop(-10,10.1,-10,10.1, variable_stars, config)
