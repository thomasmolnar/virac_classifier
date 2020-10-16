from config import *
from functools import partial
from trainset.gaia_extraction import generate_gaia_training_set
from trainset.variable_training_set import load_all_variable_stars
from initial_classif import binary_classification

sizel, sizeb = np.float64(config['sizel']), np.float64(config['sizeb'])


def train_classification_region(grid, sizel, sizeb, variable_stars, index):
    
    l,b = grid[i]
    gaia = generate_gaia_training_set(l, b, sizel * 60., sizeb * 60.)
    gaia['class']='CONST'
    
    full_data = pd.concat([variable_stars, gaia], axis=0)
    
    classfier = binary_classification(full_data)
    
    with open(config['binary_output_dir'] + 'binary_%i%s.pkl'%(index,''+'_test'*config['test']), 'wb') as f:
        pickle.dump(classifier.model, f)

        
def run_loop(data, lstart, lend, bstart, bend):
    
    l_arr, b_arr = np.linspace(-10.,10.1,sizel), np.linspace(-10.,5.1,sizeb)
    l_arr, b_arr = .5*(l_arr[1:]+l_arr[:-1]), .5*(b_arr[1:]+b_arr[:-1])
    
    L,B = np.meshgrid(l_arr, b_arr)
    grid = np.vstack([L.flatten(), B.flatten()]).T
    
    pd.DataFrame({'index':np.arange(len(grid)), 'l':L.flatten(),B.flatten()}).to_pickle(
        config['binary_output_dir'] + 'grid%s.pkl'%(''+'_test'*config['test'])
    )
    
    p = Pool(32)
    p.map(partial(train_binary_classification, grid, sizel, sizeb, variable_stars),
          np.arange(len(grid)))
    p.close()
    p.join()
        
    
if __name__=="__main__":
    
    variable_stars = load_all_variable_stars()
    variable_star['class']='VAR'
    
    if config['test']:
        run_loop(-10,-10,-8.9,-8.9)
    else:
        run_loop(-10,-10,10.1,10.1)
