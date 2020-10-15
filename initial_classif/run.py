from functools import partial
from trainset.gaia_extraction import generate_gaia_training_set
from trainset.variable_training_set import load_all_variable_stars

output_dir = '/data/jls/binary_classification/%0.4f_%0.4f'


def train_classification_region(grid, sizel, sizeb, variable_stars, index):
    
    l,b = grid[i]
    gaia = generate_gaia_training_set(l, b, sizel * 60., sizeb * 60.)
    gaia['class']='CONST'
    
    full_data = pd.concat([variable_stars, gaia], axis=0)
    
    classfier = binary_classification(full_data)
    
    with open(output_dir + 'binary_%i.pkl'%index, 'wb') as f:
        pickle.dump(classifier.model, f)
    
    
if __name__=="__main__":
    
    variable_stars = load_all_variable_stars()
    variable_star['class']='VAR'
    
    sizel, sizeb = 1., 1.
    
    l_arr, b_arr = np.linspace(-10.,10.1,sizel), np.linspace(-10.,5.1,sizeb)
    l_arr, b_arr = .5*(l_arr[1:]+l_arr[:-1]), .5*(b_arr[1:]+b_arr[:-1])
    
    L,B = np.meshgrid(l_arr, b_arr)
    grid = np.vstack([L.flatten(), B.flatten()]).T
    
    p = Pool(32)
    p.map(partial(train_binary_classification, grid, sizel, sizeb, variable_stars),
          np.arange(len(grid)))
    p.close()
    p.join()