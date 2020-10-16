from config import *
from run_binary_classifier import sizel, sizeb, output_dir as binary_output_dir
from run_variable_classifier import output_dir as variable_output_dir
from wsdb_utils.wsdb_cred import wsdb_kwargs


def grab_virac_with_stats(l,b,sizel,sizeb):
    
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
            select t.*, s.*
            from leigh_smith.virac2 as t
            inner join leigh_smith.virac2_var_indices_tmp as s on s.sourceid=t.sourceid
            where %s and duplicate=0 and astfit_params=5"""%poly_string, 
                                    password=config['password'],**wsdb_kwargs))
    return data


def classify_region(grid, variable_classifier, index):
    
    with open(config['binary_output_dir'] + 'binary_%i%s.pkl'%(index,''+'_test'*test), 'rb') as f:
        binary_classifier = pickle.load(f)
    
    input_data = grab_virac_with_stats(grid['l'], grid['b'], sizel, sizeb)
    classes = binary_classifier.pred(input_data[binary_classifier.data_cols])
    probability = binary_classifier.pred(input_data[binary_classifier.data_cols])
    
    variable_candidates = input_data[classes=='VAR'].reset_index(drop=True)
    
    variable_candidates = add_periodic_features(variable_candidates)
    
    variable_classes = variable_classifier.pred(variable_candidates[variable_classifier.data_cols])
    
    results = pd.DataFrame({'class':variable_classes, 'period':variable_candidates['period']})
    results.to_pickle(config['results_dir'] + 'results_%i%s.pkl'%(index,''+'_test'*test))
    
    
if __name__=="__main__":
    
    grid = pd.read_pickle('grid%s.pkl'%(''+'_test'*test))
    
    with open(variable_output_dir + 'variable%s.pkl'%(index,''+'_test'*test), 'rb') as f:
        variable_classifier = pickle.load(f)
    
    p = Pool(32)
    p.map(partial(classify_region, grid, variable_classifier),
          np.arange(len(grid)))
    p.close()
    p.join()