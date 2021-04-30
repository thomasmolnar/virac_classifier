import sqlutilpy
from config import *
import pickle
from classify import save_cols, save_cols_types
from fine_classif.feat_extract.extract_feats import construct_final_df
import pandas as pd
import numpy as np
from interface_utils.add_stats import *
from multiprocessing import Pool

def load_data(config):
    data = pd.DataFrame(
             sqlutilpy.get('''
             select * from thomas_molnar.virac2_variable_data as t 
             left join leigh_smith.virac2_var_indices as v on t.sourceid=v.sourceid 
             left join leigh_smith.virac2_photstats as b on t.sourceid=b.sourceid
             ''',
                        **config.wsdb_kwargs)
           )
    construct_final_df(data)
    data = pct_diff(data)
    data = error_ratios(data)
    translate = {'z_model':'Z_model', 'h_scale':'H_scale', 
                 'z_scale':'Z_scale', 'y_scale':'Y_scale', 
                 'y_model':'Y_model', 'j_scale':'J_scale',
                 'hk_col':'HK_col', 'j_model':'J_model', 
                 'jk_col':'JK_col', 'h_model':'H_model'}
    data = data.rename(columns=translate)
    return data

if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    output_file = config['results_dir'] + 'variables_reclassified.csv'
    
    data = load_data(config)
    
    with open(config['variable_output_dir'] + 'variable_classifier.pkl', 'rb') as f:
            variable_classifier = pickle.load(f)

    splits = np.array_split(data.index.values, len(data)//1000)
    chunks = [data.iloc[d].reset_index(drop=True) for d in splits]
    p = Pool(32)
    rr = p.map(variable_classifier.predict, chunks)
    p.close()
    p.join()
    variable_output = pd.concat(rr,axis=0)
    
    variable_output[save_cols].astype(save_cols_types).to_csv(
        output_file, index=False)
    
