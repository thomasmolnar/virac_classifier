from config import *
import numpy as np
import pandas as pd
import pickle
import sqlutilpy as sqlutil
from interface_utils.light_curve_loader import lightcurve_loader
from initial_classif.trainset.variable_training_sets import load_all_variable_stars, load_mira_sample
from initial_classif.trainset.gaia_extraction import generate_gaia_training_set_random
from fine_classif.classifier.classifier import variable_classification
from fine_classif.feat_extract.extract_feats import extract_per_feats


def get_periodic_features_var(data, config, serial=True):
    """
    Periodic feature extracter - to be used for variable classification
    """
    variable_output_dir = str(config['variable_output_dir'])
    
    if os.path.isfile(variable_output_dir+'variable_features_total{0}.pkl'.format(''+'_test'*int(config['test']))):
        with open(variable_output_dir+'variable_features_total{0}.pkl'.format(''+'_test'*int(config['test'])), 'rb') as f:
            total_features = pickle.load(f)
        
        return total_features
    
    print("Loading Timeseries data...")
    lcs = pd.DataFrame(sqlutil.get('''select sourceid, 
                                unnest(mjdobs) as mjdobs,
                                unnest(mag) as mag,
                                unnest(emag) as emag,
                                unnest(filterid) as filterid,
                                unnest(chi) as chi,
                                unnest(ast_res_chisq) as ast_res_chisq,
                                unnest(ambiguous_match) as ambiguous_match  
                                from leigh_smith.virac2_ts_tmolnar_train_zyjhk''' + ' limit 100000000'*int(config['test']),
                     **config.wsdb_kwargs))
    lcsE = pd.DataFrame(sqlutil.get('''select sourceid, 
                                unnest(mjdobs) as mjdobs,
                                unnest(mag) as mag,
                                unnest(emag) as emag,
                                unnest(filterid) as filterid,
                                unnest(chi) as chi,
                                unnest(ast_res_chisq) as ast_res_chisq,
                                unnest(ambiguous_match) as ambiguous_match  
                                from leigh_smith.virac2_ts_tmolnar_variables_extended''' + ' limit 100000000'*int(config['test']),
                     **config.wsdb_kwargs))

    lcs = pd.concat([lcs, lcsE], axis=0, sort=False)

    print("---Timeseries data loaded.")
    print(len(data))
 
    lcs = pd.merge(lcs, data[['sourceid']], how='inner', on='sourceid').reset_index(drop=True)
    data = pd.merge(data, pd.DataFrame({'sourceid':np.unique(lcs['sourceid'].values)}), 
                    how='inner', on='sourceid').reset_index(drop=True)
    
    lcs = lcs.sort_values(by=['sourceid', 'mjdobs']).reset_index(drop=True)
    uniq_ids, indices, inv_ids = np.unique(lcs['sourceid'], return_index=True, return_inverse=True)
    indices = indices[1:]
    
    data = data.sort_values(by='sourceid').reset_index(drop=True)

    # Add sky position of sources to be passed into periodic computation
    print(len(data), len(uniq_ids))
    assert len(data)==len(uniq_ids)
    assert all(data['sourceid'].values==uniq_ids)
        
    ras_full = data['ra'].values[inv_ids]
    decs_full = data['dec'].values[inv_ids]
    
    ts_dict = {c: np.split(lcs[c], indices) for c in list(lcs.keys())}
    ra_dict = dict(ra=np.split(ras_full, indices))
    dec_dict = dict(dec=np.split(decs_full, indices))

    datadict = {**ts_dict, **ra_dict, **dec_dict}
    lightcurves = [
        pd.DataFrame(dict(list(zip(datadict, t))))
        for t in zip(*list(datadict.values()))
    ]
    
    # Universal frequency grid conditions 
    ls_kwargs = dict(maximum_frequency=np.float64(config['ls_max_freq']),
                     minimum_frequency=np.float64(config['ls_min_freq']))
    method_kwargs = dict(irreg=True)
    
    if int(config['test'])==1:
        lightcurves = lightcurves[:2000]
    
    #Extract features
    split = np.array_split(np.arange(len(lightcurves)), 10 - 8 * int(config['test']))
    total_features = pd.DataFrame()
    for batch_numb, indices in enumerate(split):
        print('Running batch %i'%batch_numb)
        # Slicing on data dataframe not necessary as cross-matching on sourceid is done
        batch_features = extract_per_feats(lightcurves[np.min(indices):np.max(indices)+1],
                                           data, ls_kwargs, method_kwargs,
                                           config, serial=serial)
        batch_features.to_pickle(variable_output_dir+'variable_features_batch{0}{1}.pkl'.format(batch_numb,''+'_test'*int(config['test'])))
        total_features = pd.concat([total_features, batch_features])
    
    total_features.to_pickle(variable_output_dir+'variable_features_total{0}.pkl'.format(''+'_test'*int(config['test'])))
    
    return total_features


def get_periodic_features_mira_sample(config, serial=False):
    
    variable_output_dir = str(config['variable_output_dir'])
    
    if os.path.isfile(variable_output_dir+'variable_features_mira.pkl'):
        with open(variable_output_dir+'variable_features_mira.pkl', 'rb') as f:
            total_features = pickle.load(f)
        return total_features
    
    dsets = load_mira_sample(config)
    
    mira_ids = ','.join(np.str(s) for s in dsets['sourceid'].values)

    lcs = pd.DataFrame(sqlutil.get('''select sourceid, 
                                unnest(mjdobs) as mjdobs,
                                unnest(mag) as mag,
                                unnest(emag) as emag,
                                unnest(chi) as chi,
                                unnest(ast_res_chisq) as ast_res_chisq,
                                unnest(filterid) as filterid,
                                unnest(ambiguous_match) as ambiguous_match  
                                from leigh_smith.virac2_ts_jason_gcen
                                where sourceid in ({0})'''.format(mira_ids,),
                     **config.wsdb_kwargs))
    
    dsets['var_class']='MIRA'
    
    dsets = dsets.sort_values(by='sourceid').reset_index(drop=True)
    lcs = lcs.sort_values(by='sourceid').reset_index(drop=True)
    uniq_ids, indices, inv_ids = np.unique(lcs['sourceid'], return_index=True, return_inverse=True)
    indices = indices[1:]
    
    ras_full = dsets['ra'].values[inv_ids]
    decs_full = dsets['dec'].values[inv_ids]
    
    ts_dict = {c: np.split(lcs[c], indices) for c in list(lcs.keys())}
    ra_dict = dict(ra=np.split(ras_full, indices))
    dec_dict = dict(dec=np.split(decs_full, indices))

    datadict = {**ts_dict, **ra_dict, **dec_dict}
    lightcurves = [
        pd.DataFrame(dict(list(zip(datadict, t))))
        for t in zip(*list(datadict.values()))
    ]
    
    print(len(dsets), len(lightcurves))
    
    # Universal frequency grid conditions 
    ls_kwargs = dict(maximum_frequency=np.float64(config['ls_max_freq']),
                     minimum_frequency=np.float64(config['ls_min_freq']))
    method_kwargs = dict(irreg=True)
    
    #Extract features
    total_features = extract_per_feats(lightcurves,
                                       dsets, ls_kwargs, method_kwargs,
                                       config, serial=serial)
    
    total_features.to_pickle(variable_output_dir+'variable_features_mira.pkl')
    
    return total_features
    
col32_save = ['amp_0', 'amp_1', 'amp_2', 'amp_3', 
              'amp_double_0', 'amp_double_1', 'amp_double_2', 'amp_double_3', 
              'amplitude', 'beyondfrac', 'delta_loglik', 'log10_fap',
              'ls_period','lsq_period', 'max_pow', 'max_phase_lag', 'pow_mean_disp', 'phase_lag_mean',
              'phi_0','phi_1','phi_2','phi_3',
              'phi_double_0','phi_double_1','phi_double_2','phi_double_3',           
              'peak_ratio_model', 'peak_ratio_data',
              'JK_col','HK_col','prob',
              'Z_scale', 'Z_model',
              'Y_scale', 'Y_model',
              'J_scale', 'J_model',
              'H_scale', 'H_model',]

save_cols_types = dict(zip(col32_save,[np.float32]*len(col32_save)))

def generate_secondstage_training(config):
    variable_stars = load_all_variable_stars(config)
    variable_stars['gaia_sourceid']=-9999
    variable_stars['gaia_sourceid'] = variable_stars['gaia_sourceid'].astype(np.int64)
    
    constant_data = generate_gaia_training_set_random(len(variable_stars)//10, config,
                                                      np.float64(config['gaia_percentile']),
                                                      600000)
    
    constant_data['var_class']='CONST'
    
    trainset = pd.concat([variable_stars, constant_data], axis=0, sort=False).reset_index(drop=True)
    
    print(np.count_nonzero(trainset['sourceid'].duplicated()), 'duplicate sources')
    
    trainset = trainset[~trainset['sourceid'].duplicated()].reset_index(drop=True)
    
    return trainset

def generate_periodic_features(config):
    
    print("Loading trainset...")
    
    trainset = generate_secondstage_training(config)
    
    variable_output_dir = str(config['variable_output_dir'])

    if ~os.path.isfile(variable_output_dir+'variable_training_set_edr3_sourceid.csv'):
        trainset.to_csv(variable_output_dir + '/variable_training_set_edr3_sourceid.csv',
                        index=False)
    
    print("---Trainset loaded - {} stars".format(len(trainset)))
    print("Loading periodic features...")
    
    features = get_periodic_features_var(trainset, config, serial=False)
    
    mira = get_periodic_features_mira_sample(config)
    
    features = pd.concat([features, mira], axis=0, sort=False).reset_index(drop=True)
    
    return features

def combine_var_class(v):
    
    for ii in ['MIRA', 'OSARG', 'SRV']:
        v.loc[v['var_class']==ii, 'var_class'] = 'LPV'
    for ii in ['RRc', 'RRd']:
        v.loc[v['var_class']==ii, 'var_class'] = 'RRcd'
        
    return v


def cm_decaps(data):
    ra = data['ra'].values
    dec = data['dec'].values
    decaps = pd.DataFrame(
        sqlutil.local_join(
            """
		select 
        1.0857362048*stdev_g/(mean_g+1e-20) as decaps_g_amp, 
        1.0857362048*stdev_r/(mean_r+1e-20) as decaps_r_amp, 
        1.0857362048*stdev_i/(mean_i+1e-20) as decaps_i_amp, 
        1.0857362048*stdev_z/(mean_z+1e-20) as decaps_z_amp, 
        nmag_ok_g,
        q3c_dist(m.ra,m.dec,tt.ra,tt.dec)*3600. as q3c_dist_decaps from mytable as m
		left join lateral (select * from decaps_dr1.main as s
		where q3c_join(m.ra, m.dec,s.ra,s.dec,0.4/3600.) 
		order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**config.wsdb_kwargs))
    for ii in ['g','r','i','z']:
        fltr = (decaps['decaps_%s_amp'%ii]==0.)|np.isnan(decaps['decaps_%s_amp'%ii])|np.isinf(decaps['decaps_%s_amp'%ii])
        decaps['log10_decaps_%s_amp'%ii]=np.nan
        decaps.loc[~fltr, 'log10_decaps_%s_amp'%ii]=np.log10(decaps['decaps_%s_amp'%ii][~fltr])
        del decaps['decaps_%s_amp'%ii]
        
    nsc2 = pd.DataFrame(
        sqlutil.local_join(
            """
		select 
        rmsvar as nsc2_rmsvar, madvar as nsc2_madvar, 
        iqrvar as nsc2_iqrvar, etavar as nsc2_etavar,
        jvar as nsc2_jvar, kvar as nsc2_kvar, 
        chivar as nsc2_chivar, romsvar as nsc2_romsvar, nsigvar as nsc2_nsigvar,
        q3c_dist(m.ra,m.dec,tt.ra,tt.dec)*3600. as q3c_dist_nsc2 from mytable as m
		left join lateral (select * from nsc_dr2.object as s
		where q3c_join(m.ra, m.dec,s.ra,s.dec,0.4/3600.) 
		order by q3c_dist(m.ra,m.dec,s.ra,s.dec) asc limit 1)
		as tt on  true  order by xid """,
            'mytable', (ra, dec, np.arange(len(dec))), ('ra', 'dec', 'xid'),**config.wsdb_kwargs))
    
    decaps = pd.concat([decaps, nsc2], axis=1, sort=False).reset_index(drop=True)
    
    return decaps

if __name__=="__main__":
    
    config = configuration()
    config.request_password()
    
    features = generate_periodic_features(config)
    features['log10_fap'] = features['log10_fap_ls']
    del features['log10_fap_ls']

    features= features[~features['error']].reset_index(drop=True)
    
    features = features[(features['var_class']=='CONST')|(features['log10_fap']<np.float64(config['log10_fap']))].reset_index(drop=True)
    
    features = features[~(features['var_class']=='DSCT')].reset_index(drop=True)
    
    for ii in np.unique(features['var_class']):
        print(ii, np.count_nonzero(features['var_class']==ii))
    
    features = combine_var_class(features)
    
    #exit()    
    #decaps_features = cm_decaps(features)
    #pd.concat([features[['var_class']], decaps_features], axis=1, sort=False).reset_index(drop=True).to_pickle(
    #    config['variable_output_dir'] + 'decaps_dataset%s.pkl'%(''+'_test'*int(config['test'])))
    #features = pd.concat([features, decaps_features], axis=1, sort=False).reset_index(drop=True)
    
    classifier = variable_classification(features, config)
    
    classifier.training_set.astype(save_cols_types).to_pickle(
        config['variable_output_dir'] + 'variable_training_set%s.pkl'%('_%i'%np.int64(config['log10_fap'])+'_test'*int(config['test'])))
    del classifier.training_set
    
    with open(config['variable_output_dir'] 
              + 'variable_classifier%s.pkl'%('_%i'%np.int64(config['log10_fap'])+'_test'*int(config['test'])), 'wb') as f:
        pickle.dump(classifier, f)
