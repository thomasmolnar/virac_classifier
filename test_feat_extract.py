from run_variable_classifier import *
import h5py

config = configuration()
config.request_password()

file_path = '/data/jls/virac/'

data = pd.DataFrame()
with h5py.File(file_path+'n512_2318830.hdf5', 'r') as f:
    randints = np.sort(np.random.randint(0,55000,50))
    for s in f['sourceList'].keys():
        data[s] = f['sourceList'][s][:][randints]

data = data[data['ks_n_detections']>20].reset_index(drop=True)

lightcurve_loader = lightcurve_loader()
test_feats = get_periodic_features(data, lightcurve_loader, config)

test_feats.to_pickle('test_var_out.pkl')
