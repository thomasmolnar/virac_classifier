from run_variable_classifier import *
import h5py

config = configuration()
config.request_password()

file_path = '/data/jls/virac/'

with h5py.File(file_path+'n512_2318830.hdf5', 'r') as f:
    randints = np.sort(np.random.randint(0,55000,20))
s=f['sourceList']['sourceid'][:][randints]
ra=f['sourceList']['ra'][:][randints]
dec=f['sourceList']['dec'][:][randints]

data = pd.DataFrame()
data['sourceid'] = s
data['ra'] = ra
data['dec'] = dec

test_feats = get_periodic_features(data, config)

test_feats.to_pickle('test_var_out.pkl')
