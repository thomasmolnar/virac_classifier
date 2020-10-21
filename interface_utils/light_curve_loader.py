import os
import numpy as np
import h5py
from healpy import ring2nest
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy_healpix import healpix_to_lonlat, level_ipix_to_uniq, nside_to_level


MAXNSIDE=1024

file_path = '/data/jls/virac/'
thispath=os.path.dirname(os.path.abspath(__file__))+'/'

import time

class lightcurve_loader(object):
    """
        Class to handle the loading of light curve data from a series of hdf5 files.
        
        The hdf files correspond to a Healpix (ring indexed) of nside = (1024, 512, 256)
        depending upon the source density. The file names are n{nside}_{healpix_index}.hdf5
        and are stored as a list in ns_hpx.dat.
        
        We use the multi-order coverage technique to find the healpix file that a sourceid will
        lie in (the integer part of sourceid/1e6 is ringed healpix of nside 1024)
        (see here https://emfollow.docs.ligo.org/userguide/tutorial/multiorder_skymaps.html)
       
        This class is most useful when we require a set of light curves dispersed across the VVV
        survey. If all light curves in a certain region are required, Leigh's methods for building
        a 2d array (catid, sourceid) are probably superior.
       
    """
    def __init__(self, input_file='ns_hpx.dat'):
        """
            Input file is the list of healpix into which the data is split (columns nside, index)
        """
        
        self.healpix_grid = pd.read_csv(thispath + 'ns_hpx.dat', names=['nside','hpx'])
        assert np.all(self.healpix_grid['nside']<=MAXNSIDE)
        ## The call code sorts on unique hpx -- if resolution is such that hpx are duplicated things will break
        assert len(self.healpix_grid)==np.count_nonzero(np.unique(self.healpix_grid['hpx']))
            
        ## Find galactic coordinates
        c=SkyCoord(*healpix_to_lonlat(self.healpix_grid['hpx'],self.healpix_grid['nside']),
                   frame='icrs').transform_to('galactic')
        self.healpix_grid['l'], self.healpix_grid['b'] = c.l.deg, c.b.deg
        
        self.max_level = nside_to_level(MAXNSIDE)
        self.healpix_grid['index']= (2**(self.max_level - nside_to_level(self.healpix_grid['nside'])))**2 * \
                                            ring2nest(self.healpix_grid['nside'],self.healpix_grid['hpx'])
        
        self.healpix_grid.sort_values(by='index', inplace=True)
        self.healpix_grid.reset_index(drop=True,inplace=True)
        
        
    def __call__(self, sourceid_input):
        """
            sourceid: an array of VIRAC source ids
            
            returns an astropy Table containing light curves for all sources (not necessarily
            in the same order as the input sourceid list)
            
            We first find the healpix the sources are in and then split into groups based on this.
            For each group, we load the appropriate light curves for the sourceids in each data file
            and complement with MJD and filter information. We then convert to pandas DataFrame and
            concat the final set of data frames from each file.
        """
        
        sourceid = np.unique(np.atleast_1d(sourceid_input))
        
        # Find healpix of source
        r_index = np.searchsorted(self.healpix_grid['index'], 
                                  ring2nest(MAXNSIDE,np.int64(sourceid/1000000)), 
                                  side='right') - 1
        indices = self.healpix_grid['hpx'].values[r_index]
        nside = self.healpix_grid['nside'].values[r_index]
        
        # Group sources based on file
        sorted_ = np.argsort(indices)
        uniq_files, uniq_indx = np.unique(indices[sorted_],return_index=True)
        uniq_bunches = np.split(sourceid[sorted_], uniq_indx[1:])
        
        def get_data_per_file(hpx, ns, sources):
            if not os.path.isfile(file_path + 'n%i_%i.hdf5' % (ns, hpx)):
                return Table([])
            with h5py.File(file_path + 'n%i_%i.hdf5' % (ns, hpx), "r") as f:
                
                ci = f["catIndex"]
                sourceids = f['sourceList']['sourceid'][:]
                
                # Get index of required entries (sorted for efficiency)
                sorter = np.argsort(sourceids)
                required_index = sorter[np.searchsorted(sourceids, sources, sorter=sorter)]
                sorter = np.argsort(required_index)
                required_index = required_index[sorter]
                
                ## These two lines take the most time
                counts = np.sum([f['sourceList']["%s_n_detections" % band.lower()][:][required_index]
                                 for band in ["Z","Y","J","H","Ks"]], axis=0)
                cols = ["catindexid","hfad_mag", "hfad_emag", "ambiguous_match", "ast_res_chisq", "chi"]

                ## This line takes the most time -- it is cheaper to select all elements with [:] and slice
                ## if the number of light curves required >~1000 else cheaper to not select all
                if len(required_index)>1000:
                    data = [np.concatenate(f["timeSeries"][col][:][required_index]) for col in cols]
                else:
                    data = [np.concatenate(f["timeSeries"][col][:][required_index]) for col in cols]
                # Insert sourceids and ra,dec
                cols.insert(0, "sourceid")
                data.insert(0, np.repeat(sources[sorter], counts))
                
                cols.insert(0, "sourceid")
                data.insert(0, np.repeat(sources[sorter], counts))
                
                # Now fill in the mjdobs and filter entries
                sorter = np.argsort(ci['catindexid'])
                indices = sorter[np.searchsorted(ci['catindexid'], data[1], sorter=sorter)]

                for fld in "mjdobs", "filter":
                    cols.insert(2, fld)
                    data.insert(2, (ci[fld][:])[indices])
                
                del data[1]
                del cols[1]
                df = Table(data, names=cols)
                df = df[df['filter']=='Ks']
                del df['filter']
                
                ## Correction from  MJD to HJD needed here
#                 coordinate = SkyCoord(ra=data['ra'] * u.deg,
#                           dec=data['dec'] * u.deg,
#                           frame='icrs')
#                 times = time.Time(data['mjdobs'][0],
#                       format='mjd',
#                       scale='utc',
#                       location=paranal)
#                 data['HJD'] = data['mjdobs'] + 2400000.5 + times.light_travel_time(
#                     coordinate).to(u.day).value
    
                 return df
        
        # Loop through files
        df = vstack([get_data_per_file(hpx,ns,sources) for hpx, ns, sources 
                     in zip(uniq_files, nside[sorted_][uniq_indx], uniq_bunches)])
        if len(df):
            df.rename_columns(["hfad_mag", "hfad_emag"],["mag","error"])
        
        return df

def split_lcs(data):
    """
    Split Astropy Table of lightcurves into list of pandas for each light curve
    (ordered sourceid column not needed)
    
    """
    
    ll = lightcurve_loader()
    lc = ll(data['sourceid'].values)
    
    #group obs based on sourceid
    lc_by_id = lc.group_by('sourceid')
    
    #split Table
    lc_df = []
    for key, group in zip(lc_by_id.groups.keys, lc_by_id.groups):
        #additional check on number of epochs 
        count = len(group)
        group_df = pd.DataFrame(group.as_array())
        if count>=20:
            lc_df.append(group_df)
        else:
            pass
            
    return lc_df    
    
def run_tests():
    test_mag = np.array([13.291, 13.272, 13.254, 13.266, 13.252, 13.263, 12.878, 13.238,
       13.281, 13.288, 13.26 , 13.237, 13.288, 13.318, 13.331, 13.312,
       12.959, 13.099, 13.268, 13.258, 13.256, 13.281, 13.277, 13.246,
       13.272, 13.329, 13.27 , 13.265, 13.278, 13.284, 13.282, 13.299,
       13.265, 13.278, 13.168, 13.271, 13.273, 13.292, 13.294, 13.215,
       13.292, 13.31 , 13.315, 13.307, 13.346, 13.295, 13.054, 13.227,
       13.284, 13.308, 13.288, 13.295, 12.988, 13.249, 13.321, 13.325,
       13.352, 13.38 , 13.303, 13.364, 13.335, 13.392, 13.348, 13.356,
       13.281, 13.293, 13.135, 13.268, 13.232, 13.251, 13.232, 13.247,
       13.207, 13.219, 13.218, 13.236, 13.23 , 13.231, 13.236, 13.253,
       13.117, 13.134, 13.233, 13.259, 13.228, 13.244, 13.247, 13.241,
       13.202, 13.236, 13.222, 13.227, 13.163, 13.122, 13.262, 13.296,
       13.292, 13.282, 13.269, 13.307, 13.279, 13.301, 13.312, 13.279,
       13.309, 13.282, 13.184, 13.181, 13.177, 13.186, 13.317, 13.291,
       13.316, 13.299, 13.329, 13.315, 13.378, 13.322, 13.342, 13.312,
       13.262, 13.314, 13.321, 13.324, 13.272, 13.289, 13.246, 13.281,
       13.333, 13.333, 12.891, 13.162, 12.951, 13.224, 13.307, 12.986,
       13.157, 12.878, 13.051, 13.242, 13.17 , 13.237, 13.251, 13.246,
       13.259, 13.233, 13.278, 13.159, 12.993, 13.264, 13.284, 13.243,
       13.256, 13.235, 13.241, 13.079, 13.11 , 13.299, 13.318, 13.333,
       13.354, 13.279, 13.281, 13.283, 13.274, 13.291, 13.262, 13.3  ,
       13.29 , 13.293, 13.286, 13.29 , 13.316, 13.314, 13.322, 13.313,
       13.314, 13.071, 13.054, 13.327, 13.33 , 13.3  , 13.297, 13.3  ,
       13.301, 13.368, 13.379, 13.116, 13.046, 13.372, 13.384, 13.373,
       13.358, 13.406, 13.392, 13.247, 13.129, 13.053, 13.051, 12.971,
       13.385, 13.338, 13.346, 13.323, 13.363, 13.805, 13.074, 13.24 ,
       13.386, 13.422])
    
    ll = lightcurve_loader()
    np.random.seed(42)
    with h5py.File(file_path+'n512_2318830.hdf5', 'r') as f:
        s=f['sourceList']['sourceid'][:][np.sort(np.random.randint(0,55000,10))]
    tt = time.time()
    rslt=ll(s)
    print((time.time()-tt)*1000., 'ms')
    
    ## Testing against expected light curve
    assert(np.abs(np.sum(rslt[rslt['sourceid']==s[4]]['mag']-test_mag))<1e-2)
    print((time.time()-tt)*1000., 'ms')
    
    # Passing a scalar
    assert(np.all(ll(s[4])==rslt[rslt['sourceid']==s[4]]))
    print((time.time()-tt)*1000., 'ms')

    # Passing a one-element array
    assert(np.all(ll(np.array([s[4]]))==rslt[rslt['sourceid']==s[4]]))
    print((time.time()-tt)*1000., 'ms')

    # Passing indices not sorted
    rl=ll(np.array([s[5],s[4]]))
    assert(np.all(rl[rl['sourceid']==s[5]]==rslt[rslt['sourceid']==s[5]]))
    print((time.time()-tt)*1000., 'ms')
    assert(np.all(rl[rl['sourceid']==s[4]]==rslt[rslt['sourceid']==s[4]]))
    print((time.time()-tt)*1000., 'ms')

    # Sourceid not in
    assert(len(ll([1]))==0)
    print((time.time()-tt)*1000., 'ms')
    assert(len(ll(1))==0)
    print((time.time()-tt)*1000., 'ms')
    
    # Sourceid not in and source in
    rl=ll(np.array([s[4],1]))
    assert(np.all(rl[rl['sourceid']==s[4]]==rslt[rslt['sourceid']==s[4]]))
    print((time.time()-tt)*1000., 'ms')
    rl=ll(np.array([1,s[4]]))
    assert(np.all(rl[rl['sourceid']==s[4]]==rslt[rslt['sourceid']==s[4]]))
    print((time.time()-tt)*1000., 'ms')

    # Sourceid not in and two source in
    rl=ll(np.array([s[5],1,s[4]]))
    assert(np.all(rl[rl['sourceid']==s[4]]==rslt[rslt['sourceid']==s[4]]))
    print((time.time()-tt)*1000., 'ms')
    
    ## Big chunk
    with h5py.File(file_path+'n512_2318830.hdf5', 'r') as f:
        s=f['sourceList']['sourceid'][:][np.sort(np.random.randint(0,55000,24000))]
    rslt=ll(s)
    print((time.time()-tt)*1000., 'ms')
    return rslt
    
if __name__=="__main__":
    run_tests()