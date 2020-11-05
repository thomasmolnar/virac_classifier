import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy_healpix import *

from healpy import ring2nest


class extinction_map_healpix(object):
    """
    Extinction map class to be queried in order to get JK/HK colour excesses
    based on galactic coords or source
    
    """
    def __init__(self, pathtofile, version='v2_NEW_JK_PHOTSTATS_FULL'):
        
        self.version = version
        self.ff = pd.read_csv(pathtofile+'_%s.csv'%self.version)
        self.ff = self.ff[self.ff['level']>0].reset_index(drop=True)
        maxlevel = np.max(self.ff['level'])
        self.maxnside = level_to_nside(maxlevel)
        level , ipix = uniq_to_level_ipix(self.ff['uniq'])
        self.index_nest = ring2nest(level_to_nside(level), ipix)
        self.index_nest = self.index_nest * (2**(maxlevel - level))**2
        self.sorter = np.argsort(self.index_nest)

    def query(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['e%s'%self.version[-2:].lower()].values[i]
    
    def query_spread(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['sigma_e%s'%self.version[-2:].lower()].values[i]

    def resolution(self, l, b):
        match_ipix = lonlat_to_healpix(l*u.deg, b*u.deg, self.maxnside, order='nested')
        i = self.sorter[np.searchsorted(self.index_nest, match_ipix, side='right', sorter=self.sorter) - 1]
        return self.ff['resolution'].values[i]
    

    
