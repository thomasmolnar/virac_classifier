### Variability indices (used in initial classifer) computation from 'mag' sparse array of chronologically ordered observations for all sources

import numpy as np


def sum_squares(magarr):
    """
    Determine sum of the squares of differences in consecutive magnitude observations for each source
    ----
    input:

    - magarr: mag sparse array (N_catids, N_sourceids) with magnitude entries
    
    returns:
    
    - ssq: sum of squares array (N_sourceids)
    
    """
    
    # Initialise array for sum of the squares
    ssq = np.zeros(magarr.shape[1], dtype='f4')
    
    # Reference magnitude array (set of first catalogue entry)
    ref_mag = magarr[0,:]
    
    # Iterate through all observations
    for n in range(1, magarr.shape[0]):
        
        # Current/subsequent observation to reference array
        curr_mag = magarr[n,:]
        
        # Boolean masks
        _det0 = np.invert(np.isnan(ref_mag)) # true where a source has a mag in the ref_mag array
        _det1 = np.invert(np.isnan(curr_mag)) # true where a source has a mag in the curr_mag array 
        _det_ = np.logical_and(_det0,_det1) # true where a source has consecutive observations
        
        # Add sum of squares of consecutive measurements
        ssq[_det_] += ((curr_mag-ref_mag)**2)[_det_]
        
        # Update reference array to current array for sources with consec. obs.
        ref_mag[_det1] = curr_mag[_det1]
    
    return ssq

    
def von_neumann(magarr):
    """
    Compute von Neumann variability index for each source under consideration. 
    Defintion taken from eq.19 of https://ui.adsabs.harvard.edu/abs/2017MNRAS.464..274S/abstract
    ----
    input: 
    
    - magarr: mag sparse array (catid, sourceid) with magnitude entries
    
    returns:
    
    - output: record array with columns ("sourceid", "v_eta")
    
    """
    
    # Sum of the squares of consec. magnitudes
    ssq = sum_squares(magarr)
    
    # Nobs for each source
    det_count = np.count_nonzero(np.invert(np.isnan(magarr)), axis=0)

    # delta and sigma squared of von Neumann index
    _del = ssq / (det_count-1)
    _var = np.nansum((magarr - np.nanmean(magarr, axis=0))**2, axis=0) / (det_count-1)

    # von Neumann
    v_eta = _del/_var
    
    output = np.rec.fromarrays([uq_sourceids, vN_eta], names=["sourceid", "v_eta"])

    return output
