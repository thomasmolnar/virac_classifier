### Code to build sparse array where each subarray represents the a catalogue (single observation) for each sourceid. Therefore, sources that are observed will have a magnitude entry and sources that aren't will have a nan entry. 

import numpy as np


def build_mag(data):
    """
    Returns 'mag' array for all sources chronologically along the whole observational period
    ---
    input:
    
    - data: record numpy array with columns (sourceid, mjdobs, mag, catid)
    
    return:
    
    - magarr: mag array described above
    
    """
    
    # Find unique catid's and indexed chronological order of observations
    uq_catids, idx, invc = np.unique(np.array(data["catid"]), return_index=True,
                                     return_inverse=True)
    order = np.argsort(mjdobs)
    ordered_catids = order[invc]
    
    # Find unique source IDs
    uq_sourceids, invs = np.unique(data["sourceid"], return_inverse=True)
    ordered_sourceids = np.arange(uq_sourceids.size)[invs]
    
    # Build empty magarr array
    source_count, cat_count = uq_sourceids.size, uq_catids.size
    magarr = np.full((cat_count, source_count), np.nan, dtype=np.float32)
    
    # Fill the magarr array where there is data
    magarr[ordered_catids,ordered_sourceids] = data["mag"]   
    
    # Ordering and observational time checks
    
    _mjdobs = np.full((cat_count, source_count), np.nan, dtype=np.float64)
    _mjdobs[ordered_catids,ordered_sourceids] = data["mjdobs"]

    singular_mjdobs = np.array([np.unique(row[~np.isnan(row)]).size for row in _mjdobs])==1
    if not all(singular_mjdobs):
        raise RuntimeError("_mjdobs are different within each catalogue row")
    
    ordered_mjdobs = [all(np.diff(column[~np.isnan(column)])>0) for column in _mjdobs.T]
    if not all(ordered_mjdobs):
        raise RuntimeError("_mjdobs for each source are not ordered")
        
    return magarr