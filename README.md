# Hierarchical classifier of virac2 catalogue into variable subtypes.

Dependent on interface with Hertforshire cluster and healpixel hdf5 archives data structure. 

## Main outline is as follows:

### - Step 1: Initial classification of VVV tile sources into candidate variable/non-variable. 

This step includes the automatic construction/training of a Random Forest classifier to determine candidate variable sources. The training set will be constructed from variability indices (relying on consecutive measurements of timeseries data). 

### - Step 2: Fine-grain classification of subdivided tile areas.

This is the second step in the hierarchical classification of virac2 sources. Based on periodic information of the source light curves, variability classes will be assigned to each source. In addition, probabilistic and goodness of fits of the periodic model will be included to sort the output into 'clean' subsets of vairble sources. 
