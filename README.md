# Hierarchical classifier of virac2 catalogue into variable subtypes.

Dependent on interface with Hertforshire cluster and healpixel hdf5 archives data structure. 

## Main outline is as follows:

### - Step 1 (dir = init_classif): Initial classification of VVV tile sources into candidate variable/non-variable. 

This step includes the automatic construction/training of a Random Forest classifier to determine candidate variable sources. The training set will be constructed from variability indices (relying on consecutive measurements of timeseries data). 

The main code is in initial_classif/run.py This loops over regions of the sky, constructing the Gaia constant variable dataset from WSDB and training a binary classifier. The classifier is then pickled.

### - Step 2 (dir = fine_classif): Fine-grain classification of subdivided tile areas.

This is the second step in the hierarchical classification of virac2 sources. Based on periodic information of the source light curves, variability classes will be assigned to each source. In addition, probabilistic and goodness of fits of the periodic model will be included to sort the output into 'clean' subsets of variable sources. 

Here the main loop will be over the fields of step 1. We load all sources from WSDB in the region, run the appropriate binary classifier over that region, and extract the variable sources. Now we have two options. Either we save this list of sourceid and then run the period finding on the Hertfordshire cluster returning the periodic data. And then run the classifier locally. Or the whole of the second step is run on the Herts cluster.

