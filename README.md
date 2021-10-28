# Hierarchical classifier of virac2 catalogue into variable subtypes.

Code related to the publication Molnar, Sanders et al. (2021).

The corresponding catalogue is available at [https://people.ast.cam.ac.uk/~jls/data/vproject/vivace_catalogue.fits](https://people.ast.cam.ac.uk/~jls/data/vproject/vivace_catalogue.fits).

Dependent on interface with Hertfordshire cluster and healpixel hdf5 archives data structure. 

## Main outline is as follows:

### - Step 1 (dir = init_classif): Initial classification of VVV tile sources into candidate variable/non-variable. 

This step includes the automatic construction/training of a Random Forest classifier to determine candidate variable sources. The training set will be constructed from variability indices (relying on consecutive measurements of timeseries data). 

The main code is in run_binary_classifer.py This loops over regions of the sky, constructing the Gaia constant variable dataset from WSDB and training a binary classifier. The classifier is then pickled.

### - Step 2 (dir = fine_classif): Fine-grain classification of subdivided tile areas.

This is the second step in the hierarchical classification of virac2 sources. Based on periodic information of the source light curves, variability classes will be assigned to each source. In addition, probabilistic and goodness of fits of the periodic model will be included to sort the output into 'clean' subsets of variable sources. 

In run_variable_classifier.py the variable star catalogue is loaded, matched with non-periodic features and then periodic features are computed. The classifier is then run and saved.

### Step 3 Classifying new sources

This involves running the binary classifiers of Step 1 and the variable classifier of Step 2 sequentially. 

Here the main loop will be over the fields of step 1. We load all sources from WSDB in the region, run the appropriate binary classifier over that region, and extract the variable sources. We then have to compute the periodic features for this data and run the variable classifier. The code is in classify.py.

We have two options for the exact order of operations in the third step. Either we save the list of candidate sourceid and then run the period finding on the Hertfordshire cluster returning the periodic data. And then run the classifier locally. Or the whole of the second step is run on the Herts cluster.

### config.cfg

The configure file contains the configuration parameters for running the three steps. The 'general' section contains paths to where the input data is stored ('variable_dir' is the directory for the input variable dataset) and results are outputted. In particular, 'binary_output_dir' gives the path to the pickled binary classifiers, 'variable_output_dir' same for the second variable step and 'results_dir' the sets of classified variable stars. Setting 'test' to True runs the limited test regions. 'sizel', 'sizeb' governs the size in degrees of the fields to divide the survey into. The 'wsdb' section gives the hostname and username needed for fetching results from WSDB. The password does not need to be entered here but is instead requested when running the code.

### plots/

The majority of the plots presented in Molnar et al. (2021) are generated using the notebooks in the [plots](plots/) folder.

1. [Figure 1](plots/OnSkyPlot.ipynb)
2. [Figure 2](plots/)
3. [Figure 3](plots/)
4. [Figure 4](plots/)
5. [Figure 5](plots/ConfusionMatrix.ipynb)
6. [Figure 6](plots/ConfusionMatrix.ipynb)
7. [Figure 7](plots/ConfusionMatrix.ipynb)
8. [Figure 8](plots/ConfusionMatrix.ipynb)
9. [Figure 9](plots/ConfusionMatrix.ipynb)
10. [Figure 10](plots/ConfusionMatrix.ipynb)
11. [Figure 11](plots/ConfusionMatrix.ipynb)
12. [Figure 12](plots/ConfusionMatrix.ipynb)
13. [Figure 13](plots/)
14. [Figure 14](plots/binary_output.ipynb)
15. [Figure 15](plots/)
16. [Figure 16](plots/)
17. [Figure 17](plots/lc_gallery.ipynb)
18. [Figure 18](plots/)
19. [Figure 19](plots/OnSkyOutputPlot.ipynb)
20. [Figure 20](plots/ogle_comparison.ipynb)
21. [Figure 21](plots/Fig22_RRL.ipynb)
22. [Figure 22](plots/Fig23_EW.ipynb)
23. [Figure A1](plots/testing_fourier_optimal_regularization.ipynb)