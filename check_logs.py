import sys
import re
import numpy as np
import datetime
import glob

#try:
#    from config import configuration
#    config = configuration()
#    results_dir = config["results_dir"]
#except:
results_dir = ""

def process_log(input_file):
    with open(input_file, mode='r') as f:
        fl = f.read()
    finished_indices = {np.int64(g.group(1)):np.float64(g.group(2)) 
                        for g in re.finditer("Healpix ([0-9]*): finished, run in ([0-9]*.[0-9])", fl)}
    var_indices = {np.int64(g.group(1)):np.int64(g.group(2)) 
                        for g in re.finditer("Healpix ([0-9]*): ([0-9]*)\/[0-9]* variable", fl)}
    predicted_times = {np.int64(g.group(1)):g.group(2)
                      for g in re.finditer("Healpix ([0-9]*): .* Predicted finish time: (.*)", fl)}

    unfinished_indices = {g: predicted_times[g] 
                          for g in list(set(predicted_times.keys())-set(finished_indices.keys()))}
    
    return finished_indices, unfinished_indices, var_indices
	
files = list(set(glob.glob(results_dir+'*.log'))-set([results_dir+'summary_log.log']))

outputs = [process_log(f) for f in files]

finished_indices = {k:v for x in outputs for k, v in x[0].items()}
unfinished_indices = {k:v for x in outputs for k, v in x[1].items()}
var_indices = {k:v for x in outputs for k, v in x[2].items()}

with open(results_dir+'summary_log.log', 'w') as f:

    f.write('Current time:{0}\n\n'.format(datetime.datetime.now()))
    
    f.write('Finished: %i healpix\n'%len(finished_indices))
    f.write("Average runtime per lc: %0.4fs\n\n"%np.nanmean([finished_indices[g]/var_indices[g] for g in finished_indices.keys()]))
    [f.write('Healpix {0}: {1}s, {2}s per lc\n'.format(g,finished_indices[g],finished_indices[g]/var_indices[g])) 
		for g in finished_indices.keys()]
    f.write('\nUnfinished predicted times\n')
    [f.write('Healpix {0}: {1}\n'.format(g,unfinished_indices[g])) for g in unfinished_indices.keys()]

