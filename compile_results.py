import pandas as pd
import glob
import re

results_dir = ""

files = set([file for file in glob.glob(results_dir+"*.csv.tar.gz") if ("compiled" not in file and "test" not in file)])

hpx_ind = []
for i in files:
    temp = re.findall(r'\d+', i)
    res = list(map(int, temp))
    _ind = res[0]
    hpx_ind.append(_ind)

hpx_ind = set(hpx_ind)    
    
assert len(files)==len(hpx_ind), "Not matching number of files and Healpix indices."
    
df_tot = pd.DataFrame()
for i,j in zip(files, hpx_ind):
    try:
        df = pd.read_csv(i, compression="gzip", error_bad_lines=False)
    except:
        try:
            df = pd.read_csv(i, compression="gzip", error_bad_lines=False, header=None, delim_whitespace=True)
        except:
            continue
        
    df = df.loc[df['class']!='CONST'].copy()
    df["hpx_index"] = np.int32(j)
    df_tot = pd.concat([df_tot, df])
    
df_tot_drop = df_tot.drop_duplicates(subset='sourceid')
print("Compiled: {} initial sources, {} duplicated sources.".format(len(df_tot), len(df_tot)-len(df_tot_drop)))

for i in set(df_tot['class'].values):
    _cls = df_tot.loc[df_tot['class']==i].copy()
    count = len(_cls)
    num_dups = count - len(_cls.drop_duplicates(subset='sourceid'))
    print(f"{i}: {count} sources compiled, {num_dups} duplicates.")
    
df_tot.to_csv(results_dir+"results_compiled.csv.tar.gz")