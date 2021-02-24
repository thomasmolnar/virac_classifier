import pandas as pd
import glob

results_dir = ""

files = [file for file in glob.glob(results_dir+"*.csv.tar.gz") if "_test" not in file]

df_tot = pd.DataFrame()
for i in files:
    df = pd.read_csv(i, compression="gzip", error_bad_lines=False)
    df = df.loc[df['class']!='CONST'].copy()
    df_tot = pd.concat([df_tot, df])

df_tot.to_csv(results_dir+"results_compiled.csv.tar.gz")    
    
for i in set(df_tot['class'].values):
    count = len(df_tot.loc[df_tot['class']==i])
    print(f"{count} sources of type {i} compiled")