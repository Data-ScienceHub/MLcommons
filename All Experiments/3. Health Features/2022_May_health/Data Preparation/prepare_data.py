# %% [markdown]
# # Import libraries

# %%
import pandas as pd, os
import sys
sys.path.append( '../TFT-pytorch')

# %%
from dataclasses import dataclass
from Class.DataMerger import DataMerger

@dataclass
class args:
    # folder where the cleaned feature file are at
    dataPath = '../dataset_raw/CovidMay17-2022'
    supportPath = '../dataset_raw/Support files'
    configPath = './health.json'
    cachePath = None # '../2022_May/Total.csv'

    # choose this carefully
    outputPath = '2022_May_health/'

# %%
# create output path if it doesn't exist
if not os.path.exists(args.outputPath):
    print(f'Creating output directory {args.outputPath}')
    os.makedirs(args.outputPath, exist_ok=True)

import json

# load config file
with open(args.configPath) as inputFile:
    config = json.load(inputFile)
    print(f'Config file loaded from {args.configPath}')
    inputFile.close()

# %% [markdown]
# # Data merger

# %% [markdown]
# ## Total features

# %%
# get merger class
dataMerger = DataMerger(config, args.dataPath, args.supportPath)

# %%
# if you have already created the total df one, and now just want to 
# reuse it to create different population or rurality cut
if args.cachePath:
    total_df = pd.read_csv(args.cachePath)
else:
    total_df = dataMerger.get_all_features()
    
    output_path_total = os.path.join(args.outputPath, 'Total.csv') 
    print(f'Writing total data to {output_path_total}\n')

    # rounding up to reduce the file size
    total_df.round(4).to_csv(output_path_total, index=False)

# %% [markdown]
# ## Population cut

# %%
# you can define 'Population cut' in 'data'->'support'
# this means how many of top counties you want to keep

if dataMerger.need_population_cut():
    population_cuts = dataMerger.population_cut(total_df)
    for index, population_cut in enumerate(population_cuts):
        top_counties = dataMerger.data_config.population_cut[index]
        filename = f"Top_{top_counties}.csv"

        output_path_population_cut = os.path.join(args.outputPath, filename)

        print(f'Writing top {top_counties} populated counties data to {output_path_population_cut}.')
        population_cuts[index].round(4).to_csv(output_path_population_cut, index=False)


