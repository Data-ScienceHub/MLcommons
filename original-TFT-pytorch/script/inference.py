# %% [markdown]
# # Imports

# %%
import os, gc
import torch
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_columns', None)

# %% [markdown]
# # Initial setup

# %% [markdown]
# ## GPU

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# %% [markdown]
# ## Pytorch lightning and forecasting

# %%
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer

# %% [markdown]
# # Load input

# %%
from dataclasses import dataclass
from argparse import ArgumentParser

parser = ArgumentParser(description='Inference using a trained TFT model')

parser.add_argument(
   '--config', default='baseline.json',
   help='config filename in the configurations folder'
)

parser.add_argument(
   '--input_file', help='path of the input feature file',
   default='../2022_May_cleaned/Top_100.csv'
)
parser.add_argument(
   '--output', default='../scratch/TFT_baseline',
   help='output result folder. Anything written in the scratch folder will be ignored by Git.'
)
parser.add_argument(
   '--show-progress', default=False, type=bool,
   help='show the progress bar.'
)
arguments = parser.parse_args()

@dataclass
class args:
    result_folder = arguments.output
    figPath = os.path.join(result_folder, 'figures')
    checkpoint_folder = os.path.join(result_folder, 'checkpoints')
    input_filePath = arguments.input_file

    configPath = os.path.join('../configurations', arguments.config)

    model_path = get_best_model_path(checkpoint_folder)

    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = arguments.show_progress

    # interpret_output has high memory requirement
    # results in out-of-memery for Total.csv and a model of hidden size 64, even with 64GB memory
    interpret_train = 'Total.csv' not in input_filePath

# %%
start = datetime.now()
print(f'Started at {start}')

total_data = pd.read_csv(args.input_filePath)
print(total_data.shape)
total_data.head()

# %% [markdown]
# # Config

# %%
import json
import sys
sys.path.append( '..' )
from Class.Parameters import Parameters
from script.utils import *

with open(args.configPath, 'r') as input_file:
  config = json.load(input_file)

parameters = Parameters(config, **config)

# %%
targets = parameters.data.targets
time_idx = parameters.data.time_idx
tft_params = parameters.model_parameters

batch_size = tft_params.batch_size
max_prediction_length = tft_params.target_sequence_length
max_encoder_length = tft_params.input_sequence_length

# %% [markdown]
# # Processing

# %%
total_data['Date'] = pd.to_datetime(total_data['Date'].values) 
total_data['FIPS'] = total_data['FIPS'].astype(str)
print(f"There are {total_data['FIPS'].nunique()} unique counties in the dataset.")

# %% [markdown]
# ## Adapt input to encoder length
# Input data length needs to be a multiple of encoder length to created batch data loaders.

# %%
train_start = parameters.data.split.train_start
total_data = total_data[total_data['Date']>=train_start]
total_data[time_idx] = (total_data["Date"] - total_data["Date"].min()).apply(lambda x: x.days)

# %% [markdown]
# ## Train validation test split and scaling

# %%
train_data, validation_data, test_data = train_validation_test_split(
    total_data, parameters
)

# %%
train_scaled, validation_scaled, test_scaled, target_scaler = scale_data(
    train_data, validation_data, test_data, parameters
)

# %% [markdown]
# ## Create dataset and dataloaders

# %%
def prepare_data(data: pd.DataFrame, pm: Parameters, train=False):
  data_timeseries = TimeSeriesDataSet(
    data,
    time_idx= time_idx,
    target=targets,
    group_ids=pm.data.id, 
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_reals=pm.data.static_features,
    time_varying_known_reals = pm.data.time_varying_known_features,
    time_varying_unknown_reals = pm.data.time_varying_unknown_features,
    target_normalizer = MultiNormalizer(
      [GroupNormalizer(groups=pm.data.id) for _ in range(len(targets))]
    )
  )

  if train:
    dataloader = data_timeseries.to_dataloader(train=True, batch_size=batch_size)
  else:
    dataloader = data_timeseries.to_dataloader(train=False, batch_size=batch_size*8)

  return data_timeseries, dataloader

# %%
_, train_dataloader = prepare_data(train_scaled, parameters)
_, validation_dataloader = prepare_data(validation_scaled, parameters)
_, test_dataloader = prepare_data(test_scaled, parameters)

del validation_scaled, test_scaled
gc.collect()

# %% [markdown]
# # Model

# %%
tft = TemporalFusionTransformer.load_from_checkpoint(args.model_path)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# %% [markdown]
# # Prediction Processor and PlotResults

# %%
from Class.PredictionProcessor import PredictionProcessor

processor = PredictionProcessor(
    time_idx, parameters.data.id[0], max_prediction_length, targets, 
    train_start, max_encoder_length
)

# %%
from Class.Plotter import *

plotter = PlotResults(args.figPath, targets, show=args.show_progress_bar)

# %% [markdown]
# # Evaluate

# %% [markdown]
# ## Train results

# %% [markdown]
# ### Average

# %%
print(f'\n---Training prediction--\n')

train_predictions, train_index = tft.predict(
    train_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)

print('\n---Training results--\n')
train_predictions = upscale_prediction(
  targets, train_predictions, target_scaler, max_prediction_length
)
train_result_merged = processor.align_result_with_dataset(
  train_data, train_predictions, train_index
)
show_result(train_result_merged, targets)

plotter.summed_plot(train_result_merged, type='Train_error', plot_error=True)
gc.collect()

# %% [markdown]
# ## Validation results

# %%
print(f'\n---Validation results--\n')
validation_predictions, validation_index = tft.predict(
    validation_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)
validation_predictions = upscale_prediction(targets, validation_predictions, target_scaler, max_prediction_length)

validation_result_merged = processor.align_result_with_dataset(validation_data, validation_predictions, validation_index)
show_result(validation_result_merged, targets)
plotter.summed_plot(validation_result_merged, type='Validation')
gc.collect()

# %% [markdown]
# ## Test results

# %% [markdown]
# ### Average

# %%
print(f'\n---Test results--\n')
test_predictions, test_index = tft.predict(
    test_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)
test_predictions = upscale_prediction(
   targets, test_predictions, target_scaler, max_prediction_length
)

test_result_merged = processor.align_result_with_dataset(
   test_data, test_predictions, test_index
)
show_result(test_result_merged, targets)
plotter.summed_plot(test_result_merged, 'Test')
gc.collect()

# %% [markdown]
# ## Dump results

# %%
train_result_merged['split'] = 'train'
validation_result_merged['split'] = 'validation'
test_result_merged['split'] = 'test'
df = pd.concat([train_result_merged, validation_result_merged, test_result_merged])
df.to_csv(os.path.join(plotter.figPath, 'predictions.csv'), index=False)

print(df.head())

# %% [markdown]
# # End

# %%
print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')


