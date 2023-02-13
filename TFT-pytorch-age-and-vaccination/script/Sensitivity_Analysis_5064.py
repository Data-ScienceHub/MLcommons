#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os, gc
import torch
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
pd.set_option('display.max_columns', None)


# # Initial setup

# ## GPU

# In[ ]:


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


# ## Google colab
# 
# Set `running_on_colab` to true if you are running on google colab. They don't have these libraries installed by default.Uncomment the codes too if needed. They might be commented out since in .py script inline commands show errors.
# 
# **Restart the kernel after installing the new libraries.**

# In[ ]:


# running_on_colab = True

# if running_on_colab:
#     !pip install torch==1.11.0
#     # !pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
#     !pip install pytorch_lightning==1.8.1
#     !pip install pytorch_forecasting==0.10.3
#     !pip install pandas==1.4.1


# In[ ]:


# if running_on_colab:
#     from google.colab import drive

#     drive.mount('/content/drive')
#     %cd /content/drive/My Drive/TFT-pytorch/notebook


# ## Pytorch lightning and forecasting

# In[ ]:


from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer


# # Load input

# In[ ]:


from dataclasses import dataclass

@dataclass
class args:
    result_folder = '../results/age_groups_5064/'
    figPath = os.path.join(result_folder, 'figures_morris')
    checkpoint_folder = os.path.join(result_folder, 'checkpoints')
    input_filePath = '../2022_May_age_groups/Top_100_5064.csv'

    configPath = '../configurations/age_groups_5064.json'
    model_path = os.path.join(checkpoint_folder, os.listdir('../results/age_groups_5064/checkpoints')[0])

    # set this to false when submitting batch script, otherwise it prints a lot of lines
    show_progress_bar = False

if not os.path.exists(args.figPath):
    os.makedirs(args.figPath, exist_ok=True)


# In[ ]:


start = datetime.now()
print(f'Started at {start}')

total_data = pd.read_csv(args.input_filePath)
print(total_data.shape)
total_data.head()


# # Config

# In[ ]:


import json
import sys
sys.path.append( '..' )
from Class.Parameters import Parameters
from script.utils import *

with open(args.configPath, 'r') as input_file:
  config = json.load(input_file)

parameters = Parameters(config, **config)


# In[ ]:


targets = parameters.data.targets
time_idx = parameters.data.time_idx
tft_params = parameters.model_parameters

batch_size = tft_params.batch_size
max_prediction_length = tft_params.target_sequence_length
max_encoder_length = tft_params.input_sequence_length


# # Processing

# In[ ]:


total_data['Date'] = pd.to_datetime(total_data['Date'].values) 
total_data['FIPS'] = total_data['FIPS'].astype(str)
print(f"There are {total_data['FIPS'].nunique()} unique counties in the dataset.")


# ## Adapt input to encoder length
# Input data length needs to be a multiple of encoder length to created batch data loaders.

# In[ ]:


train_start = parameters.data.split.train_start
total_data = total_data[total_data['Date']>=train_start]
total_data[time_idx] = (total_data["Date"] - total_data["Date"].min()).apply(lambda x: x.days)


# ## Train validation test split and scaling

# In[ ]:


train_data, validation_data, test_data = train_validation_test_split(
    total_data, parameters
)


# In[ ]:


train_scaled, validation_scaled, test_scaled, target_scaler = scale_data(
    train_data, validation_data, test_data, parameters
)


# ## Create dataset and dataloaders

# In[ ]:


def prepare_data(data: pd.DataFrame, pm: Parameters, train=False):
  data_timeseries = TimeSeriesDataSet(
    data,
    time_idx= time_idx,
    target=targets,
    group_ids=pm.data.id, 
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_reals=pm.data.static_features,
    # static_categoricals=['FIPS'],
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

  return dataloader


# In[ ]:


train_dataloader = prepare_data(train_scaled, parameters)
validation_dataloader = prepare_data(validation_scaled, parameters)
test_dataloader = prepare_data(test_scaled, parameters)

# del validation_scaled, test_scaled
gc.collect()


# # Model

# In[ ]:


tft = TemporalFusionTransformer.load_from_checkpoint(args.model_path)

print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


# # Prediction Processor and PlotResults

# In[ ]:


from Class.PredictionProcessor import PredictionProcessor

processor = PredictionProcessor(
    time_idx, parameters.data.id[0], max_prediction_length, targets, 
    train_start, max_encoder_length
)


# In[ ]:


from Class.Plotter import *

plotter = PlotResults(args.figPath, targets, show=args.show_progress_bar)


# # Evaluate

# ## Train results

# ### Average

# In[ ]:


print(f'\n---Training results--\n')

# [number of targets (2), number of examples, prediction length (15)]
train_raw_predictions, train_index = tft.predict(
    train_dataloader, return_index=True, show_progress_bar=args.show_progress_bar
)

train_predictions = upscale_prediction(
    targets, train_raw_predictions, target_scaler, max_prediction_length
)
gc.collect()


# # Morris method

# ## Scale

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler

features = parameters.data.static_features + parameters.data.dynamic_features

minmax_scaler = MinMaxScaler()
train_minmax_scaled = minmax_scaler.fit_transform(train_data[features])

target_minmax_scaler = MinMaxScaler().fit(train_data[targets])

standard_scaler = StandardScaler()
standard_scaler.fit(train_data[features])


# ## Calculate

# In[ ]:


# delta_values = [1e-2, 1e-3, 5e-3, 9e-3, 5e-4, 1e-4, 5e-5, 1e-5]
delta_values = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
results = {
    'Delta': [],
    'Feature': [],
    'Mu_star':[],
    'Morris_sensitivity':[] 
}


# In[ ]:


for delta in delta_values:
    print(f'Delta {delta}.')
    for index, feature in enumerate(features):
        # this mimics how TF1 did it
        data = train_minmax_scaled.copy()
        data[index] += delta
        data = minmax_scaler.inverse_transform(data) # return to original scale

        # replace the value in normalized data
        data = standard_scaler.transform(data)
        train_scaled_copy = train_scaled.copy()
        train_scaled_copy[feature] = data[:, index]

        # inference on delta changed data
        dataloader = prepare_data(train_scaled_copy, parameters)
        new_predictions = tft.predict(
            dataloader, show_progress_bar=args.show_progress_bar
        )
        new_predictions = upscale_prediction(
            targets, new_predictions, target_scaler, max_prediction_length
        )

        # sum up the change in prediction
        prediction_change = np.sum([
            abs(train_predictions[target_index] - new_predictions[target_index])
                for target_index in range(len(targets)) 
        ])
        mu_star = prediction_change / (data.shape[0]*delta)

        # since delta is added to min max normalized value, std from same scaling is needed
        standard_deviation = train_minmax_scaled[:, index].std()
        scaled_morris_index = mu_star * standard_deviation

        print(f'Feature {feature}, mu_star {mu_star:0.5g}, sensitivity {scaled_morris_index:0.5g}')

        results['Delta'].append(delta)
        results['Feature'].append(feature)
        results['Mu_star'].append(mu_star)
        results['Morris_sensitivity'].append(scaled_morris_index)
    print()
    # break


# ## Dump

# In[ ]:


import pandas as pd
result_df = pd.DataFrame(results)
result_df.to_csv(os.path.join(args.figPath, 'Morris_5064.csv'), index=False)
result_df


# ## Plot

# In[ ]:


from Class.PlotConfig import *


# In[ ]:


for delta in delta_values:
    print(delta)
    fig = plt.figure(figsize = (20, 10))
    plt.bar(features, result_df[result_df['Delta']==delta]['Morris_sensitivity'])
    
    plt.ylabel("Scaled Morris Index")
    plt.tight_layout()
    plt.savefig(os.path.join(args.figPath, f'delta_{delta}.jpg'), dpi=200)
    plt.show()
    # break


# # End

# In[ ]:


print(f'Ended at {datetime.now()}. Elapsed time {datetime.now() - start}')

