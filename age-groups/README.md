# Introduction

The Age-groups folder contains the `Temporal Fushion Transformer` implemented in [`PytorchForecasting`](https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html) framework. This study aimed to identify the most influential age groups in COVID-19 infection rates at the US county level using the Modified Morris Method and deep learning for time-series. The approach involved training the state-of-the-art time-series model Temporal Fusion Transformer on different age groups as a static feature and the population vaccination status as the dynamic feature. The impact of those age groups were analyzed on COVID-19 infection rates by perturbing individual input features and ranked them based on their Morris sensitivity scores, which quantify their contribution to COVID-19 transmission rates.The data contains the population by age subgroups for each of the 3,142 US counties, along with the daily vaccination rate of the population and COVID-19 case report from March 1, 2020, to Dec 27, 2021. The eight age subgroups are 0-4, 5-17, 18-29, 30-39, 40-49, 50-64, 65-74, and 75 and older for all counties.

## Folder structure
* `Class`
  * `DataMerger.py`
  * `Parameters.py`
  * `PlotConfig.py`
  * `Plotter.py`
  * `PredictionProcessor.py`

* `Ground Truth`: Truth data from the CDC and US Census, which provide the true infection rates for each age group.
  * `2019gender_table1.csv`
  * `COVID-19_Weekly_Cases_and_Deaths_by_Age__Race_Ethnicity__and_Sex.csv`
  * `nc-est2021-agesex-res.csv`

* `configurations`: Folder to save some common configurations.

* `results`: Contains files for each Age Group showing the Morris Sensistivity results.

* `script`: Contains scripts for submitting batch jobs.
  * `outputs`
  * `prepare_data.py`: Prepare merged data from raw feature files.
  * `sensitivity_analysis.py`: Calculating Morris Sensitivity
  * `slurm-sensitivity.sh`: For when using singularity.
  * `slurm-train.sh`: For when using singularity.
  * `train_age_group.py`: Train Age Group model then interpret using the best model by validation loss.
  * `utils.py`: Contains utility methods.

* `Plotting All Results.ipynb`: Plotting Morris Index Results and Final Rankings.

* `fig1.png`: Model Sensitivity to Age.

* `fig2.png`: Weekly COVID-19 Cases by Age Group.
