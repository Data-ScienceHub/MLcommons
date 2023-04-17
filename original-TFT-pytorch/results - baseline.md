# Experiment with Baseline model
The following experiments are on inputs and outputs where outliers were removed. The raw target has very high spikes during covid peak times, specially during late 2021 and early 2022 due to dominant mutants. Removing the outliers show improved loss metrics.

The results are on all 3,142 counties.

## Train

![daily-cases](/original-TFT-pytorch/results/TFT_baseline/figures/Summed_plot_Cases_Train_error.jpg)

## Validation

![daily-cases](/original-TFT-pytorch/results/TFT_baseline/figures/Summed_plot_Cases_Validation.jpg)

## Test

![daily-cases](/original-TFT-pytorch/results/TFT_baseline/figures/Summed_plot_Cases_Test.jpg)