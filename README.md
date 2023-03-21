# Interpreting County Level COVID-19 Infection and Feature Sensitivity using Deep Learning Time Series Models

## Introduction
This work combines sensitivity analysis with heterogeneous time-series deep learning model prediction, which corresponds to the interpretations of Spatio-temporal features from what the model has actually learned. We forecast county-level COVID-19 infection using the Temporal Fusion Transformer (TFT). We then use the sensitivity analysis extending Morris Method to see how sensitive the outputs are with respect to perturbation to our static and dynamic input features. We have collected more than 2.5 years of socioeconomic and health features over 3142 US counties. Using the proposed framework, we conduct extensive experiments and show our model can learn complex interactions and perform predictions for daily infection at the county level. 

## Folder Structure
* **dataset_raw**: Contains the collected raw dataset and the supporting files. To update use the [Update dynamic dataset](/dataset_raw/Update%20dynamic%20features.ipynb) notebook. Static dataset is already update till the onset of COVID-19 using [Update static dataset](/dataset_raw/Update%20static%20features.ipynb) notebook.
* **TFT-pytorch**: Contains all codes and merged feature files used during the TFT experimentation setup and interpretation. For more details, check the [README.md](/TFT-pytorch/README.md) file inside it. The primary results are highlighted in [results.md](/TFT-pytorch/results.md). 


## How to Reproduce

### Virtual Environment
To create the virtual environment
* By pip, use the [requirement.txt](/requirements.txt).
* By anaconda, use the [environment.yml](/environment.yml).

### Singularity
You can either pull the singularity container from the remote library,
```bash
singularity pull tft_pytorch.sif library://khairulislam/collection/tft_pytorch:latest
```
Or create the container locally using the [singularity.def](/TFT-pytorch/singularity.def) file. Executeg the following command. This uses the definition file to create the container from scratch. Note that is uses `sudo` and requires root privilege. After compilation, you'll get a container named `tft_pytorch.sif`. 

```bash
sudo singularity build singultft_pytorchatft_pytorchrity.sif singularity.def
```

## Input Features

Note that, past values target and known futures are also used as observed inputs by TFT.

<div align="center">

| Feature        | Type       |
|:------------------------:|:------------:|
| Age Group               | Static     |
| Vaccination Full Dose   | Dynamic    |
| SinWeekly | Known Future |

</div>

<h2 class="accordion-toggle accordion-toggle-icon">Details of Input Features</h4>
<div class="accordion-content">
<table class="pop_up_table">
<thead>
<tr>
<th scope="col">Data Domain  <br /> Component(s)</th>
<th colspan="2" scope="col">Update Freq.</th>
<th scope="col">Description/Rationale</th>
<th scope="col">Source(s)</th>
</tr>

</thead>
<tbody>

<tr>
<td colspan="4"><strong>Age Distribution</strong></td>
</tr>
<tr>
<td>Age Group</td>
<td>Static</td>
<td style="background: #9A42C8;"></td>
<td>Percent of people in the age group 0-4, 5-17, 18-29, 30-39, 40-49, 50-59, 60-74, or 75+ from 2016-2020 American Community Survey (ACS).</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td colspan="4"><strong>Vaccination Full Dose</strong></td>
</tr>
<tr>
<td>Series_Complete_Pop_Pct</td>
<td>Daily</td>
<td style="background: #4258C9;"></td>
<td> Percent of people who are fully vaccinated (have second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">CDC</a></span></td>
</tr>

</tbody>
</table>
