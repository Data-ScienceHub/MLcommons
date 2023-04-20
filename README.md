# Introduction

The COVID-19 pandemic has created unprecedented challenges for governments and healthcare systems worldwide, highlighting the critical importance of understanding the factors that contribute to virus transmission. This study aimed to identify the most influential age groups in COVID-19 infection rates at the US county level using the Modified Morris Method and deep learning for time series. Our approach involved training the state-of-the-art time-series model Temporal Fusion Transformer on different age groups as a static feature and the population vaccination status as the dynamic feature. We analyzed the impact of those age groups on COVID-19 infection rates by perturbing individual input features and ranked them based on their Morris sensitivity scores, which quantify their contribution to COVID-19 transmission rates. The findings are verified using ground truth data from the CDC and US Census, which provide the true infection rates for each age group. The results suggest that young adults were the most influential age group in COVID-19 transmission at the county level between March 1, 2020, and November 27, 2021. Using these results can inform public health policies and interventions, such as targeted vaccination strategies, to better control the spread of the virus. Our approach demonstrates the utility of feature sensitivity analysis in identifying critical factors contributing to COVID-19 transmission and can be applied in other public health domains.

## Folder structure
* `Literature`: Related Work on Parameter Sensitivity 

* `Notes`: Extra documentation and notes

* `age-groups`: Age groups, ground truth, and results with corresponding scripts

* `dataset_raw`: raw data set in CSV files

* `original-TFT-pytorch`: Previous work on TFT models

* `race-groups`: Race groups, ground truth, and results with corresponding scripts

* `.gitignore`: Slurm script

* `Feature_Sensitivity.zip`: Paper describing model and results

* `singilarity.def`: For when using singularity.

# Data

Dataset: https://drive.google.com/drive/folders/1_AMG_qreffMI1oqegfG-nAwPAs8xK8Gw

## Runtime

Currently on Rivanna with batch size 64, each epoch with

* Top 100 counties takes around 2-3 minutes.
* Top 500 counties takes around 12-13 minutes, memory 24GB.
* Total 3,142 counties takes around 40-45 minutes, memory 32GB.

## How to Reproduce

### Singularity
You can either pull the singularity container from the remote library,
```bash
singularity pull tft_pytorch.sif library://khairulislam/collection/tft_pytorch:latest
```
Or create the container locally using the [singularity.def](/TFT-pytorch/singularity.def) file. Executeg the following command. This uses the definition file to create the container from scratch. Note that is uses `sudo` and requires root privilege. After compilation, you'll get a container named `tft_pytorch.sif`. 

```bash
sudo singularity build tft_pytorch.sif singularity.def
```

Then you can use the container to run the scripts. For example, 
```bash
cd original-TFT-baseline/script/

singularity run --nv ../../tft_pytorch.sif python train.py --config=baseline.json --output=../scratch/TFT_baseline
```

### Virtual Environment
To create the virtual environment by
* Pip, use the [requirement.txt](/requirements.txt).
* Anaconda, use the [environment.yml](/environment.yml).

You can directly create a python virtual environment using the [environment.yml](environment.yml) file and Anaconda. Copy this file to your home directory and run the following command,

```bash
# this creates a virtual environment named ml
conda create --name ml --file environment.yml

# then activate the environment with
conda activate ml
# now you should be able to run the files from your cmd line without error
# if you are on notebook select this environment as your kernel
```

You can also create the environment in this current directory, then the virtual environment will be saved in this folder instead, not in the home directory.

### UVA Rivanna/CS server

On **Rivanna**, the default python environment doesn't have all the libraries we need. The [requirements.txt](requirements.txt) file contains a list of libraries we need. There are two ways you can run the training there

#### Default Environment

Rivanna provides a bunch of python kernels readily available. You can check them from an interactive Jupyterlab session, on the top-right side of the notebook. I have tested with the `Tensorflow 2.8.0/Keras Py3.9` kernel and uncommented the following snippet in the code.

```python
!pip install pytorch_lightning==1.8.6
!pip install pytorch_forecasting==0.10.3
```

You can choose different kernels and install the additional libraries. 

#### GPU 

Next, you might face issues getting GPU running on Rivanna. Even on a GPU server the code might not recognize the GPU hardware if cuda and cudnn are not properly setup. Try to log into an interactive session in a GPU server, then run the following command

```bash
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If this is still 0, then you'll have to install the cuda and cudnn versions that match version in `nvidia-smi` command output. Also see if you tensorflow version is for CPU or GPU.

### Google Colab

A training notebook run on colab is shared [here](https://colab.research.google.com/drive/1yhI1PesOXYlB6iYXHre9zXMks1a4P6U2?usp=sharing). Feel free to copy and run on your colab and let me know if there are any issues.

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Uncomment the installation commands in the code or set `running_on_colab` to `True` in the code. Upload the TFT-pytorch folder in your drive and set that path in the notebook colab section.

```python
!pip install pytorch_lightning==1.8.6
!pip install pytorch_forecasting==0.10.3
```

If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the path accordingly in the notebook.

## Features

The following table lists the features with their source and description. Note that, past values of the target and known futures are also used as observed inputs by TFT.


<div align="center" style="overflow-x:auto;text-align:center;vertical-align: middle;">
<table border="1">
<caption> <h2>Details of Features </h2> </caption>
<thead>
<tr>
<th>Feature</th>
<th>Type</th>
<th>Update Frequency</th>
<th>Description</th>
<th>Source(s)</th>
</tr>

</thead>

<tbody>

<tr>
<td><strong>Age Groups</strong> <br>( UNDER5, AGE517, AGE1829, AGE3039, AGE4049, AGE5064, AGE6574, AGE75PLUS )</td>
<td>Static</td>
<td>Once</td>
<td>Percent of population in each age group.</td>
<td><span><a href="https://www.census.gov/data/tables/time-series/demo/popest/2020s-national-detail.html" target="_blank">2020 Govt Census</a></span></td>
</tr>

<tr>
<td><strong>Vaccination Full Dose</strong> <br>(Series_Complete_Pop_Pct)</td>
<td>Observed</td>
<td rowspan=3>Daily</td>
<td> Percent of people who are fully vaccinated (have second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where recipient lives.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">CDC</a></span></td>
</tr>

<tr>
<td><strong>SinWeekly</strong></td>
<td>Known Future</td>
<td> <em>Sin (day of the week / 7) </em>.</td>
<td>Date</td>
</tr>

<tr>
<td><strong>Case</strong></td>
<td>Target</td>
<td> COVID-19 infection at county level.</td>
<td><span><a href="https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/" target="_blank">USA Facts</a></span></td>
</tr>

</tbody>
</table>

</div>

## Usage guideline

* Please do not add temporarily generated files in this repository.
* Make sure to clean your tmp files before pushing any commits.
* In the .gitignore file you will find some paths in this directory are excluded from git tracking. So if you create anything in those folders, they won't be tracked by git.
  * To check which files git says untracked `git status -u`. 
  * If you have folders you want to exclude add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.
