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

### Runtime

Currently on Rivanna with batch size 64, each epoch with

* Top 100 counties takes around 2-3 minutes.
* Top 500 counties takes around 12-13 minutes, memory 24GB.
* Total 3,142 counties takes around 40-45 minutes, memory 32GB.

### Google Colab

A training notebook run on colab is shared [here](https://colab.research.google.com/drive/1yhI1PesOXYlB6iYXHre9zXMks1a4P6U2?usp=sharing). Feel free to copy and run on your colab and let me know if there are any issues.

If you are running on **Google colab**, most libraries are already installed there. You'll only have to install the pytorch forecasting and lightning module. Uncomment the installation commands in the code or set `running_on_colab` to `True` in the code. Upload the TFT-pytorch folder in your drive and set that path in the notebook colab section.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

If you want to run the data preparation notebook, upload the [CovidMay17-2022](../dataset_raw/CovidMay17-2022/) folder too. Modify the path accordingly in the notebook.

### Rivanna/CS server

On **Rivanna**, the default python environment doesn't have all the libraries we need. The [requirements.txt](requirements.txt) file contains a list of libraries we need. There are two ways you can run the training there

#### Default Environment

Rivanna provides a bunch of python kernels readily available. You can check them from an interactive Jupyterlab session, on the top-right side of the notebook. I have tested with the `Tensorflow 2.8.0/Keras Py3.9` kernel and uncommented the following snippet in the code.

```python
!pip install pytorch_lightning
!pip install pytorch_forecasting
```

You can choose different kernels and install the additional libraries. 

#### Virtual Environment

You can directly create a python virtual environment using the [environment.yml](environment.yml) file and Anaconda. Then you won't have to install the libraries each time. Copy this file to your home directory and run the following command,

```bash
conda create --name <env> --file <this file>

# for example
conda create --name ml --file environment.yml

# then activate the environment with
conda activate ml
# now you should be able to run the files from your cmd line without error
# if you are on notebook select this environment as your kernel
```

You can also create the environment in this current directory, then the virtual environment will be saved in this folder instead, not in the home directory.

#### GPU 

Next, you might face issues getting GPU running on Rivanna. Even on a GPU server the code might not recognize the GPU hardware if cuda and cudnn are not properly setup. Try to log into an interactive session in a GPU server, then run the following command

```bash
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

If this is still 0, then you'll have to install the cuda and cudnn versions that match version in `nvidia-smi` command output. Also see if you tensorflow version is for CPU or GPU.
