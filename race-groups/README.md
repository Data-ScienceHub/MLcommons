# Introduction

The COVID-19 pandemic has laid bare longstanding disparities in health outcomes across different demographic groups, with evidence showing that certain racial and ethnic populations have been disproportionately impacted by the virus. To better understand the extent of these disparities and inform targeted interventions, researchers have turned to various measures of health inequality, including the Morris Index.

The Morris Index is a statistical measure used in epidemiology to quantify the degree of concentration of a particular health condition within a specific population subgroup. In the context of COVID-19, the Morris Index can be used to measure the sensitivity of COVID-19 cases to race and ethnicity, by taking into account the size of the population subgroup and the prevalence of COVID-19 cases within that group.

The Centers for Disease Control and Prevention (CDC) has been tracking COVID-19 cases by race and ethnicity since the beginning of the pandemic, providing a wealth of data for researchers to analyze using the Morris Index. By examining this data through the lens of the Morris Index, researchers can identify whether COVID-19 cases are disproportionately affecting certain racial and ethnic groups, and determine the extent to which underlying social and economic factors may be contributing to these disparities.

## Folder structure
* `Class`:
  * `DataMerger.py`: 

* `Ground Truth`:

* `configurations`:

* `results`:

* `script`:

* `Plotting All Results.ipynb`:

* `fig1.png`:

* `weekly_ground_truth.jpg`:


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

