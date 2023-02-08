## 1. Introduction
This is an open-source python package which attempts to replicate the [HAIM](https://www.nature.com/articles/s41746-022-00689-4) study. 
It uses the [HAIM multimodal dataset](https://physionet.org/content/haim-multimodal/1.0.1/) containing data of 4 modalities 
(tabular, time-series, text and images) and 11 unique sources
to perform 12 predictive tasks (10 chest pathologies, length-of-stay and 48 h mortality predictions).

This package is our own adaptation of the [HAIM GitHub package](https://github.com/lrsoenksen/HAIM.git). 

## 2. How to use the package ?
The dataset used to replicate this study is publicly available in [physionet](https://physionet.org/content/haim-multimodal/1.0.1/). To run this package:
- Download the dataset and move it to [csvs](csvs).
- Install the requirements under **Python 3.9.13** as following:
```
$ pip install requirements.txt
```
The package can be used with different sources combinations to predict one of the 12 predictive tasks defined above. Here is a code snippet which uses one 
combination of sources to predict patient's length-of-stay:
```python 
# Import the function needed to run an experiment
from run_experiments import run_single_experiment
# Import constants where the task name and the sources types to use for prediction are stored
from src.data import constants

# For each source type (demographic, chart events, lab events), get all the predictors 
# (age, gender, insurance, etc.),
sources = constants.DEMOGRAPHIC.sources + constants.CHART.sources + constants.LAB.sources
# Get the modalities to which belong the sources types we will use for prediction
modalities = unique([source.modality for source in sources])

# Run one single experiment with one sources combination (demographic, chart events, lab events) 
# to predict the length-of-stay of each patient
run_single_experiment(prediction_task=constants.LOS, sources_predictors=sources, sources_modalities=modalities, 
                      evaluation_name='length_of_stay_exp')
```

The following code predicts the 48 hours mortality using all the 11 sources:
```python
# Import the function needed to run an experiment
from run_experiments import run_single_experiment
# Import constants where the task name, all the sources types predictors and the modalities are stored
from src.data import constants 

# Run one single experiment with one combination of all the 11 sources to predict the 48h mortality
run_single_experiment(prediction_task=constants.MORTALITY, sources_predictors=constants.ALL_PREDICTORS, 
                      sources_modalities=constants.ALL_MODALITIES, evaluation_name='48h_mortality_exp')
```
All data sources and modalities are stored as [constants](src/data/constants.py), here is a summary of the possible data modalities and sources to import for prediction (refer to page 3 from the [Supplementary Material](https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-022-00689-4/MediaObjects/41746_2022_689_MOESM1_ESM.pdf) for more details):
Modalities | Sources | 
---------| -----------| 
constants.TAB | constants.DEMOGRAPHIC.sources |
constants.TS | constants.CHART.sources |
constants.TS | constants.LAB.sources |
constants.TS | constants.PROC.sources |
constants.TXT | constants.RAD.sources |
constants.TXT | constants.ECG.sources |
constants.TXT | constants.ECHO.sources |
constants.IMG | constants.VP.sources |
constants.IMG | constants.VMP.sources |
constants.IMG | constants.VD.sources |
constants.IMG | constants.VMD.sources |
constants.ALL_MODALITIES | constants.ALL_PREDICTORS | 

To run the HAIM experiment which performs the 12 predictive tasks on all sources combinations 
(refer to page 7 from the [Supplementary Material](https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-022-00689-4/MediaObjects/41746_2022_689_MOESM1_ESM.pdf)),
run the following command: 
```
$ python run_experiments.py
```
 
> **Warning**
> 
> The HAIM experiment performs 14324 evaluations (1023 evaluations for each of the chest pathologies prediction tasks and 2047 for the length-of-stay and 48h mortality). We didn't run the experiment but we approximate the execution time to 200 days run with the current implementation using only 10 CPUs.

The experiments results (metrics values and figures) will be stored in the [``experiments``](experiments) directory where the name of each folder is structured as ``TaskName_NumberOfTheExperiment``
(ex. Fracture_25). For each prediction task, the sources combination with the best AUC will be stored in the directory ``TaskName_best_experiment``.

To reproduce the HAIM exepriment on one single predictive task, run the following command:
```
$ python run_experiments.py -t "task_name"
```
Tasks names can be found in ``src/data/constants.py``and are summarized in the following table: 
Task | Argument | Constant to import
---------| -----------| -----------| 
Fracture | "Fracture" | constants.FRACTURE |
Pneumothorax| "Pneumothorax" | constants.PNEUMOTHORAX |
Pneumonia       | "Pneumonia" | constants.PNEUMONIA
Lung opacity       | 	"Lung Opacity" | constants.LUNG_OPACITY |
Lung lesion    | "Lung Lesion" | constants.LUNG_LESION |
Enlarged Cardiomediastinum      | "Enlarged Cardiomediastinum" | constants.ENLARGED_CARDIOMEDIASTINUM |
Edema      | "Edema" | constants.EDEMA |
Consolidation    | "Consolidation" | constants.CONSOLIDATION |
Cardiomegaly      | "Cardiomegaly" | constants.CARDIOMEGALY |
Atelectasis     | "Atelectasis" | constants.ATELECTASIS |
Length of stay     | "48h los" | constants.LOS |
48 hours mortality     | "48h mortality" | constants.MORTALITY |
## 3. Prediction of the 12 tasks using the 4 modalities 
Experiments using all the sources from the 4 modalities to predict the 12 tasks can be found in the [``notebooks``](notebooks) directory. Each notebook is named after the prediction task it performs.

> **Note**
> 
> All the 11 sources were used to predict the length-of-stay and 48 hours mortality but the radiology notes were excluded to predict the chest pathologies to avoid data leakage.



Below are the ``AUC`` values reported from our experiments compared to those reported in the HAIM paper (refer to page 4 from the [paper](https://www.nature.com/articles/s41746-022-00689-4))



Task | AUC from our experiment | AUC from the paper |
---------| -----------| ----------- |
Fracture | 0.828 +- 0.110 | 0.838 |
Pneumothorax| 0.811 +- 0.021 | 0.836 |
Pneumonia       | 0.871 +- 0.013 | 0.883    |
Lung opacity       | 	0.797 +- 0.015 | 0.816   |
Lung lesion    | 0.829 +- 0.053	 | 0.844   |
Enlarged Cardiomediastinum      | 0.877 +- 0.035	 | 0.876  |
Edema      | 0.915 +- 0.007		 |0.917	 |
Consolidation    | 0.918 +- 0.018		 | 0.929 |
Cardiomegaly      | 0.908 +- 0.004	 | 0.914 |
Atelectasis     | 0.765 +- 0.013	 | 0.779	 |
Length of stay     | 0.932 +- 0.012		 | 0.939|
48 hours mortality     | 0.907 +- 0.007		 | 0.912	|

More statistics and metrics are reported from each of the 12 experiments above and can be found in the ``experiments`` directory. Each experiment directory is named after the task on which the prediction model was evaluated.

> **Note**
> 
> The paper reported the best AUC value among all the experiments (all possible sources combinations for each predictive task) for each task while we reported the AUC value resulting from the evaluation using all the sources for each predictive task.


## 4. Prediction of one single task using all sources combinations
We tried to reproduce the HAIM experiment and used all the 1023 possible sources combinations to predict the presence or absence of a fracture in a patient and select the one resulting in the best ``AUC``.

Below the ``AUC`` value reported from our experiments compared to the one reported in the HAIM paper. 
 AUC from our experiment | AUC from the paper |
 -----------| ----------- |
0.862 +- 0.112 | 0.838 |
 
 
The above experiment can be performed using the following command
```
$ python run_experiments.py -t "Fracture"
```
A recap of the experiment named [``Fracture_best_experiment``](experiments/Fracture_best_experiment) is generated at the end of the experiment containing more statistics and metrics values.

## 5. Issues 
While working on reproducing HAIM experiments, we observed some problems on the published embedded dataset. While img_id is supposed to uniquely identify each image, redundant img_ids belonging to different patients were found in the dataset. See [``corrupted_ids.ipynb``](corrupted_ids.ipynb) for further details. 

## 6. Future work
The next step of our package is to regenerate the embeddings for each source type. For each modality (tabular, time-series, image, text), we will also explore new embeddings generators. 

## Project Tree
```
├── csvs                         <- CSV file of the dataset used in the study
├── experiments                  <- Directories with statistics and metrics values from each evaluation
├── notebooks                    <- Notebooks with experiments using all sources for each prediction task
├── src                          <- All project modules
│   ├── data
│   │   ├── constants.py           <- Constants related to the HAIM study
│   │   ├── datasets.py           <- Custom dataset implementation for the HAIM study
│   │   └── sampling.py           <- Samples the dataset to test, train and validation
│   ├── evaluation
│   │   ├── tuning.py             <- Hyper-parameters optimizations using different optimizers
│   │   └── evaluating.py         <- Skeleton of each experiment process 
│   └── utils                     
│   │   └── metric_scores.py      <- Custom metrics implementations and wrappers
├── corrupted_ids.ipynb           <- Notebook to highlight some issues in the dataset
├── requirements.txt              <- All the requirements to install to run the project
├── run_experiments.py            <- Main script used to replicate the experiments of the HAIM study
└── README.md
