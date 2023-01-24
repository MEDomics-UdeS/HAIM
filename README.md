## 1. Introduction
This is an open-source python package which replicates the [HAIM](https://www.nature.com/articles/s41746-022-00689-4) study. 
It uses the [HAIM multimodal dataset](https://physionet.org/content/haim-multimodal/1.0.1/) containing data of 4 modalities 
(tabular, time-series, text and images) and 11 unique sources
to perform 12 predictive tasks (10 chest pathologies, length-of-stay and 48 h mortality predictions).

## 2. How to use the package ?
The package can be used with different sources combinations to predict one of the 12 predictive tasks defined above. Here is a code snippet which uses one 
combination of sources to predict patient's length-of-stay:
```python
from src.data.constants import LOS, DEMOGRAPHIC, CHART, LAB
from run_experiments import run_single_experiment

sources = DEMOGRAPHIC.sources + CHART.sources + LAB.sources
modalities = unique([source.modality for source in sources])

run_single_experiment(prediction_task=LOS, sources_predictors=sources, sources_modalities=modalities, 
                      evaluation_name='length_of_stay_exp')
```
To reproduce the HAIM experiment which performs the 12 predictive tasks on all sources combinations 
(refer to page 7 from the [Supplementary Material](https://static-content.springer.com/esm/art%3A10.1038%2Fs41746-022-00689-4/MediaObjects/41746_2022_689_MOESM1_ESM.pdf)),
run the following command: 
```
$ python run_experiments.py
```
The experiments results (metrics values and figures) will be stored in the ``experiments`` directory where the name of each folder is structured as ``TaskName_NumberOfTheExperiment``
(ex. Fracture_25).

To reproduce the HAIM exepriment on one single predictive, run the following command:
```
$ python run_experiments.py -t task_name
```
Tasks names can be found in ``src/data/constants.py`` 

## 3. Prediction of the 12 tasks using the 4 modalities 
Experiments using the 4 modalities to predict the 12 tasks can be found in the ``notebooks`` directory, each notebook is named after the prediction task it performs.
Below are the ``AUC`` values reported from our experiments compared to those reported in the HAIM paper (refer to page 4 from the [paper](https://www.nature.com/articles/s41746-022-00689-4))
Task | AUC from our experiment | AUC from the paper |
---------| -----------| ----------- |
Fracture | 0.828 +- 0.1103 | 0.838 |
Pneumothorax| 0.8114 +- 0.0208 | 0.836 |
Pneumonia       | 0.8714 +- 0.0126 | 0.883    |
Lung opacity       | 	0.7971 +- 0.0152 | 0.816   |
Lung lesion    | 0.8286 +- 0.0529	 | 0.844   |
Enlarged Cardiomediastinum      | 0.8768 +- 0.035	 | 0.876  |
Edema      | 0.9147 +- 0.0072		 |0.917	 |
Consolidation    | 0.9181 +- 0.0183		 | 0.929 |
Cardiomegaly      | 0.908 +- 0.0038		 | 0.914 |
Atelectasis     | 0.7654 +- 0.0132	 | 0.779	 |
Lenght of stay     | 0.9323 +- 0.0115		 | 0.939|
48 hours mortality     | 0.9066 +- 0.0072		 | 0.912	|

More statistics and metrics are reported from each of the 12 experiments above and can be found in the ``experiments`` directory, each experiment directory is named after the task on which the prediction model was evaluated.

## 4. Prediction of one single task using all sources combinations
We tried to reproduce the HAIM experiment and use all the possible sources combinations to predict the presence or absence of a fracture in a patient. 
Below the ``AUC`` values reported from our experiments compared to the one reported in the HAIM paper. 
 AUC from our experiment | AUC from the paper |
 -----------| ----------- |
 // | 0.838 |
 
 
The above experiment can be performed using the following command
```
$ python run_experiments.py -t Fracture
```
A ``json file`` named ``Fracture_all_sources_recap.json`` is generated at the end of the experiment containing more statistics and metrics values.
## 5. Issues 
While working on reproducing HAIM experiments, we observed some problems on the published embedded dataset. While img_id is supposed to uniquely identify each image, redundant img_ids belonging to different patients were found in the dataset, see ``corrupted_ids.py`` for further details. 
