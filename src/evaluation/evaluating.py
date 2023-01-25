"""
Filename: evaluating.py

Author : Hakima Laribi

Description: This file is used to store Evaluator object which performs different evaluations on the dataset

Date of last modification : 2023/01/11

"""
import shutil
import os
from typing import Any, Callable, Dict, Union, List, Optional
from src.evaluation.tuning import SklearnTuner, SklearnHpsOptimizer
from src.data.dataset import HAIMDataset
from sklearn.metrics import roc_curve
from numpy import argmax, array, mean, std, median, min, max
from os import makedirs, path
from time import strftime
import json
import pandas as pd
import matplotlib.pyplot as plt
from re import search
import numpy as np
from src.data.constants import *


class Evaluator:
    """
        Object used to perform models evaluations
    """

    def __init__(self,
                 dataset: HAIMDataset,
                 masks: Dict[int, Dict[str, List[int]]],
                 metrics: List[Callable],
                 model: Callable,
                 tuner: Callable,
                 tuning_metric: Callable,
                 hps: Dict[str, Union[List[Any], Any]],
                 n_tuning_splits: int,
                 fixed_params: Dict[str, Any],
                 filepath: str,
                 model_selector: str = SklearnHpsOptimizer.GS,
                 parallel_tuning: bool = True,
                 weight: Optional[str] = None,
                 evaluation_name: Optional[str] = None,
                 stratified_sampling=False):
        """
            Sets protected attributes of the Evaluator

            Args:
                 dataset: custom HAIM dataset
                 masks: dictionary with train, test and valid set at each split
                 metrics: list of metrics to report at the end of the experiment
                 model: model to evaluate
                 tuner: tuner which perform hyper-parameter tuning
                 tuning_metric: metric to optimize (maximize/minimize) during the hyper-parameter tuning
                 n_tuning_splits: number of inner data splits, used in the tuning
                 fixed_params: model's fixed parameters
                 filepath: path to the directory where to store the experiment
                 model_selector: hyper-parameter optimizer
                 parallel_tuning: boolean to specify if the tuning would be in parallel
                 weight: weight parameter
                 evaluation_name: name of current evaluation
                 stratified_sampling: if the sampling performed was stratified

        """

        if evaluation_name is not None:
            if path.exists(path.join(filepath, evaluation_name)):
                raise ValueError("evaluation with this name already exists")
        else:
            makedirs(filepath, exist_ok=True)
            evaluation_name = f"{strftime('%Y%m%d-%H%M%S')}"

        # Set protected attributes
        self._dataset = dataset
        self._masks = masks
        self._metrics = metrics
        self._model = model
        self._tuner = tuner(tuning_metric, hps, n_tuning_splits, model_selector, parallel_tuning)
        self._inner_splits = n_tuning_splits
        self._fixed_params = fixed_params
        self._weighted_param = weight
        self._filepath = path.join(filepath, evaluation_name)
        self.evaluation_name = evaluation_name
        self._stratified_sampling = stratified_sampling

    def evaluate(self):
        """
            Performs nested evaluation and saves the result of each experiment in a json file
        """
        # Perform the evaluation over all the splits of the dataset
        for i, mask in self._masks.items():
            # Extract masks
            train, test, valid = mask['train'], mask['test'], mask['valid']
            # Get data and targets for each mask
            x, y = {}, {}
            for (mask_, idx) in [('train', train), ('test', test), ('valid', valid)]:
                if idx is not None:
                    x[mask_], y[mask_] = self._dataset[idx]
                else:
                    x[mask_], y[mask_] = None, None

            if self._weighted_param is not None:
                # Compute weights to assign to the positive class
                positive_weight = (len(y['train']) - sum(y['train'])) / sum(y['train'])
                # Update the fixed parameters of the model
                self._fixed_params[self._weighted_param] = positive_weight
            if self._inner_splits > 0:
                # Perform the tuning to extract the model with the best hyper-pramaters
                best_model = self._tuner.tune(self._model(**self._fixed_params), x['train'], y['train'])
                best_hps = self._tuner.get_best_hps()

            else:
                best_model = self._model(**self._fixed_params)
                best_hps = self._fixed_params

            # Get probabilities predicted for each mask
            y_proba = {}
            for mask_ in ['train', 'test', 'valid']:
                y_proba[mask_] = best_model.predict_proba(x[mask_])[:, 1] if x[mask_] is not None else None

            # Get predictions on the training set and compute the optimal threshold on the training set
            threshold = self.optimize_j_statistic(y['train'], y_proba['train'])

            # Save the experiment of the current split in a json file
            self.record_experiment(y_proba, y, threshold, mask, i, best_hps)

        # Summarize experiment over all the splits
        self.summarize_experiment(i + 1)

    @staticmethod
    def optimize_j_statistic(targets: list,
                             pred: list) -> float:
        """
        find the optimal threshold from ROC curve that separates the negative and positive classes by optimizing
        he function the Youden's J statistics J = TruePositiveRate – FalsePositiveRate
        Args:
            targets: ground truth labels
            pred: predicted probabilities to belong to the positive class (OYM=1)

        Returns a float representing the optimal threshold
        """
        # Calculate roc curves
        fpr, tpr, thresholds = roc_curve(targets, pred, pos_label=1)

        # Get the best threshold
        J = tpr - fpr
        threshold = thresholds[argmax(J)]

        return threshold

    def record_experiment(self,
                          predictions: Dict[str, List[float]],
                          targets: Dict[str, List[int]],
                          threshold: float,
                          masks: Dict[str, List[int]],
                          split: int,
                          hps: Dict[Any, Any]
                          ):
        """
            Records the results of one single experiment on the test, train and valid sets in a json file

            Args:
                predictions: probabilities predicted on all the sets
                targets: ground truth labels of observation in each set
                threshold: prediction threshold
                masks: train, test and valid masks
                split: index of the current split
                hps: best hyper-parameters selected after the tuning

        """
        # Create the saving directory
        saving_path = self._filepath + '/split_' + str(split)
        makedirs(saving_path)

        # Initialize the file structure
        summary = {'split': str(split),
                   'sources': str(self._dataset.sources),
                   'task': self._dataset.task,
                   'threshold': str(threshold),
                   'hyper-parameters': hps}
        # Save statistics of each set
        for mask, idx in masks.items():
            if idx is not None:
                # Save number of elements in current set
                summary['N_' + mask + 'ed'] = len(idx)

                # If observations has a global_id according to which the sampling was performed
                if (self._dataset.global_ids is not None) and (not self._stratified_sampling):
                    summary['N_' + mask + 'ed_global_ids'] = len(self._dataset.get_global_ids(idx))

                summary[f"proportion_positive_class_{mask}ed"] = ''

                # Initialize the metrics recorded
                summary[mask + '_metrics'] = {}

        # Get metrics and prediction values for each mask
        for mask, idx in masks.items():
            if idx is not None:

                # Get predicted probabilities and ground truth labels and predicted classes
                y_proba, target = predictions[mask], targets[mask]
                y_pred = (y_proba >= threshold).astype(float)

                # Save proportion of classes
                summary[f"proportion_positive_class_{mask}ed"] = f"{round(np.sum(target) / len(target), 4) * 100} %"

                # Save metrics for current mask
                for metric in self._metrics:
                    summary[mask + '_metrics'][metric.name] = str(metric(y_proba, target, threshold))

                # Initialize the predictions recorded
                summary[mask + '_predictions'] = {}

                # Save predictions for each element
                if (self._dataset.global_ids is not None) and (not self._stratified_sampling):
                    # Map indexes to global ids
                    map_idx_global_ids = self._dataset.map_idx_to_global_ids()

                    # Get the global_ids of the indexes present in the current set
                    global_ids = self._dataset.get_global_ids(idx)

                    # Map indexes to their position in the current mask
                    map_idx_positions = {index: i for i, index in enumerate(idx)}

                    # Map each index to its id in the dataset
                    map_idx_to_ids = self.reverse_map(self._dataset.map_idx_to_ids())

                    # Save predictions for each global id
                    for global_id in global_ids:
                        summary[mask + '_predictions'][str(global_id)] = {}

                        # Initialize the information structure for each global_id
                        summary[mask + '_predictions'][str(global_id)] = {}

                        # Get the indexes of the observations present in the global id
                        indexes = [i for i in map_idx_global_ids[global_id] if i in idx]

                        # Save predictions for each observation
                        for index in indexes:
                            summary[mask + '_predictions'][str(global_id)][str(index)] = {
                                'id': str(map_idx_to_ids[index]),
                                'prediction': str(y_pred[map_idx_positions[index]]),
                                'probability': str(y_proba[map_idx_positions[index]]),
                                'target': str(target[map_idx_positions[index]])
                            }
                else:
                    # Map each index to its id in the dataset
                    map_idx_to_ids = self.reverse_map(self._dataset.map_idx_to_ids())

                    # Save predictions of each observation independently
                    for i, index in enumerate(idx):
                        summary[mask + '_predictions'][str(index)] = {
                            'index': str(index),
                            'id': str(map_idx_to_ids[index]),
                            'prediction': str(y_pred[i]),
                            'probability': str(y_proba[i]),
                            'target': str(target[i])
                        }

                # Generate ROC curve
                self.plot_roc_curve(saving_path, target, y_proba, mask)

        # Generate the Json file of the split

        with open(path.join(saving_path, 'records.json'), "w") as file:
            json.dump(summary, file, indent=True)

    def summarize_experiment(self,
                             n_splits: int
                             ):
        """
            Summarizes an experiment performed on different splits of the dataset ans saves it in a json file, the mean,
            standard deviation, min and max are computed for each metric.

            Args:
                n_splits: number of splits the model was evaluated on
        """
        metrics_values = {}
        # Get the folders where each split evaluation was saved
        folders = [path.join(self._filepath, 'split_' + str(i)) for i in range(n_splits)]

        for folder in folders:
            with open(path.join(folder, 'records.json'), "r") as read_file:
                split_data = json.load(read_file)

            # Get metric values over all the splits
            for section, data in split_data.items():
                # Get the sections where the metrics are saved
                if search("(metric)", section):
                    # For each split, get the value of the metric
                    for metric, value in data.items():
                        try:
                            metrics_values[section][metric].append(float(value))
                        except KeyError:
                            try:
                                metrics_values[section][metric] = [float(value)]
                            except KeyError:
                                metrics_values[section] = {metric: [float(value)]}

        # Save statistics on the metrics
        recap = {}
        for section, data in metrics_values.items():
            recap[section] = {}
            for metric, values in data.items():
                values = array(values)
                mean_, std_ = round(mean(values), 4), round(std(values), 4)
                med_, min_, max_ = round(median(values), 4), round(min(values), 4), round(max(values), 4)
                recap[section][metric] = {
                    'info': f"{mean_} +- {std_} [{med_}; {min_}-{max_}]",
                    'mean': mean_,
                    'std': std_,
                }

        # Save the file in disk
        with open(path.join(self._filepath, 'recap.json'), "w") as file:
            json.dump(recap, file, indent=True)

    @staticmethod
    def visualize_results(file_path: str,
                          task: str,
                          recap_file: str = None,
                          ) -> pd.DataFrame:
        """
            Saves metrics scores regrouped in the recap json file in a dataframe and prints it

            Args:
                recap_file: recap json file of the experiment
                task: prediction task
                file_path: directory where the experiment is saved

        """
        recap_file = 'recap.json' if recap_file is None else recap_file
        with open(path.join(file_path, recap_file), "r") as file:
            recap = json.load(file)

        metrics = {}
        # Get the mean and std for each metric in all the sets (train, test and valid)
        for _set, values in recap.items():
            for metric, stats in values.items():
                try:
                    metrics[metric][_set] = str(stats['mean']) + ' +- ' + str(stats['std'])
                except KeyError:
                    metrics[metric] = {_set: str(stats['mean']) + ' +- ' + str(stats['std'])}

        for _set in ['HAIM', 'NON_HAIM']:
            for metric in metrics.keys():
                if metric == 'AUC':
                    metrics[metric][_set] = str(AUC[_set][task])
                else:
                    metrics[metric][_set] = '--'

        # Transform the dictionary to a dataframe
        df_metrics = pd.DataFrame(metrics)

        return df_metrics

    @staticmethod
    def get_best_of_experiments(file_format: str,
                                path_file: str,
                                n_files: int) -> None:
        metric_values = []
        # Get the folders where each recap evaluation was saved
        folders = [path.join(path_file, file_format + str(i)) for i in range(n_files)]

        for folder in folders:
            with open(path.join(folder, 'recap.json'), "r") as read_file:
                recap_data = json.load(read_file)
                # Get AUC values of all experiments
                infos = recap_data["test_metrics"]['AUC']
                metric_values.append(float(infos['mean']))

        best_experiment = np.argmax(np.array(metric_values))

        # Copy the files of the best experiment to the directory file_format_best_experiment
        shutil.copytree(folders[best_experiment], path.join(path_file, file_format+'_best_experiment'))

    @staticmethod
    def reverse_map(map_: Dict[Any, Any]):
        """
            Reverses the keys and values of a dictionary

            Args:
                map_: dictionary
        """
        reversed_map = {}
        for k, v in map_.items():
            for value in v:
                reversed_map[value] = k
        return reversed_map

    @staticmethod
    def plot_roc_curve(
            saving_path: str,
            targets: array,
            y_proba: array,
            mask: str):

        """
            Plots the Area Under AUC curve and saves it

            Args:
                saving_path: path where to save the figure
                targets: ground truth labels
                y_proba: probabilities predicted
                mask: label of the current mask
        """

        fpr, tpr, _ = roc_curve(targets, y_proba, pos_label=1)

        # create ROC curve
        plt.clf()
        plt.plot(fpr, tpr)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        # Save the figure in the disk
        plt.savefig(path.join(saving_path, 'roc_curve_' + mask + '.png'))
