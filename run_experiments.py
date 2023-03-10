"""
Filename: run_experiments.py

Author : Hakima Laribi

Description: This file is used to perform all the HAIM experiments presented
                in the paper: https://doi.org/10.1038/s41746-022-00689-4

Date of last modification : 2023/02/07
"""

import argparse
from itertools import combinations
from tqdm import tqdm
from typing import List, Callable, Optional

from numpy import unique
from pandas import read_csv, DataFrame
from xgboost import XGBClassifier

from src.data import constants
from src.data.dataset import Task, HAIMDataset
from src.data.sampling import Sampler
from src.evaluation.evaluating import Evaluator
from src.evaluation.tuning import SklearnTuner
from src.utils.metric_scores import *


def get_all_sources_combinations(sources: List[str]) -> List[List[str]]:
    """
        Function to extract all possible combinations of sources

        Args:
            sources(List[str]): list of sources types

        Returns: list of combinations
    """
    comb = []
    for i in range(len(sources)):
        combination = combinations(sources, i + 1)
        for c in combination:
            comb.append(list(c))

    return comb


def run_single_experiment(prediction_task: str,
                          sources_predictors: List[str],
                          sources_modalities: List[str],
                          dataset: Optional[DataFrame] = None,
                          evaluation_name: Optional[str] = None) -> None:
    """
        Function to perform one single experiment

        Args:
            prediction_task(task): task label, must be a HAIM prediction task
            sources_predictors(List[str]): predictors to use for prediction, each source has one or more predictors
            sources_modalities(List[str]): the modalities of the sources used for prediction
            dataset(Optional[DataFrame]): HAIM dataframe
            evaluation_name(Optional[str]): name of the experiment
    """
    dataset = read_csv(constants.FILE_DF, nrows=constants.N_DATA) if dataset is None else dataset

    # Create the HAIMDataset
    dataset = HAIMDataset(dataset,
                          sources_predictors,
                          sources_modalities,
                          prediction_task,
                          constants.IMG_ID,
                          constants.GLOBAL_ID)

    # Sample the dataset using a 5-folds cross-validation method
    sampler = Sampler(dataset, constants.GLOBAL_ID, 5)
    _, masks = sampler()

    # Initialization of the list containing the evaluation metrics
    evaluation_metrics = [BinaryAccuracy(),
                          BinaryBalancedAccuracy(),
                          BinaryBalancedAccuracy(Reduction.GEO_MEAN),
                          Sensitivity(),
                          Specificity(),
                          AUC(),
                          BrierScore(),
                          BinaryCrossEntropy()]

    # Define the grid of hyper-parameters for the tuning
    grid_hps = {'max_depth': [5, 6, 7, 8],
                'n_estimators': [200, 300],
                'learning_rate': [0.3, 0.1, 0.05],
                }

    # Save the fixed parameters of the model
    fixed_params = {'seed': 42,
                    'eval_metric': 'logloss',
                    'verbosity': 1
                    }

    # Launch the evaluation
    evaluation = Evaluator(dataset=dataset,
                           masks=masks,
                           metrics=evaluation_metrics,
                           model=XGBClassifier,
                           tuner=SklearnTuner,
                           tuning_metric=AUC(),
                           hps=grid_hps,
                           n_tuning_splits=5,
                           fixed_params=fixed_params,
                           filepath=constants.EXPERIMENT_PATH,
                           weight='scale_pos_weight',
                           evaluation_name=evaluation_name
                           )
    evaluation.evaluate()


if __name__ == '__main__':
    # Get arguments passed
    parser = argparse.ArgumentParser(description='Select a specific task')
    parser.add_argument('-t', '--task', help='prediction task to evaluate through all sources combinations',
                        default=None, dest='task')
    args = parser.parse_args()

    # Load the dataframe from disk
    df = read_csv(constants.FILE_DF, nrows=constants.N_DATA)

    all_tasks = Task() if args.task is None else [args.task]

    for task in all_tasks:
        print("#"*23, f"{task} experiment", "#"*23)
        # Get all possible combinations of sources for the current task
        sources_comb = get_all_sources_combinations(constants.SOURCES) if task in [constants.MORTALITY, constants.LOS] \
            else get_all_sources_combinations(constants.CHEST_SOURCES)

        with tqdm(total=len(sources_comb)) as bar:
            for count, combination in enumerate(sources_comb):

                # Get all predictors and modalities for each source
                predictors = []
                for c in combination:
                    predictors = predictors + c.sources
                modalities = unique([c.modality for c in combination])

                run_single_experiment(prediction_task=task, sources_predictors=predictors, sources_modalities=modalities,
                                      dataset=df, evaluation_name=task + '_' + str(count))
                bar.update()

        Evaluator.get_best_of_experiments(task, constants.EXPERIMENT_PATH, count)
