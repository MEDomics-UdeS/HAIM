"""
Filename: Tuning.py

Author : Hakima Laribi

Description: This file is used to store objects used for tuning

Date of last modification : 2023/01/09

"""
from abc import abstractmethod, ABC
from typing import Callable, Dict, List, Any, Union
from numpy import array
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer


class SklearnHpsOptimizer:
    """
        Object used to store Scikit-learn hyper-parameter optimizers labels
    """
    GS = 'grid_search'
    RS = 'random_search'

    def __iter__(self):
        return iter([self.GS, self.RS])

    def __getitem__(self, item: str) -> Union[Callable, None]:
        if item == self.GS:
            return GridSearchCV
        elif item == self.RS:
            return RandomizedSearchCV
        else:
            raise ValueError(f"{item}: hyper-parameters optimizer not supported")


class Tuner(ABC):
    """
        Abstract class used to define tuner skeleton
    """

    def __init__(self,
                 metric: Callable,
                 hps: Dict[str, List[Any]],
                 n_splits: int,
                 parallel: bool = True):
        """
            Sets protected attributes

            Args:
                metric: callable function to optimize
                hps: dictionary with the hyper-parameters to optimize and the corresponding values to explore
                n_splits: number of inner splits on which test each combination of hyper-parameters
                parallel: weather to run the tuning in parallel or not
        """
        # Set protected attributes
        self._metric = metric
        self._hps = hps
        self._n_splits = n_splits
        self._n_cpus = -1 if parallel else 1  # Use all cpus for parallel tuning or a single one only

    @abstractmethod
    def tune(self,
             model: Any,
             x: array,
             y: array):
        """
            Performs the tuning of a model on specific data

            Args:
                model: the model to which find the best combination of hyper-parameters
                x: (N, D) array where N is the number of observations and D the number of predictors
                y: (N, 1) ground truth labels
        """
        raise NotImplementedError

    @abstractmethod
    def get_best_hps(self):
        """
            Returns the combination of hyper-parameters which optimized the metric value
        """
        raise NotImplementedError


class SklearnTuner(Tuner):
    """
        Object used to perform hyper-parameter tuning using Scikit-learn optimizers
    """

    def __init__(self,
                 metric: Callable,
                 hps: Dict[str, List[Any]],
                 n_splits: int,
                 model_selector: str = SklearnHpsOptimizer.GS,
                 parallel: bool = True):
        """
             Sets protected attributes

             Args:
                metric: callable function to optimize
                hps: dictionary with the hyper-parameters to optimize and the corresponding values to explore
                n_splits: number of inner splits on which test each combination of hyper-parameters
                parallel: weather to run the tuning in parallel or not

        """
        # Validation of inputs
        if model_selector not in SklearnHpsOptimizer():
            raise ValueError(f"{model_selector}: unsupported hyper-parameters optimizer in Scikit-Learn")

        # Create a custom scikit-learn score metric
        metric = make_scorer(metric, greater_is_better=(metric.direction == 'maximize'))

        # Call parent constructor
        super().__init__(metric, hps, n_splits, parallel)
        # Get the Hyper-parameter optimizer in Scikit-learn
        self._hps_optimizer = SklearnHpsOptimizer()[model_selector]
        self._optimizer = None

    def tune(self,
             model: BaseEstimator,
             x: array,
             y: array) -> BaseEstimator:
        """
            Performs the tuning of a model on specific data using a Scikit-learn Hyper-parameters optimizer

            Args:
                model: the model to which find the best combination of hyper-parameters
                x: (N, D) array where N is the number of observations and D the number of predictors
                y: (N, 1) ground truth labels

            Returns:
                Scikit-Learn optimized model
        """
        if not isinstance(model, BaseEstimator):
            raise ValueError(f"{model} is not a Scikit-Learn estimator, cannot perform tuning with SKlearnTuner")

        # instantiate the hyper-parameter Sklearn optimizer
        self._optimizer = self._hps_optimizer(model, self._hps, scoring=self._metric, cv=self._n_splits,
                                              n_jobs=self._n_cpus, refit=True, verbose=0)
        # Launch the hyper-parameter optimization
        self._optimizer.fit(x, y)

        # return the model with best hyper-parameters
        return self._optimizer.best_estimator_

    def get_best_hps(self):
        """
            Returns the combination of hyper-parameters which optimized the metric value
        """
        return self._optimizer.best_params_
