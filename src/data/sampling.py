"""
Filename: sampling.py

Author : Hakima Laribi

Description: This file is used to define the sampler class which creates train, test and valid masks

Date of last modification : 2023/02/06
"""
from tqdm import tqdm
from typing import Callable, List, Tuple, Union, Optional, Dict

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from src.data.dataset import HAIMDataset


class Sampler:
    """
        Object used in order to generate lists of indexes to use as train, valid and test masks
    """

    def __init__(self,
                 dataset: HAIMDataset,
                 split_column: str,
                 n_splits: int = 5,
                 test_size: float = 0.2,
                 valid_size: float = 0.0,
                 random_state: int = 1101,
                 cross_validation: bool = True,
                 stratify_column: str = None):
        """
            Sets public and protected attributed of the sampler

            Args:
                dataset(HAIMDataset): custom HAIM dataset
                split_column(str): the column according to which the data is sampled
                n_splits(int): number of distinct splits
                test_size(float): size of the test set
                valid_size(float): size of the valid set
                random_state(int): integer for reproducibility of experiments
                cross_validation(bool): boolean to specify weather perform a cross-validation or a sampling with
                 replacement
                stratify_column(str): if not None, specifies the column according to which perform a stratified sampling

        """

        # Validation of inputs
        if split_column not in dataset.task_dataset.columns:
            raise ValueError(f"{split_column} is not in the dataset columns.")

        if stratify_column is not None and stratify_column not in dataset.task_dataset.columns:
            raise ValueError(f"{stratify_column} is not in the dataset columns.")

        if cross_validation and (test_size * n_splits != 1.):
            raise ValueError(f"Cannot perform {n_splits} folds cross-validation with a testing set of size "
                             f"{test_size * 100}%")

        # Set the protected attributes
        self._dataset = dataset
        self._split_column = split_column
        self._n_splits = n_splits
        self._test_size = test_size if not cross_validation else 1 / float(n_splits)
        self._valid_size = valid_size
        self._random_state = random_state
        self._cv = cross_validation
        self._stratify = stratify_column

        # Set public attributes
        self.split = self.__define_split_function()

    def __call__(self) -> Tuple[Dict, Dict]:
        """
            Returns global_masks which contains the values of the split_column in each set, and the masks containing
            the indexes of observations in each set
        """
        # Get unique values to sample from the splitting column, either it's indexes or specific ids
        samples = np.array(self._dataset[self._split_column].unique().tolist()) if self._stratify is None \
            else np.array(self._dataset.task_dataset.index[self._dataset[self._dataset.ids is not None]].tolist())

        # Get targets values for stratified sampling
        targets = np.array(self._dataset[self._stratify].tolist()) if self._stratify is not None else None

        return self.split(samples, targets)

    def __define_split_function(self) -> Callable:
        """
            Defines the split function which creates the masks according to the type of sampling : cross-validation or
            with replacement, stratified or not.
        """
        # Initialize masks
        global_masks, masks = {}, {}

        # Define the split function, it can be a cross-validation with no replacement among all the testing sets
        # (i.e. intersection of all testing sets is null) or a sampler with replacement, both can be stratified or not
        if self._cv and self._stratify is not None:
            splitter = StratifiedKFold(n_splits=self._n_splits, shuffle=True, random_state=self._random_state)

            def split(indexes: np.array, targets: np.array) -> Tuple[Dict, Dict]:
                for i, (train, test) in enumerate(splitter.split(indexes, targets)):
                    # Get valid set
                    train, test = indexes[train], indexes[test]
                    train, valid = self.__get_valid_set(train, stratify=targets[train.tolist()],
                                                        random_state=self._random_state + i)
                    # Update masks
                    global_masks[i] = {'train': train.tolist(), 'test': test.tolist(),
                                       'valid': valid.tolist() if valid is not None else None}
                    masks[i] = {'train': train.tolist(), 'test': test.tolist(),
                                'valid': valid.tolist() if valid is not None else None}

                return global_masks, masks

        elif self._cv and self._stratify is None:
            splitter = KFold(n_splits=self._n_splits, shuffle=True, random_state=self._random_state)

            def split(indexes: np.array, targets: np.array) -> Tuple[Dict, Dict]:
                for i, (train, test) in enumerate(splitter.split(indexes, targets)):
                    # Get valid set
                    train, test = indexes[train], indexes[test]
                    train, valid = self.__get_valid_set(train, random_state=self._random_state + i)

                    # Get indexes sampled in each set
                    train_idx, test_idx, valid_idx = self.__get_idx(train, test, valid)

                    # Update masks
                    global_masks[i] = {'train': train.tolist(), 'test': test.tolist(),
                                       'valid': valid.tolist() if valid is not None else None}
                    masks[i] = {'train': train_idx, 'test': test_idx, 'valid': valid_idx}

                return global_masks, masks

        else:
            def split(indexes: np.array, targets: np.array) -> Tuple[Dict, Dict]:
                for i in range(self._n_splits):
                    # Split the dataset to train and test
                    train, test = train_test_split(indexes, test_size=self._test_size,
                                                   random_state=self._random_state + i, stratify=targets)
                    # Get valid set
                    train_targets = targets[train.tolist()] if targets is not None else None
                    train, valid = self.__get_valid_set(train, stratify=train_targets,
                                                        random_state=self._random_state + i)

                    # Get indexes sampled in each set
                    train_idx, test_idx, valid_idx = self.__get_idx(train, test, valid)

                    # Update masks
                    global_masks[i] = {'train': train.tolist(), 'test': test.tolist(),
                                       'valid': valid.tolist() if valid is not None else None}
                    masks[i] = {'train': train_idx, 'test': test_idx, 'valid': valid_idx}

                return global_masks, masks

        return split

    def __get_idx(self,
                  train_mask: List[int],
                  test_mask: List[int],
                  valid_mask: List[int] = None) -> Tuple[List[int], List[int], Union[List[int], None]]:
        """
            Gets indexes associated with the global masks

            Args:
                train_mask(List[int]): global ids in the train set
                test_mask(List[int]): global ids in the test set
                valid_mask(List[int]): global ids in the valid set

            Returns: Tuple of lists
        """

        # Get observations indexes in the dataframe corresponding to IDS of the split column selected in each set
        # for HAIM experiment, once the patients ids sampled between test, valid and train sets, indexes of the
        # different stays and visits as present in the dataframe are then collected
        m = self._dataset.task_dataset[self._split_column].isin(train_mask)
        train_idx = self._dataset.task_dataset.index[m].tolist()
        test_idx = self._dataset.task_dataset.index[self._dataset.task_dataset[self._split_column].isin(test_mask)]. \
            tolist()
        valid_idx = self._dataset.task_dataset.index[self._dataset.task_dataset[self._split_column].isin(valid_mask)]. \
            tolist() if valid_mask is not None else None

        return train_idx, test_idx, valid_idx

    def __get_valid_set(self,
                        train_mask: np.array,
                        stratify: Optional[np.array] = None,
                        random_state: int = None) -> Tuple[np.array, Union[np.array, None]]:

        """
            Splits the train set to the final train and valid sets

            Args:
                train_mask(np.array): (N,) ids of observations in the train set
                stratify(Optional[np.array]): (N,) if not None, data is split in a stratified fashion, using this as
                the class labels.
                random_state(int): seed to reproduce the results

            Returns: Tuple
        """

        train, valid = train_mask, None
        if self._valid_size > 0.:
            train, valid = train_test_split(train, test_size=self._valid_size, random_state=random_state,
                                            stratify=stratify)

        return train, valid
