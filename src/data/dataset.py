"""
Filename: dataset.py
Author : Hakima Laribi
Description: This file is used to define classes related to the dataset
Date of last modification : 2023/01/11
"""

import pandas as pd
from typing import List, Union, Tuple, Optional, Dict
from src.data.constants import *
import numpy as np


class Task:
    """
        Stores the constants related to the task name
    """

    def __iter__(self):
        return iter([FRACTURE, LUNG_LESION, ENLARGED_CARDIOMEDIASTINUM, CONSOLIDATION,
                     PNEUMONIA, LUNG_OPACITY, ATELECTASIS, PNEUMOTHORAX, EDEMA,
                     CARDIOMEGALY, MORTALITY, LOS])


class HAIMDataset:
    """
            Custom dataset for the HAIM experiment
    """

    def __init__(self,
                 original_dataset: pd.DataFrame,
                 sources: List[str],
                 modalities: List[str],
                 task: str,
                 ids: str,
                 global_ids: Optional[str] = None,
                 ):
        """
           Sets the protected attributes of the custom dataset
            Args:
                   original_dataset: datatframe with the original data
                   sources: list with the predictors to use in prediction, each predictor belongs to a specific source
                   modalities: list with the modalities of the sources
                   task: the prediction task to perform
                   ids: id column in the dataset
                   global_ids: id column which is not unique for each observation, but a unique ids is associated with
                                each global_id
        """

        # Validation of inputs
        for column in ['img_length_of_stay', 'death_status']:
            if column not in original_dataset.columns:
                raise ValueError(f"HAIM dataset must contain {column} column")

        if task not in Task():
            raise ValueError(f"{task} is not a supported prediction problem")

        for source in sources:
            if source not in original_dataset.columns:
                raise ValueError(f"{source} can't be resolved to a column name")

        if ids not in original_dataset.columns:
            raise ValueError(
                f"{ids} column missing from the dataset. The dataset must contain observations' identifiers")

        if global_ids is not None and global_ids not in original_dataset.columns:
            raise ValueError(f"{global_ids} column missing from the dataset.")

        # Set protected attributes
        self._sources = sources
        self._modalities = modalities
        self._original_dataset = original_dataset
        self._task = task
        self._ids = ids
        self._global_ids = global_ids
        self._x, self._y, self._task_dataset = None, None, None

        # Extract x and y from the dataset
        self.__create_dataset()

    def __len__(self):
        return self._task_dataset.shape[0]

    def __getitem__(self, idx: Union[int, List[int], str, pd.DataFrame, pd.Series]
                    ) -> Union[Tuple[np.array, np.array], pd.DataFrame, pd.Series]:
        """
           Gets specific rows in the dataset

            Args:
                idx: list of int or int of indexes from which to get associated rows in the dataset, or a dataframe /
                    serie of booleans with same length as the dataset or a string representing the name of a column
                    in the dataset

            Returns: (array of datas and targets) or a dataframe

        """

        if isinstance(idx, int):
            return self.x[idx], self.y[idx]
        if isinstance(idx, list) and isinstance(idx[0], int):
            return self.x[idx], self.y[idx]
        else:
            return self.task_dataset[idx]

    @property
    def modalities(self) -> List[str]:
        return self._modalities

    @property
    def sources(self) -> List[str]:
        return self._sources

    @property
    def task(self) -> str:
        return self._task

    @property
    def ids(self) -> str:
        return self._ids

    @property
    def global_ids(self) -> str:
        return self._global_ids

    @property
    def task_dataset(self) -> pd.DataFrame:
        return self._task_dataset

    @property
    def x(self) -> np.array:
        return self._x

    @property
    def y(self) -> np.array:
        return self._y

    def __create_dataset(self):
        """
            Creates the HAIM dataset to use in downstream prediction tasks
        """

        # Compute targets for non-chest pathologies tasks
        if self._task == MORTALITY:
            self._original_dataset[self._task] = None
            # If the patient died within 48 hours after the hospital admission
            self._original_dataset.loc[((self._original_dataset['img_length_of_stay'] < 48) &
                                        (self._original_dataset['death_status'] == 1)), self._task] = 1

            # self._original_dataset[((self._original_dataset['img_length_of_stay'] < 48) &
            #                        (self._original_dataset['death_status'] == 1))][self._task] = 1

            # If the patient survived
            self._original_dataset.loc[self._original_dataset['death_status'] == 0, self._task] = 0

            # If the patient died after 48 hours of the hospital admission
            self._original_dataset.loc[((self._original_dataset['img_length_of_stay'] >= 48) &
                                        (self._original_dataset['death_status'] == 1)), self._task] = 0

        elif self._task == LOS:
            # If the patient was discharged less than 48 hours after the hospital admission alive
            self._original_dataset.loc[((self._original_dataset['img_length_of_stay'] < 48) &
                                    (self._original_dataset['death_status'] == 0)), self._task] = 1

            # If the patient died within 48 hours after the hospital admission
            self._original_dataset.loc[((self._original_dataset['img_length_of_stay'] < 48) &
                                    (self._original_dataset['death_status'] == 1)), self._task] = 0

            # If the patient stayed more than 48 hours in the hospital
            self._original_dataset.loc[self._original_dataset['img_length_of_stay'] >= 48, self._task] = 0

        # Select the lines where the target y = {0, 1}
        self._task_dataset = (self._original_dataset[((self._original_dataset[self._task] == 0) |
                                                      (self._original_dataset[self._task] == 1))]).reset_index(
            drop=True)

        # Free the space associated to the original dataset
        self._original_dataset = None

        # Extract x and y
        self._x = np.array(self._task_dataset[self._sources], float)
        self._y = np.array(self._task_dataset[self._task], int)

    def map_idx_to_global_ids(self) -> Union[Dict[int, List[int]], None]:
        """
            Maps the global_ids to all the observations indexes
        """

        if self._global_ids is not None:
            map_ids = {}
            global_ids = self.task_dataset[self._global_ids].tolist()

            for id_ in global_ids:
                map_ids[id_] = self.task_dataset.index[self.task_dataset[self._global_ids] == id_].tolist()

            return map_ids

        else:
            return None

    def map_idx_to_ids(self) -> Dict[int, List[int]]:
        """
            Maps the ids to all the observations indexes
        """
        map_ids = {}
        ids = self.task_dataset[self._ids].tolist()

        for id_ in ids:
            map_ids[id_] = self.task_dataset.index[self.task_dataset[self._ids] == id_].tolist()

        return map_ids

    def get_global_ids(self, indexes: List[int]) -> Union[List[int], None]:
        """
            Gets the list of global_ids
        """
        if self.global_ids is not None:
            return self.task_dataset.iloc[indexes][self.global_ids].unique().tolist()
        else:
            return None
