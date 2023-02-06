"""
Filename: constants.py
Author : Hakima Laribi
Description: This file is used to store helpful constants
Date of last modification : 2023/01/12
"""

# All data modalities
TAB = 'tabular'
TS = 'time_series'
TXT = 'text'
IMG = 'images'


# Store each source type in a class structure, for each class:
# name: the name of the source type in the dataset
# n_embeddings: number of embeddings extracted for this source type in the dataset
# sources: column names of all extracted embeddings of the source type
# modality: the modality to which the source type belongs

class DEMOGRAPHIC:
    name = 'de'
    n_embeddings = 6
    sources = [f"de_{i}" for i in range(n_embeddings)]
    modality = TAB


class CHART:
    name = 'ce'
    n_embeddings = 99
    sources = [f"ts_ce_{i}" for i in range(n_embeddings)]
    modality = TS


class LAB:
    name = 'le'
    n_embeddings = 242
    sources = [f"ts_le_{i}" for i in range(n_embeddings)]
    modality = TS


class PROC:
    name = 'pe'
    n_embeddings = 110
    sources = [f"ts_pe_{i}" for i in range(n_embeddings)]
    modality = TS


class RAD:
    name = 'rad'
    n_embeddings = 768
    sources = [f"n_rad_{i}" for i in range(n_embeddings)]
    modality = TXT


class ECG:
    name = 'ecg'
    n_embeddings = 768
    sources = [f"n_ecg_{i}" for i in range(n_embeddings)]
    modality = TXT


class ECHO:
    name = 'echo'
    n_embeddings = 768
    sources = [f"n_ech_{i}" for i in range(n_embeddings)]
    modality = TXT


class VP:
    name = 'vp'
    n_embeddings = 18
    sources = [f"vp_{i}" for i in range(n_embeddings)]
    modality = IMG


class VMP:
    name = 'vmp'
    n_embeddings = 18
    sources = [f"vmp_{i}" for i in range(n_embeddings)]
    modality = IMG


class VD:
    name = 'vd'
    n_embeddings = 1024
    sources = [f"vd_{i}" for i in range(n_embeddings)]
    modality = IMG


class VMD:
    name = 'vmd'
    n_embeddings = 1024
    sources = [f"vmd_{i}" for i in range(n_embeddings)]
    modality = IMG


# Group all sources types in a list
SOURCES = [DEMOGRAPHIC, CHART, LAB, PROC, RAD, ECG, ECHO, VP, VMP, VD, VMD]

# Group all predictors of all sources in a list
ALL_PREDICTORS = DEMOGRAPHIC.sources + CHART.sources + LAB.sources + PROC.sources + RAD.sources + ECG.sources + \
                 ECHO.sources + VP.sources + VMP.sources + VD.sources + VMD.sources

# Group all chest sources types in a list
CHEST_SOURCES = [DEMOGRAPHIC, CHART, LAB, PROC, ECG, ECHO, VP, VMP, VD, VMD]

# Group all predictors of chest sources in a list
CHEST_PREDICTORS = DEMOGRAPHIC.sources + CHART.sources + LAB.sources + PROC.sources + ECG.sources + \
                   ECHO.sources + VP.sources + VMP.sources + VD.sources + VMD.sources


# Group all modalities in a list
ALL_MODALITIES = [TAB, TS, TXT, IMG]

# ID columns
IMG_ID = 'img_id'

GLOBAL_ID = 'haim_id'

# Number of valid data in the HAIM dataset
N_DATA = 45050

# File where the dataset is stored
FILE_DF = 'datasets/cxr_ic_fusion_1103.csv'

EXPERIMENT_PATH = 'experiments'

# All tasks names
FRACTURE = 'Fracture'
LUNG_LESION = 'Lung Lesion'
ENLARGED_CARDIOMEDIASTINUM = 'Enlarged Cardiomediastinum'
CONSOLIDATION = 'Consolidation'
PNEUMONIA = 'Pneumonia'
LUNG_OPACITY = 'Lung Opacity'
ATELECTASIS = 'Atelectasis'
PNEUMOTHORAX = 'Pneumothorax'
EDEMA = 'Edema'
CARDIOMEGALY = 'Cardiomegaly'
MORTALITY = '48h mortality'
LOS = '48h los'

# AUC values from the paper

AUC = {'HAIM': {FRACTURE: 0.838,
                LUNG_LESION: 0.844,
                ENLARGED_CARDIOMEDIASTINUM: 0.876,
                CONSOLIDATION: 0.929,
                PNEUMONIA: 0.883,
                ATELECTASIS: 0.779,
                LUNG_OPACITY: 0.816,
                PNEUMOTHORAX: 0.836,
                EDEMA: 0.917,
                CARDIOMEGALY: 0.914,
                LOS: 0.939,
                MORTALITY: 0.912},

       'NON_HAIM': {FRACTURE: 0.787,
                    LUNG_LESION: 0.831,
                    ENLARGED_CARDIOMEDIASTINUM: 0.868,
                    CONSOLIDATION: 0.920,
                    PNEUMONIA: 0.876,
                    ATELECTASIS: 0.767,
                    LUNG_OPACITY: 0.813,
                    PNEUMOTHORAX: 0.804,
                    EDEMA: 0.912,
                    CARDIOMEGALY: 0.912,
                    LOS: 0.919,
                    MORTALITY: 0.889}
       }
