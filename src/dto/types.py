from enum import Enum


class DatasetType(str, Enum):
    ETHICS = "ethics"
    SOCIAL_CHEMISTRY_101 = "social_chemistry_101"
    MIXED_PVA = "mixed_pva"
