from __future__ import absolute_import
import warnings

from .dukemtmc import DukeMTMC
from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .unreal import UnrealPerson
from .cuhk03 import CUHK03
from .ilids import iLIDS
from .grid import GRID
from .viper import VIPeR
from .prid import PRID
from .cuhk02 import CUHK02
from .cuhksysu import CUHKSYSU


__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMC,
    'msmt17': MSMT17,
    'personx': PersonX,
    'unreal': UnrealPerson,
    'cuhk03': CUHK03,
    'ilids': iLIDS,
    'grid': GRID,
    'viper': VIPeR,
    'prid': PRID,
    'cuhk02': CUHK02,
    'cuhksysu': CUHKSYSU,
}


def names():
    return sorted(__factory.keys())


def create(name, root, combineall=False, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. 
    root : str
        The path to the dataset directory.
    split_id : int, optional
        The index of data split. Default: 0
    num_val : int or float, optional
        When int, it means the number of validation identities. When float,
        it means the proportion of validation to all the trainval. Default: 100
    download : bool, optional
        If True, will download the dataset. Default: False
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, combineall=combineall, *args, **kwargs)


def get_dataset(name, root, combineall=False, *args, **kwargs):
    warnings.warn("get_dataset is deprecated. Use create instead.")
    return create(name, root, combineall=combineall, *args, **kwargs)
