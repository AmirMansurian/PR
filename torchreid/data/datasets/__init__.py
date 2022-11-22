from __future__ import print_function, absolute_import

from .image import (
    GRID, PRID, CUHK01, CUHK02, CUHK03, MSMT17, VIPeR, SenseReID, Market1501,
    DukeMTMCreID, iLIDS, Synergy, EpflSport, OccludedDuke, OccludedReID, Partial_iLIDS, Partial_REID, PDukemtmcReid,
    P_ETHZ, SynergySequences, Dartfish, SoccerNet
)
from .image.de_challenge_synergy import DEChallengeSynergy
from .image.motchallenge import get_sequence_class, MOTChallenge
from .video import PRID2011, Mars, DukeMTMCVidReID, iLIDSVID
from .dataset import Dataset, ImageDataset, VideoDataset

__image_datasets = {
    'soccernet': SoccerNet,
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'viper': VIPeR,
    'grid': GRID,
    'cuhk01': CUHK01,
    'ilids': iLIDS,
    'sensereid': SenseReID,
    'prid': PRID,
    'cuhk02': CUHK02,
    'synergy': Synergy,
    'occluded_duke': OccludedDuke,
    'occluded_reid': OccludedReID,
    'partial_reid': Partial_REID,
    'partial_ilids': Partial_iLIDS,
    'p_ETHZ': P_ETHZ,
    'p_dukemtmc_reid': PDukemtmcReid,
    'epflsport': EpflSport,
    'synergy_sequences': SynergySequences,
    'de_challenge_synergy': DEChallengeSynergy,
    'dartfish': Dartfish,
    'MOTChallenge': MOTChallenge,
    'MOT17-02': get_sequence_class('MOT17-02-FRCNN'),
    'MOT17-04': get_sequence_class('MOT17-04-FRCNN'),
    'MOT17-05': get_sequence_class('MOT17-05-FRCNN'),
    'MOT17-09': get_sequence_class('MOT17-09-FRCNN'),
    'MOT17-10': get_sequence_class('MOT17-10-FRCNN'),
    'MOT17-11': get_sequence_class('MOT17-11-FRCNN'),
    'MOT17-13': get_sequence_class('MOT17-13-FRCNN'),
}

__datasets_nicknames = {
    'market1501': 'mk',
    'cuhk03': 'c03',
    'dukemtmcreid': 'du',
    'msmt17': 'ms',
    'viper': 'vi',
    'grid': 'gr',
    'cuhk01': 'c01',
    'ilids': 'il',
    'sensereid': 'se',
    'prid': 'pr',
    'cuhk02': 'c02',
    'synergy': 'sy',
    'occluded_duke': 'od',
    'occluded_reid': 'or',
    'partial_reid': 'pr',
    'partial_ilids': 'pi',
    'p_ETHZ': 'pz',
    'p_dukemtmc_reid': 'pd',
    'epflsport': 'epfl',
    'synergy_sequences': 'ss',
    'soccernet': 'sn',
    'de_challenge_synergy': 'dec',
    'dartfish': 'df',
    'MOT17-02': 'mc2',
    'MOT17-04': 'mc4',
    'MOT17-05': 'mc5',
    'MOT17-09': 'mc9',
    'MOT17-10': 'mc10',
    'MOT17-11': 'mc11',
    'MOT17-13': 'mc13',
}

__video_datasets = {
    'mars': Mars,
    'ilidsvid': iLIDSVID,
    'prid2011': PRID2011,
    'dukemtmcvidreid': DukeMTMCVidReID
}

def get_dataset_nickname(name):
    return __datasets_nicknames.get(name, name)


def get_image_dataset(name):
    avai_datasets = list(__image_datasets.keys())
    import time
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __image_datasets[name]


def init_image_dataset(name, **kwargs):
    """Initializes an image dataset."""
    return get_image_dataset(name)(**kwargs)


def init_video_dataset(name, **kwargs):
    """Initializes a video dataset."""
    avai_datasets = list(__video_datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __video_datasets[name](**kwargs)


def register_image_dataset(name, dataset):
    """Registers a new image dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_image_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.ImageDataManager(
            root='reid-data',
            sources=['new_dataset', 'dukemtmcreid']
        )
    """
    global __image_datasets
    curr_datasets = list(__image_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __image_datasets[name] = dataset


def register_video_dataset(name, dataset):
    """Registers a new video dataset.

    Args:
        name (str): key corresponding to the new dataset.
        dataset (Dataset): the new dataset class.

    Examples::
        
        import torchreid
        import NewDataset
        torchreid.data.register_video_dataset('new_dataset', NewDataset)
        # single dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources='new_dataset'
        )
        # multiple dataset case
        datamanager = torchreid.data.VideoDataManager(
            root='reid-data',
            sources=['new_dataset', 'ilidsvid']
        )
    """
    global __video_datasets
    curr_datasets = list(__video_datasets.keys())
    if name in curr_datasets:
        raise ValueError(
            'The given name already exists, please choose '
            'another name excluding {}'.format(curr_datasets)
        )
    __video_datasets[name] = dataset
