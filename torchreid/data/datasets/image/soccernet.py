from __future__ import division, print_function, absolute_import
import glob
import os.path as osp
import os
import numpy as np
import time

from ..dataset import ImageDataset

class SoccerNet(ImageDataset):
    """Synergy Dataset.
    """
    _junk_pids = [0, -1]
    dataset_dir = 'reid_dataset'
    dataset_url = None
    masks_base_dir = 'masks'
    #eval_metric = 'soccernetv3'

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        'pifpaf': (36, False, '.confidence_fields.npy')
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in SoccerNet.masks_dirs:
            return None
        else:
            return SoccerNet.masks_dirs[masks_dir]

    def infer_masks_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path + self.masks_suffix
        return masks_path

    def infer_keypoints_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path + self.keypoints_suffix
        return masks_path

    def infer_segmentation_path(self, img_path): # FIXME remove when all datasets migrated
        masks_path = img_path + self.segmentation_suffix
        return masks_path

    def __init__(self, root='', masks_dir=None, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        #self.download_dataset(self.dataset_dir, self.dataset_url)
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None

         
        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        self.train_dir = osp.join(self.data_dir, 'train2')
        self.query_dir = osp.join(self.data_dir, 'query2')
        self.gallery_dir = osp.join(self.data_dir, 'gallery2')

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]

        self.check_before_run(required_files)
        np.random.seed(0)
        train, _ = self.process_dir(self.train_dir, 1)
        gallery, mapping = self.process_dir(self.gallery_dir, 2)
        query, _ = self.process_dir(self.query_dir, 3, mapping)


        super(SoccerNet, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, main_path, mode=1, mapping=[]):

        sequences_dir_list = [f for f in os.scandir(main_path) if f.is_dir()]
        data = []
        data2 = []
        id_dict = {}
        for seq_dir in sequences_dir_list:
            seq_name = seq_dir.name
            dir_path = osp.join(main_path, seq_dir)

            img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
            for img_path in img_paths:
                pid = img_path.split('/')[-1][:-4].split('_')[0]
                camid = img_path.split('/')[-1][:-4].split('_')[1]
                masks_path = self.infer_masks_path(img_path)

                if (int(pid)) not in id_dict.keys():
                    id_dict[int(pid)] = []

                id_dict[int(pid)].append({'img_path': img_path, 'pid': int(pid), 'masks_path': masks_path, 'camid': camid})

                data.append({'img_path': img_path,
                                'pid': int(pid),
                                'masks_path': masks_path,
                                'camid': camid})

        if mode == 1:
            maps = list(set([player['pid'] for player in data]))
            for i, player in enumerate(data):
                idx = maps.index(player['pid'])
                data[i]['pid'] = idx

            return data, None

        if mode == 2:
            ids = list(set([i['pid'] for i in data]))
            index = np.random.permutation(len(ids))[:350]
            indice = [ids[j] for j in index]
            for player in indice:
                idx = np.random.permutation(len(id_dict[player]))[:10]
                temp = [id_dict[player][f] for f in idx]
                for t in temp:
                    data2.append(t)

            for i, player in enumerate(data2):
                idx = indice.index(player['pid'])
                data2[i]['pid'] = idx

            return data2, indice

        if mode == 3:
            indice = list(set([i['pid'] for i in data]))
            indice = [i for i in indice if i in mapping]
            for player in indice:
                idx = np.random.permutation(len(id_dict[player]))
                temp = [id_dict[player][f] for f in idx]
                for t in temp:
                    data2.append(t)

            for i, player in enumerate(data2):
                idx = mapping.index(player['pid'])
                data2[i]['pid'] = idx

            return data2, None

