import unittest

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio


class TestOther(unittest.TestCase):

    def test_qg_body_part_distances_boxplot(self):
        test = np.array([[0, 1, 2, 3], [2, 3, 4]], dtype=object)
        m = test.mean()
        print(m)

    def test_qg_body_part_distances_boxplot(self):
        test = np.array([[0.2, 0.8, 0.9, 0.5], [0.1, 0.3, 0.6, 0.5]])
        test = torch.from_numpy(test)
        r = F.normalize(test, 1, 1)
        print(test)
        print(r)
        print(r.sum(dim=1))

    def test_load_mat(self):
        market_attribute = sio.loadmat(
            '/Users/vladimirsomers/datasets/reid/market1501/Market-1501_Attribute-master/market_attribute.mat')
        gallery_market = sio.loadmat(
            '/Users/vladimirsomers/datasets/reid/market1501/Market-1501_Attribute-master/gallery_market.mat')
        import numpy as np
        import h5py
        # f = h5py.File(
        #     '/Users/vladimirsomers/datasets/reid/market1501/Market-1501_Attribute-master/market_attribute.mat',
        #     'r')
        # data = f.get('data/variable1')
        # data = np.array(data)  # For converting to a NumPy array
        #
        from mat4py import loadmat

        data = loadmat('/Users/vladimirsomers/datasets/reid/market1501/Market-1501_Attribute-master/market_attribute.mat')

        print()
