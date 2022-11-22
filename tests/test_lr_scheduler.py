import sys
import unittest

import torch
from torch.optim import SGD

from torchreid.optim.lr_scheduler import WarmupMultiStepLR

sys.path.append('.')

"""Source: https://github.com/michuanhaohao/reid-strong-baseline"""
class MyTestCase(unittest.TestCase):
    def test_something(self):
        model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
        optimizer = SGD(model, 0.00035)
        lr_scheduler = WarmupMultiStepLR(optimizer, [40, 70], warmup_factor=0.01, warmup_iters=10, warmup_method='linear')
        for i in range(120):
            for j in range(1):
                print(i, lr_scheduler.get_lr()[0])
                optimizer.step()
            lr_scheduler.step()


if __name__ == '__main__':
    unittest.main()


# GOAL :
# 0 3.5e-06
# 1 3.8150000000000006e-05
# 2 7.280000000000001e-05
# 3 0.00010745
# 4 0.0001421
# 5 0.00017675
# 6 0.0002114
# 7 0.00024605
# 8 0.0002807
# 9 0.00031535
# 10 0.00035
# 11 0.00035