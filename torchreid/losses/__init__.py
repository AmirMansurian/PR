from __future__ import division, print_function, absolute_import

from .inter_parts_triplet_loss import InterPartsTripletLoss
from .part_based_triplet_loss_max import PartBasedTripletLossMax
from .part_based_triplet_loss_max_min import PartBasedTripletLossMaxMin
from .part_based_triplet_loss_mean import PartBasedTripletLossMean
from .part_based_triplet_loss_min import PartBasedTripletLossMin
from .part_based_triplet_loss_random_max_min import PartBasedTripletLossRandomMaxMin
from .intra_parts_triplet_loss import IntraPartsTripletLoss
from .cross_entropy_loss import CrossEntropyLoss
from .hard_mine_triplet_loss import TripletLoss

__body_parts_losses = {
    'intra_parts_triplet_loss': IntraPartsTripletLoss,
    'inter_parts_triplet_loss': InterPartsTripletLoss,
    'part_based_triplet_loss_max': PartBasedTripletLossMax,
    'part_based_triplet_loss_min': PartBasedTripletLossMin,
    'part_based_triplet_loss_random_max_min': PartBasedTripletLossRandomMaxMin,
    'part_based_triplet_loss_max_min': PartBasedTripletLossMaxMin,
    'part_based_triplet_loss_mean': PartBasedTripletLossMean
}


def init_body_parts_triplet_loss(name, **kwargs):
    """Initializes a body parts loss."""
    avai_body_parts_losses = list(__body_parts_losses.keys())
    if name not in avai_body_parts_losses:
        raise ValueError(
            'Invalid loss name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_body_parts_losses)
        )
    return __body_parts_losses[name](**kwargs)


def deep_supervision(criterion, xs, y):
    """DeepSupervision

    Applies criterion to each element in a list.

    Args:
        criterion: loss function
        xs: tuple of inputs
        y: ground truth
    """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    return loss
