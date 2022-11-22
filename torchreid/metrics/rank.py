from __future__ import division, print_function, absolute_import
import numpy as np
import warnings
from collections import defaultdict

try:
    from torchreid.metrics.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )


def eval_soccernetv3(distmat, q_pids, g_pids, q_action_indices, g_action_indices, max_rank):
    """Evaluation with market1501 metric
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    smallest_ranking_size = max_rank

    for q_idx in range(num_q):
        # get query pid and action_idx
        q_pid = q_pids[q_idx]
        q_action_idx = q_action_indices[q_idx]

        # remove gallery samples from different action than the query
        order = indices[q_idx]
        remove = (g_action_indices[order] != q_action_idx)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            print("Does not appear in gallery: q_idx {} - q_pid {} - q_action_idx {}".format(q_idx, q_pid, q_action_idx))
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        cmc = cmc[:max_rank]
        if smallest_ranking_size > cmc.size:
            smallest_ranking_size = cmc.size

        all_cmc.append(cmc)
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    all_cmc = [np.concatenate((np.array(cmc[:smallest_ranking_size]), np.zeros(max_rank-smallest_ranking_size, dtype=np.int64)))
               for cmc in all_cmc] # np.cat(cmc[:smallest_ranking_size], np.zeros(max_rank-smallest_ranking_size))
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q  # size = 174
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc = 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)

        cmc /= num_repeats
        all_cmc.append(cmc)
        # compute AP
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return {
        'cmc': cmc,
        'mAP': mAP,
    }


def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return {
        'cmc': cmc,
        'mAP': mAP,
    }


def eval_motchallenge(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, q_anns, g_anns):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
            format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][
            keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    all_AP = np.array(all_AP)
    mAP = np.mean(all_AP)

    sample_visibility_scores = q_anns['vis']
    low_vis = sample_visibility_scores <= 0.25
    medium_vis = np.logical_and(sample_visibility_scores >= 0.25, sample_visibility_scores <= 0.75)
    high_vis = sample_visibility_scores >= 0.75

    low_vis_mAP = np.mean(all_AP[low_vis]) if np.any(low_vis) else None
    medium_vis_mAP = np.mean(all_AP[medium_vis]) if np.any(medium_vis) else None
    high_vis_mAP = np.mean(all_AP[high_vis]) if np.any(high_vis) else None

    low_vis_cmc = all_cmc[low_vis].sum(0) / low_vis.sum() if np.any(low_vis) else None
    medium_vis_cmc = all_cmc[medium_vis].sum(0) / medium_vis.sum() if np.any(medium_vis) else None
    high_vis_cmc = all_cmc[high_vis].sum(0) / high_vis.sum() if np.any(high_vis) else None

    return {
        'cmc': cmc,
        'mAP': mAP,
        'Low visibility mAP': (low_vis_mAP, low_vis.sum()),
        'Medium visibility mAP': (medium_vis_mAP, medium_vis.sum()),
        'High visibility mAP': (high_vis_mAP, high_vis.sum()),
        'Low visibility rank-1': (low_vis_cmc[0] if low_vis_cmc is not None else None, low_vis.sum()),
        'Medium visibility rank-1': (medium_vis_cmc[0] if medium_vis_cmc is not None else None, medium_vis.sum()),
        'High visibility rank-1': (high_vis_cmc[0] if high_vis_cmc is not None else None, high_vis.sum()),
    }


def eval_dartfish(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with market1501 metric, but for each query identity,
    all gallery images are used.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print(
            'Note: number of gallery samples is quite small, got {}'.
                format(num_g)
        )

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query

    for q_idx in range(num_q):
        # compute cmc curve
        raw_cmc = matches[q_idx]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return {
        'cmc': cmc,
        'mAP': mAP,
    }


def evaluate_py(
    distmat, q_pids, g_pids, q_camids, g_camids, max_rank, eval_metric, q_anns=None, g_anns=None,
):
    if eval_metric == 'default':
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    elif eval_metric == 'cuhk03':
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    elif eval_metric == 'soccernetv3':
        return eval_soccernetv3(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    elif eval_metric == 'dartfish':
        return eval_dartfish(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    elif eval_metric == 'motchallenge':
        return eval_motchallenge(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, q_anns, g_anns)
    else:
        raise ValueError("Incorrect eval_metric value '{}'".format(eval_metric))


def evaluate_rank(
    distmat,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    max_rank=50,
    eval_metric='default',
    q_anns=None,
    g_anns=None,
    use_cython=True
):
    """Evaluates CMC rank.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        q_pids (numpy.ndarray): 1-D array containing person identities
            of each query instance.
        g_pids (numpy.ndarray): 1-D array containing person identities
            of each gallery instance.
        q_camids (numpy.ndarray): 1-D array containing camera views under
            which each query instance is captured.
        g_camids (numpy.ndarray): 1-D array containing camera views under
            which each gallery instance is captured.
        max_rank (int, optional): maximum CMC rank to be computed. Default is 50.
        eval_metric (str, optional): use multi-gallery-shot setting with 'default', single-gallery-shot
            setting with 'cuhk03' or action-to-replay setting with 'soccernetv3'.
            Default is 'default'.
        use_cython (bool, optional): use cython code for evaluation. Default is True.
            This is highly recommended as the cython code can speed up the cmc computation
            by more than 10x. This requires Cython to be installed.
    """
    if use_cython and IS_CYTHON_AVAI and (eval_metric == 'default' or eval_metric == 'cuhk03'):
        return evaluate_py(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            eval_metric
        )
    else:
        return evaluate_py(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank,
            eval_metric, q_anns=q_anns, g_anns=g_anns
        )
