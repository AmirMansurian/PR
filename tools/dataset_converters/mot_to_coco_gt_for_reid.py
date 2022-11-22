import argparse
import configparser
import datetime
import json
import os
import os.path as osp

import numpy as np
import pycocotools.mask as rletools
import tqdm
from PIL import Image
from pathlib import Path

# Source: https://github.com/phil-bergmann/tracking_wo_bnw
SPLITS = ['train']


class SegmentedObject:
    """
    Helper class for segmentation objects.
    """
    def __init__(self, mask: dict, class_id: int, track_id: int) -> None:
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id


def load_mots_gt(path: str) -> dict:
    """Load MOTS ground truth from path."""
    objects_per_frame = {}
    track_ids_per_frame = {}  # Check that no frame contains two objects with same id
    combined_mask_per_frame = {}  # Check that no frame contains overlapping masks

    with open(path, "r") as gt_file:
        for line in gt_file:
            line = line.strip()
            fields = line.split(" ")

            frame = int(fields[0])
            if frame not in objects_per_frame:
                objects_per_frame[frame] = []
            if frame not in track_ids_per_frame:
                track_ids_per_frame[frame] = set()
            if int(fields[1]) in track_ids_per_frame[frame]:
                assert False, f"Multiple objects with track id {fields[1]} in frame {fields[0]}"
            else:
                track_ids_per_frame[frame].add(int(fields[1]))

            class_id = int(fields[2])
            if not(class_id == 1 or class_id == 2 or class_id == 10):
                assert False, "Unknown object class " + fields[2]

            mask = {
                'size': [int(fields[3]), int(fields[4])],
                'counts': fields[5].encode(encoding='UTF-8')}
            if frame not in combined_mask_per_frame:
                combined_mask_per_frame[frame] = mask
            elif rletools.area(rletools.merge([
                    combined_mask_per_frame[frame], mask],
                    intersect=True)):
                assert False, "Objects with overlapping masks in frame " + fields[0]
            else:
                combined_mask_per_frame[frame] = rletools.merge(
                    [combined_mask_per_frame[frame], mask],
                    intersect=False)
            objects_per_frame[frame].append(SegmentedObject(
                mask,
                class_id,
                int(fields[1])
            ))

    return objects_per_frame


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def save_json(data, path):
    with open(path, 'w') as json_file:
        json.dump(data, json_file)


def get_im_anns_dict(anns):
    im_anns = {}
    for im_ann in anns['images']:
        im_anns[im_ann['id']] = im_ann
    return im_anns


def crop_box(im, bbox):
    x1, y1, w, h = bbox
    x2, y2 = x1+ w, y1+ h
    return im.crop((x1, y1, x2, y2))


def ped_im_from_anno(data_root, anno, im_anns):
    im_path = osp.join(data_root, im_anns[anno['image_id']]['file_name'])
    im = Image.open(im_path)

    if anno['mask'] is not None:
        mask = rletools.decode(anno['mask'].mask)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

        im = im * mask
        im = Image.fromarray(im)

    crop_im = crop_box(im, anno['bbox'])
    return crop_im


def get_img_id(dataset, seq, fname):
    return int(f"{dataset[3:5]}{seq.split('-')[1]}{int(fname.split('.')[0]):06}")


def read_seqinfo(path):
    cp = configparser.ConfigParser()
    cp.read(path)

    return {'height': int(cp.get('Sequence', 'imHeight')),
            'width': int(cp.get('Sequence', 'imWidth')),
            'fps': int(cp.get('Sequence', 'frameRate')),
            'seq_length': int(cp.get('Sequence', 'seqLength'))}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', required=True)
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--no_bg', action='store_true')
    parser.add_argument('--mask_dir', default='')

    args = parser.parse_args()

    if args.no_bg:
        assert args.mask_dir

    print(f'DATASET: {args.dataset}')
    print(f'NO_BG: {args.no_bg}')

    for split in SPLITS:
        data_path = osp.join(args.data_root, args.dataset, split)
        seqs = os.listdir(data_path)
        seqs = [s for s in seqs
                if not s.endswith('GT') and not s.startswith('.') and not s.endswith('.json') and not s == 'reid' and 'DPM' not in s and 'SDP' not in s]
        seqs = sorted(seqs)

        # generate reid data
        reid_path = osp.join(data_path, 'reid')
        reid_imgs_path = osp.join(reid_path, 'imgs')
        reid_anns_path = osp.join(reid_path, 'anns')
        os.makedirs(reid_imgs_path, exist_ok=True)
        os.makedirs(reid_anns_path, exist_ok=True)

        for seq in seqs:
            print(f"Processing sequence {seq} in dataset {args.dataset}")

            seq_path = osp.join(data_path, seq)
            seqinfo_path = osp.join(seq_path, 'seqinfo.ini')
            gt_path = osp.join(seq_path, 'gt/gt.txt')
            im_dir = osp.join(seq_path, 'img1')

            seqinfo = read_seqinfo(seqinfo_path)
            data = {'info': {'sequence': seq,
                                'dataset': args.dataset,
                                'split': split,
                                'creation_date': datetime.datetime.today().strftime('%Y-%m-%d-%H-%M'),
                                **seqinfo},
                    'images': [],
                    'annotations': [],
                    'categories': [{'id': 1, 'name': 'person', 'supercategory': 'person'}]}

            # Load Bounding Box annotations
            gt = np.loadtxt(gt_path, dtype=np.float32, delimiter=',')
            # keep_classes = [1, 2, 7, 8, 12]
            keep_classes = [1]
            mask = np.isin(gt[:, 7], keep_classes)
            gt = gt[mask]
            #break
            anns = [{'ped_id': int(row[1]),
                        'frame_n': row[0],
                        'category_id': 1,
                        'id': f"{get_img_id(args.dataset, seq, f'{int(row[0]):06}.jpg')}{int(row_i):010}{'_NO_BG' if args.no_bg else ''}",
                        'image_id': get_img_id(args.dataset, seq, f'{int(row[0]):06}.jpg'),
                        'bbox': row[2:6].tolist(),
                        'area': row[4]*row[5],
                        'vis': row[8],
                        'iscrowd': 1 - row[6],
                        'mask': None}
                    for row_i, row in enumerate(gt.astype(float))]

            for anno in anns:
                anno['filename'] = f"fr{int(anno['frame_n'])}_id{anno['id']}_x{int(anno['bbox'][0])}_y{int(anno['bbox'][1])}_w{int(anno['bbox'][2])}_h{int(anno['bbox'][3])}_vis{anno['vis']:.4f}_cr{anno['iscrowd']}.png"

            if args.no_bg:
                # mots_data_path = osp.join(args.data_root, MOTS_DIR, split)
                # mots_seq_gt_path = osp.join(mots_data_path, seq.replace('MOT17', 'MOTS20'), 'gt/gt.txt')
                mots_seq_gt_path = osp.join(args.mask_dir, f'{seq}.txt')

                if not os.path.isfile(mots_seq_gt_path):
                    print(f"No mask information at {mots_seq_gt_path} to remove background for {seq}.")
                else:
                    mask_objects_per_frame = load_mots_gt(mots_seq_gt_path)

                    for frame_id, mask_objects in mask_objects_per_frame.items():
                        # frame_data = args.dataset.data[frame_id - 1]
                        frame_data = [a for a in anns if a['frame_n'] == frame_id]

                        for obj_data in frame_data:
                            mask_object = [
                                mask_object
                                for mask_object in mask_objects
                                if mask_object.track_id % 1000 == obj_data['ped_id']]

                            if len(mask_object):
                                obj_data['mask'] = mask_object[0]
                            else:
                                obj_data['iscrowd'] = 1

            # Load Image information
            all_img_ids = list(set([aa['image_id'] for aa in anns]))
            imgs = [{'file_name': osp.join(args.dataset, split, seq, 'img1', fname),
                        'height': seqinfo['height'],
                        'width': seqinfo['width'],
                        'id': get_img_id(args.dataset, seq, fname)}
                    for fname in os.listdir(im_dir) if get_img_id(args.dataset, seq, fname) in all_img_ids]
            assert len(set([im['id'] for im in imgs])) == len(imgs)
            data['images'].extend(imgs)

            assert len(str(imgs[0]['id'])) == len(str(anns[0]['image_id']))

            data['annotations'].extend(anns)

            # generate reid data
            im_anns = get_im_anns_dict(data)

            for anno in tqdm.tqdm(data['annotations']):
                box_im = ped_im_from_anno(args.data_root, anno, im_anns)
                box_dir = osp.join(reid_imgs_path, str(seq), str(anno['ped_id']))
                Path(box_dir).mkdir(parents=True, exist_ok=True)
                box_path = osp.join(box_dir, anno['filename'])
                box_im.save(box_path)

            # save annotation file
            ann_dir = reid_anns_path
            if not osp.exists(ann_dir):
                os.makedirs(ann_dir)
            os.makedirs(ann_dir, exist_ok=True)

            ann_file = osp.join(ann_dir, f"{seq}.json")
            if args.no_bg:
                ann_file = osp.join(ann_dir, f"{seq}_NO_BG.json")

                # remove mask before saving
                for i in range(len(data['annotations'])):
                    data['annotations'][i]['mask'] = None

            save_json(data, ann_file)
            print(f"Saving annotation file in {ann_file}.\n")


if __name__ == '__main__':
    main()
