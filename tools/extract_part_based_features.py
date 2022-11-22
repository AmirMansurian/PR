import torch
import tqdm
import glob
import os
from scripts.default_config import get_default_config, display_config_diff
from torchreid.utils import FeatureExtractor
import numpy as np

def extract_part_based_features(extractor, image_list, batch_size=400):

    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    all_embeddings = []
    all_visibility_scores = []
    all_masks = []

    images_chunks = chunks(image_list, batch_size)
    for chunk in tqdm.tqdm(images_chunks):
        embeddings, visibility_scores, masks = extractor(chunk)

        embeddings = embeddings.cpu().detach()
        visibility_scores = visibility_scores.cpu().detach()
        masks = masks.cpu().detach()

        all_embeddings.append(embeddings)
        all_visibility_scores.append(visibility_scores)
        all_masks.append(masks)

    all_embeddings = torch.cat(all_embeddings, 0).numpy()
    all_visibility_scores = torch.cat(all_visibility_scores, 0).numpy()
    all_masks = torch.cat(all_masks, 0).numpy()

    return {
        "parts_embeddings": all_embeddings,
        "parts_visibility_scores": all_visibility_scores,
        "parts_masks": all_masks,
    }


def extract_det_idx(img_path):
    return int(os.path.basename(img_path).split("_")[0])


def extract_reid_features(cfg, base_folder, out_path, model=None, model_path=None, num_classes=None):
    extractor = FeatureExtractor(
        cfg,
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_classes=num_classes,
        model=model
    )

    print("Looking for video folders with images crops in {}".format(base_folder))
    folder_list = glob.glob(base_folder + '/*')
    for folder in folder_list:
        image_list = glob.glob(os.path.join(folder, "*.png"))
        image_list.sort(key=extract_det_idx)
        print("{} images to process for folder {}".format(len(image_list), folder))
        results = extract_part_based_features(extractor, image_list, batch_size=50)

        # dump to disk
        video_name = os.path.splitext(os.path.basename(folder))[0]
        parts_embeddings_filename = os.path.join(out_path, "embeddings_" + video_name + ".npy")
        parts_visibility_scores_filanme = os.path.join(out_path, "visibility_scores_" + video_name + ".npy")
        parts_masks_filename = os.path.join(out_path, "masks_" + video_name + ".npy")

        os.makedirs(os.path.dirname(parts_embeddings_filename), exist_ok=True)
        os.makedirs(os.path.dirname(parts_visibility_scores_filanme), exist_ok=True)
        os.makedirs(os.path.dirname(parts_masks_filename), exist_ok=True)

        np.save(parts_embeddings_filename, results['parts_embeddings'])
        np.save(parts_visibility_scores_filanme, results['parts_visibility_scores'])
        np.save(parts_masks_filename, results['parts_masks'])

        print("features saved to {}".format(out_path))


if __name__ == '__main__':
    project_root_torchreid = "/home/vso/projects/deep-person-reid"
    project_root_strongsort = "/home/vso/projects/StrongSORT"
    model_path = '/home/vso/log/bpbreid_market/2022_04_29_02_59_33_59S8be4228f-685e-4b62-bb99-53d8f43e105e/2022_04_29_02_59_33_59S8be4228f-685e-4b62-bb99-53d8f43e105emodel/model.pth.tar-120'

    # project_root_torchreid= "/Users/vladimirsomers/Code/deep-person-reid"
    # project_root_strongsort = "/Users/vladimirsomers/Code/MOT/StrongSORT"
    # model_path = '/Users/vladimirsomers/Models/BPBReID/hrnet_jobid_8728_model.pth.tar-120'

    config_name = "hrnet_jobid_8728_model"
    base_folder = os.path.join(project_root_strongsort, "pregenerated_files/MOT17_val_YOLOX_crops_for_reid")
    out_path = os.path.join(project_root_strongsort, "pregenerated_files/MOT17_val_YOLOX_features", config_name)
    config_file_path = os.path.join(project_root_torchreid, "configs/bpbreid/remote_bpbreid_dukemtmc_train.yaml")

    cfg = get_default_config()
    cfg.data.parts_num = 5
    cfg.use_gpu = torch.cuda.is_available()
    default_cfg_copy = cfg.clone()
    cfg.merge_from_file(config_file_path)
    cfg.project.config_file = os.path.basename(config_file_path)
    display_config_diff(cfg, default_cfg_copy)
    cfg.model.pretrained = False
    num_classes = 702  # for model trained on DukeMTMC

    extract_reid_features(cfg, base_folder, out_path, model_path=model_path, num_classes=num_classes)
