import glob
import json
import ntpath
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import timedelta
import torch
import torch.nn.functional as F

from torchreid.data.data_augmentation import CombinePifPafIntoEightBodyMasks


# skeletons_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/pifpaf/queries_results"
# segmentation_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/detectron2/MaskR-CNNX152-025-full_queries"
# # target_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/ankle_issue/*.jpg"
# # target_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/samples/*.jpg"
# target_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/queries/*.jpg"
# base_result_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/deep-reid-skeleton-filtering/display_heatmaps_final"
# base_visualization_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/deep-reid-skeleton-filtering/display_heatmaps_final"
# skeletons_image_extension = ".predictions.png"
# isp_attention_masks_folder = "/Users/vladimirsomers/Downloads/isp_attention_weights_as_numpy"

# dataset_dir = "Market-1501-v15.09.15"
# dataset_dir = "DukeMTMC-reID"
# dataset_dir = "Occluded_Duke"
# sub_folder = "query"
# sub_folder = "bounding_box_test"
# sub_folder = "bounding_box_train"

# dataset_dir = "Occluded_REID_COPY"
# sub_folder = "occluded_body_images"
# sub_folder = "whole_body_images"
# img_filemane_pattern = "*.tif"

# dataset_dir = "P-DukeMTMC-reID"
# sub_folder = "train/occluded_body_images"
# sub_folder = "train/whole_body_images"
# sub_folder = "test/occluded_body_images"
# sub_folder = "test/whole_body_images"
# img_filemane_pattern = "*.jpg"

dataset_dir = "MSMT17_V1"
sub_folder = "test"
# sub_folder = "train"
img_filemane_pattern = "*.jpg"



# dataset_dir = "synergy_sequences_dataset"
# sub_folder = "bbox_test_full"
# sub_folder = "bbox_val_full"
# sub_folder = "bbox_train_full"


run_name = "pifpaf_maskrcnn_filtering"

dataset_folder = "/home/vso/datasets/reid/" + dataset_dir
skeletons_folder = os.path.join(dataset_folder, "external_annotation/pifpaf", sub_folder)
skeletons_images_folder = os.path.join(dataset_folder, "figures/pifpaf", sub_folder)
masks_folder = os.path.join(dataset_folder, "external_annotation", "pifpaf", sub_folder)
# masks_folder = os.path.join(dataset_folder, "masks", "pifpaf", sub_folder)
segmentation_folder = os.path.join(dataset_folder, "external_annotation/detectron_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv_conf_tresh_025", sub_folder)
# segmentation_images_folder = os.path.join(dataset_folder, "figures/detectron_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv_conf_tresh_025", sub_folder)
segmentation_images_folder = os.path.join(dataset_folder, "external_annotation/detectron_cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv_conf_tresh_025", sub_folder)

# "/home/vso/datasets/reid/DukeMTMC-reID/figures/pifpaf_maskrcnn_filtering/bounding_box_test/0115_c1_f0072787.jpg_skeleton_filtering_results.jpg"
# img_filemane_pattern = "*c1_f0072787*.jpg"
# img_filemane_pattern = "*c1_f0073027*.jpg"
# img_filemane_pattern = "*c1_f0073387*.jpg"
# img_filemane_pattern = "*c1_f0073147*.jpg"
# img_filemane_pattern = "*c5_f0181653*.jpg"
# img_filemane_pattern = "*f0191448*.jpg"
# img_filemane_pattern = "*0008516*.jpg"
# img_filemane_pattern = "*0073027*.jpg"
# img_filemane_pattern = "*0115_*.jpg"
# img_filemane_pattern = "*0188383*.jpg" # women behind car pink jacket

target_folder = os.path.join(dataset_folder, "images", sub_folder, img_filemane_pattern)

# dataset_folder = "/home/vso/experiments/skeleton_filtering"
base_result_folder = os.path.join(dataset_folder, "masks", run_name, sub_folder)
base_visualization_folder = os.path.join(dataset_folder, "figures", run_name, sub_folder)
skeletons_result_folder = os.path.join(dataset_folder, "external_annotation", "pifpaf_keypoints_" + run_name, sub_folder)


# skeletons_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/test_1_filtering"
# segmentation_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/test_1_filtering"
# target_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/test_1/*.jpg"
# base_result_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/skeletons_filtering_results"
# base_visualization_folder = "/Users/vladimirsomers/Experiments/occluded_duke_queries/skeletons_filtering_visualization"

skeletons_image_extension = ".predictions.jpg"
paf_maps_extension = ".paf.npy"
pif_maps_extension = ".pif.npy"
skeletons_extension = ".predictions.json"
# skeletons_image_extension = ".predictions.jpg"
results_extension = ".npy"
body_part_maps_extension = ".png"

scores_extension = "_scores.npy"
segmentation_classes_extension = "_pred_classes.npy"
segmentation_masks_extension = "_pred_masks.npy"
segmentation_masks_image_extension = ""
heads_image_extension = "_heads.jpg"
skeleton_convex_shape_extension = "_skeleton_convex_shape.jpg"
keypoints_json_output_extension = "_keypoints.json"

OK = "OK"
NO_SKELETON = "NO_SKELETON"
NO_HEAD = "NO_HEAD"
NO_MASK = "NO_MASK"
NO_CONFIDENT_KEYPOINT = "NO_CONFIDENT_KEYPOINT"
NO_SKELETON_MASK_INTERSECTION = "NO_SKELETON_MASK_INTERSECTION"
ERRORS = [OK, NO_SKELETON, NO_HEAD, NO_MASK, NO_CONFIDENT_KEYPOINT, NO_SKELETON_MASK_INTERSECTION]

device = torch.device('cpu')
DISPLAY_SIZE = (300, 750)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CIRCLE_SIZE = 6
LINE_WIDTH = 2
PERSON_CATEGORY = 0
HEAT_MAP_TRESHOLD = 0.15
KEYPOINT_CONFIDENCE_THRESHOLD = 0.4

def load_skeletons(filename, masks_folder, skeletons_folder):
    paf_maps_path = os.path.join(masks_folder, filename + paf_maps_extension)
    pif_maps_path = os.path.join(masks_folder, filename + pif_maps_extension)
    skeletons_path = os.path.join(skeletons_folder, filename + skeletons_extension)
    paf_maps = np.load(paf_maps_path)
    pif_maps = np.load(pif_maps_path)
    with open(skeletons_path, 'r') as j:
        skeletons = json.loads(j.read())
    # keypoints to list of tuple
    for skeleton in skeletons:
        keypoints = skeleton["keypoints"]
        keypoints_tuples = []
        for i in range(0, int(len(keypoints)/3)):
            idx = i*3
            keypoints_tuples.append((keypoints[idx], keypoints[idx+1], keypoints[idx+2]))
        skeleton["keypoints"] = keypoints_tuples

    return paf_maps, pif_maps, skeletons


def load_segmentation(filename, folder):
    scores_path = os.path.join(folder, filename + scores_extension)
    classes_path = os.path.join(folder, filename + segmentation_classes_extension)
    masks_path = os.path.join(folder, filename + segmentation_masks_extension)

    # scores = torch.load(scores_path, map_location=device).numpy().astype(np.uint8)
    # classes = torch.load(classes_path, map_location=device).numpy().astype(np.uint8)
    # masks = torch.load(masks_path, map_location=device).numpy().astype(np.uint8)

    scores = np.load(scores_path).astype(np.uint8)
    classes = np.load(classes_path).astype(np.uint8)
    masks = np.load(masks_path).astype(np.uint8)

    # keep only person segmentation masks
    scores = scores[classes == PERSON_CATEGORY]
    masks = masks[classes == PERSON_CATEGORY]

    if masks.size == 0:
        return None, None
    return scores, masks


def map_coordinates(coord, in_shape, out_shape):
    (w, h) = coord
    return int(w/in_shape[1]*out_shape[1]), int(h/in_shape[0]*out_shape[0])


def dump_body_part_confidence_maps(skeletons, body_part_confidence_maps, maps_result_folder, skeleton_result_folder, input_filename):
    if not os.path.exists(maps_result_folder):
        os.makedirs(maps_result_folder)

    if not os.path.exists(skeleton_result_folder):
        os.makedirs(skeleton_result_folder)

    # can have zero or one skeleton
    # dump keypoints json
    json_out_name = os.path.join(skeleton_result_folder, input_filename + keypoints_json_output_extension)
    # print('json output = {}'.format(json_out_name))
    with open(json_out_name, 'w') as f:
        json.dump(skeletons, f)
    print(json_out_name)
    # dump body part heatmaps
    masks_out_name = os.path.join(maps_result_folder, input_filename.rsplit('.', 1)[0] + results_extension)
    np.save(masks_out_name, body_part_confidence_maps)
    print(masks_out_name)


def dump_body_part_maps(body_part_maps, result_folder, input_filename):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # dump body part maps
    cv2.imwrite(os.path.join(result_folder, input_filename.rsplit('.', 1)[0] + body_part_maps_extension), body_part_maps)


def pifpaf_to_confidence(pif_maps, paf_maps): # FIXME OPTIMIZE
    body_part_confidence_maps = []
    for pif_field in pif_maps:
        pif_confidence = pif_field[0, :, :]
        body_part_confidence_maps.append(pif_confidence)
    for paf_field in paf_maps:
        paf_confidence = paf_field[0, :, :]
        body_part_confidence_maps.append(paf_confidence)
    body_part_confidence_maps = np.array(body_part_confidence_maps)
    return body_part_confidence_maps


def filter_pifpaf_maps(paf_maps, pif_maps, target_skeleton_mask):
    low_res_size = (pif_maps.shape[3], pif_maps.shape[2])
    target_skeleton_mask = cv2.resize(target_skeleton_mask, low_res_size, interpolation=cv2.INTER_LINEAR)
    assert target_skeleton_mask.min() >= 0 and target_skeleton_mask.max() <= 1
    target_pif_maps = target_skeleton_mask * pif_maps
    target_paf_maps = target_skeleton_mask * paf_maps
    return target_paf_maps, target_pif_maps


def find_target_segmentation_mask(target_skeleton_contour, img_shape, masks, scores):
    H, W, C = img_shape
    skeleton_convex_shape = np.zeros((H, W), dtype=np.uint8) # TODO work in smaller resolution for efficiency?
    skeleton_convex_shape = cv2.fillConvexPoly(skeleton_convex_shape, target_skeleton_contour, 1)  # FIXME WHAT IF ONE KEYPOINT

    max_iou = -1
    target_masks_idx = None
    segmentation_contours_image = np.copy(image)
    masks_contours = []
    for i, mask in enumerate(masks):
        intersection = np.logical_and(skeleton_convex_shape, mask)
        intersection_area = intersection.sum()
        union = np.logical_or(skeleton_convex_shape, mask)
        union_area = union.sum()
        iou = intersection_area / union_area

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        masks_contours.append(contours)
        cv2.drawContours(segmentation_contours_image, contours, -1, RED, LINE_WIDTH)

        if iou > max_iou:
            max_iou = iou
            target_masks_idx = i

    return max_iou, masks_contours, target_masks_idx


def find_target_head(head_positions, expected_head_position):
    smallest_distance = sys.maxsize
    target_head_idx = None
    for i, head_position in enumerate(head_positions):
        if head_position is not None:
            (wd, hd) = np.abs(np.array(head_position) - np.array(expected_head_position))
            # 1.1 exponent to give more weight to horizontal distance: it's more important for the horizontal
            # distance to be small than the vertical distance: target must be at the center of the bounding box!
            wd = wd ** 1.1
            distance = np.linalg.norm((wd, hd))
            if distance < smallest_distance:
                smallest_distance = distance
                target_head_idx = i

    return target_head_idx


def get_heads_positions(skeletons):
    # keypoints = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    head_keypoints_indices = [0, 1, 2, 3, 4, 5, 6]  # nose left_eye right_eye left_ear left_shoulder right_shoulder
    head_positions = []
    for i, skeleton in enumerate(skeletons):
        h = 0
        w = 0
        c_sum = 0
        for j in head_keypoints_indices:
            kp = skeleton["keypoints"][j]
            h += kp[0] * kp[2]
            w += kp[1] * kp[2]
            c_sum += kp[2]
        if c_sum == 0:
            # FIXME compute centroid of other keypoints? Edge case: just legs visible on top of the image
            # FIXME What if head occluded by something such as umbrella?
            head_positions.append(None)
        else:
            h /= c_sum
            w /= c_sum
            head_position = (h, w)
            head_positions.append(head_position)

    return head_positions

def compute_target_skeleton_contour(target_skeleton):
    target_skeleton_contour = []
    for kp in target_skeleton["keypoints"]:
        if kp[2] > KEYPOINT_CONFIDENCE_THRESHOLD:
            target_skeleton_contour.append([kp[0], kp[1]])
    target_skeleton_contour = np.array(target_skeleton_contour).astype(int)
    if len(target_skeleton_contour) != 0:
        target_skeleton_contour = cv2.convexHull(target_skeleton_contour)
    return target_skeleton_contour


def compute_target_pifpaf_maps(img_shape, paf_maps, pif_maps, skeletons, scores, masks):
    if len(skeletons) == 0:
        return paf_maps, pif_maps, None, None, None, None, None, None, None, NO_SKELETON

    head_positions = get_heads_positions(skeletons)

    if all(h is None for h in head_positions):
        return paf_maps, pif_maps, None, None, None, None, None, None, None, NO_HEAD

    H, W, C = img_shape
    expected_head_position = (int(0.5 * W), int(0.05 * H))
    target_head_idx = find_target_head(head_positions, expected_head_position)

    target_skeleton = skeletons[target_head_idx]

    target_skeleton_contour = compute_target_skeleton_contour(target_skeleton)
    if len(target_skeleton_contour) == 0:
        return paf_maps, pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, None, None, None, NO_CONFIDENT_KEYPOINT

    if masks is None:
        return paf_maps, pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, target_skeleton_contour, None, None, NO_MASK

    max_iou, masks_contours, target_masks_idx = \
        find_target_segmentation_mask(target_skeleton_contour, img_shape, masks, scores)

    target_skeleton_mask = masks[target_masks_idx]
    if max_iou == 0:
        return paf_maps, pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, target_skeleton_contour, None, None, NO_SKELETON_MASK_INTERSECTION

    target_paf_maps, target_pif_maps = filter_pifpaf_maps(paf_maps, pif_maps, target_skeleton_mask)

    return target_paf_maps, target_pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, target_skeleton_contour, masks_contours, target_masks_idx, OK


def display_results(filename, image, masks, paf_maps, pif_maps, target_body_part_confidence_maps_list, body_parts_maps_list, target_paf_maps, target_pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, target_skeleton_contour, segmentation_masks_contours, target_masks_idx):

    # confidence_map
    body_part_confidence_maps = pifpaf_to_confidence(pif_maps, paf_maps)
    body_part_confidence_maps = np.moveaxis(body_part_confidence_maps, 0, -1)
    confidence_map = cv2.resize(np.max(body_part_confidence_maps, axis=2), dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    confidence_map = cv2.resize(confidence_map, dsize=DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

    # skeletons_image
    skeletons_image_path = os.path.join(skeletons_images_folder, filename + skeletons_image_extension)
    skeletons_image = cv2.imread(skeletons_image_path)

    if head_positions is not None:
        for i, head_position in enumerate(head_positions):
            if head_position is not None:
                new_head_position = map_coordinates(head_position, image.shape, skeletons_image.shape)
                skeletons_image = cv2.circle(skeletons_image, new_head_position, CIRCLE_SIZE, RED, -1)

        new_expected_head_position = map_coordinates(expected_head_position, image.shape, skeletons_image.shape)
        skeletons_image = cv2.circle(skeletons_image, new_expected_head_position, CIRCLE_SIZE, BLUE, -1)

        target_head_position = head_positions[target_head_idx]
        new_target_head_position = map_coordinates(target_head_position, image.shape, skeletons_image.shape)
        skeletons_image = cv2.circle(skeletons_image, new_target_head_position, CIRCLE_SIZE, GREEN, -1)
        skeletons_image = cv2.resize(skeletons_image, dsize=DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)

    # original_segmentation_masks_image
    original_segmentation_masks_image_path = os.path.join(segmentation_images_folder, filename + segmentation_masks_image_extension)
    original_segmentation_masks_image = cv2.imread(original_segmentation_masks_image_path)

    # segmentation_contours_image
    segmentation_contours_image = np.copy(image)
    if target_masks_idx is not None:
        target_countours = segmentation_masks_contours[target_masks_idx]
        cv2.drawContours(segmentation_contours_image, [target_skeleton_contour], -1, BLUE, LINE_WIDTH)
        cv2.drawContours(segmentation_contours_image, target_countours, -1, GREEN, LINE_WIDTH)
        segmentation_contours_image = cv2.resize(segmentation_contours_image, dsize=DISPLAY_SIZE,
                                                 interpolation=cv2.INTER_LINEAR)

    # target_confidence_map
    target_body_part_confidence_maps = target_body_part_confidence_maps_list[0]
    target_body_part_confidence_maps = np.moveaxis(target_body_part_confidence_maps, 0, -1)
    target_confidence_map = cv2.resize(np.max(target_body_part_confidence_maps, axis=2),
                                       dsize=(image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    target_confidence_map = cv2.resize(target_confidence_map, dsize=DISPLAY_SIZE, interpolation=cv2.INTER_LINEAR)


    # NO_SKELETON, NO_HEAD, NO_MASK, NO_CONFIDENT_KEYPOINT, NO_SKELETON_MASK_INTERSECTION
    visualization_folder = base_visualization_folder
    if not os.path.exists(visualization_folder):
        os.makedirs(visualization_folder)

    mode = 1
    if mode == 1:
        fig, axs = plt.subplots(1, 5 + len(body_parts_maps_list))
        fig.set_size_inches(25, 100)
        axs[0].imshow(color_and_resize(image, DISPLAY_SIZE))
        axs[0].imshow(confidence_map, cmap='jet', vmin=0, vmax=1, alpha=0.4)
        axs[1].imshow(color_and_resize(skeletons_image, DISPLAY_SIZE))
        axs[2].imshow(color_and_resize(original_segmentation_masks_image, DISPLAY_SIZE))
        axs[3].imshow(color_and_resize(segmentation_contours_image, DISPLAY_SIZE))
        axs[4].imshow(color_and_resize(image, DISPLAY_SIZE))
        axs[4].imshow(target_confidence_map, cmap='jet', vmin=0, vmax=1, alpha=0.4)
        for i, body_parts_maps in enumerate(body_parts_maps_list):
            axs[5+i].imshow(color_and_resize(image, DISPLAY_SIZE))
            axs[5+i].imshow(resize(body_parts_maps[0], DISPLAY_SIZE), cmap='gist_rainbow', vmin=0, vmax=body_parts_maps[2], alpha=resize(body_parts_maps[1], DISPLAY_SIZE))
        for ax in axs:
            ax.axis('off')
        plt_out_name = os.path.join(visualization_folder, filename + "_skeleton_filtering_results.jpg")
        plt.savefig(plt_out_name, bbox_inches='tight',
                    pad_inches=0)
        plt.close()
        print(plt_out_name)

    if mode == 2:
        cols = 2 + target_body_part_confidence_maps_list[0].shape[0]
        fig, axs = plt.subplots(2, cols)
        fig.set_size_inches(50, 25)
        axs[0, 0].imshow(color_and_resize(image, DISPLAY_SIZE))
        body_parts_maps = body_parts_maps_list[0]
        axs[0, 1].imshow(color_and_resize(image, DISPLAY_SIZE))
        axs[0, 1].imshow(resize(body_parts_maps[0], DISPLAY_SIZE), cmap='prism', vmin=0, vmax=body_parts_maps[2],
                          alpha=resize(body_parts_maps[1], DISPLAY_SIZE))

        for i, body_parts_heatmaps in enumerate(target_body_part_confidence_maps_list[0]):
            body_parts_heatmaps = cv2.resize(body_parts_heatmaps, dsize=DISPLAY_SIZE,
                                               interpolation=cv2.INTER_NEAREST)
            axs[0, 2 + i].imshow(color_and_resize(image, DISPLAY_SIZE))
            axs[0, 2 + i].imshow(body_parts_heatmaps, cmap='jet', vmin=0, vmax=1, alpha=0.4) # TODO fix min max

        isp_masks = np.load(os.path.join(isp_attention_masks_folder, filename.rsplit('.', 1)[0] + '.npy'))

        isp_masks_bp = isp_masks[1::]
        isp_masks_bp = np.moveaxis(isp_masks_bp, 0, -1)
        isp_masks_argmax = np.argmax(isp_masks_bp, axis=2)
        # body_part_map_alpha = np.max(body_parts_map, axis=2)
        isp_masks_alpha = (np.max(isp_masks_bp, axis=2) != 0) * 0.6

        axs[1, 0].imshow(color_and_resize(image, DISPLAY_SIZE))
        axs[1, 1].imshow(color_and_resize(image, DISPLAY_SIZE))
        axs[1, 1].imshow(resize(isp_masks_argmax, DISPLAY_SIZE), cmap='prism', vmin=0, vmax=isp_masks_bp.shape[2],
                          alpha=resize(isp_masks_alpha, DISPLAY_SIZE))

        for i, body_parts_heatmaps in enumerate(isp_masks):
            body_parts_heatmaps = cv2.resize(body_parts_heatmaps, dsize=DISPLAY_SIZE,
                                               interpolation=cv2.INTER_NEAREST)
            axs[1, 2 + i].imshow(color_and_resize(image, DISPLAY_SIZE))
            axs[1, 2 + i].imshow(body_parts_heatmaps, cmap='jet', vmin=0, vmax=1, alpha=0.4)

        # TODO DISPLAY STATS ABOUT MASKS: min,mean,max values, sum
        pifpaf_min_max_mean = "[{}, {}, {}]".format(target_body_part_confidence_maps_list[0].min(),
                                                    target_body_part_confidence_maps_list[0].mean(),
                                                    target_body_part_confidence_maps_list[0].max())
        isp_min_max_mean = "[{}, {}, {}]".format(isp_masks.min(),
                                                    isp_masks.mean(),
                                                    isp_masks.max())
        plt.title("pifpaf " + pifpaf_min_max_mean + " - isp " + isp_min_max_mean)

        for ax in axs[0]:
            ax.axis('off')

        for ax in axs[1]:
            ax.axis('off')
        plt.savefig(os.path.join(visualization_folder, filename + "_body_part_heatmaps.jpg"),
                    bbox_inches='tight',
                    pad_inches=0)
        plt.close()


def color_and_resize(image, size):
    return cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=size, interpolation=cv2.INTER_LINEAR)


def resize(image, size):
    type = image.dtype
    return cv2.resize(image.astype(float), dsize=size, interpolation=cv2.INTER_NEAREST).astype(type)


def combine_body_parts(transforms, target_body_part_confidence_maps):
    target_body_part_confidence_maps_list = []
    # target_body_part_confidence_maps_list.append(target_body_part_confidence_maps)
    for transform in transforms:
        body_part_confidence_maps = transform(mask=torch.from_numpy(target_body_part_confidence_maps), image=torch.from_numpy(target_body_part_confidence_maps))['mask'].numpy()
        target_body_part_confidence_maps_list.append(body_part_confidence_maps)
    return target_body_part_confidence_maps_list


def compute_body_part_maps(target_body_part_confidence_maps_list):
    body_parts_maps_list = []
    for target_body_part_confidence_maps in target_body_part_confidence_maps_list:
        body_parts_map = np.moveaxis(target_body_part_confidence_maps, 0, -1)
        body_parts_map[body_parts_map < HEAT_MAP_TRESHOLD] = 0
        body_parts_map = np.concatenate((np.zeros((body_parts_map.shape[0], body_parts_map.shape[1], 1)), body_parts_map),
                                     axis=2)
        body_parts_map_argmax = np.argmax(body_parts_map, axis=2)
        # body_part_map_alpha = np.max(body_parts_map, axis=2)
        body_part_map_alpha = (np.max(body_parts_map, axis=2) != 0) * 0.6

        body_parts_maps_list.append((body_parts_map_argmax, body_part_map_alpha, body_parts_map.shape[2]))
    return body_parts_maps_list


def transform_confidence_map(target_confidence_maps, weight=5):
    # return np.concatenate((np.expand_dims(np.clip(1-target_confidence_maps.sum(axis=0), 0, 1), 0), target_confidence_maps), axis=0)
    cf = np.concatenate((np.expand_dims(np.clip(1-target_confidence_maps.sum(axis=0), 0, 1), 0), target_confidence_maps), axis=0)
    cf = F.softmax(torch.from_numpy(cf)*weight, dim=0)
    return cf.numpy()


if __name__ == "__main__":
    start = time.time()
    no_confident_keypoint = 0
    images = glob.glob(target_folder)
    print("Found {} images in {}".format(len(images), target_folder))
    errors = {}
    for error in ERRORS:
        errors[error] = [0, []]
    for img_idx, path in enumerate(images):
        filename = ntpath.basename(path)
        print("Processing input {}/{} {}".format(img_idx, len(images), path))
        image = cv2.imread(path)

        paf_maps, pif_maps, skeletons = load_skeletons(filename, masks_folder, skeletons_folder)
        scores, masks = load_segmentation(filename, segmentation_folder)

        target_paf_maps, target_pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, target_skeleton_contour, segmentation_masks_contours, target_masks_idx, error \
            = compute_target_pifpaf_maps(image.shape, paf_maps, pif_maps, skeletons, scores, masks)

        errors[error][0] += 1
        errors[error][1].append(filename)

        target_body_part_confidence_maps = pifpaf_to_confidence(target_pif_maps, target_paf_maps)
        dump_body_part_confidence_maps(target_skeleton, target_body_part_confidence_maps, base_result_folder, skeletons_result_folder, filename)
        # STOP HERE for generating training dataset
        transforms = [
            # CombinePifPafIntoFourBodyMasks(),
            # CombinePifPafIntoFourBodyMasksNoOverlap(),
            # NewCombinePifPafIntoFourBodyMasks(),
            # CombinePifPafIntoFourVerticalParts(),
            # CombinePifPafIntoFourVerticalPartsPif(),
            # CombinePifPafIntoFiveVerticalParts(),
            # CombinePifPafIntoSixBodyMasks(),
            # CombinePifPafIntoSixBodyMasksSum(),
            # CombinePifPafIntoSevenBodyMasks(),
            CombinePifPafIntoEightBodyMasks(),
            # CombinePifPafIntoSixVerticalParts(),
            # CombinePifPafIntoSixBodyMasksSimilarToEight(),
        ]
        target_body_part_confidence_maps_list = combine_body_parts(transforms, target_body_part_confidence_maps)

        transformed_target_body_part_confidence_maps_list = []
        for target_confidence_maps in target_body_part_confidence_maps_list:
            confidence_maps = transform_confidence_map(target_confidence_maps)
            transformed_target_body_part_confidence_maps_list.append(confidence_maps)
            # dump_body_part_confidence_maps(target_skeleton, confidence_maps, base_result_folder, filename)

        body_part_maps_list = compute_body_part_maps(target_body_part_confidence_maps_list)

        # transformed_target_body_part_confidence_maps_list = target_body_part_confidence_maps_list

        # for body_part_maps, _, _ in body_part_maps_list:
        #     dump_body_part_maps(body_part_maps, base_result_folder, filename)

        display_results(filename, image, masks, paf_maps, pif_maps, transformed_target_body_part_confidence_maps_list, body_part_maps_list, target_paf_maps, target_pif_maps, target_skeleton, expected_head_position, head_positions, target_head_idx, target_skeleton_contour, segmentation_masks_contours, target_masks_idx)


    for error_name, error in errors.items():
        print("{} = {}".format(error_name, error[1]))

    end = time.time()
    time_elapsed = timedelta(seconds=end - start)
    print("Done, {} files processed in {}".format(len(images), time_elapsed))
    for error_name, error in errors.items():
        print("{} = {}".format(error_name, error[0]))


# FIXME TOO MANY VALUES TO UNPACK WITH INDEITITy MASK
# put skeleton filtering as transform
# refactor code: extract all visualization code to be used outside
# output image for edge cases
# add option to just generate classic pifpaf confidence fields

# finish script for generating pifpaf + detectron annotations
# Compare with SCHP
# check all these in small resolution
# Ask maxim about panoptic segmentation in Detectron2: seems à chier????
# how to handle mismatch between segmentation masks and skeletons? Use only confidence fields + skeletons to output same result? Skeletons are always better than segmentation masks: use only skeletons!
# Try to output masks binary body part masks: body part of one pixel = background if outside segmentation mask and max accross all body parts if inside
# write evernote guide to output skeleton + masks + this filtering
# Try to use joint segmentation mask + skeleton model (Gab said mask-rcnn can do it)
# Try CDCL-human-part-segmentation
