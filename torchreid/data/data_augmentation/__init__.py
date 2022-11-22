from __future__ import print_function, absolute_import

from torchreid.data.data_augmentation.mask_transform import CombinePifPafIntoFullBodyMask, IdentityMask, \
    PCBMasks2, PCBMasks3, PCBMasks4, PCBMasks5, CombinePifPafIntoSixBodyMasksSum, PCBMasks6, PCBMasks7, PCBMasks8, \
    AddFullBodyMaskToBaseMasks, AddFullBodyMaskAndFullBoundingBoxToBaseMasks, CombinePifPafIntoMultiScaleBodyMasks, \
    CombinePifPafIntoOneBodyMasks, CombinePifPafIntoTwoBodyMasks, CombinePifPafIntoThreeBodyMasks, \
    CombinePifPafIntoFourBodyMasks, CombinePifPafIntoFourBodyMasksNoOverlap, CombinePifPafIntoFourVerticalParts, \
    CombinePifPafIntoFourVerticalPartsPif, CombinePifPafIntoFiveVerticalParts, CombinePifPafIntoFiveBodyMasks, \
    CombinePifPafIntoSixBodyMasks, CombinePifPafIntoSixVerticalParts, CombinePifPafIntoSixBodyMasksSimilarToEight, \
    CombinePifPafIntoSevenVerticalBodyMasks, CombinePifPafIntoSevenBodyMasksSimilarToEight, \
    CombinePifPafIntoEightBodyMasks, CombinePifPafIntoEightVerticalBodyMasks, CombinePifPafIntoTenMSBodyMasks, \
    CombinePifPafIntoElevenBodyMasks, CombinePifPafIntoFourteenBodyMasks

masks_preprocess_transforms = {
    'full': CombinePifPafIntoFullBodyMask,
    'bs_fu': AddFullBodyMaskToBaseMasks,
    'bs_fu_bb': AddFullBodyMaskAndFullBoundingBoxToBaseMasks,
    'mu_sc': CombinePifPafIntoMultiScaleBodyMasks,
    'one': CombinePifPafIntoOneBodyMasks,
    'two_v': CombinePifPafIntoTwoBodyMasks,
    'three_v': CombinePifPafIntoThreeBodyMasks,
    'four': CombinePifPafIntoFourBodyMasks,
    'four_no': CombinePifPafIntoFourBodyMasksNoOverlap,
    'four_v': CombinePifPafIntoFourVerticalParts,
    'four_v_pif': CombinePifPafIntoFourVerticalPartsPif,
    'five_v': CombinePifPafIntoFiveVerticalParts,
    'five': CombinePifPafIntoFiveBodyMasks,
    'six': CombinePifPafIntoSixBodyMasks,
    'six_v': CombinePifPafIntoSixVerticalParts,
    'six_no': CombinePifPafIntoSixBodyMasksSum,
    'six_new': CombinePifPafIntoSixBodyMasksSimilarToEight,
    'seven_v': CombinePifPafIntoSevenVerticalBodyMasks,
    'seven_new': CombinePifPafIntoSevenBodyMasksSimilarToEight,
    'eight': CombinePifPafIntoEightBodyMasks,
    'eight_v': CombinePifPafIntoEightVerticalBodyMasks,
    'ten_ms': CombinePifPafIntoTenMSBodyMasks,
    'eleven': CombinePifPafIntoElevenBodyMasks,
    'fourteen': CombinePifPafIntoFourteenBodyMasks,
}

masks_preprocess_fixed = {
    'id': IdentityMask,
    'strp_2': PCBMasks2,
    'strp_3': PCBMasks3,
    'strp_4': PCBMasks4,
    'strp_5': PCBMasks5,
    'strp_6': PCBMasks6,
    'strp_7': PCBMasks7,
    'strp_8': PCBMasks8,
}

masks_preprocess_all = {**masks_preprocess_transforms, **masks_preprocess_fixed}
