from __future__ import division, absolute_import
import torch
from torch import nn
import torch.nn.functional as F
from torchreid import models

__all__ = [
    'bpbreid'
]

from torchreid.utils.constants import *


class BPBreID(nn.Module):
    """Posed based feature extraction network
    """
    # TODO Parts embeddings fusion strategy: concat vs concat + fc vs element wise sum
    def __init__(self, num_classes, pretrained, loss, config=None, **kwargs):
        super(BPBreID, self).__init__()

        # Init config
        self.config = config
        self.num_classes = num_classes
        self.parts_num = self.config.data.parts_num
        self.shared_parts_id_classifier = self.config.model.bpbreid.shared_parts_id_classifier
        self.test_embeddings = self.config.model.bpbreid.test_embeddings
        self.test_use_target_segmentation = self.config.model.bpbreid.test_use_target_segmentation
        self.training_binary_visibility_score = self.config.model.bpbreid.training_binary_visibility_score
        self.testing_binary_visibility_score = self.config.model.bpbreid.testing_binary_visibility_score
        self.normalized_bned_embeddings = self.config.model.bpbreid.normalized_bned_embeddings

        # Init backbone
        self.backbone_appearance_feature_extractor = models.build_model(self.config.model.bpbreid.backbone,
                                                                        num_classes,
                                                                        loss=loss,
                                                                        pretrained=pretrained,
                                                                        last_stride=config.model.bpbreid.last_stride,
                                                                        enable_dim_reduction=(self.config.model.bpbreid.dim_reduce=='before_pooling'),
                                                                        dim_reduction_channels=self.config.model.bpbreid.dim_reduce_output,
                                                                        config=self.config
                                                                        )
        self.spatial_feature_size = self.backbone_appearance_feature_extractor.feature_dim

        # Init dim reduce layers
        self.after_pooling_dim_reduce = False
        self.before_pooling_dim_reduce = None
        self.dim_reduce_output = self.config.model.bpbreid.dim_reduce_output
        # TODO clean all of this
        if self.config.model.bpbreid.dim_reduce == 'before_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.spatial_feature_size = self.dim_reduce_output
        elif self.config.model.bpbreid.dim_reduce == 'after_pooling':
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
        elif self.config.model.bpbreid.dim_reduce == 'before_and_after_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output*2)
            self.spatial_feature_size = self.dim_reduce_output*2
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output)
        elif self.config.model.bpbreid.dim_reduce == 'after_pooling_with_dropout':
            self.after_pooling_dim_reduce = True
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output, 0.5)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output, 0.5)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output, 0.5)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(self.spatial_feature_size, self.dim_reduce_output, 0.5)
        else:
            self.dim_reduce_output = self.spatial_feature_size

        # Init pooling layers
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.foreground_global_attention_pooling_head = GlobalAveragePoolingHead('identity', self.dim_reduce_output)
        self.background_global_attention_pooling_head = GlobalAveragePoolingHead('identity', self.dim_reduce_output)
        # TODO still usefull with softmax?
        # TODO clean those Pooling heads, keep gap gmp gwap

        pooling = config.model.bpbreid.pooling
        normalization = config.model.bpbreid.normalization
        if pooling == 'gap':
            self.global_attention_pooling_head = GlobalAveragePoolingHead(normalization, self.dim_reduce_output)
        elif pooling == 'gmp':
            self.global_attention_pooling_head = GlobalMaxPoolingHead(normalization, self.dim_reduce_output)
        elif pooling == 'gwap':
            self.global_attention_pooling_head = GlobalWeightedAveragePoolingHead(normalization, self.dim_reduce_output)
        elif pooling == 'softmax':
            self.global_attention_pooling_head = SoftmaxAveragePoolingHead(normalization, self.dim_reduce_output)
        elif pooling == 'gwap2':
            self.global_attention_pooling_head = GlobalWeightedAveragePoolingHead2(normalization, self.dim_reduce_output)
        else:
            raise ValueError('pooling type {} not supported'.format(pooling))

        # Init parts classifier
        self.learnable_attention_enabled = self.config.model.bpbreid.learnable_attention_enabled
        self.pixel_classifier = PixelToPartClassifier(self.spatial_feature_size, self.parts_num)

        # Init id classifier
        self.global_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.background_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.foreground_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        self.concat_parts_identity_classifier = BNClassifier(self.parts_num * self.dim_reduce_output, self.num_classes)
        if self.shared_parts_id_classifier:
            self.parts_identity_classifier = BNClassifier(self.dim_reduce_output, self.num_classes)
        else:
            self.parts_identity_classifier = nn.ModuleList(
                [
                    BNClassifier(self.dim_reduce_output, self.num_classes)
                    for _ in range(self.parts_num)
                ]
            )

    def forward(self, images, external_parts_masks=None):
        """ Outputs one feature vector of depth D for each of the M parts for each of the N input images of the batch
        """

        # Global spatial_features
        spatial_features = self.backbone_appearance_feature_extractor(images)  # [N, D, Hf, Wf]
        N, _, Hf, Wf = spatial_features.shape
        # images.shape = [N, C, Hi, Wi], ex. [32, 3, 256, 128]
        # external_parts_masks.shape =  [N, M, Hm, Wm], ex. [32, 36, 17, 9]
        # spatial_features.shape = [N, D, Hf, Wf], ex. [32, 2048, 16, 8]

        if self.before_pooling_dim_reduce is not None:
            spatial_features = self.before_pooling_dim_reduce(spatial_features)  # [N, dim_reduce_output, Hf, Wf]

        global_embeddings = self.global_avgpool(spatial_features).view(N, -1)  # [N, D]

        # Pixels classification and parts attention weights
        if self.learnable_attention_enabled:
            pixels_cls_scores = self.pixel_classifier(spatial_features)  # [N, P, Hf, Wf]
            pixels_parts_probabilities = F.softmax(pixels_cls_scores, dim=1)
        else:
            assert external_parts_masks is not None
            pixels_parts_probabilities = nn.functional.interpolate(external_parts_masks, (Hf, Wf), mode='bilinear', align_corners=True)
            assert pixels_parts_probabilities.max() <= 1 and pixels_parts_probabilities.min() >= 0
            pixels_cls_scores = None

        background_masks = pixels_parts_probabilities[:, 0]
        parts_masks = pixels_parts_probabilities[:, 1:]

        # Explicit pixels segmentation of re-id target using external part masks
        if not self.training and self.test_use_target_segmentation == 'hard':
            assert external_parts_masks is not None
            # hard masking
            external_parts_masks = nn.functional.interpolate(external_parts_masks, (Hf, Wf), mode='bilinear',
                                                                   align_corners=True)
            target_segmentation_mask = external_parts_masks[:, 1::].max(dim=1)[0] > external_parts_masks[:, 0]
            background_masks = ~target_segmentation_mask
            parts_masks[background_masks.unsqueeze(1).expand_as(parts_masks)] = 1e-12 # 1e-12 # TODO why do not work with 0?

        if not self.training and self.test_use_target_segmentation == 'soft':
            assert external_parts_masks is not None
            # soft masking
            external_parts_masks = nn.functional.interpolate(external_parts_masks, (Hf, Wf), mode='bilinear',
                                                                   align_corners=True)
            parts_masks = parts_masks * external_parts_masks[:, 1::]

        # foreground_masks = parts_masks.sum(dim=1)
        foreground_masks = parts_masks.max(dim=1)[0]
        global_masks = torch.ones_like(foreground_masks)

        # Parts visibility
        # if self.training or self.binary_visibility_score:
        if (self.training and self.training_binary_visibility_score) or (not self.training and self.testing_binary_visibility_score):
            pixels_parts_predictions = pixels_parts_probabilities.argmax(dim=1)  # [N, Hf, Wf]
            pixels_parts_predictions_one_hot = F.one_hot(pixels_parts_predictions, self.parts_num + 1).permute(0, 3, 1, 2)  # [N, P+1, Hf, Wf]
            parts_visibility = pixels_parts_predictions_one_hot.amax(dim=(2, 3)).to(torch.bool)  # [N, P+1]
        else:
            parts_visibility = pixels_parts_probabilities.amax(dim=(2, 3))  # [N, P+1]
        background_visibility = parts_visibility[:, 0]  # [N]
        foreground_visibility = parts_visibility.amax(dim=1)  # [N]
        parts_visibility = parts_visibility[:, 1:]  # [N, P]
        concat_parts_visibility = foreground_visibility
        global_visibility = torch.ones_like(foreground_visibility)  # [N]

        # Foreground and background embeddings
        foreground_embeddings = self.foreground_global_attention_pooling_head(spatial_features, foreground_masks.unsqueeze(1)).flatten(1, 2)  # [N, D]
        background_embeddings = self.background_global_attention_pooling_head(spatial_features, background_masks.unsqueeze(1)).flatten(1, 2)  # [N, D]

        # Part features
        parts_embeddings = self.global_attention_pooling_head(spatial_features, parts_masks)  # [N, P, D]

        # Dim reduction  # FIXME apply dim reduce on foreground, background and global
        if self.after_pooling_dim_reduce:  # No improvement on PCB, but big improvement on ResNet (PCB uses dropout before conv, ResNet uses not dropout)
            global_embeddings = self.global_after_pooling_dim_reduce(global_embeddings)  # [N, dim_reduce_output]
            foreground_embeddings = self.foreground_after_pooling_dim_reduce(foreground_embeddings)  # [N, dim_reduce_output]
            background_embeddings = self.background_after_pooling_dim_reduce(background_embeddings)  # [N, dim_reduce_output]
            parts_embeddings = self.parts_after_pooling_dim_reduce(parts_embeddings)  # [N, M, dim_reduce_output]

        # Concatenated part features
        concat_parts_embeddings = parts_embeddings.flatten(1, 2)  # [N, P*D]

        # Identity classification scores
        bned_global_embeddings, global_cls_score = self.global_identity_classifier(global_embeddings)  # [N, D], [N, num_classes]
        bned_background_features, background_cls_score = self.background_identity_classifier(background_embeddings)  # [N, D], [N, num_classes]
        bned_foreground_embeddings, foreground_cls_score = self.foreground_identity_classifier(foreground_embeddings)  # [N, D], [N, num_classes]
        bned_concat_parts_embeddings, concat_parts_cls_score = self.concat_parts_identity_classifier(concat_parts_embeddings) # [N, P*D], [N, num_classes]
        bned_parts_embeddings, parts_cls_score = self.parts_identity_classification(self.dim_reduce_output, N, parts_embeddings)  # [N, P, D], [N*P, num_classes] or [N, num_classes], [N*P, D] or [N, P*D]

        # Outputs
        if self.training:
            if self.normalized_bned_embeddings:
                embeddings = {GLOBAL: F.normalize(bned_global_embeddings, p=2, dim=-1),  # [N, D]
                              BACKGROUND: F.normalize(bned_background_features, p=2, dim=-1),  # [N, D]
                              FOREGROUND: F.normalize(bned_foreground_embeddings, p=2, dim=-1),  # [N, D]
                              CONCAT_PARTS: F.normalize(bned_concat_parts_embeddings, p=2, dim=-1),  # [N, P*D] or [N, P, D]
                              PARTS: F.normalize(bned_parts_embeddings, p=2, dim=-1),  # [N, P*D] or [N, P, D]
                              }
            else:
                embeddings = {GLOBAL: global_embeddings,  # [N, D]
                              BACKGROUND: background_embeddings,  # [N, D]
                              FOREGROUND: foreground_embeddings,  # [N, D]
                              CONCAT_PARTS: concat_parts_embeddings,  # [N, P*D] or [N, P, D]
                              PARTS: parts_embeddings,  # [N, P*D] or [N, P, D]
                              }
            visibility_scores = {GLOBAL: global_visibility,  # [N]
                                 BACKGROUND: background_visibility,  # [N]
                                 FOREGROUND: foreground_visibility,  # [N]
                                 CONCAT_PARTS: concat_parts_visibility,  # [N] or [N, P]
                                 PARTS: parts_visibility,  # [N] or [N, P]
                                 }
            id_cls_scores = {GLOBAL: global_cls_score,  # [N, num_classes]
                             BACKGROUND: background_cls_score,  # [N, num_classes]
                             FOREGROUND: foreground_cls_score,  # [N, num_classes]
                             CONCAT_PARTS: concat_parts_cls_score,  # [N, num_classes] or [N, P, num_classes]
                             PARTS: parts_cls_score,  # [N, num_classes] or [N, P, num_classes]
                             }
            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, parts_masks
        else:
            embeddings = []
            visibility_scores = []
            embeddings_masks = []

            # Global embedding
            if GLOBAL in self.test_embeddings:
                embeddings.append(global_embeddings.unsqueeze(1))
            if BN_GLOBAL in self.test_embeddings:
                embeddings.append(bned_global_embeddings.unsqueeze(1))
            if GLOBAL in self.test_embeddings or BN_GLOBAL in self.test_embeddings:
                visibility_scores.append(global_visibility.view(N, 1))
                embeddings_masks.append(global_masks.unsqueeze(1))

            # Foreground embedding
            if FOREGROUND in self.test_embeddings:
                embeddings.append(foreground_embeddings.unsqueeze(1))
            if BN_FOREGROUND in self.test_embeddings:
                embeddings.append(bned_foreground_embeddings.unsqueeze(1))
            if FOREGROUND in self.test_embeddings or BN_FOREGROUND in self.test_embeddings:
                visibility_scores.append(foreground_visibility.view(N, 1))
                embeddings_masks.append(foreground_masks.unsqueeze(1))

            # Concat embedding
            if CONCAT_PARTS in self.test_embeddings:
                embeddings.append(concat_parts_embeddings.unsqueeze(1))
            if BN_CONCAT_PARTS in self.test_embeddings:
                embeddings.append(bned_concat_parts_embeddings.unsqueeze(1))
            if CONCAT_PARTS in self.test_embeddings or BN_CONCAT_PARTS in self.test_embeddings:
                visibility_scores.append(concat_parts_visibility.view(N, 1))
                embeddings_masks.append(foreground_masks.unsqueeze(1))

            # Parts embeddings
            if PARTS in self.test_embeddings:
                embeddings.append(parts_embeddings)
            if BN_PARTS in self.test_embeddings:
                embeddings.append(bned_parts_embeddings)
            if PARTS in self.test_embeddings or BN_PARTS in self.test_embeddings:
                visibility_scores.append(parts_visibility)
                embeddings_masks.append(parts_masks)

            assert len(embeddings) != 0

            embeddings = torch.cat(embeddings, dim=1)  # [N, P+2, D]
            visibility_scores = torch.cat(visibility_scores, dim=1)  # [N, P+2]
            embeddings_masks = torch.cat(embeddings_masks, dim=1)  # [N, P+2, Hf, Wf]
            all_masks = torch.cat([global_masks.unsqueeze(1),
                                   foreground_masks.unsqueeze(1),
                                   foreground_masks.unsqueeze(1),
                                   parts_masks], dim=1)  # [N, P+2, Hf, Wf]

            return embeddings, visibility_scores, embeddings_masks, pixels_cls_scores

    def parts_identity_classification(self, D, N, parts_embeddings):
        if self.shared_parts_id_classifier:
            # apply the same classifier on each part embedding, classifier weights are therefore shared across parts
            parts_embeddings = parts_embeddings.flatten(0, 1)  # [N*P, D]
            bned_part_embeddings, part_cls_score = self.parts_identity_classifier(parts_embeddings)
            bned_part_embeddings = bned_part_embeddings.view([N, self.parts_num, D])
            part_cls_score = part_cls_score.view([N, self.parts_num, -1])
        else:
            # apply P classifiers on each of the P part embedding, each part has therefore it's own classifier weights
            scores = []
            embeddings = []
            for i, parts_identity_classifier in enumerate(self.parts_identity_classifier):
                bned_part_embeddings, part_cls_score = parts_identity_classifier(parts_embeddings[:, i])
                scores.append(part_cls_score.unsqueeze(1))
                embeddings.append(bned_part_embeddings.unsqueeze(1))
            part_cls_score = torch.cat(scores, 1)
            bned_part_embeddings = torch.cat(embeddings, 1)

        return bned_part_embeddings, part_cls_score

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer

        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


class BeforePoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BeforePoolingDimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                input_dim, output_dim, 1, stride=1, padding=0
            )
        )
        layers.append(nn.BatchNorm2d(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        return self.layers(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AfterPoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=None):
        super(AfterPoolingDimReduceLayer, self).__init__()
        # dim reduction used in ResNet and PCB
        layers = []
        layers.append(
            nn.Linear(
                input_dim, output_dim, bias=True
            )
        )
        layers.append(nn.BatchNorm1d(output_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))

        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        if len(x.size()) == 3:
            N, P, _ = x.size()  # [N, P, input_dim]
            x = x.flatten(0, 1)
            x = self.layers(x)
            x = x.view(N, P, -1)
        else:
            x = self.layers(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class PixelToPartClassifier(nn.Module):
    def __init__(self, dim_reduce_output, parts_num):
        super(PixelToPartClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm2d(dim_reduce_output)
        self.classifier = nn.Conv2d(in_channels=dim_reduce_output, out_channels=parts_num + 1, kernel_size=1, stride=1, padding=0)
        self._init_params()

    def forward(self, x):
        x = self.bn(x)
        return self.classifier(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class BNClassifier(nn.Module):
    # Source: https://github.com/upgirlnana/Pytorch-Person-REID-Baseline-Bag-of-Tricks
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()

        self.in_dim = in_dim
        self.class_num = class_num

        self.bn = nn.BatchNorm1d(self.in_dim)
        self.bn.bias.requires_grad_(False) # BoF: this doesn't have a big impact on perf according to author on github
        self.classifier = nn.Linear(self.in_dim, self.class_num, bias=False)

        self._init_params()

    def forward(self, x):
        # self.bn.bias.requires_grad_(False)  # previous call might be canceled by torchtools.open_layers call # FIXME
        feature = self.bn(x)
        cls_score = self.classifier(feature)
        return feature, cls_score

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalMaskWeightedPoolingHead(nn.Module): # TODO (need other name: weighted pooling by masks)
    def __init__(self, normalization, depth):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        else:
            raise ValueError('normalization type {} not supported'.format(normalization))
        # self._init_params()

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        parts_features = torch.mul(part_masks, features)  # [N, M, D, Hf, Wf]

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)  # [N*M, D, Hf, Wf]
        parts_features = self.normalization(parts_features)  # [N*M, D, Hf, Wf] # TODO one BN per bp?
        parts_features = self.global_pooling(parts_features)  # [N*M, D, 1, 1] # TODO BN 1D after GP?
        parts_features = parts_features.view(N, M, -1)  # [N, M, D]
        return parts_features

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)  # ResNet = 0.01, Bof and ISP-reid = 0.001
                # Try nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class GlobalMaxPoolingHead(GlobalMaskWeightedPoolingHead):
    def __init__(self, normalization, depth):
        super().__init__(normalization, depth)
        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        # self._init_params()


class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def __init__(self, normalization, depth):
        super().__init__(normalization, depth)
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        # self._init_params()


# TODO compare parts_features norm distribution with avg pooling and weighted pooling
# TODO fix size annotation
class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        parts_features = torch.mul(part_masks, features)  # [N, M, D, Hf, Wf]

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)  # [N*M, D, Hf, Wf]
        parts_features = self.normalization(parts_features)  # [N*M, D, Hf, Wf] # TODO why here?
        parts_features = torch.sum(parts_features, dim=(-2, -1))  # [N, M, D]
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))  # [N, M, 1]
        parts_features_avg = torch.div(parts_features, part_masks_sum)  # [N, M, D]
        parts_features = parts_features_avg.view(N, M, -1)  # [N, M, D]
        return parts_features


class GlobalWeightedAveragePoolingHead2(GlobalMaskWeightedPoolingHead): # TODO write toy tests?
    # Should be the same as above with identity as normalization: TODO test it!
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        N, M, _, H, W = part_masks.shape
        # normalized_part_masks = F.softmax(part_masks.reshape((N, M, -1)), dim=2).reshape((N, M, 1, H, W)) # [N, M, 1, Hf, Wf]
        normalized_part_masks = torch.div(part_masks, torch.sum(part_masks, dim=(-2, -1), keepdim=True))
        parts_features = torch.mul(normalized_part_masks, features)  # [N, M, D, Hf, Wf]

        parts_features = parts_features.flatten(0, 1)  # [N*M, D, Hf, Wf]

        parts_features = self.normalization(parts_features)  # [N*M, D, Hf, Wf] # TODO why here?
        parts_features = torch.sum(parts_features, [-2, -1])
        parts_features = parts_features.view(N, M, -1)  # [N, M, D]

        return parts_features


# TODO compare parts_features norm distribution with avg pooling and weighted pooling
class GlobalWeightedAveragePoolingHead1D(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        parts_features = torch.mul(part_masks, features)  # [N, M, D, Hf, Wf]

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)  # [N*M, D, Hf, Wf]
        parts_features = torch.sum(parts_features, dim=(-2, -1))  # [N, M, D]
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))  # [N, M, 1]
        parts_features = torch.div(parts_features, part_masks_sum)  # [N, M, D]

        parts_features = self.normalization(parts_features)  # [N*M, D, Hf, Wf] # TODO why here?
        parts_features = parts_features.view(N, M, -1)  # [N, M, D]
        return parts_features

class GlobalWeightedAveragePoolingHead21D(GlobalMaskWeightedPoolingHead): # TODO write toy tests?
    # Should be the same as above with identity as normalization: TODO test it!
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        N, M, _, H, W = part_masks.shape
        # normalized_part_masks = F.softmax(part_masks.reshape((N, M, -1)), dim=2).reshape((N, M, 1, H, W)) # [N, M, 1, Hf, Wf]
        normalized_part_masks = torch.div(part_masks, torch.sum(part_masks, dim=(-2, -1), keepdim=True))
        parts_features = torch.mul(normalized_part_masks, features)  # [N, M, D, Hf, Wf]

        parts_features = parts_features.flatten(0, 1)  # [N*M, D, Hf, Wf]
        parts_features = torch.sum(parts_features, [-2, -1])

        parts_features = self.normalization(parts_features)  # [N*M, D, Hf, Wf] # TODO why here?
        parts_features = parts_features.view(N, M, -1)  # [N, M, D]
        return parts_features


# TODO compare parts_features norm distribution with avg pooling and weighted pooling
class GlobalWeightedAveragePoolingHead3D(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        parts_features = torch.mul(part_masks, features)  # [N, M, D, Hf, Wf]

        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.permute(0, 2, 1, 3, 4)  # [N, D, M, Hf, Wf]
        parts_features = self.normalization(parts_features)  # [N, D, M, Hf, Wf] # TODO why here? no sense to do that before torch.div (?)
        parts_features = parts_features.permute(0, 2, 1, 3, 4)  # [N, M, D, Hf, Wf]
        parts_features = torch.sum(parts_features, dim=(-2, -1))  # [N, M, D]
        part_masks_sum = torch.sum(part_masks, dim=(-2, -1))  # [N, M, 1]
        parts_features = torch.div(parts_features, part_masks_sum)  # [N, M, D]
        return parts_features


class SoftmaxAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)  # [N, M, 1, Hf, Wf]
        features = torch.unsqueeze(features, 1)  # [N, 1, D, Hf, Wf]
        N, M, _, H, W = part_masks.shape
        normalized_part_masks = F.softmax(part_masks.reshape((N, M, -1)), dim=2).reshape((N, M, 1, H, W)) # [N, M, 1, Hf, Wf]
        parts_features = torch.mul(normalized_part_masks, features)  # [N, M, D, Hf, Wf]

        parts_features = parts_features.flatten(0, 1)  # [N*M, D, Hf, Wf]
        parts_features = self.normalization(parts_features)  # [N*M, D, Hf, Wf]
        parts_features = torch.sum(parts_features, [-2, -1])
        parts_features = parts_features.view(N, M, -1)  # [N, M, D]
        return parts_features


def bpbreid(num_classes, loss='part_based', pretrained=True, **kwargs):
    model = BPBreID(
        num_classes,
        pretrained,
        loss,
        **kwargs
    )
    return model
