from torchreid import models, utils

model = models.build_model(name='resnet50', num_classes=1000, loss='softmax')
num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)

model = models.build_model(name='resnet50', num_classes=1000, loss='triplet')
num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)

model = models.build_model(name='resnet50', num_classes=1000, loss='part_based')
num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)

model = models.build_model(name='bpbreid', num_classes=1000, loss='part_based')
num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)

