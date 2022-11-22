import torch
import torchreid
from torchreid import models, utils

from custom_engine import CustomImageSoftmaxEngine

torchreid.models.show_avai_models()

# Load data manager
datamanager = torchreid.data.ImageDataManager(
    root='reid-data',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=100,
    transforms=['random_flip', 'random_crop'],
    use_gpu=True
)

print(datamanager.num_train_pids)
print(datamanager.num_train_cams)

# Build model, optimizer and lr_scheduler
model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True,
    use_gpu=True
)

num_params, flops = utils.compute_model_complexity(model, (1, 3, 256, 128))

# count flops for all layers including ReLU and BatchNorm
utils.compute_model_complexity(model, (1, 3, 256, 128), verbose=True, only_conv_linear=False)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim='adam',
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler='single_step',
    stepsize=1
)
# Build engine
engine = CustomImageSoftmaxEngine(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    use_gpu=True,
    label_smooth=True
)

# Run training and test
engine.run(
    save_dir='log/resnet50',
    max_epoch=10,
    start_epoch=0,
    print_freq=1,
    fixbase_epoch=0,
    open_layers=None,
    start_eval=0,
    eval_freq=-1,
    test_only=False,
    dist_metric='euclidean',
    normalize_feature=False,
    visrank=False,
    visrank_topk=10,
    use_metric_cuhk03=False,
    ranks=[1, 5, 10, 20],
    rerank=False
)