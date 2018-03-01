# RefineNet


* RefineNet was originally proposed in:
[1] Guosheng Lin, Anton Milan, Chunhua Shen, Ian Reid
    RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentatiion. arXiv:1611.06612

* The RefineNet implemented in this module is based on [1]

* Since RefineNet relies on ResNet, we use ResNet-50 for this implementation
    * `resnet_v2.py` and `resnet_utils.py` are copied from github repo `tensorflow/models/research/slim/nets`
    * resnet_v2_50 in `resnet_v2.py` is used to create the 50-layer ResNet

* The key differences between this implementation and the one proposed in [1]:
    * [1] uses ResNet pretrained on ImageNet recognition tasks, while this implementaton is trained end-to-end
    * [1] uses 512 filters for each conv layer of RefineNet-4 block, while this implementation uses 256 instead, to keep it consistent with the remaining RefineNet blocks

* This implementation only supports input images that are 512x512x3. Other sizes might not work.

* tensorflow 1.5.0 or above is required. Using lower versions of tensorflow may generate "incompatible dimension" errors.

* `pretrain_resnet.py` can be used to pretrain ResNet50 defined by slim's resnet_v2_50 in `resnet_v2.py`. `wrangle_tiny_imagenet.py` is used to prepare the raw Tiny ImageNet dataset to the file structure that Keras image generator supports.
