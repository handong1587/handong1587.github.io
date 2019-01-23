---
layout: post
category: deep_learning
title: Deep Learning Resources
date: 2015-10-09
---

# ImageNet

Single-model on 224x224

| Method              | top1        | top5        | Model Size  | Speed       |
|:-------------------:|:-----------:|:-----------:|:-----------:|:-----------:|
| ResNet-101          | 78.0%       | 94.0%       |             |             |
| ResNet-200          | 78.3%       | 94.2%       |             |             |
| Inception-v3        |             |             |             |             |
| Inception-v4        |             |             |             |             |
| Inception-ResNet-v2 |             |             |             |             |
| ResNet-50           | 77.8%       |             |             |             |
| ResNet-101          | 79.6%       | 94.7%       |             |             |

Single-model on 320×320 / 299×299

| Method              | top1        | top5        | Model Size  | Speed       |
|:-------------------:|:-----------:|:-----------:|:-----------:|:-----------:|
| ResNet-101          |             |             |             |             |
| ResNet-200          | 79.9%       | 95.2%       |             |             |
| Inception-v3        | 78.8%       | 94.4%       |             |             |
| Inception-v4        | 80.0%       | 95.0%       |             |             |
| Inception-ResNet-v2 | 80.1%       | 95.1%       |             |             |
| ResNet-50           |             |             |             |             |
| ResNet-101          | 80.9%       | 95.6%       |             |             |

## AlexNet

**ImageNet Classification with Deep Convolutional Neural Networks**

- nips-page: [http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-)
- paper: [http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- slides: [http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf)
- code: [https://code.google.com/p/cuda-convnet/](https://code.google.com/p/cuda-convnet/)
- github: [https://github.com/dnouri/cuda-convnet](https://github.com/dnouri/cuda-convnet)
- code: [https://code.google.com/p/cuda-convnet2/](https://code.google.com/p/cuda-convnet2/)

## Network In Network

**Network In Network**

- intro: ICLR 2014
- arxiv: [http://arxiv.org/abs/1312.4400](http://arxiv.org/abs/1312.4400)
- gitxiv: [http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin](http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin)
- code(Caffe, official): [https://gist.github.com/mavenlin/d802a5849de39225bcc6](https://gist.github.com/mavenlin/d802a5849de39225bcc6)

**Batch-normalized Maxout Network in Network**

- arxiv: [http://arxiv.org/abs/1511.02583](http://arxiv.org/abs/1511.02583)

## GoogLeNet (Inception V1)

**Going Deeper with Convolutions**

- arxiv: [http://arxiv.org/abs/1409.4842](http://arxiv.org/abs/1409.4842)
- github: [https://github.com/google/inception](https://github.com/google/inception)
- github: [https://github.com/soumith/inception.torch](https://github.com/soumith/inception.torch)

**Building a deeper understanding of images**

- blog: [http://googleresearch.blogspot.jp/2014/09/building-deeper-understanding-of-images.html](http://googleresearch.blogspot.jp/2014/09/building-deeper-understanding-of-images.html)

## VGGNet

**Very Deep Convolutional Networks for Large-Scale Image Recognition**

- homepage: [http://www.robots.ox.ac.uk/~vgg/research/very_deep/](http://www.robots.ox.ac.uk/~vgg/research/very_deep/)
- arxiv: [http://arxiv.org/abs/1409.1556](http://arxiv.org/abs/1409.1556)
- slides: [http://llcao.net/cu-deeplearning15/presentation/cc3580_Simonyan.pptx](http://llcao.net/cu-deeplearning15/presentation/cc3580_Simonyan.pptx)
- slides: [http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf](http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf)
- slides: [http://deeplearning.cs.cmu.edu/slides.2015/25.simonyan.pdf](http://deeplearning.cs.cmu.edu/slides.2015/25.simonyan.pdf)
- github(official, deprecated Caffe API): [https://gist.github.com/ksimonyan/211839e770f7b538e2d8](https://gist.github.com/ksimonyan/211839e770f7b538e2d8)
- github: [https://github.com/ruimashita/caffe-train](https://github.com/ruimashita/caffe-train)

**Tensorflow VGG16 and VGG19**

- github: [https://github.com/machrisaa/tensorflow-vgg](https://github.com/machrisaa/tensorflow-vgg)

## Inception-V2

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**

- intro: ImageNet top-5 error: 4.82%
- keywords: internal covariate shift problem
- arxiv: [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167)
- blog: [https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/](https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/)
- notes: [http://blog.csdn.net/happynear/article/details/44238541](http://blog.csdn.net/happynear/article/details/44238541)
- github: [https://github.com/lim0606/caffe-googlenet-bn](https://github.com/lim0606/caffe-googlenet-bn)

**ImageNet pre-trained models with batch normalization**

- arxiv: [https://arxiv.org/abs/1612.01452](https://arxiv.org/abs/1612.01452)
- project page: [http://www.inf-cv.uni-jena.de/Research/CNN+Models.html](http://www.inf-cv.uni-jena.de/Research/CNN+Models.html)
- github: [https://github.com/cvjena/cnn-models](https://github.com/cvjena/cnn-models)

## Inception-V3

Inception-V3 = Inception-V2 + BN-auxiliary (fully connected layer of the auxiliary classifier is also batch-normalized, 
not just the convolutions)

**Rethinking the Inception Architecture for Computer Vision**

- intro: "21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network; 
3.5% top-5 error and 17.3% top-1 error With an ensemble of 4 models and multi-crop evaluation."
- arxiv: [http://arxiv.org/abs/1512.00567](http://arxiv.org/abs/1512.00567)
- github: [https://github.com/Moodstocks/inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch)

**Inception in TensorFlow**

- intro: demonstrate how to train the Inception v3 architecture
- github: [https://github.com/tensorflow/models/tree/master/inception](https://github.com/tensorflow/models/tree/master/inception)

**Train your own image classifier with Inception in TensorFlow**

- intro: Inception-v3
- blog: [https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html](https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html)

**Notes on the TensorFlow Implementation of Inception v3**

[https://pseudoprofound.wordpress.com/2016/08/28/notes-on-the-tensorflow-implementation-of-inception-v3/](https://pseudoprofound.wordpress.com/2016/08/28/notes-on-the-tensorflow-implementation-of-inception-v3/)

**Training an InceptionV3-based image classifier with your own dataset**

- github: [https://github.com/danielvarga/keras-finetuning](https://github.com/danielvarga/keras-finetuning)

**Inception-BN full for Caffe: Inception-BN ImageNet (21K classes) model for Caffe**

- github: [https://github.com/pertusa/InceptionBN-21K-for-Caffe](https://github.com/pertusa/InceptionBN-21K-for-Caffe)

## ResNet

**Deep Residual Learning for Image Recognition**

- intro: CVPR 2016 Best Paper Award
- arxiv: [http://arxiv.org/abs/1512.03385](http://arxiv.org/abs/1512.03385)
- slides: [http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)
- gitxiv: [http://gitxiv.com/posts/LgPRdTY3cwPBiMKbm/deep-residual-learning-for-image-recognition](http://gitxiv.com/posts/LgPRdTY3cwPBiMKbm/deep-residual-learning-for-image-recognition)
- github: [https://github.com/KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)
- github: [https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)

**Third-party re-implementations**

[https://github.com/KaimingHe/deep-residual-networks#third-party-re-implementations](https://github.com/KaimingHe/deep-residual-networks#third-party-re-implementations)

**Training and investigating Residual Nets**

- intro: Facebook AI Research
- blog: [http://torch.ch/blog/2016/02/04/resnets.html](http://torch.ch/blog/2016/02/04/resnets.html)
- github: [https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)

**resnet.torch: an updated version of fb.resnet.torch with many changes.**

- github: [https://github.com/erogol/resnet.torch](https://github.com/erogol/resnet.torch)

**Highway Networks and Deep Residual Networks**

- blog: [http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html](http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html)

**Interpretating Deep Residual Learning Blocks as Locally Recurrent Connections**

- blog: [https://matrixmashing.wordpress.com/2016/01/29/interpretating-deep-residual-learning-blocks-as-locally-recurrent-connections/](https://matrixmashing.wordpress.com/2016/01/29/interpretating-deep-residual-learning-blocks-as-locally-recurrent-connections/)

**Lab41 Reading Group: Deep Residual Learning for Image Recognition**

- blog: [https://gab41.lab41.org/lab41-reading-group-deep-residual-learning-for-image-recognition-ffeb94745a1f](https://gab41.lab41.org/lab41-reading-group-deep-residual-learning-for-image-recognition-ffeb94745a1f)

**50-layer ResNet, trained on ImageNet, classifying webcam**

- homepage: [https://ml4a.github.io/demos/keras.js/](https://ml4a.github.io/demos/keras.js/)

**Reproduced ResNet on CIFAR-10 and CIFAR-100 dataset.**

- github: [https://github.com/tensorflow/models/tree/master/resnet](https://github.com/tensorflow/models/tree/master/resnet)

## ResNet-V2

**Identity Mappings in Deep Residual Networks**

- intro: ECCV 2016. ResNet-v2
- arxiv: [http://arxiv.org/abs/1603.05027](http://arxiv.org/abs/1603.05027)
- github: [https://github.com/KaimingHe/resnet-1k-layers](https://github.com/KaimingHe/resnet-1k-layers)
- github: [https://github.com/tornadomeet/ResNet](https://github.com/tornadomeet/ResNet)

**Deep Residual Networks for Image Classification with Python + NumPy**

![](https://dnlcrl.github.io/assets/thesis-post/Diagramma.png)

- blog: [https://dnlcrl.github.io/projects/2016/06/22/Deep-Residual-Networks-for-Image-Classification-with-Python+NumPy.html](https://dnlcrl.github.io/projects/2016/06/22/Deep-Residual-Networks-for-Image-Classification-with-Python+NumPy.html)

## Inception-V4 / Inception-ResNet-V2

**Inception-V4, Inception-Resnet And The Impact Of Residual Connections On Learning**

- intro: Workshop track - ICLR 2016. 3.08 % top-5 error on ImageNet CLS
- intro: "achieve 3.08% top-5 error on the test set of the ImageNet classification (CLS) challenge"
- arxiv: [http://arxiv.org/abs/1602.07261](http://arxiv.org/abs/1602.07261)
- github(Keras): [https://github.com/kentsommer/keras-inceptionV4](https://github.com/kentsommer/keras-inceptionV4)

**The inception-resnet-v2 models trained from scratch via torch**

- github: [https://github.com/lim0606/torch-inception-resnet-v2](https://github.com/lim0606/torch-inception-resnet-v2)

**Inception v4 in Keras**

- intro: Inception-v4, Inception - Resnet-v1 and v2
- github: [https://github.com/titu1994/Inception-v4](https://github.com/titu1994/Inception-v4)

## ResNeXt

**Aggregated Residual Transformations for Deep Neural Networks**

- intro: CVPR 2017. UC San Diego & Facebook AI Research
- arxiv: [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
- github(Torch): [https://github.com/facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt)
- github: [https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol/resnext.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol/resnext.py)
- dataset: [http://data.dmlc.ml/models/imagenet/resnext/](http://data.dmlc.ml/models/imagenet/resnext/)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/5haml9/p_implementation_of_aggregated_residual/](https://www.reddit.com/r/MachineLearning/comments/5haml9/p_implementation_of_aggregated_residual/)

## Residual Networks Variants

**Resnet in Resnet: Generalizing Residual Architectures**

- paper: [http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g](http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g)
- arxiv: [http://arxiv.org/abs/1603.08029](http://arxiv.org/abs/1603.08029)

**Residual Networks are Exponential Ensembles of Relatively Shallow Networks**

- arxiv: [http://arxiv.org/abs/1605.06431](http://arxiv.org/abs/1605.06431)

**Wide Residual Networks**

- intro: BMVC 2016
- arxiv: [http://arxiv.org/abs/1605.07146](http://arxiv.org/abs/1605.07146)
- github: [https://github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)
- github: [https://github.com/asmith26/wide_resnets_keras](https://github.com/asmith26/wide_resnets_keras)
- github: [https://github.com/ritchieng/wideresnet-tensorlayer](https://github.com/ritchieng/wideresnet-tensorlayer)
- github: [https://github.com/xternalz/WideResNet-pytorch](https://github.com/xternalz/WideResNet-pytorch)
- github(Torch): [https://github.com/meliketoy/wide-residual-network](https://github.com/meliketoy/wide-residual-network)

**Residual Networks of Residual Networks: Multilevel Residual Networks**

- arxiv: [http://arxiv.org/abs/1608.02908](http://arxiv.org/abs/1608.02908)

**Multi-Residual Networks**

- arxiv: [http://arxiv.org/abs/1609.05672](http://arxiv.org/abs/1609.05672)
- github: [https://github.com/masoudabd/multi-resnet](https://github.com/masoudabd/multi-resnet)

**Deep Pyramidal Residual Networks**

- intro: PyramidNet
- arxiv: [https://arxiv.org/abs/1610.02915](https://arxiv.org/abs/1610.02915)
- github: [https://github.com/jhkim89/PyramidNet](https://github.com/jhkim89/PyramidNet)

**Learning Identity Mappings with Residual Gates**

- arxiv: [https://arxiv.org/abs/1611.01260](https://arxiv.org/abs/1611.01260)

**Wider or Deeper: Revisiting the ResNet Model for Visual Recognition**

- intro: image classification, semantic image segmentation
- arxiv: [https://arxiv.org/abs/1611.10080](https://arxiv.org/abs/1611.10080)
- github: [https://github.com/itijyou/ademxapp](https://github.com/itijyou/ademxapp)

**Deep Pyramidal Residual Networks with Separated Stochastic Depth**

- arxiv: [https://arxiv.org/abs/1612.01230](https://arxiv.org/abs/1612.01230)

**Spatially Adaptive Computation Time for Residual Networks**

- intro: Higher School of Economics & Google & CMU
- arxiv: [https://arxiv.org/abs/1612.02297](https://arxiv.org/abs/1612.02297)

**ShaResNet: reducing residual network parameter number by sharing weights**

- arxiv: [https://arxiv.org/abs/1702.08782](https://arxiv.org/abs/1702.08782)
- github: [https://github.com/aboulch/sharesnet](https://github.com/aboulch/sharesnet)

**Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks**

- intro: Collective Residual Networks
- arxiv: [https://arxiv.org/abs/1703.02180](https://arxiv.org/abs/1703.02180)
- github(MXNet): [https://github.com/cypw/CRU-Net](https://github.com/cypw/CRU-Net)

**Residual Attention Network for Image Classification**

- intro: CVPR 2017 Spotlight. SenseTime Group Limited & Tsinghua University & The Chinese University of Hong Kong
- intro: ImageNet (4.8% single model and single crop, top-5 error)
- arxiv: [https://arxiv.org/abs/1704.06904](https://arxiv.org/abs/1704.06904)
- github(Caffe): [https://github.com/buptwangfei/residual-attention-network](https://github.com/buptwangfei/residual-attention-network)

**Dilated Residual Networks**

- intro: CVPR 2017. Princeton University & Intel Labs
- keywords: Dilated Residual Networks (DRN)
- project page: [http://vladlen.info/publications/dilated-residual-networks/](http://vladlen.info/publications/dilated-residual-networks/)
- arxiv: [https://arxiv.org/abs/1705.09914](https://arxiv.org/abs/1705.09914)
- paper: [http://vladlen.info/papers/DRN.pdf](http://vladlen.info/papers/DRN.pdf)

**Dynamic Steerable Blocks in Deep Residual Networks**

- intro: University of Amsterdam & ESAT-PSI
- arxiv: [https://arxiv.org/abs/1706.00598](https://arxiv.org/abs/1706.00598)

**Learning Deep ResNet Blocks Sequentially using Boosting Theory**

- intro: Microsoft Research & Princeton University
- arxiv: [https://arxiv.org/abs/1706.04964](https://arxiv.org/abs/1706.04964)

**Learning Strict Identity Mappings in Deep Residual Networks**

- keywords: epsilon-ResNet
- arxiv: [https://arxiv.org/abs/1804.01661](https://arxiv.org/abs/1804.01661)

**Spiking Deep Residual Network**

[https://arxiv.org/abs/1805.01352](https://arxiv.org/abs/1805.01352)

**Norm-Preservation: Why Residual Networks Can Become Extremely Deep?**

- intro: University of Central Florida
- arxiv: [https://arxiv.org/abs/1805.07477](https://arxiv.org/abs/1805.07477)

## DenseNet

**Densely Connected Convolutional Networks**

![](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)

- intro: CVPR 2017 best paper. Cornell University & Tsinghua University. DenseNet
- arxiv: [http://arxiv.org/abs/1608.06993](http://arxiv.org/abs/1608.06993)
- github: [https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)
- github(Lasagne): [https://github.com/Lasagne/Recipes/tree/master/papers/densenet](https://github.com/Lasagne/Recipes/tree/master/papers/densenet)
- github(Keras): [https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet)
- github(Caffe): [https://github.com/liuzhuang13/DenseNetCaffe](https://github.com/liuzhuang13/DenseNetCaffe)
- github(Tensorflow): [https://github.com/YixuanLi/densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow)
- github(Keras): [https://github.com/titu1994/DenseNet](https://github.com/titu1994/DenseNet)
- github(PyTorch): [https://github.com/bamos/densenet.pytorch](https://github.com/bamos/densenet.pytorch)
- github(PyTorch): [https://github.com/andreasveit/densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- github(Tensorflow): [https://github.com/ikhlestov/vision_networks](https://github.com/ikhlestov/vision_networks)

**Memory-Efficient Implementation of DenseNets**

- intro: Cornell University & Fudan University & Facebook AI Research
- arxiv: [https://arxiv.org/abs/1707.06990](https://arxiv.org/abs/1707.06990)
- github: [https://github.com/liuzhuang13/DenseNet/tree/master/models](https://github.com/liuzhuang13/DenseNet/tree/master/models)
- github: [https://github.com/gpleiss/efficient_densenet_pytorch](https://github.com/gpleiss/efficient_densenet_pytorch)
- github: [https://github.com/taineleau/efficient_densenet_mxnet](https://github.com/taineleau/efficient_densenet_mxnet)
- github: [https://github.com/Tongcheng/DN_CaffeScript](https://github.com/Tongcheng/DN_CaffeScript)

## DenseNet 2.0

**CondenseNet: An Efficient DenseNet using Learned Group Convolutions**

- arxiv: [https://arxiv.org/abs/1711.09224](https://arxiv.org/abs/1711.09224)
- github: [https://github.com//ShichenLiu/CondenseNet](https://github.com//ShichenLiu/CondenseNet)

**Multimodal Densenet**

[https://arxiv.org/abs/1811.07407](https://arxiv.org/abs/1811.07407)

## Xception

**Deep Learning with Separable Convolutions**

**Xception: Deep Learning with Depthwise Separable Convolutions**

- intro: CVPR 2017. Extreme Inception
- arxiv: [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)
- code: [https://keras.io/applications/#xception](https://keras.io/applications/#xception)
- github(Keras): [https://github.com/fchollet/deep-learning-models/blob/master/xception.py](https://github.com/fchollet/deep-learning-models/blob/master/xception.py)
- github: [https://gist.github.com/culurciello/554c8e56d3bbaf7c66bf66c6089dc221](https://gist.github.com/culurciello/554c8e56d3bbaf7c66bf66c6089dc221)
- github: [https://github.com/kwotsin/Tensorflow-Xception](https://github.com/kwotsin/Tensorflow-Xception)
- github: [https://github.com//bruinxiong/xception.mxnet](https://github.com//bruinxiong/xception.mxnet)
- notes: [http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1610.02357](http://www.shortscience.org/paper?bibtexKey=journals%2Fcorr%2F1610.02357)

**Towards a New Interpretation of Separable Convolutions**

- arxiv: [https://arxiv.org/abs/1701.04489](https://arxiv.org/abs/1701.04489)

## MobileNets

**MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications**

- intro: Google
- arxiv: [https://arxiv.org/abs/1704.04861](https://arxiv.org/abs/1704.04861)
- github: [https://github.com/rcmalli/keras-mobilenet](https://github.com/rcmalli/keras-mobilenet)
- github: [https://github.com/marvis/pytorch-mobilenet](https://github.com/marvis/pytorch-mobilenet)
- github(Tensorflow): [https://github.com/Zehaos/MobileNet](https://github.com/Zehaos/MobileNet)
- github: [https://github.com/shicai/MobileNet-Caffe](https://github.com/shicai/MobileNet-Caffe)
- github: [https://github.com/hollance/MobileNet-CoreML](https://github.com/hollance/MobileNet-CoreML)
- github: [https://github.com/KeyKy/mobilenet-mxnet](https://github.com/KeyKy/mobilenet-mxnet)

**MobileNets: Open-Source Models for Efficient On-Device Vision**

- blog: [https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html](https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html)
- github: [https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md](https://github.com/tensorflow/models/blob/master/slim/nets/mobilenet_v1.md)

**Google’s MobileNets on the iPhone**

- blog: [http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/](http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/)
- github: [https://github.com/hollance/MobileNet-CoreML](https://github.com/hollance/MobileNet-CoreML)

**Depth_conv-for-mobileNet**

[https://github.com//LamHoCN/Depth_conv-for-mobileNet](https://github.com//LamHoCN/Depth_conv-for-mobileNet)

**The Enhanced Hybrid MobileNet**

[https://arxiv.org/abs/1712.04698](https://arxiv.org/abs/1712.04698)

**FD-MobileNet: Improved MobileNet with a Fast Downsampling Strategy**

[https://arxiv.org/abs/1802.03750](https://arxiv.org/abs/1802.03750)

**A Quantization-Friendly Separable Convolution for MobileNets**

- intro: THE 1ST WORKSHOP ON ENERGY EFFICIENT MACHINE LEARNING AND COGNITIVE COMPUTING FOR EMBEDDED APPLICATIONS (EMC2)
- arxiv: [https://arxiv.org/abs/1803.08607](https://arxiv.org/abs/1803.08607)

## MobileNetV2

**Inverted Residuals and Linear Bottlenecks: Mobile Networks forClassification, Detection and Segmentation**

- intro: Google
- keywords: MobileNetV2, SSDLite, DeepLabv3
- arxiv: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- github: [https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet](https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet)
- github: [https://github.com/liangfu/mxnet-mobilenet-v2](https://github.com/liangfu/mxnet-mobilenet-v2)
- blog: [https://research.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html](https://research.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html)

**PydMobileNet: Improved Version of MobileNets with Pyramid Depthwise Separable Convolution**

[https://arxiv.org/abs/1811.07083](https://arxiv.org/abs/1811.07083)

## ShuffleNet

**ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices**

- intro: Megvii Inc (Face++)
- arxiv: [https://arxiv.org/abs/1707.01083](https://arxiv.org/abs/1707.01083)

## ShuffleNet V2

**ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design**

- intro: ECCV 2018. Megvii Inc (Face++) & Tsinghua University
- arxiv: [https://arxiv.org/abs/1807.11164]（https://arxiv.org/abs/1807.11164

## SENet

**Squeeze-and-Excitation Networks**

- intro: CVPR 2018
- intro: ILSVRC 2017 image classification winner. Momenta & University of Oxford
- arxiv: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
- github(official, Caffe): [https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)
- github: [https://github.com/bruinxiong/SENet.mxnet](https://github.com/bruinxiong/SENet.mxnet)

**Competitive Inner-Imaging Squeeze and Excitation for Residual Network**

- arxiv: [https://arxiv.org/abs/1807.08920](https://arxiv.org/abs/1807.08920)
- github: [https://github.com/scut-aitcm/CompetitiveSENet](https://github.com/scut-aitcm/CompetitiveSENet)

## GENet

**Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks**

- intro: NIPS 2018
- github: [https://github.com/hujie-frank/GENet](https://github.com/hujie-frank/GENet)

## ImageNet Projects

**Training an Object Classifier in Torch-7 on multiple GPUs over ImageNet**

- intro: an imagenet example in torch
- github: [https://github.com/soumith/imagenet-multiGPU.torch](https://github.com/soumith/imagenet-multiGPU.torch)

# Pre-training

**Exploring the Limits of Weakly Supervised Pretraining**

- intro: report the highest ImageNet-1k single-crop, top-1 accuracy to date: 85.4% (97.6% top-5)
- paper: [https://research.fb.com/publications/exploring-the-limits-of-weakly-supervised-pretraining/](https://research.fb.com/publications/exploring-the-limits-of-weakly-supervised-pretraining/)

**Rethinking ImageNet Pre-training**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/1811.08883](https://arxiv.org/abs/1811.08883)

**Revisiting Pre-training: An Efficient Training Method for Image Classification**

[https://arxiv.org/abs/1811.09347](https://arxiv.org/abs/1811.09347)

# Semi-Supervised Learning

**Semi-Supervised Learning with Graphs**

- intro: Label Propagation
- paper: [http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf)
- blog("标签传播算法（Label Propagation）及Python实现"): [http://blog.csdn.net/zouxy09/article/details/49105265](http://blog.csdn.net/zouxy09/article/details/49105265)

**Semi-Supervised Learning with Ladder Networks**

- arxiv: [http://arxiv.org/abs/1507.02672](http://arxiv.org/abs/1507.02672)
- github: [https://github.com/CuriousAI/ladder](https://github.com/CuriousAI/ladder)
- github: [https://github.com/rinuboney/ladder](https://github.com/rinuboney/ladder)

**Semi-supervised Feature Transfer: The Practical Benefit of Deep Learning Today?**

- blog: [http://www.kdnuggets.com/2016/07/semi-supervised-feature-transfer-deep-learning.html](http://www.kdnuggets.com/2016/07/semi-supervised-feature-transfer-deep-learning.html)

**Temporal Ensembling for Semi-Supervised Learning**

- intro: ICLR 2017
- arxiv: [https://arxiv.org/abs/1610.02242](https://arxiv.org/abs/1610.02242)
- github: [https://github.com/smlaine2/tempens](https://github.com/smlaine2/tempens)

**Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data**

- intro: ICLR 2017 best paper award
- arxiv: [https://arxiv.org/abs/1610.05755](https://arxiv.org/abs/1610.05755)
- github: [https://github.com/tensorflow/models/tree/8505222ea1f26692df05e65e35824c6c71929bb5/privacy](https://github.com/tensorflow/models/tree/8505222ea1f26692df05e65e35824c6c71929bb5/privacy)

**Infinite Variational Autoencoder for Semi-Supervised Learning**

- arxiv: [https://arxiv.org/abs/1611.07800](https://arxiv.org/abs/1611.07800)

# Multi-label Learning

**CNN: Single-label to Multi-label**

- arxiv: [http://arxiv.org/abs/1406.5726](http://arxiv.org/abs/1406.5726)

**Deep Learning for Multi-label Classification**

- arxiv: [http://arxiv.org/abs/1502.05988](http://arxiv.org/abs/1502.05988)
- github: [http://meka.sourceforge.net](http://meka.sourceforge.net)

**Predicting Unseen Labels using Label Hierarchies in Large-Scale Multi-label Learning**

- intro: ECML 2015
- paper: [https://www.kdsl.tu-darmstadt.de/fileadmin/user_upload/Group_KDSL/PUnL_ECML2015_camera_ready.pdf](https://www.kdsl.tu-darmstadt.de/fileadmin/user_upload/Group_KDSL/PUnL_ECML2015_camera_ready.pdf)

**Learning with a Wasserstein Loss**

- project page: [http://cbcl.mit.edu/wasserstein/](http://cbcl.mit.edu/wasserstein/)
- arxiv: [http://arxiv.org/abs/1506.05439](http://arxiv.org/abs/1506.05439)
- code: [http://cbcl.mit.edu/wasserstein/yfcc100m_labels.tar.gz](http://cbcl.mit.edu/wasserstein/yfcc100m_labels.tar.gz)
- MIT news: [http://news.mit.edu/2015/more-flexible-machine-learning-1001](http://news.mit.edu/2015/more-flexible-machine-learning-1001)

**From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification**

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1602.02068](http://arxiv.org/abs/1602.02068)
- github: [https://github.com/gokceneraslan/SparseMax.torch](https://github.com/gokceneraslan/SparseMax.torch)
- github: [https://github.com/Unbabel/sparsemax](https://github.com/Unbabel/sparsemax)

**CNN-RNN: A Unified Framework for Multi-label Image Classification**

- arxiv: [http://arxiv.org/abs/1604.04573](http://arxiv.org/abs/1604.04573)

**Improving Multi-label Learning with Missing Labels by Structured Semantic Correlations**

- arxiv: [http://arxiv.org/abs/1608.01441](http://arxiv.org/abs/1608.01441)

**Extreme Multi-label Loss Functions for Recommendation, Tagging, Ranking & Other Missing Label Applications**

- intro: Indian Institute of Technology Delhi & MSR
- paper: [https://manikvarma.github.io/pubs/jain16.pdf](https://manikvarma.github.io/pubs/jain16.pdf)

**Multi-Label Image Classification with Regional Latent Semantic Dependencies**

- intro: Regional Latent Semantic Dependencies model (RLSD), RNN, RPN
- arxiv: [https://arxiv.org/abs/1612.01082](https://arxiv.org/abs/1612.01082)

**Privileged Multi-label Learning**

- intro: Peking University & University of Technology Sydney & University of Sydney
- arxiv: [https://arxiv.org/abs/1701.07194](https://arxiv.org/abs/1701.07194)

# Multi-task Learning

**Multitask Learning / Domain Adaptation**

- homepage: [http://www.cs.cornell.edu/~kilian/research/multitasklearning/multitasklearning.html](http://www.cs.cornell.edu/~kilian/research/multitasklearning/multitasklearning.html)

**multi-task learning**

- discussion: [https://github.com/memect/hao/issues/93](https://github.com/memect/hao/issues/93)

**Learning and Transferring Multi-task Deep Representation for Face Alignment**

- arxiv: [http://arxiv.org/abs/1408.3967](http://arxiv.org/abs/1408.3967)

**Multi-task learning of facial landmarks and expression**

- paper: [http://www.uoguelph.ca/~gwtaylor/publications/gwtaylor_crv2014.pdf](http://www.uoguelph.ca/~gwtaylor/publications/gwtaylor_crv2014.pdf)

**Multi-Task Deep Visual-Semantic Embedding for Video Thumbnail Selection**

- intro:  CVPR 2015
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Multi-Task_Deep_Visual-Semantic_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Multi-Task_Deep_Visual-Semantic_2015_CVPR_paper.pdf)

**Learning Multiple Tasks with Deep Relationship Networks**

- arxiv: [https://arxiv.org/abs/1506.02117](https://arxiv.org/abs/1506.02117)

**Learning deep representation of multityped objects and tasks**

- arxiv: [http://arxiv.org/abs/1603.01359](http://arxiv.org/abs/1603.01359)

**Cross-stitch Networks for Multi-task Learning**

- arxiv: [http://arxiv.org/abs/1604.03539](http://arxiv.org/abs/1604.03539)

**Multi-Task Learning in Tensorflow (Part 1)**

- blog: [https://jg8610.github.io/Multi-Task/](https://jg8610.github.io/Multi-Task/)

**Deep Multi-Task Learning with Shared Memory**

- intro: EMNLP 2016
- arxiv: [http://arxiv.org/abs/1609.07222](http://arxiv.org/abs/1609.07222)

**Learning to Push by Grasping: Using multiple tasks for effective learning**

- arxiv: [http://arxiv.org/abs/1609.09025](http://arxiv.org/abs/1609.09025)

**Identifying beneficial task relations for multi-task learning in deep neural networks**

- intro: EACL 2017
- arxiv: [https://arxiv.org/abs/1702.08303](https://arxiv.org/abs/1702.08303)
- github: [https://github.com/jbingel/eacl2017_mtl](https://github.com/jbingel/eacl2017_mtl)

**Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics**

- intro: University of Cambridge
- arxiv: [https://arxiv.org/abs/1705.07115](https://arxiv.org/abs/1705.07115)

**One Model To Learn Them All**

- intro: Google Brain & University of Toronto
- arxiv: [https://arxiv.org/abs/1706.05137](https://arxiv.org/abs/1706.05137)
- github: [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor)

**MultiModel: Multi-Task Machine Learning Across Domains**

[https://research.googleblog.com/2017/06/multimodel-multi-task-machine-learning.html](https://research.googleblog.com/2017/06/multimodel-multi-task-machine-learning.html)

**An Overview of Multi-Task Learning in Deep Neural Networks**

- intro: Aylien Ltd
- arxiv: [https://arxiv.org/abs/1706.05098](https://arxiv.org/abs/1706.05098)

**PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning**

- arxiv: [https://arxiv.org/abs/1711.05769](https://arxiv.org/abs/1711.05769)
- github: [https://github.com/arunmallya/packnet](https://github.com/arunmallya/packnet)

**End-to-End Multi-Task Learning with Attention**

- intro: Imperial College London
- arxiv: [https://arxiv.org/abs/1803.10704](https://arxiv.org/abs/1803.10704)

**Cross-connected Networks for Multi-task Learning of Detection and Segmentation**

[https://arxiv.org/abs/1805.05569](https://arxiv.org/abs/1805.05569)

**Auxiliary Tasks in Multi-task Learning**

[https://arxiv.org/abs/1805.06334](https://arxiv.org/abs/1805.06334)

**K For The Price Of 1: Parameter Efficient Multi-task And Transfer Learning**

- intro: The University of Chicago & Google
- arxiv: [https://arxiv.org/abs/1810.10703](https://arxiv.org/abs/1810.10703)

# Multi-modal Learning

**Multimodal Deep Learning**

- paper: [http://ai.stanford.edu/~ang/papers/nipsdlufl10-MultimodalDeepLearning.pdf](http://ai.stanford.edu/~ang/papers/nipsdlufl10-MultimodalDeepLearning.pdf)

**Multimodal Convolutional Neural Networks for Matching Image and Sentence**

- homepage: [http://mcnn.noahlab.com.hk/project.html](http://mcnn.noahlab.com.hk/project.html)
- paper: [http://mcnn.noahlab.com.hk/ICCV2015.pdf](http://mcnn.noahlab.com.hk/ICCV2015.pdf)
- arxiv: [http://arxiv.org/abs/1504.06063](http://arxiv.org/abs/1504.06063)

**A C++ library for Multimodal Deep Learning**

- arxiv: [http://arxiv.org/abs/1512.06927](http://arxiv.org/abs/1512.06927)
- github: [https://github.com/Jian-23/Deep-Learning-Library](https://github.com/Jian-23/Deep-Learning-Library)

**Multimodal Learning for Image Captioning and Visual Question Answering**

- slides: [http://research.microsoft.com/pubs/264769/UCB_XiaodongHe.6.pdf](http://research.microsoft.com/pubs/264769/UCB_XiaodongHe.6.pdf)

**Multi modal retrieval and generation with deep distributed models**

- slides: [http://www.slideshare.net/roelofp/multi-modal-retrieval-and-generation-with-deep-distributed-models](http://www.slideshare.net/roelofp/multi-modal-retrieval-and-generation-with-deep-distributed-models)
- slides: [http://pan.baidu.com/s/1kUSjn4z](http://pan.baidu.com/s/1kUSjn4z)

**Learning Aligned Cross-Modal Representations from Weakly Aligned Data**

![](http://projects.csail.mit.edu/cmplaces/imgs/teaser.png)

- homepage: [http://projects.csail.mit.edu/cmplaces/index.html](http://projects.csail.mit.edu/cmplaces/index.html)
- paper: [http://projects.csail.mit.edu/cmplaces/content/paper.pdf](http://projects.csail.mit.edu/cmplaces/content/paper.pdf)

**Variational methods for Conditional Multimodal Deep Learning**

- arxiv: [http://arxiv.org/abs/1603.01801](http://arxiv.org/abs/1603.01801)

**Training and Evaluating Multimodal Word Embeddings with Large-scale Web Annotated Images**

- intro: NIPS 2016. University of California & Pinterest
- project page: [http://www.stat.ucla.edu/~junhua.mao/multimodal_embedding.html](http://www.stat.ucla.edu/~junhua.mao/multimodal_embedding.html)
- arxiv: [https://arxiv.org/abs/1611.08321](https://arxiv.org/abs/1611.08321)

**Deep Multi-Modal Image Correspondence Learning**

- arxiv: [https://arxiv.org/abs/1612.01225](https://arxiv.org/abs/1612.01225)

**Multimodal Deep Learning (D4L4 Deep Learning for Speech and Language UPC 2017)**

- slides: [http://www.slideshare.net/xavigiro/multimodal-deep-learning-d4l4-deep-learning-for-speech-and-language-upc-2017](http://www.slideshare.net/xavigiro/multimodal-deep-learning-d4l4-deep-learning-for-speech-and-language-upc-2017)

# Debugging Deep Learning

**Some tips for debugging deep learning**

- blog: [http://www.lab41.org/some-tips-for-debugging-in-deep-learning-2/](http://www.lab41.org/some-tips-for-debugging-in-deep-learning-2/)

**Introduction to debugging neural networks**

- blog: [http://russellsstewart.com/notes/0.html](http://russellsstewart.com/notes/0.html)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/4du7gv/introduction_to_debugging_neural_networks](https://www.reddit.com/r/MachineLearning/comments/4du7gv/introduction_to_debugging_neural_networks)

**How to Visualize, Monitor and Debug Neural Network Learning**

- blog: [http://deeplearning4j.org/visualization](http://deeplearning4j.org/visualization)

**Learning from learning curves**

- intro: Kaggle
- blog: [https://medium.com/@dsouza.amanda/learning-from-learning-curves-1a82c6f98f49#.o5synrvvl](https://medium.com/@dsouza.amanda/learning-from-learning-curves-1a82c6f98f49#.o5synrvvl)

# Understanding CNN

**Understanding the Effective Receptive Field in Deep Convolutional Neural Networks**

- intro: NIPS 2016
- paper: [http://www.cs.toronto.edu/~wenjie/papers/nips16/top.pdf](http://www.cs.toronto.edu/~wenjie/papers/nips16/top.pdf)

# Deep Learning Networks

**PCANet: A Simple Deep Learning Baseline for Image Classification?**

- arixv: [http://arxiv.org/abs/1404.3606](http://arxiv.org/abs/1404.3606)
- code(Matlab): [http://mx.nthu.edu.tw/~tsunghan/download/PCANet_demo_pyramid.rar](http://mx.nthu.edu.tw/~tsunghan/download/PCANet_demo_pyramid.rar)
- mirror: [http://pan.baidu.com/s/1mg24b3a](http://pan.baidu.com/s/1mg24b3a)
- github(C++): [https://github.com/Ldpe2G/PCANet](https://github.com/Ldpe2G/PCANet)
- github(Python): [https://github.com/IshitaTakeshi/PCANet](https://github.com/IshitaTakeshi/PCANet)

**Convolutional Kernel Networks**

- intro: NIPS 2014
- arxiv: [http://arxiv.org/abs/1406.3332](http://arxiv.org/abs/1406.3332)

**Deeply-supervised Nets**

- intro: DSN
- arxiv: [http://arxiv.org/abs/1409.5185](http://arxiv.org/abs/1409.5185)
- homepage: [http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/](http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/)
- github: [https://github.com/s9xie/DSN](https://github.com/s9xie/DSN)
- notes: [http://zhangliliang.com/2014/11/02/paper-note-dsn/](http://zhangliliang.com/2014/11/02/paper-note-dsn/)

**FitNets: Hints for Thin Deep Nets**

- arxiv: [https://arxiv.org/abs/1412.6550](https://arxiv.org/abs/1412.6550)
- github: [https://github.com/adri-romsor/FitNets](https://github.com/adri-romsor/FitNets)

**Striving for Simplicity: The All Convolutional Net**

- intro: ICLR-2015 workshop
- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

**How these researchers tried something unconventional to come out with a smaller yet better Image Recognition.**

- intro: All Convolutional Network: (https://arxiv.org/abs/1412.6806#) implementation in Keras
- blog: [https://medium.com/@matelabs_ai/how-these-researchers-tried-something-unconventional-to-came-out-with-a-smaller-yet-better-image-544327f30e72#.pfdbvdmuh](https://medium.com/@matelabs_ai/how-these-researchers-tried-something-unconventional-to-came-out-with-a-smaller-yet-better-image-544327f30e72#.pfdbvdmuh)
- blog: [https://github.com/MateLabs/All-Conv-Keras](https://github.com/MateLabs/All-Conv-Keras)

**Pointer Networks**

- arxiv: [https://arxiv.org/abs/1506.03134](https://arxiv.org/abs/1506.03134)
- github: [https://github.com/vshallc/PtrNets](https://github.com/vshallc/PtrNets)
- github(TensorFlow): [https://github.com/ikostrikov/TensorFlow-Pointer-Networks](https://github.com/ikostrikov/TensorFlow-Pointer-Networks)
- github(TensorFlow): [https://github.com/devsisters/pointer-network-tensorflow](https://github.com/devsisters/pointer-network-tensorflow)
- notes: [https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/pointer-networks.md](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/pointer-networks.md)

**Pointer Networks in TensorFlow (with sample code)**

- blog: [https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264#.sxipqfj30](https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264#.sxipqfj30)
- github: [https://github.com/devnag/tensorflow-pointer-networks](https://github.com/devnag/tensorflow-pointer-networks)

**Rectified Factor Networks**

- arxiv: [http://arxiv.org/abs/1502.06464](http://arxiv.org/abs/1502.06464)
- github: [https://github.com/untom/librfn](https://github.com/untom/librfn)

**Correlational Neural Networks**

- arxiv: [http://arxiv.org/abs/1504.07225](http://arxiv.org/abs/1504.07225)
- github: [https://github.com/apsarath/CorrNet](https://github.com/apsarath/CorrNet)

**Diversity Networks**

- arxiv: [http://arxiv.org/abs/1511.05077](http://arxiv.org/abs/1511.05077)

**Competitive Multi-scale Convolution**

- arxiv: [http://arxiv.org/abs/1511.05635](http://arxiv.org/abs/1511.05635)
- blog: [https://zhuanlan.zhihu.com/p/22377389](https://zhuanlan.zhihu.com/p/22377389)

**A Unified Approach for Learning the Parameters of Sum-Product Networks (SPN)**

- intro: "The Sum-Product Network (SPN) is a new type of machine learning model 
with fast exact probabilistic inference over many layers."
- arxiv: [http://arxiv.org/abs/1601.00318](http://arxiv.org/abs/1601.00318)
- homepage: [http://spn.cs.washington.edu/index.shtml](http://spn.cs.washington.edu/index.shtml)
- code: [http://spn.cs.washington.edu/code.shtml](http://spn.cs.washington.edu/code.shtml)

**Awesome Sum-Product Networks**

- github: [https://github.com/arranger1044/awesome-spn](https://github.com/arranger1044/awesome-spn)

**Recombinator Networks: Learning Coarse-to-Fine Feature Aggregation**

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1511.07356](http://arxiv.org/abs/1511.07356)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Honari_Recombinator_Networks_Learning_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Honari_Recombinator_Networks_Learning_CVPR_2016_paper.pdf)
- github: [https://github.com/SinaHonari/RCN](https://github.com/SinaHonari/RCN)

**Dynamic Capacity Networks**

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1511.07838](http://arxiv.org/abs/1511.07838)
- github(Tensorflow): [https://github.com/beopst/dcn.tf](https://github.com/beopst/dcn.tf)
- review: [http://www.erogol.com/1314-2/](http://www.erogol.com/1314-2/)

**Bitwise Neural Networks**

- paper: [http://paris.cs.illinois.edu/pubs/minje-icmlw2015.pdf](http://paris.cs.illinois.edu/pubs/minje-icmlw2015.pdf)
- demo: [http://minjekim.com/demo_bnn.html](http://minjekim.com/demo_bnn.html)

**Learning Discriminative Features via Label Consistent Neural Network**

- arxiv: [http://arxiv.org/abs/1602.01168](http://arxiv.org/abs/1602.01168)

**A Theory of Generative ConvNet**

- project page: [http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html)
- arxiv: [http://arxiv.org/abs/1602.03264](http://arxiv.org/abs/1602.03264)
- code: [http://www.stat.ucla.edu/~ywu/GenerativeConvNet/doc/code.zip](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/doc/code.zip)

**How to Train Deep Variational Autoencoders and Probabilistic Ladder Networks**

- arxiv: [http://arxiv.org/abs/1602.02282](http://arxiv.org/abs/1602.02282)

**Group Equivariant Convolutional Networks (G-CNNs)**

- arxiv: [http://arxiv.org/abs/1602.07576](http://arxiv.org/abs/1602.07576)

**Deep Spiking Networks**

- arxiv: [http://arxiv.org/abs/1602.08323](http://arxiv.org/abs/1602.08323)
- github: [https://github.com/petered/spiking-mlp](https://github.com/petered/spiking-mlp)

**Low-rank passthrough neural networks**

- arxiv: [http://arxiv.org/abs/1603.03116](http://arxiv.org/abs/1603.03116)
- github: [https://github.com/Avmb/lowrank-gru](https://github.com/Avmb/lowrank-gru)

**Single Image 3D Interpreter Network**

- intro: ECCV 2016 (oral)
- arxiv: [https://arxiv.org/abs/1604.08685](https://arxiv.org/abs/1604.08685)

**Deeply-Fused Nets**

- arxiv: [http://arxiv.org/abs/1605.07716](http://arxiv.org/abs/1605.07716)

**SNN: Stacked Neural Networks**

- arxiv: [http://arxiv.org/abs/1605.08512](http://arxiv.org/abs/1605.08512)

**Universal Correspondence Network**

- intro: NIPS 2016 full oral presentation. Stanford University & NEC Laboratories America
- project page: [http://cvgl.stanford.edu/projects/ucn/](http://cvgl.stanford.edu/projects/ucn/)
- arxiv: [https://arxiv.org/abs/1606.03558](https://arxiv.org/abs/1606.03558)

**Progressive Neural Networks**

- intro: Google DeepMind
- arxiv: [https://arxiv.org/abs/1606.04671](https://arxiv.org/abs/1606.04671)
- github: [https://github.com/synpon/prog_nn](https://github.com/synpon/prog_nn)
- github: [https://github.com/yao62995/A3C](https://github.com/yao62995/A3C)

**Holistic SparseCNN: Forging the Trident of Accuracy, Speed, and Size**

- arxiv: [http://arxiv.org/abs/1608.01409](http://arxiv.org/abs/1608.01409)

**Mollifying Networks**

- author: Caglar Gulcehre, Marcin Moczulski, Francesco Visin, Yoshua Bengio
- arxiv: [http://arxiv.org/abs/1608.04980](http://arxiv.org/abs/1608.04980)

**Domain Separation Networks**

- intro: NIPS 2016. Google Brain & Imperial College London & Google Research
- arxiv: [https://arxiv.org/abs/1608.06019](https://arxiv.org/abs/1608.06019)
- github: [https://github.com/tensorflow/models/tree/master/domain_adaptation](https://github.com/tensorflow/models/tree/master/domain_adaptation)

**Local Binary Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.06049](http://arxiv.org/abs/1608.06049)

**CliqueCNN: Deep Unsupervised Exemplar Learning**

- intro: NIPS 2016
- arxiv: [http://arxiv.org/abs/1608.08792](http://arxiv.org/abs/1608.08792)
- github: [https://github.com/asanakoy/cliquecnn](https://github.com/asanakoy/cliquecnn)

**Convexified Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1609.01000](http://arxiv.org/abs/1609.01000)

**Multi-scale brain networks**

- arxiv: [http://arxiv.org/abs/1608.08828](http://arxiv.org/abs/1608.08828)

[https://arxiv.org/abs/1711.11473](https://arxiv.org/abs/1711.11473)

**Input Convex Neural Networks**

- arxiv: [http://arxiv.org/abs/1609.07152](http://arxiv.org/abs/1609.07152)
- github: [https://github.com/locuslab/icnn](https://github.com/locuslab/icnn)

**HyperNetworks**

- arxiv: [https://arxiv.org/abs/1609.09106](https://arxiv.org/abs/1609.09106)
- blog: [http://blog.otoro.net/2016/09/28/hyper-networks/](http://blog.otoro.net/2016/09/28/hyper-networks/)
- github: [https://github.com/hardmaru/supercell/blob/master/assets/MNIST_Static_HyperNetwork_Example.ipynb](https://github.com/hardmaru/supercell/blob/master/assets/MNIST_Static_HyperNetwork_Example.ipynb)

**HyperLSTM**

- github: [https://github.com/hardmaru/supercell/blob/master/supercell.py](https://github.com/hardmaru/supercell/blob/master/supercell.py)

**X-CNN: Cross-modal Convolutional Neural Networks for Sparse Datasets**

- arxiv: [https://arxiv.org/abs/1610.00163](https://arxiv.org/abs/1610.00163)

**Tensor Switching Networks**

- intro: NIPS 2016
- arixiv: [https://arxiv.org/abs/1610.10087](https://arxiv.org/abs/1610.10087)
- github: [https://github.com/coxlab/tsnet](https://github.com/coxlab/tsnet)

**BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks**

- intro: Harvard University
- paper: [http://www.eecs.harvard.edu/~htk/publication/2016-icpr-teerapittayanon-mcdanel-kung.pdf](http://www.eecs.harvard.edu/~htk/publication/2016-icpr-teerapittayanon-mcdanel-kung.pdf)
- github: [https://github.com/kunglab/branchynet](https://github.com/kunglab/branchynet)

**Spectral Convolution Networks**

- arxiv: [https://arxiv.org/abs/1611.05378](https://arxiv.org/abs/1611.05378)

**DelugeNets: Deep Networks with Massive and Flexible Cross-layer Information Inflows**

- arxiv: [https://arxiv.org/abs/1611.05552](https://arxiv.org/abs/1611.05552)
- github: [https://github.com/xternalz/DelugeNets](https://github.com/xternalz/DelugeNets)

**PolyNet: A Pursuit of Structural Diversity in Very Deep Networks**

- arxiv: [https://arxiv.org/abs/1611.05725](https://arxiv.org/abs/1611.05725)
- poster: [http://mmlab.ie.cuhk.edu.hk/projects/cu_deeplink/polynet_poster.pdf](http://mmlab.ie.cuhk.edu.hk/projects/cu_deeplink/polynet_poster.pdf)

**Weakly Supervised Cascaded Convolutional Networks**

- arxiv: [https://arxiv.org/abs/1611.08258](https://arxiv.org/abs/1611.08258)

**DeepSetNet: Predicting Sets with Deep Neural Networks**

- intro: multi-class image classification and pedestrian detection
- arxiv: [https://arxiv.org/abs/1611.08998](https://arxiv.org/abs/1611.08998)

**Steerable CNNs**

- intro: University of Amsterdam
- arxiv: [https://arxiv.org/abs/1612.08498](https://arxiv.org/abs/1612.08498)

**Feedback Networks**

- project page: [http://feedbacknet.stanford.edu/](http://feedbacknet.stanford.edu/)
- arxiv: [https://arxiv.org/abs/1612.09508](https://arxiv.org/abs/1612.09508)
- youtube: [https://youtu.be/MY5Uhv38Ttg](https://youtu.be/MY5Uhv38Ttg)

**Oriented Response Networks**

- arxiv: [https://arxiv.org/abs/1701.01833](https://arxiv.org/abs/1701.01833)

**OptNet: Differentiable Optimization as a Layer in Neural Networks**

- arxiv: [https://arxiv.org/abs/1703.00443](https://arxiv.org/abs/1703.00443)
- github: [https://github.com/locuslab/optnet](https://github.com/locuslab/optnet)

**A fast and differentiable QP solver for PyTorch**

- github: [https://github.com/locuslab/qpth](https://github.com/locuslab/qpth)

**Meta Networks**

[https://arxiv.org/abs/1703.00837](https://arxiv.org/abs/1703.00837)

**Deformable Convolutional Networks**

- intro: ICCV 2017 oral. Microsoft Research Asia
- keywords: deformable convolution, deformable RoI pooling
- arxiv: [https://arxiv.org/abs/1703.06211](https://arxiv.org/abs/1703.06211)
- sliedes: [http://www.jifengdai.org/slides/Deformable_Convolutional_Networks_Oral.pdf](http://www.jifengdai.org/slides/Deformable_Convolutional_Networks_Oral.pdf)
- github(official): [https://github.com/msracver/Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets)
- github: [https://github.com/felixlaumon/deform-conv](https://github.com/felixlaumon/deform-conv)
- github: [https://github.com/oeway/pytorch-deform-conv](https://github.com/oeway/pytorch-deform-conv)

Deformable ConvNets v2: More Deformable, Better Results**

- intro: University of Science and Technology of China & Microsoft Research Asia
- keywords: DCNv2
- arxiv: [https://arxiv.org/abs/1811.11168](https://arxiv.org/abs/1811.11168)
- github: [https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op](https://github.com/msracver/Deformable-ConvNets/tree/master/DCNv2_op)

**Second-order Convolutional Neural Networks**

[https://arxiv.org/abs/1703.06817](https://arxiv.org/abs/1703.06817)

**Gabor Convolutional Networks**

[https://arxiv.org/abs/1705.01450](https://arxiv.org/abs/1705.01450)

**Deep Rotation Equivariant Network**

[https://arxiv.org/abs/1705.08623](https://arxiv.org/abs/1705.08623)

**Dense Transformer Networks**

- intro: Washington State University & University of California, Davis
- arxiv: [https://arxiv.org/abs/1705.08881](https://arxiv.org/abs/1705.08881)
- github: [https://github.com/divelab/dtn](https://github.com/divelab/dtn)

**Deep Complex Networks**

- intro: [Université de Montréal & INRS-EMT & Microsoft Maluuba
- arxiv: [https://arxiv.org/abs/1705.09792](https://arxiv.org/abs/1705.09792)
- github: [https://github.com/ChihebTrabelsi/deep_complex_networks](https://github.com/ChihebTrabelsi/deep_complex_networks)

**Deep Quaternion Networks**

- intro: University of Louisiana
- arxiv: [https://arxiv.org/abs/1712.04604](https://arxiv.org/abs/1712.04604)

**DiracNets: Training Very Deep Neural Networks Without Skip-Connections**

- intro: Université Paris-Est
- arxiv: [https://arxiv.org/abs/1706.00388](https://arxiv.org/abs/1706.00388)
- github: [https://github.com/szagoruyko/diracnets](https://github.com/szagoruyko/diracnets)

**Dual Path Networks**

- intro: National University of Singapore
- arxiv: [https://arxiv.org/abs/1707.01629](https://arxiv.org/abs/1707.01629)
- github(MXNet): [https://github.com/cypw/DPNs](https://github.com/cypw/DPNs)

**Primal-Dual Group Convolutions for Deep Neural Networks**

**Interleaved Group Convolutions for Deep Neural Networks**

- intro: ICCV 2017
- keywords: interleaved group convolutional neural networks (IGCNets), IGCV1
- arxiv: [https://arxiv.org/abs/1707.02725](https://arxiv.org/abs/1707.02725)
- gihtub: [https://github.com/hellozting/InterleavedGroupConvolutions](https://github.com/hellozting/InterleavedGroupConvolutions)

**IGCV2: Interleaved Structured Sparse Convolutional Neural Networks**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1804.06202](https://arxiv.org/abs/1804.06202)

**IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks**

- intro: University of Scinence and Technology of China & Microsoft Reserach Asia
- arxiv: [https://arxiv.org/abs/1806.00178](https://arxiv.org/abs/1806.00178)
- github(official): [https://github.com/homles11/IGCV3](https://github.com/homles11/IGCV3)

**Sensor Transformation Attention Networks**

[https://arxiv.org/abs/1708.01015](https://arxiv.org/abs/1708.01015)

**Sparsity Invariant CNNs**

[https://arxiv.org/abs/1708.06500](https://arxiv.org/abs/1708.06500)

**SPARCNN: SPAtially Related Convolutional Neural Networks**

[https://arxiv.org/abs/1708.07522](https://arxiv.org/abs/1708.07522)

**BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks**

[https://arxiv.org/abs/1709.01686](https://arxiv.org/abs/1709.01686)

**Polar Transformer Networks**

[https://arxiv.org/abs/1709.01889](https://arxiv.org/abs/1709.01889)

**Tensor Product Generation Networks**

[https://arxiv.org/abs/1709.09118](https://arxiv.org/abs/1709.09118)

**Deep Competitive Pathway Networks**

- intro: ACML 2017
- arxiv: [https://arxiv.org/abs/1709.10282](https://arxiv.org/abs/1709.10282)
- github: [https://github.com/JiaRenChang/CoPaNet](https://github.com/JiaRenChang/CoPaNet)

**Context Embedding Networks**

[https://arxiv.org/abs/1710.01691](https://arxiv.org/abs/1710.01691)

**Generalization in Deep Learning**

- intro: MIT & University of Montreal
- arxiv: [https://arxiv.org/abs/1710.05468](https://arxiv.org/abs/1710.05468)

**Understanding Deep Learning Generalization by Maximum Entropy**

- intro: University of Science and Technology of China & Beijing Jiaotong University & Chinese Academy of Sciences
- arxiv: [https://arxiv.org/abs/1711.07758](https://arxiv.org/abs/1711.07758)

**Do Convolutional Neural Networks Learn Class Hierarchy?**

- intro: Bosch Research North America & Michigan State University
- arxiv: [https://arxiv.org/abs/1710.06501](https://arxiv.org/abs/1710.06501)
- video demo: [https://vimeo.com/228263798](https://vimeo.com/228263798)

**Deep Hyperspherical Learning**

- intro: NIPS 2017
- arxiv: [https://arxiv.org/abs/1711.03189](https://arxiv.org/abs/1711.03189)

**Beyond Sparsity: Tree Regularization of Deep Models for Interpretability**

- intro: AAAI 2018
- arxiv: [https://arxiv.org/abs/1711.06178](https://arxiv.org/abs/1711.06178)

**Neural Motifs: Scene Graph Parsing with Global Context**

- keywords: Stacked Motif Networks
- arxiv: [https://arxiv.org/abs/1711.06640](https://arxiv.org/abs/1711.06640)

**Priming Neural Networks**

[https://arxiv.org/abs/1711.05918](https://arxiv.org/abs/1711.05918)

**Three Factors Influencing Minima in SGD**

[https://arxiv.org/abs/1711.04623](https://arxiv.org/abs/1711.04623)

**BPGrad: Towards Global Optimality in Deep Learning via Branch and Pruning**

[https://arxiv.org/abs/1711.06959](https://arxiv.org/abs/1711.06959)

**BlockDrop: Dynamic Inference Paths in Residual Networks**

- intro: UMD & UT Austin & IBM Research & Fusemachines Inc.
- arxiv: [https://arxiv.org/abs/1711.08393](https://arxiv.org/abs/1711.08393)

**Wasserstein Introspective Neural Networks**

[https://arxiv.org/abs/1711.08875](https://arxiv.org/abs/1711.08875)

**SkipNet: Learning Dynamic Routing in Convolutional Networks**

[https://arxiv.org/abs/1711.09485](https://arxiv.org/abs/1711.09485)

**Do Convolutional Neural Networks act as Compositional Nearest Neighbors?**

- intro: CMU & West Virginia University
- arxiv: [https://arxiv.org/abs/1711.10683](https://arxiv.org/abs/1711.10683)

**ConvNets and ImageNet Beyond Accuracy: Explanations, Bias Detection, Adversarial Examples and Model Criticism**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/1711.11443](https://arxiv.org/abs/1711.11443)

**Broadcasting Convolutional Network**

[https://arxiv.org/abs/1712.02517](https://arxiv.org/abs/1712.02517)

**Point-wise Convolutional Neural Network**

- intro: Singapore University of Technology and Design
- arxiv: [https://arxiv.org/abs/1712.05245](https://arxiv.org/abs/1712.05245)

**ScreenerNet: Learning Curriculum for Neural Networks**

- intro: Intel Corporation & Allen Institute for Artificial Intelligence
- keywords: curricular learning, deep learning, deep q-learning
- arxiv: [https://arxiv.org/abs/1801.00904](https://arxiv.org/abs/1801.00904)

**Sparsely Connected Convolutional Networks**

[https://arxiv.org/abs/1801.05895](https://arxiv.org/abs/1801.05895)

**Spherical CNNs**

- intro: ICLR 2018 best paper award. University of Amsterdam & EPFL
- arxiv: [https://arxiv.org/abs/1801.10130](https://arxiv.org/abs/1801.10130)
- github(official, PyTorch): [https://github.com/jonas-koehler/s2cnn](https://github.com/jonas-koehler/s2cnn)

**Going Deeper in Spiking Neural Networks: VGG and Residual Architectures**

- intro: Purdue University & Oculus Research & Facebook Research
- arxiv: [https://arxiv.org/abs/1802.02627](https://arxiv.org/abs/1802.02627)

**Rotate your Networks: Better Weight Consolidation and Less Catastrophic Forgetting**

[https://arxiv.org/abs/1802.02950](https://arxiv.org/abs/1802.02950)

**Convolutional Neural Networks with Alternately Updated Clique**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1802.10419](https://arxiv.org/abs/1802.10419)
- github: [https://github.com/iboing/CliqueNet](https://github.com/iboing/CliqueNet)

**Decoupled Networks**

- intro: CVPR 2018 (Spotlight)
- arxiv: [https://arxiv.org/abs/1804.08071](https://arxiv.org/abs/1804.08071)

**Optical Neural Networks**

[https://arxiv.org/abs/1805.06082](https://arxiv.org/abs/1805.06082)

**Regularization Learning Networks**

- intro: Weizmann Institute of Science
- keywords: Regularization Learning Networks (RLNs), Counterfactual Loss, tabular datasets
- arxiv: [https://arxiv.org/abs/1805.06440](https://arxiv.org/abs/1805.06440)

**Bilinear Attention Networks**

[https://arxiv.org/abs/1805.07932](https://arxiv.org/abs/1805.07932)

**Cautious Deep Learning**

[https://arxiv.org/abs/1805.09460](https://arxiv.org/abs/1805.09460)

**Perturbative Neural Networks**

- intro: CVPR 2018
- intro: We introduce a very simple, yet effective, module called a perturbation layer as an alternative to a convolutional layer
- project page: [http://xujuefei.com/pnn.html](http://xujuefei.com/pnn.html)
- arxiv: [https://arxiv.org/abs/1806.01817](https://arxiv.org/abs/1806.01817)

**Lightweight Probabilistic Deep Networks**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1805.11327](https://arxiv.org/abs/1805.11327)

**Channel Gating Neural Networks**

[https://arxiv.org/abs/1805.12549](https://arxiv.org/abs/1805.12549)

**Evenly Cascaded Convolutional Networks**

[https://arxiv.org/abs/1807.00456](https://arxiv.org/abs/1807.00456)

**SGAD: Soft-Guided Adaptively-Dropped Neural Network**

[https://arxiv.org/abs/1807.01430](https://arxiv.org/abs/1807.01430)

**Explainable Neural Computation via Stack Neural Module Networks**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1807.08556](https://arxiv.org/abs/1807.08556)

**Rank-1 Convolutional Neural Network**

[https://arxiv.org/abs/1808.04303](https://arxiv.org/abs/1808.04303)

**Neural Network Encapsulation**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1808.03749](https://arxiv.org/abs/1808.03749)

**Penetrating the Fog: the Path to Efficient CNN Models**

[https://arxiv.org/abs/1810.04231](https://arxiv.org/abs/1810.04231)

**A2-Nets: Double Attention Networks**

- intro: NIPS 2018
- arxiv: [https://arxiv.org/abs/1810.11579](https://arxiv.org/abs/1810.11579)
**
Global Second-order Pooling Neural Networks**

[https://arxiv.org/abs/1811.12006](https://arxiv.org/abs/1811.12006)

**ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network**

- intro: University of Washington & Allen Institute for AI (AI2) & XNOR.AI
- arxiv: [https://arxiv.org/abs/1811.11431](https://arxiv.org/abs/1811.11431)
- github: [https://github.com/sacmehta/ESPNetv2](https://github.com/sacmehta/ESPNetv2)

**Kernel Transformer Networks for Compact Spherical Convolution**

[https://arxiv.org/abs/1812.03115](https://arxiv.org/abs/1812.03115)

## Convolutions / Filters

**Warped Convolutions: Efficient Invariance to Spatial Transformations**

- arxiv: [http://arxiv.org/abs/1609.04382](http://arxiv.org/abs/1609.04382)

**Coordinating Filters for Faster Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1703.09746](https://arxiv.org/abs/1703.09746)
- github: [https://github.com/wenwei202/caffe/tree/sfm](https://github.com/wenwei202/caffe/tree/sfm)

**Shift: A Zero FLOP, Zero Parameter Alternative to Spatial Convolutions**

- intro: UC Berkeley
- arxiv: [https://arxiv.org/abs/1711.08141](https://arxiv.org/abs/1711.08141)

**Spatially-Adaptive Filter Units for Deep Neural Networks**

- intro: University of Ljubljana & University of Birmingham
- arxiv: [https://arxiv.org/abs/1711.11473](https://arxiv.org/abs/1711.11473)

**clcNet: Improving the Efficiency of Convolutional Neural Network using Channel Local Convolutions**

[https://arxiv.org/abs/1712.06145](https://arxiv.org/abs/1712.06145)

**DCFNet: Deep Neural Network with Decomposed Convolutional Filters**

[https://arxiv.org/abs/1802.04145](https://arxiv.org/abs/1802.04145)

**Fast End-to-End Trainable Guided Filter**

- intro: CVPR 2018
- project page: [http://wuhuikai.me/DeepGuidedFilterProject/](http://wuhuikai.me/DeepGuidedFilterProject/)
- gtihub(official, PyTorch): [https://github.com/wuhuikai/DeepGuidedFilter](https://github.com/wuhuikai/DeepGuidedFilter)

**Diagonalwise Refactorization: An Efficient Training Method for Depthwise Convolutions**

- arxiv: [https://arxiv.org/abs/1803.09926](https://arxiv.org/abs/1803.09926)
- github: [https://github.com/clavichord93/diagonalwise-refactorization-tensorflow](https://github.com/clavichord93/diagonalwise-refactorization-tensorflow)

**Use of symmetric kernels for convolutional neural networks**

- intro: ICDSIAI 2018
- arxiv: [https://arxiv.org/abs/1805.09421](https://arxiv.org/abs/1805.09421)

**EasyConvPooling: Random Pooling with Easy Convolution for Accelerating Training and Testing**

[https://arxiv.org/abs/1806.01729](https://arxiv.org/abs/1806.01729)

**Targeted Kernel Networks: Faster Convolutions with Attentive Regularization**

[https://arxiv.org/abs/1806.00523](https://arxiv.org/abs/1806.00523)

**An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution**

- intro: 1Uber AI Labs & Uber Technologies
- arxiv: [https://arxiv.org/abs/1807.03247](https://arxiv.org/abs/1807.03247)
- github: [https://github.com/mkocabas/CoordConv-pytorch](https://github.com/mkocabas/CoordConv-pytorch)
- youtube: [https://www.youtube.com/watch?v=8yFQc6elePA](https://www.youtube.com/watch?v=8yFQc6elePA)

**Network Decoupling: From Regular to Depthwise Separable Convolutions**

[https://arxiv.org/abs/1808.05517](https://arxiv.org/abs/1808.05517)

**Partial Convolution based Padding**

- intro: NVIDIA Corporation
- arxiv; [https://arxiv.org/abs/1811.11718](https://arxiv.org/abs/1811.11718)
- github: [https://github.com/NVIDIA/partialconv](https://github.com/NVIDIA/partialconv)

**DSConv: Efficient Convolution Operator**

[https://arxiv.org/abs/1901.01928](https://arxiv.org/abs/1901.01928)

## Highway Networks

**Highway Networks**

- intro: ICML 2015 Deep Learning workshop
- intro: shortcut connections with gating functions. These gates are data-dependent and have parameters
- arxiv: [http://arxiv.org/abs/1505.00387](http://arxiv.org/abs/1505.00387)
- github(PyTorch): [https://github.com/analvikingur/pytorch_Highway](https://github.com/analvikingur/pytorch_Highway)

**Highway Networks with TensorFlow**

- blog: [https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.71fgztsb6](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.71fgztsb6)

**Very Deep Learning with Highway Networks**

- homepage(papers+code+FAQ): [http://people.idsia.ch/~rupesh/very_deep_learning/](http://people.idsia.ch/~rupesh/very_deep_learning/)

**Training Very Deep Networks**

- intro: Extends [Highway Networks](https://arxiv.org/abs/1505.00387)
- project page: [http://people.idsia.ch/~rupesh/very_deep_learning/](http://people.idsia.ch/~rupesh/very_deep_learning/)
- arxiv: [http://arxiv.org/abs/1507.06228](http://arxiv.org/abs/1507.06228)

## Spatial Transformer Networks

**Spatial Transformer Networks**

![](https://camo.githubusercontent.com/bb81d6267f2123d59979453526d958a58899bb4f/687474703a2f2f692e696d6775722e636f6d2f4578474456756c2e706e67)

- intro: NIPS 2015
- arxiv: [http://arxiv.org/abs/1506.02025](http://arxiv.org/abs/1506.02025)
- gitxiv: [http://gitxiv.com/posts/5WTXTLuEA4Hd8W84G/spatial-transformer-networks](http://gitxiv.com/posts/5WTXTLuEA4Hd8W84G/spatial-transformer-networks)
- github: [https://github.com/daerduoCarey/SpatialTransformerLayer](https://github.com/daerduoCarey/SpatialTransformerLayer)
- github: [https://github.com/qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd)
- github: [https://github.com/skaae/transformer_network](https://github.com/skaae/transformer_network)
- github(Caffe): [https://github.com/happynear/SpatialTransformerLayer](https://github.com/happynear/SpatialTransformerLayer)
- github: [https://github.com/daviddao/spatial-transformer-tensorflow](https://github.com/daviddao/spatial-transformer-tensorflow)
- caffe-issue: [https://github.com/BVLC/caffe/issues/3114](https://github.com/BVLC/caffe/issues/3114)
- code: [https://lasagne.readthedocs.org/en/latest/modules/layers/special.html#lasagne.layers.TransformerLayer](https://lasagne.readthedocs.org/en/latest/modules/layers/special.html#lasagne.layers.TransformerLayer)
- ipn(Lasagne): [http://nbviewer.jupyter.org/github/Lasagne/Recipes/blob/master/examples/spatial_transformer_network.ipynb](http://nbviewer.jupyter.org/github/Lasagne/Recipes/blob/master/examples/spatial_transformer_network.ipynb)
- notes: [https://www.evernote.com/shard/s189/sh/ad8a38de-9e98-4e06-b09e-574bd62893ff/32f72798c095dd7672f4cb017a32d9b4](https://www.evernote.com/shard/s189/sh/ad8a38de-9e98-4e06-b09e-574bd62893ff/32f72798c095dd7672f4cb017a32d9b4)
- youtube: [https://www.youtube.com/watch?v=6NOQC_fl1hQ](https://www.youtube.com/watch?v=6NOQC_fl1hQ)

**The power of Spatial Transformer Networks**

- blog: [http://torch.ch/blog/2015/09/07/spatial_transformers.html](http://torch.ch/blog/2015/09/07/spatial_transformers.html)
- github: [https://github.com/moodstocks/gtsrb.torch](https://github.com/moodstocks/gtsrb.torch)

**Recurrent Spatial Transformer Networks**

- paper: [http://arxiv.org/abs/1509.05329](http://arxiv.org/abs/1509.05329)

**Deep Learning Paper Implementations: Spatial Transformer Networks - Part I**

- blog: [https://kevinzakka.github.io/2017/01/10/stn-part1/](https://kevinzakka.github.io/2017/01/10/stn-part1/)
- github: [https://github.com/kevinzakka/blog-code/tree/master/spatial_transformer](https://github.com/kevinzakka/blog-code/tree/master/spatial_transformer)

**Top-down Flow Transformer Networks**

[https://arxiv.org/abs/1712.02400](https://arxiv.org/abs/1712.02400)

**Non-Parametric Transformation Networks**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1801.04520](https://arxiv.org/abs/1801.04520)

**Hierarchical Spatial Transformer Network**

[https://arxiv.org/abs/1801.09467](https://arxiv.org/abs/1801.09467)

**Spatial Transformer Introspective Neural Network**

- intro: Johns Hopkins University & Shanghai University
- arxiv: [https://arxiv.org/abs/1805.06447](https://arxiv.org/abs/1805.06447)

**DeSTNet: Densely Fused Spatial Transformer Networks**

- intro: BMVC 2018
- arxiv: [https://arxiv.org/abs/1807.04050](https://arxiv.org/abs/1807.04050)

**MIST: Multiple Instance Spatial Transformer Network**

[https://arxiv.org/abs/1811.10725](https://arxiv.org/abs/1811.10725)

## FractalNet

**FractalNet: Ultra-Deep Neural Networks without Residuals**

![](http://people.cs.uchicago.edu/~larsson/fractalnet/overview.png)

- project: [http://people.cs.uchicago.edu/~larsson/fractalnet/](http://people.cs.uchicago.edu/~larsson/fractalnet/)
- arxiv: [http://arxiv.org/abs/1605.07648](http://arxiv.org/abs/1605.07648)
- github: [https://github.com/gustavla/fractalnet](https://github.com/gustavla/fractalnet)
- github: [https://github.com/edgelord/FractalNet](https://github.com/edgelord/FractalNet)
- github(Keras): [https://github.com/snf/keras-fractalnet](https://github.com/snf/keras-fractalnet)

## Graph Convolutional Networks

**Learning Convolutional Neural Networks for Graphs**

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1605.05273](http://arxiv.org/abs/1605.05273)

**Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering**

- arxiv: [https://arxiv.org/abs/1606.09375](https://arxiv.org/abs/1606.09375)
- github: [https://github.com/mdeff/cnn_graph](https://github.com/mdeff/cnn_graph)
- github: [https://github.com/pfnet-research/chainer-graph-cnn](https://github.com/pfnet-research/chainer-graph-cnn)

**Semi-Supervised Classification with Graph Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1609.02907](http://arxiv.org/abs/1609.02907)
- github: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)
- blog: [http://tkipf.github.io/graph-convolutional-networks/](http://tkipf.github.io/graph-convolutional-networks/)

**Graph Based Convolutional Neural Network**

- intro: BMVC 2016
- arxiv: [http://arxiv.org/abs/1609.08965](http://arxiv.org/abs/1609.08965)

**How powerful are Graph Convolutions? (review of Kipf & Welling, 2016)**

[http://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2/](http://www.inference.vc/how-powerful-are-graph-convolutions-review-of-kipf-welling-2016-2/)

**Graph Convolutional Networks**

![](http://tkipf.github.io/graph-convolutional-networks/images/gcn_web.png)

- blog: [http://tkipf.github.io/graph-convolutional-networks/](http://tkipf.github.io/graph-convolutional-networks/)

**DeepGraph: Graph Structure Predicts Network Growth**

- arxiv: [https://arxiv.org/abs/1610.06251](https://arxiv.org/abs/1610.06251)

**Deep Learning with Sets and Point Clouds**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1611.04500](https://arxiv.org/abs/1611.04500)

**Deep Learning on Graphs**

- lecture: [https://figshare.com/articles/Deep_Learning_on_Graphs/4491686](https://figshare.com/articles/Deep_Learning_on_Graphs/4491686)

**Robust Spatial Filtering with Graph Convolutional Neural Networks**

[https://arxiv.org/abs/1703.00792](https://arxiv.org/abs/1703.00792)

**Modeling Relational Data with Graph Convolutional Networks**

[https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)

**Distance Metric Learning using Graph Convolutional Networks: Application to Functional Brain Networks**

- intro: Imperial College London
- arxiv: [https://arxiv.org/abs/1703.02161](https://arxiv.org/abs/1703.02161)

**Deep Learning on Graphs with Graph Convolutional Networks**

- slides: [http://tkipf.github.io/misc/GCNSlides.pdf](http://tkipf.github.io/misc/GCNSlides.pdf)

**Deep Learning on Graphs with Keras**

- intro:; Keras implementation of Graph Convolutional Networks
- github: [https://github.com/tkipf/keras-gcn](https://github.com/tkipf/keras-gcn)

**Learning Graph While Training: An Evolving Graph Convolutional Neural Network**

[https://arxiv.org/abs/1708.04675](https://arxiv.org/abs/1708.04675)

**Graph Attention Networks**

- intro: ICLR 2018
- intro: University of Cambridge & Centre de Visio per Computador, UAB & Montreal Institute for Learning Algorithms
- project page: [http://petar-v.com/GAT/](http://petar-v.com/GAT/)
- arxiv: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- github: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT)

**Residual Gated Graph ConvNets**

[https://arxiv.org/abs/1711.07553](https://arxiv.org/abs/1711.07553)

**Probabilistic and Regularized Graph Convolutional Networks**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1803.04489](https://arxiv.org/abs/1803.04489)

**Videos as Space-Time Region Graphs**

[https://arxiv.org/abs/1806.01810](https://arxiv.org/abs/1806.01810)

**Relational inductive biases, deep learning, and graph networks**

- intro: DeepMind & Google Brain & MIT & University of Edinburgh
- arxiv: [https://arxiv.org/abs/1806.01261](https://arxiv.org/abs/1806.01261)

# Generative Models

**Max-margin Deep Generative Models**

- intro: NIPS 2015
- arxiv: [http://arxiv.org/abs/1504.06787](http://arxiv.org/abs/1504.06787)
- github: [https://github.com/zhenxuan00/mmdgm](https://github.com/zhenxuan00/mmdgm)

**Discriminative Regularization for Generative Models**

- arxiv: [http://arxiv.org/abs/1602.03220](http://arxiv.org/abs/1602.03220)
- github: [https://github.com/vdumoulin/discgen](https://github.com/vdumoulin/discgen)

**Auxiliary Deep Generative Models**

- arxiv: [http://arxiv.org/abs/1602.05473](http://arxiv.org/abs/1602.05473)
- github: [https://github.com/larsmaaloee/auxiliary-deep-generative-models](https://github.com/larsmaaloee/auxiliary-deep-generative-models)

**Sampling Generative Networks: Notes on a Few Effective Techniques**

- arxiv: [http://arxiv.org/abs/1609.04468](http://arxiv.org/abs/1609.04468)
- paper: [https://github.com/dribnet/plat](https://github.com/dribnet/plat)

**Conditional Image Synthesis With Auxiliary Classifier GANs**

- arxiv: [https://arxiv.org/abs/1610.09585](https://arxiv.org/abs/1610.09585)
- github: [https://github.com/buriburisuri/ac-gan](https://github.com/buriburisuri/ac-gan)
- github(Keras): [https://github.com/lukedeo/keras-acgan](https://github.com/lukedeo/keras-acgan)

**On the Quantitative Analysis of Decoder-Based Generative Models**

- intro: University of Toronto & OpenAI & CMU
- arxiv: [https://arxiv.org/abs/1611.04273](https://arxiv.org/abs/1611.04273)
- github: [https://github.com/tonywu95/eval_gen](https://github.com/tonywu95/eval_gen)

**Boosted Generative Models**

- arxiv: [https://arxiv.org/abs/1702.08484](https://arxiv.org/abs/1702.08484)
- paper: [https://openreview.net/pdf?id=HyY4Owjll](https://openreview.net/pdf?id=HyY4Owjll)

**An Architecture for Deep, Hierarchical Generative Models**

- intro: NIPS 2016
- arxiv: [https://arxiv.org/abs/1612.04739](https://arxiv.org/abs/1612.04739)
- github: [https://github.com/Philip-Bachman/MatNets-NIPS](https://github.com/Philip-Bachman/MatNets-NIPS)

**Deep Learning and Hierarchal Generative Models**

- intro: NIPS 2016. MIT
- arxiv: [https://arxiv.org/abs/1612.09057](https://arxiv.org/abs/1612.09057)

**Probabilistic Torch**

- intro: Probabilistic Torch is library for deep generative models that extends PyTorch
- github: [https://github.com/probtorch/probtorch](https://github.com/probtorch/probtorch)

**Tutorial on Deep Generative Models**

- intro: UAI 2017 Tutorial: Shakir Mohamed & Danilo Rezende (DeepMind)
- youtube: [https://www.youtube.com/watch?v=JrO5fSskISY](https://www.youtube.com/watch?v=JrO5fSskISY)
- mirror: [https://www.bilibili.com/video/av16428277/](https://www.bilibili.com/video/av16428277/)
- slides: [http://www.shakirm.com/slides/DeepGenModelsTutorial.pdf](http://www.shakirm.com/slides/DeepGenModelsTutorial.pdf)

**A Note on the Inception Score**

- intro: Stanford University
- arxiv: [https://arxiv.org/abs/1801.01973](https://arxiv.org/abs/1801.01973)

**Gradient Layer: Enhancing the Convergence of Adversarial Training for Generative Models**

- intro: AISTATS 2018. The University of Tokyo
- arxiv: [https://arxiv.org/abs/1801.02227](https://arxiv.org/abs/1801.02227)

**Batch Normalization in the final layer of generative networks**

[https://arxiv.org/abs/1805.07389](https://arxiv.org/abs/1805.07389)

**Deep Structured Generative Models**

- intro: Tsinghua University
- arxiv: [https://arxiv.org/abs/1807.03877](https://arxiv.org/abs/1807.03877)

**VFunc: a Deep Generative Model for Functions**

- intro: ICML 2018 workshop on Prediction and Generative Modeling in Reinforcement Learning. Microsoft Research & McGill University
- arxiv: [https://arxiv.org/abs/1807.04106](https://arxiv.org/abs/1807.04106)

# Deep Learning and Robots

**Robot Learning Manipulation Action Plans by "Watching" Unconstrained Videos from the World Wide Web**

- intro: AAAI 2015
- paper: [http://www.umiacs.umd.edu/~yzyang/paper/YouCookMani_CameraReady.pdf](http://www.umiacs.umd.edu/~yzyang/paper/YouCookMani_CameraReady.pdf)
- author page: [http://www.umiacs.umd.edu/~yzyang/](http://www.umiacs.umd.edu/~yzyang/)

**End-to-End Training of Deep Visuomotor Policies**

- arxiv: [http://arxiv.org/abs/1504.00702](http://arxiv.org/abs/1504.00702)

**Comment on Open AI’s Efforts to Robot Learning**

- blog: [https://gridworld.wordpress.com/2016/07/28/comment-on-open-ais-efforts-to-robot-learning/](https://gridworld.wordpress.com/2016/07/28/comment-on-open-ais-efforts-to-robot-learning/)

**The Curious Robot: Learning Visual Representations via Physical Interactions**

- arxiv: [http://arxiv.org/abs/1604.01360](http://arxiv.org/abs/1604.01360)

**How to build a robot that “sees” with $100 and TensorFlow**

![](https://d3ansictanv2wj.cloudfront.net/Figure_5-5b104cf7a53a9c1ee95110b78fb14256.jpg)

- blog: [https://www.oreilly.com/learning/how-to-build-a-robot-that-sees-with-100-and-tensorflow](https://www.oreilly.com/learning/how-to-build-a-robot-that-sees-with-100-and-tensorflow)

**Deep Visual Foresight for Planning Robot Motion**

- project page: [https://sites.google.com/site/brainrobotdata/](https://sites.google.com/site/brainrobotdata/)
- arxiv: [https://arxiv.org/abs/1610.00696](https://arxiv.org/abs/1610.00696)
- video: [https://sites.google.com/site/robotforesight/](https://sites.google.com/site/robotforesight/)

**Sim-to-Real Robot Learning from Pixels with Progressive Nets**

- intro: Google DeepMind
- arxiv: [https://arxiv.org/abs/1610.04286](https://arxiv.org/abs/1610.04286)

**Towards Lifelong Self-Supervision: A Deep Learning Direction for Robotics**

- arxiv: [https://arxiv.org/abs/1611.00201](https://arxiv.org/abs/1611.00201)

**A Differentiable Physics Engine for Deep Learning in Robotics**

- paper: [http://openreview.net/pdf?id=SyEiHNKxx](http://openreview.net/pdf?id=SyEiHNKxx)

**Deep-learning in Mobile Robotics - from Perception to Control Systems: A Survey on Why and Why not**

- intro: City University of Hong Kong & Hong Kong University of Science and Technology
- arxiv: [https://arxiv.org/abs/1612.07139](https://arxiv.org/abs/1612.07139)

**Deep Robotic Learning**

- intro: [https://simons.berkeley.edu/talks/sergey-levine-01-24-2017-1](https://simons.berkeley.edu/talks/sergey-levine-01-24-2017-1)
- youtube: [https://www.youtube.com/watch?v=jtjW5Pye_44](https://www.youtube.com/watch?v=jtjW5Pye_44)

**Deep Learning in Robotics: A Review of Recent Research**

[https://arxiv.org/abs/1707.07217](https://arxiv.org/abs/1707.07217)

**Deep Learning for Robotics**

- intro: by Pieter Abbeel
- video: [https://www.facebook.com/nipsfoundation/videos/1554594181298482/](https://www.facebook.com/nipsfoundation/videos/1554594181298482/)
- mirror: [https://www.bilibili.com/video/av17078186/](https://www.bilibili.com/video/av17078186/)
- slides: [https://www.dropbox.com/s/4fhczb9cxkuqalf/2017_11_xx_BARS-Abbeel.pdf?dl=0](https://www.dropbox.com/s/4fhczb9cxkuqalf/2017_11_xx_BARS-Abbeel.pdf?dl=0)

**DroNet: Learning to Fly by Driving**

- project page: [http://rpg.ifi.uzh.ch/dronet.html](http://rpg.ifi.uzh.ch/dronet.html)
- paper: [http://rpg.ifi.uzh.ch/docs/RAL18_Loquercio.pdf](http://rpg.ifi.uzh.ch/docs/RAL18_Loquercio.pdf)
- github: [https://github.com/uzh-rpg/rpg_public_dronet](https://github.com/uzh-rpg/rpg_public_dronet)

**A Survey on Deep Learning Methods for Robot Vision**

[https://arxiv.org/abs/1803.10862](https://arxiv.org/abs/1803.10862)

# Deep Learning on Mobile / Embedded Devices

**Convolutional neural networks on the iPhone with VGGNet**

- blog: [http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/](http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/)
- github: [https://github.com/hollance/VGGNet-Metal](https://github.com/hollance/VGGNet-Metal)

**TensorFlow for Mobile Poets**

- blog: [https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/](https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/)

**The Convolutional Neural Network(CNN) for Android**

- intro: CnnForAndroid:A Classification Project using Convolutional Neural Network(CNN) in Android platform。It also support Caffe Model
- github: [https://github.com/zhangqianhui/CnnForAndroid](https://github.com/zhangqianhui/CnnForAndroid)

**TensorFlow on Android**

- blog: [https://www.oreilly.com/learning/tensorflow-on-android](https://www.oreilly.com/learning/tensorflow-on-android)

**Experimenting with TensorFlow on Android**

- part 1: [https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-pt-1-362683b31838#.5gbp2d4st](https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-pt-1-362683b31838#.5gbp2d4st)
- part 2: [https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-part-2-12f3dc294eaf#.2gx3o65f5](https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-part-2-12f3dc294eaf#.2gx3o65f5)
- github: [https://github.com/MostafaGazar/tensorflow](https://github.com/MostafaGazar/tensorflow)

**XNOR.ai frees AI from the prison of the supercomputer**

- blog: [https://techcrunch.com/2017/01/19/xnor-ai-frees-ai-from-the-prison-of-the-supercomputer/](https://techcrunch.com/2017/01/19/xnor-ai-frees-ai-from-the-prison-of-the-supercomputer/)

**Embedded Deep Learning with NVIDIA Jetson**

- youtube: [https://www.youtube.com/watch?v=_4tzlXPQWb8](https://www.youtube.com/watch?v=_4tzlXPQWb8)
- mirror: [https://pan.baidu.com/s/1pKCDXkZ](https://pan.baidu.com/s/1pKCDXkZ)

**Embedded and mobile deep learning research resources**

[https://github.com/csarron/emdl](https://github.com/csarron/emdl)

**Modeling the Resource Requirements of Convolutional Neural Networks on Mobile Devices**

[https://arxiv.org/abs/1709.09118](https://arxiv.org/abs/1709.09118)

# Benchmarks

**Deep Learning’s Accuracy**

- blog: [http://deeplearning4j.org/accuracy.html](http://deeplearning4j.org/accuracy.html)

**Benchmarks for popular CNN models**

- intro: Benchmarks for popular convolutional neural network models on CPU and different GPUs, with and without cuDNN.
- github: [https://github.com/jcjohnson/cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks)

**Deep Learning Benchmarks**

[http://add-for.com/deep-learning-benchmarks/](http://add-for.com/deep-learning-benchmarks/)

**cudnn-rnn-benchmarks**

- github: [https://github.com/MaximumEntropy/cudnn_rnn_theano_benchmarks](https://github.com/MaximumEntropy/cudnn_rnn_theano_benchmarks)

# Papers

**Reweighted Wake-Sleep**

- paper: [http://arxiv.org/abs/1406.2751](http://arxiv.org/abs/1406.2751)
- github: [https://github.com/jbornschein/reweighted-ws](https://github.com/jbornschein/reweighted-ws)

**Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks**

- paper: [http://arxiv.org/abs/1502.05336](http://arxiv.org/abs/1502.05336)
- github: [https://github.com/HIPS/Probabilistic-Backpropagation](https://github.com/HIPS/Probabilistic-Backpropagation)

**Deeply-Supervised Nets**

- paper: [http://arxiv.org/abs/1409.5185](http://arxiv.org/abs/1409.5185)
- github: [https://github.com/mbhenaff/spectral-lib](https://github.com/mbhenaff/spectral-lib)

**Deep learning**

- intro: Nature 2015
- author: Yann LeCun, Yoshua Bengio & Geoffrey Hinton
- paper: [http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)

**On the Expressive Power of Deep Learning: A Tensor Analysis**

- paper: [http://arxiv.org/abs/1509.05009](http://arxiv.org/abs/1509.05009)

**Understanding and Predicting Image Memorability at a Large Scale**

- intro: MIT. ICCV 2015
- homepage: [http://memorability.csail.mit.edu/](http://memorability.csail.mit.edu/)
- paper: [https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf](https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf)
- code: [http://memorability.csail.mit.edu/download.html](http://memorability.csail.mit.edu/download.html)
- reviews: [http://petapixel.com/2015/12/18/how-memorable-are-times-top-10-photos-of-2015-to-a-computer/](http://petapixel.com/2015/12/18/how-memorable-are-times-top-10-photos-of-2015-to-a-computer/)

**Towards Open Set Deep Networks**

- arxiv: [http://arxiv.org/abs/1511.06233](http://arxiv.org/abs/1511.06233)
- github: [https://github.com/abhijitbendale/OSDN](https://github.com/abhijitbendale/OSDN)

**Structured Prediction Energy Networks**

- intro: ICML 2016. SPEN
- arxiv: [http://arxiv.org/abs/1511.06350](http://arxiv.org/abs/1511.06350)
- github: [https://github.com/davidBelanger/SPEN](https://github.com/davidBelanger/SPEN)

**Deep Neural Networks predict Hierarchical Spatio-temporal Cortical Dynamics of Human Visual Object Recognition**

- arxiv: [http://arxiv.org/abs/1601.02970](http://arxiv.org/abs/1601.02970)
- demo: [http://brainmodels.csail.mit.edu/dnn/drawCNN/](http://brainmodels.csail.mit.edu/dnn/drawCNN/)

**Recent Advances in Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1512.07108](http://arxiv.org/abs/1512.07108)

**Understanding Deep Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1601.04920](http://arxiv.org/abs/1601.04920)

**DeepCare: A Deep Dynamic Memory Model for Predictive Medicine**

- arxiv: [http://arxiv.org/abs/1602.00357](http://arxiv.org/abs/1602.00357)

**Exploiting Cyclic Symmetry in Convolutional Neural Networks**

![](http://benanne.github.io/images/cyclicroll.png)

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1602.02660](http://arxiv.org/abs/1602.02660)
- github(Winning solution for the National Data Science Bowl competition on Kaggle (plankton classification)): [https://github.com/benanne/kaggle-ndsb](https://github.com/benanne/kaggle-ndsb)
- ref(use Cyclic pooling): [http://benanne.github.io/2015/03/17/plankton.html](http://benanne.github.io/2015/03/17/plankton.html)

**Cross-dimensional Weighting for Aggregated Deep Convolutional Features**

- arxiv: [http://arxiv.org/abs/1512.04065](http://arxiv.org/abs/1512.04065)
- github: [https://github.com/yahoo/crow](https://github.com/yahoo/crow)

**Understanding Visual Concepts with Continuation Learning**

- project page: [http://willwhitney.github.io/understanding-visual-concepts/](http://willwhitney.github.io/understanding-visual-concepts/)
- arxiv: [http://arxiv.org/abs/1602.06822](http://arxiv.org/abs/1602.06822)
- github: [https://github.com/willwhitney/understanding-visual-concepts](https://github.com/willwhitney/understanding-visual-concepts)

**Learning Efficient Algorithms with Hierarchical Attentive Memory**

- arxiv: [http://arxiv.org/abs/1602.03218](http://arxiv.org/abs/1602.03218)
- github: [https://github.com/Smerity/tf-ham](https://github.com/Smerity/tf-ham)

**DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1601.00917](http://arxiv.org/abs/1601.00917)
- github: [https://github.com/bigaidream-projects/drmad](https://github.com/bigaidream-projects/drmad)

**Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?**

- arxiv: [http://arxiv.org/abs/1603.05691](http://arxiv.org/abs/1603.05691)
- review: [http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/)

**Harnessing Deep Neural Networks with Logic Rules**

- arxiv: [http://arxiv.org/abs/1603.06318](http://arxiv.org/abs/1603.06318)

**Degrees of Freedom in Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1603.09260](http://arxiv.org/abs/1603.09260)

**Deep Networks with Stochastic Depth**

- arxiv: [http://arxiv.org/abs/1603.09382](http://arxiv.org/abs/1603.09382)
- github: [https://github.com/yueatsprograms/Stochastic_Depth](https://github.com/yueatsprograms/Stochastic_Depth)
- notes("Stochastic Depth Networks will Become the New Normal"): [http://deliprao.com/archives/134](http://deliprao.com/archives/134)
- github: [https://github.com/dblN/stochastic_depth_keras](https://github.com/dblN/stochastic_depth_keras)
- github: [https://github.com/yasunorikudo/chainer-ResDrop](https://github.com/yasunorikudo/chainer-ResDrop)
- review: [https://medium.com/@tim_nth/review-deep-networks-with-stochastic-depth-51bd53acfe72](https://medium.com/@tim_nth/review-deep-networks-with-stochastic-depth-51bd53acfe72)

**LIFT: Learned Invariant Feature Transform**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1603.09114](http://arxiv.org/abs/1603.09114)
- github(official): [https://github.com/cvlab-epfl/LIFT](https://github.com/cvlab-epfl/LIFT)

**Bridging the Gaps Between Residual Learning, Recurrent Neural Networks and Visual Cortex**

- arxiv: [https://arxiv.org/abs/1604.03640](https://arxiv.org/abs/1604.03640)
- slides: [http://prlab.tudelft.nl/sites/default/files/rnnResnetCortex.pdf](http://prlab.tudelft.nl/sites/default/files/rnnResnetCortex.pdf)

**Understanding How Image Quality Affects Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1604.04004](http://arxiv.org/abs/1604.04004)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/4exk3u/dcnns_are_more_sensitive_to_blur_and_noise_than/](https://www.reddit.com/r/MachineLearning/comments/4exk3u/dcnns_are_more_sensitive_to_blur_and_noise_than/)

**Deep Embedding for Spatial Role Labeling**

- arxiv: [http://arxiv.org/abs/1603.08474](http://arxiv.org/abs/1603.08474)
- github: [https://github.com/oswaldoludwig/visually-informed-embedding-of-word-VIEW-](https://github.com/oswaldoludwig/visually-informed-embedding-of-word-VIEW-)

**Unreasonable Effectiveness of Learning Neural Nets: Accessible States and Robust Ensembles**

- arxiv: [http://arxiv.org/abs/1605.06444](http://arxiv.org/abs/1605.06444)

**Learning Deep Representation for Imbalanced Classification**

![](http://mmlab.ie.cuhk.edu.hk/projects/LMLE/method.png)

- intro: CVPR 2016
- keywords: Deep Learning Large Margin Local Embedding (LMLE)
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/LMLE.html](http://mmlab.ie.cuhk.edu.hk/projects/LMLE.html)
- paper: [http://personal.ie.cuhk.edu.hk/~ccloy/files/cvpr_2016_imbalanced.pdf](http://personal.ie.cuhk.edu.hk/~ccloy/files/cvpr_2016_imbalanced.pdf)
- code: [http://mmlab.ie.cuhk.edu.hk/projects/LMLE/lmle_code.zip](http://mmlab.ie.cuhk.edu.hk/projects/LMLE/lmle_code.zip)

**Newtonian Image Understanding: Unfolding the Dynamics of Objects in Static Images**

![](http://allenai.org/images/projects/plato_newton.png?cb=1466683222538)

- homepage: [http://allenai.org/plato/newtonian-understanding/](http://allenai.org/plato/newtonian-understanding/)
- arxiv: [http://arxiv.org/abs/1511.04048](http://arxiv.org/abs/1511.04048)
- github: [https://github.com/roozbehm/newtonian](https://github.com/roozbehm/newtonian)

**DeepMath - Deep Sequence Models for Premise Selection**

- arxiv: [https://arxiv.org/abs/1606.04442](https://arxiv.org/abs/1606.04442)
- github: [https://github.com/tensorflow/deepmath](https://github.com/tensorflow/deepmath)

**Convolutional Neural Networks Analyzed via Convolutional Sparse Coding**

- arxiv: [http://arxiv.org/abs/1607.08194](http://arxiv.org/abs/1607.08194)

**Systematic evaluation of CNN advances on the ImageNet**

- arxiv: [http://arxiv.org/abs/1606.02228](http://arxiv.org/abs/1606.02228)
- github: [https://github.com/ducha-aiki/caffenet-benchmark](https://github.com/ducha-aiki/caffenet-benchmark)

**Why does deep and cheap learning work so well?**

- intro: Harvard and MIT
- arxiv: [http://arxiv.org/abs/1608.08225](http://arxiv.org/abs/1608.08225)
- review: [https://www.technologyreview.com/s/602344/the-extraordinary-link-between-deep-neural-networks-and-the-nature-of-the-universe/](https://www.technologyreview.com/s/602344/the-extraordinary-link-between-deep-neural-networks-and-the-nature-of-the-universe/)

**A scalable convolutional neural network for task-specified scenarios via knowledge distillation**

- arxiv: [http://arxiv.org/abs/1609.05695](http://arxiv.org/abs/1609.05695)

**Alternating Back-Propagation for Generator Network**

- project page(code+data): [http://www.stat.ucla.edu/~ywu/ABP/main.html](http://www.stat.ucla.edu/~ywu/ABP/main.html)
- paper: [http://www.stat.ucla.edu/~ywu/ABP/doc/arXivABP.pdf](http://www.stat.ucla.edu/~ywu/ABP/doc/arXivABP.pdf)

**A Novel Representation of Neural Networks**

- arxiv: [https://arxiv.org/abs/1610.01549](https://arxiv.org/abs/1610.01549)

**Optimization of Convolutional Neural Network using Microcanonical Annealing Algorithm**

- intro: IEEE ICACSIS 2016
- arxiv: [https://arxiv.org/abs/1610.02306](https://arxiv.org/abs/1610.02306)

**Uncertainty in Deep Learning**

- intro: PhD Thesis. Cambridge Machine Learning Group
- blog: [http://mlg.eng.cam.ac.uk/yarin/blog_2248.html](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)
- thesis: [http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)

**Deep Convolutional Neural Network Design Patterns**

- arxiv: [https://arxiv.org/abs/1611.00847](https://arxiv.org/abs/1611.00847)
- github: [https://github.com/iPhysicist/CNNDesignPatterns](https://github.com/iPhysicist/CNNDesignPatterns)

**Extensions and Limitations of the Neural GPU**

- arxiv: [https://arxiv.org/abs/1611.00736](https://arxiv.org/abs/1611.00736)
- github: [https://github.com/openai/ecprice-neural-gpu](https://github.com/openai/ecprice-neural-gpu)

**Neural Functional Programming**

- arxiv: [https://arxiv.org/abs/1611.01988](https://arxiv.org/abs/1611.01988)

**Deep Information Propagation**

- arxiv: [https://arxiv.org/abs/1611.01232](https://arxiv.org/abs/1611.01232)

**Compressed Learning: A Deep Neural Network Approach**

- arxiv: [https://arxiv.org/abs/1610.09615](https://arxiv.org/abs/1610.09615)

**A backward pass through a CNN using a generative model of its activations**

- arxiv: [https://arxiv.org/abs/1611.02767](https://arxiv.org/abs/1611.02767)

**Understanding deep learning requires rethinking generalization**

- intro: ICLR 2017 best paper. MIT & Google Brain & UC Berkeley & Google DeepMind
- arxiv: [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)
- example code: [https://github.com/pluskid/fitting-random-labels](https://github.com/pluskid/fitting-random-labels)
- notes: [https://theneuralperspective.com/2017/01/24/understanding-deep-learning-requires-rethinking-generalization/](https://theneuralperspective.com/2017/01/24/understanding-deep-learning-requires-rethinking-generalization/)

**Learning the Number of Neurons in Deep Networks**

- intro: NIPS 2016
- arxiv: [https://arxiv.org/abs/1611.06321](https://arxiv.org/abs/1611.06321)

**Survey of Expressivity in Deep Neural Networks**

- intro: Presented at NIPS 2016 Workshop on Interpretable Machine Learning in Complex Systems
- intro: Google Brain & Cornell University & Stanford University
- arxiv: [https://arxiv.org/abs/1611.08083](https://arxiv.org/abs/1611.08083)

**Designing Neural Network Architectures using Reinforcement Learning**

- intro: MIT
- project page: [https://bowenbaker.github.io/metaqnn/](https://bowenbaker.github.io/metaqnn/)
- arxiv: [https://arxiv.org/abs/1611.02167](https://arxiv.org/abs/1611.02167)

**Towards Robust Deep Neural Networks with BANG**

- intro: University of Colorado
- arxiv: [https://arxiv.org/abs/1612.00138](https://arxiv.org/abs/1612.00138)

**Deep Quantization: Encoding Convolutional Activations with Deep Generative Model**

- intro: University of Science and Technology of China & MSR
- arxiv: [https://arxiv.org/abs/1611.09502](https://arxiv.org/abs/1611.09502)

**A Probabilistic Theory of Deep Learning**

- arxiv: [https://arxiv.org/abs/1504.00641](https://arxiv.org/abs/1504.00641)

**A Probabilistic Framework for Deep Learning**

- intro: Rice University
- arxiv: [https://arxiv.org/abs/1612.01936](https://arxiv.org/abs/1612.01936)

**Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer**

- arxiv: [https://arxiv.org/abs/1612.03928](https://arxiv.org/abs/1612.03928)
- github(PyTorch): [https://github.com/szagoruyko/attention-transfer](https://github.com/szagoruyko/attention-transfer)

**Risk versus Uncertainty in Deep Learning: Bayes, Bootstrap and the Dangers of Dropout**

- intro: Google Deepmind
- paper: [http://bayesiandeeplearning.org/papers/BDL_4.pdf](http://bayesiandeeplearning.org/papers/BDL_4.pdf)

**Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer**

- intro: Google Brain & Jagiellonian University
- keywords: Sparsely-Gated Mixture-of-Experts layer (MoE), language modeling and machine translation
- arxiv: [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/5pud72/research_outrageously_large_neural_networks_the/](https://www.reddit.com/r/MachineLearning/comments/5pud72/research_outrageously_large_neural_networks_the/)

**Deep Network Guided Proof Search**

- intro: Google Research & University of Innsbruck
- arxiv: [https://arxiv.org/abs/1701.06972](https://arxiv.org/abs/1701.06972)

**PathNet: Evolution Channels Gradient Descent in Super Neural Networks**

- intro: Google DeepMind & Google Brain
- arxiv: [https://arxiv.org/abs/1701.08734](https://arxiv.org/abs/1701.08734)
- notes: [https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273#.8f0o6w3en](https://medium.com/intuitionmachine/pathnet-a-modular-deep-learning-architecture-for-agi-5302fcf53273#.8f0o6w3en)

**Reluplex: An Efficient SMT Solver for Verifying Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1702.01135](https://arxiv.org/abs/1702.01135)

**The Power of Sparsity in Convolutional Neural Networks**

- arxiv: [https://arxiv.org/abs/1702.06257](https://arxiv.org/abs/1702.06257)

**Learning across scales - A multiscale method for Convolution Neural Networks**

- arxiv: [https://arxiv.org/abs/1703.02009](https://arxiv.org/abs/1703.02009)

**Stacking-based Deep Neural Network: Deep Analytic Network on Convolutional Spectral Histogram Features**

- arxiv: [https://arxiv.org/abs/1703.01396](https://arxiv.org/abs/1703.01396)

**A Compositional Object-Based Approach to Learning Physical Dynamics**

- intro: ICLR 2017. Neural Physics Engine
- paper: [https://openreview.net/pdf?id=Bkab5dqxe](https://openreview.net/pdf?id=Bkab5dqxe)
- github: [https://github.com/mbchang/dynamics](https://github.com/mbchang/dynamics)

**Genetic CNN**

- arxiv: [https://arxiv.org/abs/1703.01513](https://arxiv.org/abs/1703.01513)
- github(Tensorflow): [https://github.com/aqibsaeed/Genetic-CNN](https://github.com/aqibsaeed/Genetic-CNN)

**Deep Sets**

- intro: Amazon Web Services & CMU
- keywords: statistic estimation, point cloud classification, set expansion, and image tagging
- arxiv: [https://arxiv.org/abs/1703.06114](https://arxiv.org/abs/1703.06114)

**Multiscale Hierarchical Convolutional Networks**

- arxiv: [https://arxiv.org/abs/1703.04140](https://arxiv.org/abs/1703.04140)
- github: [https://github.com/jhjacobsen/HierarchicalCNN](https://github.com/jhjacobsen/HierarchicalCNN)

**Deep Neural Networks Do Not Recognize Negative Images**

[https://arxiv.org/abs/1703.06857](https://arxiv.org/abs/1703.06857)

**Failures of Deep Learning**

- arxiv: [https://arxiv.org/abs/1703.07950](https://arxiv.org/abs/1703.07950)
- github: [https://github.com/shakedshammah/failures_of_DL](https://github.com/shakedshammah/failures_of_DL)

**Multi-Scale Dense Convolutional Networks for Efficient Prediction**

- intro: Cornell University & Tsinghua University & Fudan University & Facebook AI Research
- arxiv: [https://arxiv.org/abs/1703.09844](https://arxiv.org/abs/1703.09844)
- github: [https://github.com/gaohuang/MSDNet](https://github.com/gaohuang/MSDNet)

**Scaling the Scattering Transform: Deep Hybrid Networks**

- arxiv: [https://arxiv.org/abs/1703.08961](https://arxiv.org/abs/1703.08961)
- github: [https://github.com/edouardoyallon/scalingscattering](https://github.com/edouardoyallon/scalingscattering)
- github(CuPy/PyTorch): [https://github.com/edouardoyallon/pyscatwave](https://github.com/edouardoyallon/pyscatwave)

**Deep Learning is Robust to Massive Label Noise**

[https://arxiv.org/abs/1705.10694](https://arxiv.org/abs/1705.10694)

**Input Fast-Forwarding for Better Deep Learning**

- intro: ICIAR 2017
- keywords: Fast-Forward Network (FFNet)
- arxiv: [https://arxiv.org/abs/1705.08479](https://arxiv.org/abs/1705.08479)

**Deep Mutual Learning**

- keywords: deep mutual learning (DML)
- arxiv: [https://arxiv.org/abs/1706.00384](https://arxiv.org/abs/1706.00384)

**Automated Problem Identification: Regression vs Classification via Evolutionary Deep Networks**

- intro: University of Cape Town
- arxiv: [https://arxiv.org/abs/1707.00703](https://arxiv.org/abs/1707.00703)

**Revisiting Unreasonable Effectiveness of Data in Deep Learning Era**

- intro: Google Research & CMU
- arxiv: [https://arxiv.org/abs/1707.02968](https://arxiv.org/abs/1707.02968)
- blog: [https://research.googleblog.com/2017/07/revisiting-unreasonable-effectiveness.html](https://research.googleblog.com/2017/07/revisiting-unreasonable-effectiveness.html)

**Deep Layer Aggregation**

- intro: UC Berkeley
- arxiv: [https://arxiv.org/abs/1707.06484](https://arxiv.org/abs/1707.06484)

**Improving Robustness of Feature Representations to Image Deformations using Powered Convolution in CNNs**

[https://arxiv.org/abs/1707.07830](https://arxiv.org/abs/1707.07830)

**Learning uncertainty in regression tasks by deep neural networks**

- intro: Free University of Berlin
- arxiv: [https://arxiv.org/abs/1707.07287](https://arxiv.org/abs/1707.07287)

**Generalizing the Convolution Operator in Convolutional Neural Networks**

[https://arxiv.org/abs/1707.09864](https://arxiv.org/abs/1707.09864)

**Convolution with Logarithmic Filter Groups for Efficient Shallow CNN**

[https://arxiv.org/abs/1707.09855](https://arxiv.org/abs/1707.09855)

**Deep Multi-View Learning with Stochastic Decorrelation Loss**

[https://arxiv.org/abs/1707.09669](https://arxiv.org/abs/1707.09669)

**Take it in your stride: Do we need striding in CNNs?**

[https://arxiv.org/abs/1712.02502](https://arxiv.org/abs/1712.02502)

**Security Risks in Deep Learning Implementation**

- intro: Qihoo 360 Security Research Lab & University of Georgia & University of Virginia
- arxiv: [https://arxiv.org/abs/1711.11008](https://arxiv.org/abs/1711.11008)

**Online Learning with Gated Linear Networks**

- intro: DeepMind
- arxiv: [https://arxiv.org/abs/1712.01897](https://arxiv.org/abs/1712.01897)

**On the Information Bottleneck Theory of Deep Learning**

[https://openreview.net/forum?id=ry_WPG-A-&noteId=ry_WPG-A](https://openreview.net/forum?id=ry_WPG-A-&noteId=ry_WPG-A)

**The Unreasonable Effectiveness of Deep Features as a Perceptual Metric**

- project page: [https://richzhang.github.io/PerceptualSimilarity/](https://richzhang.github.io/PerceptualSimilarity/)
- arxiv: [https://arxiv.org/abs/1801.03924](https://arxiv.org/abs/1801.03924)
- github: [https://github.com//richzhang/PerceptualSimilarity](https://github.com//richzhang/PerceptualSimilarity)

**Less is More: Culling the Training Set to Improve Robustness of Deep Neural Networks**

- intro: University of California, Davis
- arxiv: [https://arxiv.org/abs/1801.02850](https://arxiv.org/abs/1801.02850)

**Towards an Understanding of Neural Networks in Natural-Image Spaces**

[https://arxiv.org/abs/1801.09097](https://arxiv.org/abs/1801.09097)

**Deep Private-Feature Extraction**

[https://arxiv.org/abs/1802.03151](https://arxiv.org/abs/1802.03151)

**Not All Samples Are Created Equal: Deep Learning with Importance Sampling**

- intro: Idiap Research Institute
- arxiv: [https://arxiv.org/abs/1803.00942](https://arxiv.org/abs/1803.00942)

**Label Refinery: Improving ImageNet Classification through Label Progression**

- intro: Using a Label Refinery improves the state-of-the-art top-1 accuracy of (1) AlexNet from 59.3 to 67.2, 
(2) MobileNet from 70.6 to 73.39, (3) MobileNet-0.25 from 50.6 to 55.59, 
(4) VGG19 from 72.7 to 75.46, and (5) Darknet19 from 72.9 to 74.47.
- intro: XNOR AI, University of Washington, Allen AI
- arxiv: [https://arxiv.org/abs/1805.02641](https://arxiv.org/abs/1805.02641)
- github: [https://github.com/hessamb/label-refinery](https://github.com/hessamb/label-refinery)

**How Many Samples are Needed to Learn a Convolutional Neural Network?**

[https://arxiv.org/abs/1805.07883](https://arxiv.org/abs/1805.07883)

**VisualBackProp for learning using privileged information with CNNs**

[https://arxiv.org/abs/1805.09474](https://arxiv.org/abs/1805.09474)

**BAM: Bottleneck Attention Module**

- intro: BMVC 2018 (oral). Lunit Inc. & Adobe Research
- arxiv: [https://arxiv.org/abs/1807.06514](https://arxiv.org/abs/1807.06514)

**CBAM: Convolutional Block Attention Module**

- intro: ECCV 2018. Lunit Inc. & Adobe Research
- arxiv: [https://arxiv.org/abs/1807.06521](https://arxiv.org/abs/1807.06521)

**Scale equivariance in CNNs with vector fields**

- intro: ICML/FAIM 2018 workshop on Towards learning with limited labels: Equivariance, Invariance, and Beyond (oral presentation)
- arxiv: [https://arxiv.org/abs/1807.11783](https://arxiv.org/abs/1807.11783)

**Downsampling leads to Image Memorization in Convolutional Autoencoders**

[https://arxiv.org/abs/1810.10333](https://arxiv.org/abs/1810.10333)

**Do Normalization Layers in a Deep ConvNet Really Need to Be Distinct?**

[https://arxiv.org/abs/1811.07727](https://arxiv.org/abs/1811.07727)

**Are All Training Examples Created Equal? An Empirical Study**

[https://arxiv.org/abs/1811.12569](https://arxiv.org/abs/1811.12569)

**ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness**

[https://arxiv.org/abs/1811.12231](https://arxiv.org/abs/1811.12231)

## Tutorials and Surveys

**A Survey: Time Travel in Deep Learning Space: An Introduction to Deep Learning Models and How Deep Learning Models Evolved from the Initial Ideas**

- arxiv: [http://arxiv.org/abs/1510.04781](http://arxiv.org/abs/1510.04781)

**On the Origin of Deep Learning**

- intro: CMU. 70 pages, 200 references
- arxiv: [https://arxiv.org/abs/1702.07800](https://arxiv.org/abs/1702.07800)

**Efficient Processing of Deep Neural Networks: A Tutorial and Survey**

- intro: MIT
- arxiv: [https://arxiv.org/abs/1703.09039](https://arxiv.org/abs/1703.09039)

**The History Began from AlexNet: A Comprehensive Survey on Deep Learning Approaches**

{https://arxiv.org/abs/1803.01164}(https://arxiv.org/abs/1803.01164)

## Mathematics of Deep Learning

**A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction**

- arxiv: [http://arxiv.org/abs/1512.06293](http://arxiv.org/abs/1512.06293)

**Mathematics of Deep Learning**

- intro: Johns Hopkins University & New York University & Tel-Aviv University & University of California, Los Angeles
- arxiv: [https://arxiv.org/abs/1712.04741](https://arxiv.org/abs/1712.04741)

## Local Minima

**Local minima in training of deep networks**

- intro: DeepMind
- arxiv: [https://arxiv.org/abs/1611.06310](https://arxiv.org/abs/1611.06310)

**Deep linear neural networks with arbitrary loss: All local minima are global**

- intro: CMU & University of Southern California & Facebook Artificial Intelligence Research
- arxiv: [https://arxiv.org/abs/1712.00779](https://arxiv.org/abs/1712.00779)

**Gradient Descent Learns One-hidden-layer CNN: Don't be Afraid of Spurious Local Minima**

- intro: Loyola Marymount University & California State University
- arxiv: [https://arxiv.org/abs/1712.01473](https://arxiv.org/abs/1712.01473)

**CNNs are Globally Optimal Given Multi-Layer Support**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1712.02501](https://arxiv.org/abs/1712.02501)

**Spurious Local Minima are Common in Two-Layer ReLU Neural Networks**

[https://arxiv.org/abs/1712.08968](https://arxiv.org/abs/1712.08968)

## Dive Into CNN

**Structured Receptive Fields in CNNs**

- arxiv: [https://arxiv.org/abs/1605.02971](https://arxiv.org/abs/1605.02971)
- github: [https://github.com/jhjacobsen/RFNN](https://github.com/jhjacobsen/RFNN)

**How ConvNets model Non-linear Transformations**

- arxiv: [https://arxiv.org/abs/1702.07664](https://arxiv.org/abs/1702.07664)

## Separable Convolutions / Grouped Convolutions

**Factorized Convolutional Neural Networks**

**Design of Efficient Convolutional Layers using Single Intra-channel Convolution, Topological Subdivisioning and Spatial "Bottleneck" Structure**

- arxiv: [http://arxiv.org/abs/1608.04337](http://arxiv.org/abs/1608.04337)

## STDP

**A biological gradient descent for prediction through a combination of STDP and homeostatic plasticity**

- arxiv: [http://arxiv.org/abs/1206.4812](http://arxiv.org/abs/1206.4812)

**An objective function for STDP**

- arxiv: [http://arxiv.org/abs/1509.05936](http://arxiv.org/abs/1509.05936)

**Towards a Biologically Plausible Backprop**

- arxiv: [http://arxiv.org/abs/1602.05179](http://arxiv.org/abs/1602.05179)

## Target Propagation

**How Auto-Encoders Could Provide Credit Assignment in Deep Networks via Target Propagation**

- arxiv: [http://arxiv.org/abs/1407.7906](http://arxiv.org/abs/1407.7906)

**Difference Target Propagation**

- arxiv: [http://arxiv.org/abs/1412.7525](http://arxiv.org/abs/1412.7525)
- github: [https://github.com/donghyunlee/dtp](https://github.com/donghyunlee/dtp)

## Zero Shot Learning

**Learning a Deep Embedding Model for Zero-Shot Learning**

- arxiv: [https://arxiv.org/abs/1611.05088](https://arxiv.org/abs/1611.05088)

**Zero-Shot (Deep) Learning**

[https://amundtveit.com/2016/11/18/zero-shot-deep-learning/](https://amundtveit.com/2016/11/18/zero-shot-deep-learning/)

**Zero-shot learning experiments by deep learning.**

[https://github.com/Elyorcv/zsl-deep-learning](https://github.com/Elyorcv/zsl-deep-learning)

**Zero-Shot Learning - The Good, the Bad and the Ugly**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1703.04394](https://arxiv.org/abs/1703.04394)

**Semantic Autoencoder for Zero-Shot Learning**

- intro: CVPR 2017
- project page: [https://elyorcv.github.io/projects/sae](https://elyorcv.github.io/projects/sae)
- arxiv: [https://arxiv.org/abs/1704.08345](https://arxiv.org/abs/1704.08345)
- github: [https://github.com/Elyorcv/SAE](https://github.com/Elyorcv/SAE)

**Zero-Shot Learning via Category-Specific Visual-Semantic Mapping**

[https://arxiv.org/abs/1711.06167](https://arxiv.org/abs/1711.06167)

**Zero-Shot Learning via Class-Conditioned Deep Generative Models**

- intro: AAAI 2018
- arxiv: [https://arxiv.org/abs/1711.05820](https://arxiv.org/abs/1711.05820)

**Feature Generating Networks for Zero-Shot Learning**

[https://arxiv.org/abs/1712.00981](https://arxiv.org/abs/1712.00981)

**Zero-Shot Visual Recognition using Semantics-Preserving Adversarial Embedding Network**

[https://arxiv.org/abs/1712.01928](https://arxiv.org/abs/1712.01928)

**Combining Deep Universal Features, Semantic Attributes, and Hierarchical Classification for Zero-Shot Learning**

- intro: extension to work published in conference proceedings of 2017 IAPR MVA Conference
- arxiv: [https://arxiv.org/abs/1712.03151](https://arxiv.org/abs/1712.03151)

**Multi-Context Label Embedding**

- keywords: Multi-Context Label Embedding (MCLE) 
- arxiv: [https://arxiv.org/abs/1805.01199](https://arxiv.org/abs/1805.01199)

## Incremental Learning

**iCaRL: Incremental Classifier and Representation Learning**

- arxiv: [https://arxiv.org/abs/1611.07725](https://arxiv.org/abs/1611.07725)

**FearNet: Brain-Inspired Model for Incremental Learning**

[https://arxiv.org/abs/1711.10563](https://arxiv.org/abs/1711.10563)

**Incremental Learning in Deep Convolutional Neural Networks Using Partial Network Sharing**

- intro: Purdue University
- arxiv: [https://arxiv.org/abs/1712.02719](https://arxiv.org/abs/1712.02719)

**Incremental Classifier Learning with Generative Adversarial Networks**

[https://arxiv.org/abs/1802.00853](https://arxiv.org/abs/1802.00853)

**Learn the new, keep the old: Extending pretrained models with new anatomy and images**

- intro: MICCAI 2018
- arxiv: [https://arxiv.org/abs/1806.00265](https://arxiv.org/abs/1806.00265)

## Ensemble Deep Learning

**Convolutional Neural Fabrics**

- intro: NIPS 2016
- arxiv: [http://arxiv.org/abs/1606.02492](http://arxiv.org/abs/1606.02492)
- github: [https://github.com/shreyassaxena/convolutional-neural-fabrics](https://github.com/shreyassaxena/convolutional-neural-fabrics)

**Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles**

- arxiv: [https://arxiv.org/abs/1606.07839](https://arxiv.org/abs/1606.07839)
- youtube: [https://www.youtube.com/watch?v=KjUfMtZjyfg&feature=youtu.be](https://www.youtube.com/watch?v=KjUfMtZjyfg&feature=youtu.be)

**Snapshot Ensembles: Train 1, Get M for Free**

- paper: [http://openreview.net/pdf?id=BJYwwY9ll](http://openreview.net/pdf?id=BJYwwY9ll)
- github(Torch): [https://github.com/gaohuang/SnapshotEnsemble](https://github.com/gaohuang/SnapshotEnsemble)
- github: [https://github.com/titu1994/Snapshot-Ensembles](https://github.com/titu1994/Snapshot-Ensembles)

**Ensemble Deep Learning**

- blog: [https://amundtveit.com/2016/12/02/ensemble-deep-learning/](https://amundtveit.com/2016/12/02/ensemble-deep-learning/)

## Domain Adaptation

**Adversarial Discriminative Domain Adaptation**

- intro: UC Berkeley & Stanford University & Boston University
- arxiv: [https://arxiv.org/abs/1702.05464](https://arxiv.org/abs/1702.05464)
- github: [https://github.com//corenel/pytorch-adda](https://github.com//corenel/pytorch-adda)

**Parameter Reference Loss for Unsupervised Domain Adaptation**

[https://arxiv.org/abs/1711.07170](https://arxiv.org/abs/1711.07170)

**Residual Parameter Transfer for Deep Domain Adaptation**

[https://arxiv.org/abs/1711.07714](https://arxiv.org/abs/1711.07714)

**Adversarial Feature Augmentation for Unsupervised Domain Adaptation**

[https://arxiv.org/abs/1711.08561](https://arxiv.org/abs/1711.08561)

**Image to Image Translation for Domain Adaptation**

[https://arxiv.org/abs/1712.00479](https://arxiv.org/abs/1712.00479)

**Incremental Adversarial Domain Adaptation**

[https://arxiv.org/abs/1712.07436](https://arxiv.org/abs/1712.07436)

**Deep Visual Domain Adaptation: A Survey**

[https://arxiv.org/abs/1802.03601](https://arxiv.org/abs/1802.03601)

**Unsupervised Domain Adaptation: A Multi-task Learning-based Method**

[https://arxiv.org/abs/1803.09208](https://arxiv.org/abs/1803.09208)

**Importance Weighted Adversarial Nets for Partial Domain Adaptation**

[https://arxiv.org/abs/1803.09210](https://arxiv.org/abs/1803.09210)

**Open Set Domain Adaptation by Backpropagation**

[https://arxiv.org/abs/1804.10427](https://arxiv.org/abs/1804.10427)

**Learning Sampling Policies for Domain Adaptation**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1805.07641](https://arxiv.org/abs/1805.07641)

**Multi-Adversarial Domain Adaptation**

- intro: AAAI 2018 Oral.
- arxiv: [https://arxiv.org/abs/1809.02176](https://arxiv.org/abs/1809.02176)

**Unsupervised Domain Adaptation: An Adaptive Feature Norm Approach**

- intro: Sun Yat-sen University
- arxiv: [https://arxiv.org/abs/1811.07456](https://arxiv.org/abs/1811.07456)
- github: [https://github.com/jihanyang/AFN/](https://github.com/jihanyang/AFN/)

## Embedding

**Learning Deep Embeddings with Histogram Loss**

- intro: NIPS 2016
- arxiv: [https://arxiv.org/abs/1611.00822](https://arxiv.org/abs/1611.00822)

**Full-Network Embedding in a Multimodal Embedding Pipeline**

[https://arxiv.org/abs/1707.09872](https://arxiv.org/abs/1707.09872)

**Clustering-driven Deep Embedding with Pairwise Constraints**

[https://arxiv.org/abs/1803.08457](https://arxiv.org/abs/1803.08457)

**Deep Mixture of Experts via Shallow Embedding**

[https://arxiv.org/abs/1806.01531](https://arxiv.org/abs/1806.01531)

**Learning to Learn from Web Data through Deep Semantic Embeddings**

- intro: ECCV MULA Workshop 2018
- arxiv: [https://arxiv.org/abs/1808.06368](https://arxiv.org/abs/1808.06368)

**Heated-Up Softmax Embedding**

[https://arxiv.org/abs/1809.04157](https://arxiv.org/abs/1809.04157)

**Virtual Class Enhanced Discriminative Embedding Learning**

- intro: NeurIPS 2018
- arxiv: [https://arxiv.org/abs/1811.12611](https://arxiv.org/abs/1811.12611)

## Regression

**A Comprehensive Analysis of Deep Regression**

[https://arxiv.org/abs/1803.08450](https://arxiv.org/abs/1803.08450)

**Neural Motifs: Scene Graph Parsing with Global Context**

- intro: CVPR 2018. University of Washington
- project page: [http://rowanzellers.com/neuralmotifs/](http://rowanzellers.com/neuralmotifs/)
- arxiv: [https://arxiv.org/abs/1711.06640](https://arxiv.org/abs/1711.06640)
- github: [https://github.com/rowanz/neural-motifs](https://github.com/rowanz/neural-motifs)
- demo: [https://rowanzellers.com/scenegraph2/](https://rowanzellers.com/scenegraph2/)

## CapsNets

**Dynamic Routing Between Capsules**

- intro: Sara Sabour, Nicholas Frosst, Geoffrey E Hinton
- intro: Google Brain, Toronto
- arxiv: [https://arxiv.org/abs/1710.09829](https://arxiv.org/abs/1710.09829)
- github(official, Tensorflow): [https://github.com/Sarasra/models/tree/master/research/capsules](https://github.com/Sarasra/models/tree/master/research/capsules)

**Capsule Networks (CapsNets) – Tutorial**

- youtube: [https://www.youtube.com/watch?v=pPN8d0E3900](https://www.youtube.com/watch?v=pPN8d0E3900)
- mirror: [http://www.bilibili.com/video/av16594836/](http://www.bilibili.com/video/av16594836/)

**Improved Explainability of Capsule Networks: Relevance Path by Agreement**

- intro: Concordia University & University of Toronto
- arxiv: [https://arxiv.org/abs/1802.10204](https://arxiv.org/abs/1802.10204)

## Computer Vision

**A Taxonomy of Deep Convolutional Neural Nets for Computer Vision**

- arxiv: [http://arxiv.org/abs/1601.06615](http://arxiv.org/abs/1601.06615)

**On the usability of deep networks for object-based image analysis**

- intro: GEOBIA 2016
- arxiv: [http://arxiv.org/abs/1609.06845](http://arxiv.org/abs/1609.06845)

**Learning Recursive Filters for Low-Level Vision via a Hybrid Neural Network**

- intro: ECCV 2016
- project page: [http://www.sifeiliu.net/linear-rnn](http://www.sifeiliu.net/linear-rnn)
- paper: [http://faculty.ucmerced.edu/mhyang/papers/eccv16_rnn_filter.pdf](http://faculty.ucmerced.edu/mhyang/papers/eccv16_rnn_filter.pdf)
- poster: [http://www.eccv2016.org/files/posters/O-3A-03.pdf](http://www.eccv2016.org/files/posters/O-3A-03.pdf)
- github: [https://github.com/Liusifei/caffe-lowlevel](https://github.com/Liusifei/caffe-lowlevel)

**Toward Geometric Deep SLAM**

- intro: Magic Leap, Inc
- arxiv: [https://arxiv.org/abs/1707.07410](https://arxiv.org/abs/1707.07410)

**Learning Dual Convolutional Neural Networks for Low-Level Vision**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1805.05020](https://arxiv.org/abs/1805.05020)

**Not just a matter of semantics: the relationship between visual similarity and semantic similarity**

[https://arxiv.org/abs/1811.07120](https://arxiv.org/abs/1811.07120)

### All-In-One Network

**HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition**

- arxiv: [https://arxiv.org/abs/1603.01249](https://arxiv.org/abs/1603.01249)
- summary: [https://github.com/aleju/papers/blob/master/neural-nets/HyperFace.md](https://github.com/aleju/papers/blob/master/neural-nets/HyperFace.md)

**UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory**

- arxiv: [http://arxiv.org/abs/1609.02132](http://arxiv.org/abs/1609.02132)
- demo: [http://cvn.ecp.fr/ubernet/](http://cvn.ecp.fr/ubernet/)

**An All-In-One Convolutional Neural Network for Face Analysis**

- intro: simultaneous face detection, face alignment, pose estimation, gender recognition, smile detection, age estimation and face recognition 
- arxiv: [https://arxiv.org/abs/1611.00851](https://arxiv.org/abs/1611.00851)

**MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving**

- intro: first place on Kitti Road Segmentation. 
joint classification, detection and semantic segmentation via a unified architecture, less than 100 ms to perform all tasks
- arxiv: [https://arxiv.org/abs/1612.07695](https://arxiv.org/abs/1612.07695)
- github: [https://github.com/MarvinTeichmann/MultiNet](https://github.com/MarvinTeichmann/MultiNet)

**Adversarial Collaboration: Joint Unsupervised Learning of Depth, Camera Motion, Optical Flow and Motion Segmentation**

[https://arxiv.org/abs/1805.09806](https://arxiv.org/abs/1805.09806)

### Deep Learning for Data Structures

**The Case for Learned Index Structures**

- intro: MIT & Google
- keywords: B-Tree-Index, Hash-Index, BitMap-Index
- arxiv: [https://arxiv.org/abs/1712.01208](https://arxiv.org/abs/1712.01208)

# Projects

**Top Deep Learning Projects**

- github: [https://github.com/aymericdamien/TopDeepLearning](https://github.com/aymericdamien/TopDeepLearning)

**deepnet: Implementation of some deep learning algorithms**

- github: [https://github.com/nitishsrivastava/deepnet](https://github.com/nitishsrivastava/deepnet)

**DeepNeuralClassifier(Julia): Deep neural network using rectified linear units to classify hand written digits from the MNIST dataset**

- github: [https://github.com/jostmey/DeepNeuralClassifier](https://github.com/jostmey/DeepNeuralClassifier)

**Clarifai Node.js Demo**

- github: [https://github.com/patcat/Clarifai-Node-Demo](https://github.com/patcat/Clarifai-Node-Demo)
- blog("How to Make Your Web App Smarter with Image Recognition"): [http://www.sitepoint.com/how-to-make-your-web-app-smarter-with-image-recognition/](http://www.sitepoint.com/how-to-make-your-web-app-smarter-with-image-recognition/)

**Deep Learning in Rust**

- blog("baby steps"): [https://medium.com/@tedsta/deep-learning-in-rust-7e228107cccc#.t0pskuwkm](https://medium.com/@tedsta/deep-learning-in-rust-7e228107cccc#.t0pskuwkm)
- blog("a walk in the park"): [https://medium.com/@tedsta/deep-learning-in-rust-a-walk-in-the-park-fed6c87165ea#.pucj1l5yx](https://medium.com/@tedsta/deep-learning-in-rust-a-walk-in-the-park-fed6c87165ea#.pucj1l5yx)
- github: [https://github.com/tedsta/deeplearn-rs](https://github.com/tedsta/deeplearn-rs)

**Implementation of state-of-art models in Torch**

- github: [https://github.com/aciditeam/torch-models](https://github.com/aciditeam/torch-models)

**Deep Learning (Python, C, C++, Java, Scala, Go)**

- github: [https://github.com/yusugomori/DeepLearning](https://github.com/yusugomori/DeepLearning)

**deepmark: THE Deep Learning Benchmarks**

- github: [https://github.com/DeepMark/deepmark](https://github.com/DeepMark/deepmark)

**Siamese Net**

- intro: "This package shows how to train a siamese network using Lasagne and Theano and includes network definitions 
for state-of-the-art networks including: DeepID, DeepID2, Chopra et. al, and Hani et. al. 
We also include one pre-trained model using a custom convolutional network."
- github: [https://github.com/Kadenze/siamese_net](https://github.com/Kadenze/siamese_net)

**PRE-TRAINED CONVNETS AND OBJECT LOCALISATION IN KERAS**

- blog: [https://blog.heuritech.com/2016/04/26/pre-trained-convnets-and-object-localisation-in-keras/](https://blog.heuritech.com/2016/04/26/pre-trained-convnets-and-object-localisation-in-keras/)
- github: [https://github.com/heuritech/convnets-keras](https://github.com/heuritech/convnets-keras)

**Deep Learning algorithms with TensorFlow: Ready to use implementations of various Deep Learning algorithms using TensorFlow**

- homepage: [http://www.gabrieleangeletti.com/](http://www.gabrieleangeletti.com/)
- github: [https://github.com/blackecho/Deep-Learning-TensorFlow](https://github.com/blackecho/Deep-Learning-TensorFlow)

**Fast Multi-threaded VGG 19 Feature Extractor**

- github: [https://github.com/coreylynch/vgg-19-feature-extractor](https://github.com/coreylynch/vgg-19-feature-extractor)

**Live demo of neural network classifying images**

![](/assets/cnn-materials/nn_classify_images_live_demo.jpg)

[http://ml4a.github.io/dev/demos/cifar_confusion.html#](http://ml4a.github.io/dev/demos/cifar_confusion.html#)

**mojo cnn: c++ convolutional neural network**

- intro: the fast and easy header only c++ convolutional neural network package
- github: [https://github.com/gnawice/mojo-cnn](https://github.com/gnawice/mojo-cnn)

**DeepHeart: Neural networks for monitoring cardiac data**

- github: [https://github.com/jisaacso/DeepHeart](https://github.com/jisaacso/DeepHeart)

**Deep Water: Deep Learning in H2O using Native GPU Backends**

- intro: Native implementation of Deep Learning models for GPU backends (mxnet, Caffe, TensorFlow, etc.)
- github: [https://github.com/h2oai/deepwater](https://github.com/h2oai/deepwater)

**Greentea LibDNN: Greentea LibDNN - a universal convolution implementation supporting CUDA and OpenCL**

- github: [https://github.com/naibaf7/libdnn](https://github.com/naibaf7/libdnn)

**Dracula: A spookily good Part of Speech Tagger optimized for Twitter**

- intro: A deep, LSTM-based part of speech tagger and sentiment analyser using character embeddings instead of words. 
Compatible with Theano and TensorFlow. Optimized for Twitter.
- homepage: [http://dracula.sentimentron.co.uk/](http://dracula.sentimentron.co.uk/)
- speech tagging demo: [http://dracula.sentimentron.co.uk/pos-demo/](http://dracula.sentimentron.co.uk/pos-demo/)
- sentiment demo: [http://dracula.sentimentron.co.uk/sentiment-demo/](http://dracula.sentimentron.co.uk/sentiment-demo/)
- github: [https://github.com/Sentimentron/Dracula](https://github.com/Sentimentron/Dracula)

**Trained image classification models for Keras**

- intro: Keras code and weights files for popular deep learning models.
- intro: VGG16, VGG19, ResNet50, Inception v3
- github: [https://github.com/fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models)

**PyCNN: Cellular Neural Networks Image Processing Python Library**

![](https://camo.githubusercontent.com/0c5fd234a144b3d2145a133466766b2ecd9d3f3c/687474703a2f2f7777772e6973697765622e65652e6574687a2e63682f6861656e6767692f434e4e5f7765622f434e4e5f666967757265732f626c6f636b6469616772616d2e676966)

- blog: [http://blog.ankitaggarwal.me/PyCNN/](http://blog.ankitaggarwal.me/PyCNN/)
- github: [https://github.com/ankitaggarwal011/PyCNN](https://github.com/ankitaggarwal011/PyCNN)

**regl-cnn: Digit recognition with Convolutional Neural Networks in WebGL**

- intro: TensorFlow, WebGL, [regl](https://github.com/mikolalysenko/regl)
- github: [https://github.com/Erkaman/regl-cnn/](https://github.com/Erkaman/regl-cnn/)
- demo: [https://erkaman.github.io/regl-cnn/src/demo.html](https://erkaman.github.io/regl-cnn/src/demo.html)

**dagstudio: Directed Acyclic Graph Studio with Javascript D3**

![](https://raw.githubusercontent.com/TimZaman/dagstudio/master/misc/20160907_dagstudio_ex.gif)

- github: [https://github.com/TimZaman/dagstudio](https://github.com/TimZaman/dagstudio)

**NEUGO: Neural Networks in Go**

- github: [https://github.com/wh1t3w01f/neugo](https://github.com/wh1t3w01f/neugo)

**gvnn: Neural Network Library for Geometric Computer Vision**

- arxiv: [http://arxiv.org/abs/1607.07405](http://arxiv.org/abs/1607.07405)
- github: [https://github.com/ankurhanda/gvnn](https://github.com/ankurhanda/gvnn)

**DeepForge: A development environment for deep learning**

- github: [https://github.com/dfst/deepforge](https://github.com/dfst/deepforge)

**Implementation of recent Deep Learning papers**

- intro: DenseNet / DeconvNet / DenseRecNet
- github: [https://github.com/tdeboissiere/DeepLearningImplementations](https://github.com/tdeboissiere/DeepLearningImplementations)

**GPU-accelerated Theano & Keras on Windows 10 native**

- github: [https://github.com/philferriere/dlwin](https://github.com/philferriere/dlwin)

**Head Pose and Gaze Direction Estimation Using Convolutional Neural Networks**

- github: [https://github.com/mpatacchiola/deepgaze](https://github.com/mpatacchiola/deepgaze)

**Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)**

- homepage: [https://01.org/mkl-dnn](https://01.org/mkl-dnn)
- github: [https://github.com/01org/mkl-dnn](https://github.com/01org/mkl-dnn)

**Deep CNN and RNN - Deep convolution/recurrent neural network project with TensorFlow**

- github: [https://github.com/tobegit3hub/deep_cnn](https://github.com/tobegit3hub/deep_cnn)

**Experimental implementation of novel neural network structures**

- intro: binarynet / ternarynet / qrnn / vae / gcnn
- github: [https://github.com/DingKe/nn_playground](https://github.com/DingKe/nn_playground)

**WaterNet: A convolutional neural network that identifies water in satellite images**

- github: [https://github.com/treigerm/WaterNet](https://github.com/treigerm/WaterNet)

**Kur: Descriptive Deep Learning**

- github: [https://github.com/deepgram/kur](https://github.com/deepgram/kur)
- docs: [http://kur.deepgram.com/](http://kur.deepgram.com/)

**Development of JavaScript-based deep learning platform and application to distributed training**

- intro: Workshop paper for ICLR2017
- arxiv: [https://arxiv.org/abs/1702.01846](https://arxiv.org/abs/1702.01846)
- github: [https://github.com/mil-tokyo](https://github.com/mil-tokyo)

**NewralNet**

- intro: A lightweight, easy to use and open source Java library for experimenting with
feed-forward neural nets and deep learning.
- gitlab: [https://gitlab.com/flimmerkiste/NewralNet](https://gitlab.com/flimmerkiste/NewralNet)

**FeatherCNN**

- intro: FeatherCNN is a high performance inference engine for convolutional neural networks
- github: [https://github.com/Tencent/FeatherCNN](https://github.com/Tencent/FeatherCNN)

# Readings and Questions

**What you wanted to know about AI**

[http://fastml.com/what-you-wanted-to-know-about-ai/](http://fastml.com/what-you-wanted-to-know-about-ai/)

**Epoch vs iteration when training neural networks**

- stackoverflow: [http://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks](http://stackoverflow.com/questions/4752626/epoch-vs-iteration-when-training-neural-networks)

**Questions to Ask When Applying Deep Learning**

[http://deeplearning4j.org/questions.html](http://deeplearning4j.org/questions.html)

**How can I know if Deep Learning works better for a specific problem than SVM or random forest?**

- github: [https://github.com/rasbt/python-machine-learning-book/blob/master/faq/deeplearn-vs-svm-randomforest.md](https://github.com/rasbt/python-machine-learning-book/blob/master/faq/deeplearn-vs-svm-randomforest.md)

**What is the difference between deep learning and usual machine learning?**

- note: [https://github.com/rasbt/python-machine-learning-book/blob/master/faq/difference-deep-and-normal-learning.md](https://github.com/rasbt/python-machine-learning-book/blob/master/faq/difference-deep-and-normal-learning.md)

# Resources

**Awesome Deep Learning**

- github: [https://github.com/ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning)

**Awesome-deep-vision: A curated list of deep learning resources for computer vision**

- website: [http://jiwonkim.org/awesome-deep-vision/](http://jiwonkim.org/awesome-deep-vision/)
- github: [https://github.com/kjw0612/awesome-deep-vision](https://github.com/kjw0612/awesome-deep-vision)

**Applied Deep Learning Resources: A collection of research articles, blog posts, slides and code snippets about deep learning in applied settings.**

- github: [https://github.com/kristjankorjus/applied-deep-learning-resources](https://github.com/kristjankorjus/applied-deep-learning-resources)

**Deep Learning Libraries by Language**

- website: [http://www.teglor.com/b/deep-learning-libraries-language-cm569/](http://www.teglor.com/b/deep-learning-libraries-language-cm569/)

**Deep Learning Resources**

[http://yanirseroussi.com/deep-learning-resources/](http://yanirseroussi.com/deep-learning-resources/)

**Deep Learning Resources**

[https://omtcyfz.github.io/2016/08/29/Deep-Learning-Resources.html](https://omtcyfz.github.io/2016/08/29/Deep-Learning-Resources.html)

**Turing Machine: musings on theory & code(DEEP LEARNING REVOLUTION, summer 2015, state of the art & topnotch links)**

[https://vzn1.wordpress.com/2015/09/01/deep-learning-revolution-summer-2015-state-of-the-art-topnotch-links/](https://vzn1.wordpress.com/2015/09/01/deep-learning-revolution-summer-2015-state-of-the-art-topnotch-links/)

**BICV Group: Biologically Inspired Computer Vision research group**

[http://www.bicv.org/deep-learning/](http://www.bicv.org/deep-learning/)

**Learning Deep Learning**

[http://rt.dgyblog.com/ref/ref-learning-deep-learning.html](http://rt.dgyblog.com/ref/ref-learning-deep-learning.html)

**Summaries and notes on Deep Learning research papers**

- github: [https://github.com/dennybritz/deeplearning-papernotes](https://github.com/dennybritz/deeplearning-papernotes)

**Deep Learning Glossary**

- intro: "Simple, opinionated explanations of various things encountered in Deep Learning / AI / ML."
- author: Ryan Dahl, author of NodeJS. 
- github: [https://github.com/ry/deep_learning_glossary](https://github.com/ry/deep_learning_glossary)

**The Deep Learning Playbook**

[https://medium.com/@jiefeng/deep-learning-playbook-c5ebe34f8a1a#.eg9cdz5ak](https://medium.com/@jiefeng/deep-learning-playbook-c5ebe34f8a1a#.eg9cdz5ak)

**Deep Learning Study: Study of HeXA@UNIST in Preparation for Submission**

- github: [https://github.com/carpedm20/deep-learning-study](https://github.com/carpedm20/deep-learning-study)

**Deep Learning Books**

- blog: [http://machinelearningmastery.com/deep-learning-books/](http://machinelearningmastery.com/deep-learning-books/)

**awesome-very-deep-learning: A curated list of papers and code about very deep neural networks (50+ layers)**

- github: [https://github.com/daviddao/awesome-very-deep-learning](https://github.com/daviddao/awesome-very-deep-learning)

**Deep Learning Resources and Tutorials using Keras and Lasagne**

- github: [https://github.com/Vict0rSch/deep_learning](https://github.com/Vict0rSch/deep_learning)

**Deep Learning: Definition, Resources, Comparison with Machine Learning**

- blog: [http://www.datasciencecentral.com/profiles/blogs/deep-learning-definition-resources-comparison-with-machine-learni](http://www.datasciencecentral.com/profiles/blogs/deep-learning-definition-resources-comparison-with-machine-learni)

**Awesome - Most Cited Deep Learning Papers**

- github: [https://github.com/terryum/awesome-deep-learning-papers](https://github.com/terryum/awesome-deep-learning-papers)

**The most cited papers in computer vision and deep learning**

- blog: [https://computervisionblog.wordpress.com/2016/06/19/the-most-cited-papers-in-computer-vision-and-deep-learning/](https://computervisionblog.wordpress.com/2016/06/19/the-most-cited-papers-in-computer-vision-and-deep-learning/)

**deep learning papers: A place to collect papers that are related to deep learning and computational biology**

- github: [https://github.com/pimentel/deep_learning_papers](https://github.com/pimentel/deep_learning_papers)

**papers-I-read**

- intro: "I am trying a new initiative - a-paper-a-week. This repository will hold all those papers and related summaries and notes."
- github: [https://github.com/shagunsodhani/papers-I-read](https://github.com/shagunsodhani/papers-I-read)

**LEARNING DEEP LEARNING - MY TOP-FIVE LIST**

- blog: [http://thegrandjanitor.com/2016/08/15/learning-deep-learning-my-top-five-resource/](http://thegrandjanitor.com/2016/08/15/learning-deep-learning-my-top-five-resource/)

**awesome-free-deep-learning-papers**

- github: [https://github.com/HFTrader/awesome-free-deep-learning-papers](https://github.com/HFTrader/awesome-free-deep-learning-papers)

**DeepLearningBibliography: Bibliography for Publications about Deep Learning using GPU**

- homepage: [http://memkite.com/deep-learning-bibliography/](http://memkite.com/deep-learning-bibliography/)
- github: [https://github.com/memkite/DeepLearningBibliography](https://github.com/memkite/DeepLearningBibliography)

**Deep Learning Papers Reading Roadmap**

- github: [https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap](https://github.com/songrotek/Deep-Learning-Papers-Reading-Roadmap)

**deep-learning-papers**

- intro: Papers about deep learning ordered by task, date. Current state-of-the-art papers are labelled.
- github: [https://github.com/sbrugman/deep-learning-papers/blob/master/README.md](https://github.com/sbrugman/deep-learning-papers/blob/master/README.md)

**Deep Learning and applications in Startups, CV, Text Mining, NLP**

- github: [https://github.com/lipiji/app-dl](https://github.com/lipiji/app-dl)

**ml4a-guides - a collection of practical resources for working with machine learning software, including code and tutorials**

[http://ml4a.github.io/guides/](http://ml4a.github.io/guides/)

**deep-learning-resources**

- intro: A Collection of resources I have found useful on my journey finding my way through the world of Deep Learning.
- github: [https://github.com/chasingbob/deep-learning-resources](https://github.com/chasingbob/deep-learning-resources)

**21 Deep Learning Videos, Tutorials & Courses on Youtube from 2016**

[https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/](https://www.analyticsvidhya.com/blog/2016/12/21-deep-learning-videos-tutorials-courses-on-youtube-from-2016/)

**Awesome Deep learning papers and other resources**

- github: [https://github.com/endymecy/awesome-deeplearning-resources](https://github.com/endymecy/awesome-deeplearning-resources)

**awesome-deep-vision-web-demo**

- intro: A curated list of awesome deep vision web demo
- github: [https://github.com/hwalsuklee/awesome-deep-vision-web-demo](https://github.com/hwalsuklee/awesome-deep-vision-web-demo)

**Summaries of machine learning papers**

[https://github.com/aleju/papers](https://github.com/aleju/papers)

**Awesome Deep Learning Resources**

[https://github.com/guillaume-chevalier/awesome-deep-learning-resources](https://github.com/guillaume-chevalier/awesome-deep-learning-resources)

**Virginia Tech Vision and Learning Reading Group**

[https://github.com//vt-vl-lab/reading_group](https://github.com//vt-vl-lab/reading_group)

**MEGALODON: ML/DL Resources At One Place**

- intro: Various ML/DL Resources organised at a single place.
- arxiv: [https://github.com//vyraun/Megalodon](https://github.com//vyraun/Megalodon)

## Arxiv Pages

**Neural and Evolutionary Computing**

[https://arxiv.org/list/cs.NE/recent](https://arxiv.org/list/cs.NE/recent)

**Learning**

[https://arxiv.org/list/cs.LG/recent](https://arxiv.org/list/cs.LG/recent)

**Computer Vision and Pattern Recognition**

[https://arxiv.org/list/cs.CV/recent](https://arxiv.org/list/cs.CV/recent)

## Arxiv Sanity Preserver

- intro: Built by @karpathy to accelerate research.
- page: [http://www.arxiv-sanity.com/](http://www.arxiv-sanity.com/)

**Today's Deep Learning**

[http://todaysdeeplearning.com/](http://todaysdeeplearning.com/)

**arXiv Analytics**

[http://arxitics.com/](http://arxitics.com/)

## Papers with Code

**Papers with Code**

[https://paperswithcode.com/](https://paperswithcode.com/)

# Tools

**DNNGraph - A deep neural network model generation DSL in Haskell**

- homepage: [http://ajtulloch.github.io/dnngraph/](http://ajtulloch.github.io/dnngraph/)

**Deep playground: an interactive visualization of neural networks, written in typescript using d3.js**

- homepage: [http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.23990&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.23990&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification)
- github: [https://github.com/tensorflow/playground](https://github.com/tensorflow/playground)

**Neural Network Package**

- intro: This package provides an easy and modular way to build and train simple or complex neural networks using Torch
- github: [https://github.com/torch/nn](https://github.com/torch/nn)

**deepdish: Deep learning and data science tools from the University of Chicago**
**deepdish: Serving Up Chicago-Style Deep Learning**

- homepage: [http://deepdish.io/](http://deepdish.io/)
- github: [https://github.com/uchicago-cs/deepdish](https://github.com/uchicago-cs/deepdish)

**AETROS CLI: Console application to manage deep neural network training in AETROS Trainer**

- intro: Create, train and monitor deep neural networks using a model designer.
- homepage: [http://aetros.com/](http://aetros.com/)
- github: [https://github.com/aetros/aetros-cli](https://github.com/aetros/aetros-cli)

**Deep Learning Studio: Cloud platform for designing Deep Learning AI without programming**

[http://deepcognition.ai/](http://deepcognition.ai/)

**cuda-on-cl: Build NVIDIA® CUDA™ code for OpenCL™ 1.2 devices**

- github: [https://github.com/hughperkins/cuda-on-cl](https://github.com/hughperkins/cuda-on-cl)

**Receptive Field Calculator**

- homepage: [http://fomoro.com/tools/receptive-fields/](http://fomoro.com/tools/receptive-fields/)
- example: [http://fomoro.com/tools/receptive-fields/#3,1,1,VALID;3,1,1,VALID;3,1,1,VALID](http://fomoro.com/tools/receptive-fields/#3,1,1,VALID;3,1,1,VALID;3,1,1,VALID)

**receptivefield**

- intro: (PyTorch/Keras/TensorFlow)Gradient based receptive field estimation for Convolutional Neural Networks
- github: [https://github.com//fornaxai/receptivefield](https://github.com//fornaxai/receptivefield)

# Challenges / Hackathons

**Open Images Challenge 2018**

[https://storage.googleapis.com/openimages/web/challenge.html](https://storage.googleapis.com/openimages/web/challenge.html)

**VisionHack 2017**

- intro: 10 - 14 Sep 2017, Moscow, Russia
- intro: a full-fledged hackathon that will last three full days
- homepage: [http://visionhack.misis.ru/](http://visionhack.misis.ru/)

**NVIDIA AI City Challenge Workshop at CVPR 2018**

[http://www.aicitychallenge.org/](http://www.aicitychallenge.org/)

# Books

**Deep Learning**

- author: Ian Goodfellow, Aaron Courville and Yoshua Bengio
- homepage: [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/)
- website: [http://goodfeli.github.io/dlbook/](http://goodfeli.github.io/dlbook/)
- github: [https://github.com/HFTrader/DeepLearningBook](https://github.com/HFTrader/DeepLearningBook)
- notes("Deep Learning for Beginners"): [http://randomekek.github.io/deep/deeplearning.html](http://randomekek.github.io/deep/deeplearning.html)

**Fundamentals of Deep Learning: Designing Next-Generation Artificial Intelligence Algorithms**

- author: Nikhil Buduma
- book review: [http://www.opengardensblog.futuretext.com/archives/2015/08/book-review-fundamentals-of-deep-learning-designing-next-generation-artificial-intelligence-algorithms-by-nikhil-buduma.html](http://www.opengardensblog.futuretext.com/archives/2015/08/book-review-fundamentals-of-deep-learning-designing-next-generation-artificial-intelligence-algorithms-by-nikhil-buduma.html)
- github: [https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)

**FIRST CONTACT WITH TENSORFLOW: Get started with with Deep Learning programming**

- author: Jordi Torres
- book: [http://www.jorditorres.org/first-contact-with-tensorflow/](http://www.jorditorres.org/first-contact-with-tensorflow/)

**《解析卷积神经网络—深度学习实践手册》**

- intro: by 魏秀参（Xiu-Shen WEI）
- homepage: [http://lamda.nju.edu.cn/weixs/book/CNN_book.html](http://lamda.nju.edu.cn/weixs/book/CNN_book.html)

**Make Your Own Neural Network: IPython Neural Networks on a Raspberry Pi Zero**

- book: [http://makeyourownneuralnetwork.blogspot.jp/2016/03/ipython-neural-networks-on-raspberry-pi.html](http://makeyourownneuralnetwork.blogspot.jp/2016/03/ipython-neural-networks-on-raspberry-pi.html)
- github: [https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork](https://github.com/makeyourownneuralnetwork/makeyourownneuralnetwork)

# Blogs

**Neural Networks and Deep Learning**

[http://neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com)

**Deep Learning Reading List**

[http://deeplearning.net/reading-list/](http://deeplearning.net/reading-list/)

**WILDML: A BLOG ABOUT MACHINE LEARNING, DEEP LEARNING AND NLP.**

[http://www.wildml.com/](http://www.wildml.com/)

**Andrej Karpathy blog**

[http://karpathy.github.io/](http://karpathy.github.io/)

**Rodrigob's github page**

[http://rodrigob.github.io/](http://rodrigob.github.io/)

**colah's blog**

[http://colah.github.io/](http://colah.github.io/)

**What My Deep Model Doesn't Know...**

[http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html](http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html)

**Christoph Feichtenhofer**

- intro: PhD Student, Graz University of Technology
- homepage: [http://feichtenhofer.github.io/](http://feichtenhofer.github.io/)

**Image recognition is not enough: As with language, photos need contextual intelligence**

[https://medium.com/@ken_getquik/image-recognition-is-not-enough-293cd7d58004#.dex817l2z](https://medium.com/@ken_getquik/image-recognition-is-not-enough-293cd7d58004#.dex817l2z)

**ResNets, HighwayNets, and DenseNets, Oh My!**

- blog: [https://medium.com/@awjuliani/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32#.pgltg8pro](https://medium.com/@awjuliani/resnets-highwaynets-and-densenets-oh-my-9bb15918ee32#.pgltg8pro)
- github: [https://github.com/awjuliani/TF-Tutorials/blob/master/Deep%20Network%20Comparison.ipynb](https://github.com/awjuliani/TF-Tutorials/blob/master/Deep%20Network%20Comparison.ipynb)

**The Frontiers of Memory and Attention in Deep Learning**

- sldies: [http://slides.com/smerity/quora-frontiers-of-memory-and-attention#/](http://slides.com/smerity/quora-frontiers-of-memory-and-attention#/)

**Design Patterns for Deep Learning Architectures**

[http://www.deeplearningpatterns.com/doku.php](http://www.deeplearningpatterns.com/doku.php)

**Building a Deep Learning Powered GIF Search Engine**

- blog: [https://medium.com/@zan2434/building-a-deep-learning-powered-gif-search-engine-a3eb309d7525](https://medium.com/@zan2434/building-a-deep-learning-powered-gif-search-engine-a3eb309d7525)

**850k Images in 24 hours: Automating Deep Learning Dataset Creation**

[https://gab41.lab41.org/850k-images-in-24-hours-automating-deep-learning-dataset-creation-60bdced04275#.xhq9feuxx](https://gab41.lab41.org/850k-images-in-24-hours-automating-deep-learning-dataset-creation-60bdced04275#.xhq9feuxx)

**How six lines of code + SQL Server can bring Deep Learning to ANY App**

- blog: [https://blogs.technet.microsoft.com/dataplatforminsider/2017/01/05/how-six-lines-of-code-sql-server-can-bring-deep-learning-to-any-app/](https://blogs.technet.microsoft.com/dataplatforminsider/2017/01/05/how-six-lines-of-code-sql-server-can-bring-deep-learning-to-any-app/)
- github: [https://github.com/Microsoft/SQL-Server-R-Services-Samples/tree/master/Galaxies](https://github.com/Microsoft/SQL-Server-R-Services-Samples/tree/master/Galaxies)

**Neural Network Architectures**

![](https://culurciello.github.io/assets/nets/acc_vs_net_vs_ops.svg)

- blog: [https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba#.m8y39oih6](https://medium.com/towards-data-science/neural-network-architectures-156e5bad51ba#.m8y39oih6)
- blog: [https://culurciello.github.io/tech/2016/06/04/nets.html](https://culurciello.github.io/tech/2016/06/04/nets.html)
