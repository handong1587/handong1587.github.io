---
layout: post
category: deep_learning
title: Deep Learning Resources
date: 2015-10-09
---

# ImageNet

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

- arxiv: [http://arxiv.org/abs/1312.4400](http://arxiv.org/abs/1312.4400)
- gitxiv: [http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin](http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin)

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
- github(Torch): [https://github.com/Moodstocks/inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch)

**Inception in TensorFlow**

- intro: demonstrate how to train the Inception v3 architecture
- github: [https://github.com/tensorflow/models/tree/master/inception](https://github.com/tensorflow/models/tree/master/inception)

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
- github: [https://github.com/alrojo/lasagne_residual_network](https://github.com/alrojo/lasagne_residual_network)
- github: [https://github.com/shuokay/resnet](https://github.com/shuokay/resnet)
- github: [https://github.com/gcr/torch-residual-networks](https://github.com/gcr/torch-residual-networks)
- github: [https://github.com/apark263/cfmz](https://github.com/apark263/cfmz)
- github: [https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_msra.py](https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_msra.py)
- github: [https://github.com/yasunorikudo/chainer-ResNet](https://github.com/yasunorikudo/chainer-ResNet)
- github: [https://github.com/raghakot/keras-resnet](https://github.com/raghakot/keras-resnet)
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

**Resnet in Resnet: Generalizing Residual Architectures**

- paper: [http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g](http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g)
- arxiv: [http://arxiv.org/abs/1603.08029](http://arxiv.org/abs/1603.08029)

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
- github: [https://github.com/bazilas/matconvnet-ResNet](https://github.com/bazilas/matconvnet-ResNet)
- github: [https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne](https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne)
- github: [https://github.com/tornadomeet/ResNet](https://github.com/tornadomeet/ResNet)

**Residual Networks are Exponential Ensembles of Relatively Shallow Networks**

- arxiv: [http://arxiv.org/abs/1605.06431](http://arxiv.org/abs/1605.06431)

**Wide Residual Networks**

- intro: BMVC 2016
- arxiv: [http://arxiv.org/abs/1605.07146](http://arxiv.org/abs/1605.07146)
- github: [https://github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)
- github: [https://github.com/asmith26/wide_resnets_keras](https://github.com/asmith26/wide_resnets_keras)

**Deep Residual Networks for Image Classification with Python + NumPy**

![](https://dnlcrl.github.io/assets/thesis-post/Diagramma.png)

- blog: [https://dnlcrl.github.io/projects/2016/06/22/Deep-Residual-Networks-for-Image-Classification-with-Python+NumPy.html](https://dnlcrl.github.io/projects/2016/06/22/Deep-Residual-Networks-for-Image-Classification-with-Python+NumPy.html)

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

## Inception-V4 / Inception-ResNet-V2

**Inception-V4, Inception-Resnet And The Impact Of Residual Connections On Learning**

- intro: Workshop track - ICLR 2016. 3.08 % top-5 error on ImageNet CLS
- intro: "achieve 3.08% top-5 error on the test set of the ImageNet classification (CLS) challenge"
- arxiv: [http://arxiv.org/abs/1602.07261](http://arxiv.org/abs/1602.07261)

**The inception-resnet-v2 models trained from scratch via torch**

- github: [https://github.com/lim0606/torch-inception-resnet-v2](https://github.com/lim0606/torch-inception-resnet-v2)

## ResNeXt

**Aggregated Residual Transformations for Deep Neural Networks**

- intro: UC San Diego & Facebook AI Research
- arxiv: [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
- github: [https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol/resnext.py](https://github.com/dmlc/mxnet/blob/master/example/image-classification/symbol/resnext.py)
- github(Caffe): [https://github.com/terrychenism/ResNeXt](https://github.com/terrychenism/ResNeXt)
- dataset: [http://data.dmlc.ml/models/imagenet/resnext/](http://data.dmlc.ml/models/imagenet/resnext/)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/5haml9/p_implementation_of_aggregated_residual/](https://www.reddit.com/r/MachineLearning/comments/5haml9/p_implementation_of_aggregated_residual/)

**Deep Pyramidal Residual Networks with Separated Stochastic Depth**

- arxiv: [https://arxiv.org/abs/1612.01230](https://arxiv.org/abs/1612.01230)

**Spatially Adaptive Computation Time for Residual Networks**

- intro: Higher School of Economics & Google & CMU
- arxiv: [https://arxiv.org/abs/1612.02297](https://arxiv.org/abs/1612.02297)

# Deep Learning And Bayesian

**Scalable Bayesian Optimization Using Deep Neural Networks**

- intro: ICML 2015
- paper: [http://jmlr.org/proceedings/papers/v37/snoek15.html](http://jmlr.org/proceedings/papers/v37/snoek15.html)
- arxiv: [http://arxiv.org/abs/1502.05700](http://arxiv.org/abs/1502.05700)
- github: [https://github.com/bshahr/torch-dngo](https://github.com/bshahr/torch-dngo)

**Bayesian Dark Knowledge**

- paper: [http://arxiv.org/abs/1506.04416](http://arxiv.org/abs/1506.04416)
- notes: [Notes on Bayesian Dark Knowledge](https://www.evernote.com/shard/s189/sh/92cc4cbf-285e-4038-af08-c6d9e4aee6ea/d505237e82dc81be9859bc82f3902f9f)

**Memory-based Bayesian Reasoning with Deep Learning**

- intro: Google DeepMind
- slides: [http://blog.shakirm.com/wp-content/uploads/2015/11/CSML_BayesDeep.pdf](http://blog.shakirm.com/wp-content/uploads/2015/11/CSML_BayesDeep.pdf)

**Towards Bayesian Deep Learning: A Survey**

- arxiv: [http://arxiv.org/abs/1604.01662](http://arxiv.org/abs/1604.01662)

**Towards Bayesian Deep Learning: A Framework and Some Existing Methods**

- intro: IEEE Transactions on Knowledge and Data Engineering (TKDE), 2016
- arxiv: [http://arxiv.org/abs/1608.06884](http://arxiv.org/abs/1608.06884)

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

**Semi-supervised Knowledge Transfer for Deep Learning from Private Training Data**

- arxiv: [https://arxiv.org/abs/1610.05755](https://arxiv.org/abs/1610.05755)
- github: [https://github.com/tensorflow/models/tree/master/differential_privacy/multiple_teachers](https://github.com/tensorflow/models/tree/master/differential_privacy/multiple_teachers)

**Infinite Variational Autoencoder for Semi-Supervised Learning**

- arxiv: [https://arxiv.org/abs/1611.07800](https://arxiv.org/abs/1611.07800)

# Transfer Learning

**How transferable are features in deep neural networks?**

- intro: NIPS 2014
- arxiv: [http://arxiv.org/abs/1411.1792](http://arxiv.org/abs/1411.1792)
- paper: [http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf](http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
- github: [https://github.com/yosinski/convnet_transfer](https://github.com/yosinski/convnet_transfer)

**Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks**

- paper: [http://research.microsoft.com/pubs/214307/paper.pdf](http://research.microsoft.com/pubs/214307/paper.pdf)

**Transferring Knowledge from a RNN to a DNN**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1504.01483](https://arxiv.org/abs/1504.01483)

**Simultaneous Deep Transfer Across Domains and Tasks**

- intro: ICCV 2015
- arxiv: [http://arxiv.org/abs/1510.02192](http://arxiv.org/abs/1510.02192)

**Net2Net: Accelerating Learning via Knowledge Transfer**

- arxiv: [http://arxiv.org/abs/1511.05641](http://arxiv.org/abs/1511.05641)
- github: [https://github.com/soumith/net2net.torch](https://github.com/soumith/net2net.torch)
- notes(by Hugo Larochelle): [https://www.evernote.com/shard/s189/sh/46414718-9663-440e-bbb7-65126b247b42/19688c438709251d8275d843b8158b03](https://www.evernote.com/shard/s189/sh/46414718-9663-440e-bbb7-65126b247b42/19688c438709251d8275d843b8158b03)

**Transfer Learning from Deep Features for Remote Sensing and Poverty Mapping**

- arxiv: [http://arxiv.org/abs/1510.00098](http://arxiv.org/abs/1510.00098)

**A theoretical framework for deep transfer learning**

- key words: transfer learning, PAC learning, PAC-Bayesian, deep learning
- homepage: [http://imaiai.oxfordjournals.org/content/early/2016/04/28/imaiai.iaw008](http://imaiai.oxfordjournals.org/content/early/2016/04/28/imaiai.iaw008)
- paper: [http://imaiai.oxfordjournals.org/content/early/2016/04/28/imaiai.iaw008.full.pdf](http://imaiai.oxfordjournals.org/content/early/2016/04/28/imaiai.iaw008.full.pdf)

**Transfer learning using neon**

- blog: [http://www.nervanasys.com/transfer-learning-using-neon/](http://www.nervanasys.com/transfer-learning-using-neon/)

**Hyperparameter Transfer Learning through Surrogate Alignment for Efficient Deep Neural Network Training**

- arxiv: [http://arxiv.org/abs/1608.00218](http://arxiv.org/abs/1608.00218)

**What makes ImageNet good for transfer learning?**

- project page: [http://minyounghuh.com/papers/analysis/](http://minyounghuh.com/papers/analysis/)
- arxiv: [http://arxiv.org/abs/1608.08614](http://arxiv.org/abs/1608.08614)

**Fine-tuning a Keras model using Theano trained Neural Network & Introduction to Transfer Learning**

- github: [https://www.analyticsvidhya.com/blog/2016/11/fine-tuning-a-keras-model-using-theano-trained-neural-network-introduction-to-transfer-learning/](https://www.analyticsvidhya.com/blog/2016/11/fine-tuning-a-keras-model-using-theano-trained-neural-network-introduction-to-transfer-learning/)

**Multi-source Transfer Learning with Convolutional Neural Networks for Lung Pattern Analysis**

- arxiv: [https://arxiv.org/abs/1612.02589](https://arxiv.org/abs/1612.02589)

# Multi-label Learning

**CNN: Single-label to Multi-label**

- arxiv: [http://arxiv.org/abs/1406.5726](http://arxiv.org/abs/1406.5726)

**Deep Learning for Multi-label Classification**

- arxiv: [http://arxiv.org/abs/1502.05988](http://arxiv.org/abs/1502.05988)
- github: [http://meka.sourceforge.net](http://meka.sourceforge.net)

**Predicting Unseen Labels using Label Hierarchies in Large-Scale Multi-label Learning(ECML2015)**

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

# Metric Learning

**Learnable Structured Clustering Framework for Deep Metric Learning**

- arxiv: [https://arxiv.org/abs/1612.01213](https://arxiv.org/abs/1612.01213)

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

# Adversarial Examples of Deep Learning

**Intriguing properties of neural networks**

- arxiv: [http://arxiv.org/abs/1312.6199](http://arxiv.org/abs/1312.6199)
- my notes: In each layer of a deep network it is the "direction" of "space" (ensemble of feature activations) 
which encodes useful class information rather than individual units (feature activations).

**Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images**

- arxiv: [http://arxiv.org/abs/1412.1897](http://arxiv.org/abs/1412.1897)
- github: [https://github.com/Evolving-AI-Lab/fooling/](https://github.com/Evolving-AI-Lab/fooling/)

**Explaining and Harnessing Adversarial Examples**

- intro: primary cause of neural networks’ vulnerability to adversarial perturbation is their linear nature
- arxiv: [http://arxiv.org/abs/1412.6572](http://arxiv.org/abs/1412.6572)

**Distributional Smoothing with Virtual Adversarial Training**

- arxiv: [http://arxiv.org/abs/1507.00677](http://arxiv.org/abs/1507.00677)
- github: [https://github.com/takerum/vat](https://github.com/takerum/vat)

**Confusing Deep Convolution Networks by Relabelling**

- arxiv: [http://arxiv.org/abs/1510.06925v1](http://arxiv.org/abs/1510.06925v1)

**Exploring the Space of Adversarial Images**

- arxiv: [http://arxiv.org/abs/1510.05328](http://arxiv.org/abs/1510.05328)
- github: [https://github.com/tabacof/adversarial](https://github.com/tabacof/adversarial)

**Learning with a Strong Adversary**

- arxiv: [http://arxiv.org/abs/1511.03034](http://arxiv.org/abs/1511.03034)

**Adversarial examples in the physical world**

- author: Alexey Kurakin, Ian Goodfellow, Samy Bengio. Google Brain & OpenAI
- arxiv: [http://arxiv.org/abs/1607.02533](http://arxiv.org/abs/1607.02533)

**DeepFool: a simple and accurate method to fool deep neural networks**

- arxiv: [http://arxiv.org/abs/1511.04599](http://arxiv.org/abs/1511.04599)
- github: [https://github.com/LTS4/DeepFool](https://github.com/LTS4/DeepFool)

**Adversarial Autoencoders**

- arxiv: [http://arxiv.org/abs/1511.05644](http://arxiv.org/abs/1511.05644)
- slides: [https://docs.google.com/presentation/d/1Lyp91JOSzXo0Kk8gPdgyQUDuqLV_PnSzJh7i5c8ZKjs/edit?pref=2&pli=1](https://docs.google.com/presentation/d/1Lyp91JOSzXo0Kk8gPdgyQUDuqLV_PnSzJh7i5c8ZKjs/edit?pref=2&pli=1)
- notes(by Dustin Tran): [http://dustintran.com/blog/adversarial-autoencoders/](http://dustintran.com/blog/adversarial-autoencoders/)
- TFD manifold: [http://www.comm.utoronto.ca/~makhzani/adv_ae/tfd.gif](http://www.comm.utoronto.ca/~makhzani/adv_ae/tfd.gif)
- SVHN style manifold: [http://www.comm.utoronto.ca/~makhzani/adv_ae/svhn.gif](http://www.comm.utoronto.ca/~makhzani/adv_ae/svhn.gif)

**Understanding Adversarial Training: Increasing Local Stability of Neural Nets through Robust Optimization**

- arxiv: [http://arxiv.org/abs/1511.05432](http://arxiv.org/abs/1511.05432)
- github: [https://github.com/yutaroyamada/RobustTraining](https://github.com/yutaroyamada/RobustTraining)

**(Deep Learning’s Deep Flaws)’s Deep Flaws (By Zachary Chase Lipton)**

- blog: [http://www.kdnuggets.com/2015/01/deep-learning-flaws-universal-machine-learning.html](http://www.kdnuggets.com/2015/01/deep-learning-flaws-universal-machine-learning.html)

**Deep Learning Adversarial Examples – Clarifying Misconceptions**

- intro: By Ian Goodfellow, Google
- blog: [http://www.kdnuggets.com/2015/07/deep-learning-adversarial-examples-misconceptions.html](http://www.kdnuggets.com/2015/07/deep-learning-adversarial-examples-misconceptions.html)

**Adversarial Machines: Fooling A.Is (and turn everyone into a Manga)**

- blog: [https://medium.com/@samim/adversarial-machines-998d8362e996#.iv3muefgt](https://medium.com/@samim/adversarial-machines-998d8362e996#.iv3muefgt)

**How to trick a neural network into thinking a panda is a vulture**

- blog: [https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture](https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture)

**Assessing Threat of Adversarial Examples on Deep Neural Networks**

- intro: pre-print version to appear in IEEE ICMLA 2016
- arxiv: [https://arxiv.org/abs/1610.04256](https://arxiv.org/abs/1610.04256)

**Safety Verification of Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1610.06940](https://arxiv.org/abs/1610.06940)

**Adversarial Machine Learning at Scale**

- intro: Google Brain & OpenAI
- arxiv: [https://arxiv.org/abs/1611.01236](https://arxiv.org/abs/1611.01236)

# Deep Learning Networks

**PCANet: A Simple Deep Learning Baseline for Image Classification?**

- arixv: [http://arxiv.org/abs/1404.3606](http://arxiv.org/abs/1404.3606)
- code(Matlab): [http://mx.nthu.edu.tw/~tsunghan/download/PCANet_demo_pyramid.rar](http://mx.nthu.edu.tw/~tsunghan/download/PCANet_demo_pyramid.rar)
- mirror: [http://pan.baidu.com/s/1mg24b3a](http://pan.baidu.com/s/1mg24b3a)
- github(C++): [https://github.com/Ldpe2G/PCANet](https://github.com/Ldpe2G/PCANet)

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

- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

**Pointer Networks**

- arxiv: [https://arxiv.org/abs/1506.03134](https://arxiv.org/abs/1506.03134)
- github: [https://github.com/vshallc/PtrNets](https://github.com/vshallc/PtrNets)
- github(TensorFlow): [https://github.com/ikostrikov/TensorFlow-Pointer-Networks](https://github.com/ikostrikov/TensorFlow-Pointer-Networks)
- notes: [https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/pointer-networks.md](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/pointer-networks.md)

**Pointer Networks in TensorFlow (with sample code)**

- blog: [https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264#.sxipqfj30](https://medium.com/@devnag/pointer-networks-in-tensorflow-with-sample-code-14645063f264#.sxipqfj30)
- github: [https://github.com/devnag/tensorflow-pointer-networks](https://github.com/devnag/tensorflow-pointer-networks)

**Rectified Factor Networks**

- arxiv: [http://arxiv.org/abs/1502.06464](http://arxiv.org/abs/1502.06464)
- github: [https://github.com/untom/librfn](https://github.com/untom/librfn)

**FlowNet: Learning Optical Flow with Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1504.06852](http://arxiv.org/abs/1504.06852)

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

- arxiv: [http://arxiv.org/abs/1511.07356](http://arxiv.org/abs/1511.07356)
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

**Value Iteration Networks**

![](https://raw.githubusercontent.com/karpathy/paper-notes/master/img/vin/Screen%20Shot%202016-08-13%20at%204.58.42%20PM.png)

- intro: NIPS 2016. University of California, Berkeley
- arxiv: [http://arxiv.org/abs/1602.02867](http://arxiv.org/abs/1602.02867)
- github: [https://github.com/avivt/VIN](https://github.com/avivt/VIN)
- notes(by Andrej Karpathy): [https://github.com/karpathy/paper-notes/blob/master/vin.md](https://github.com/karpathy/paper-notes/blob/master/vin.md)

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

**Convolutional Neural Fabrics**

- arxiv: [http://arxiv.org/abs/1606.02492](http://arxiv.org/abs/1606.02492)

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

**Factorized Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.04337](http://arxiv.org/abs/1608.04337)

**Mollifying Networks**

- author: Caglar Gulcehre, Marcin Moczulski, Francesco Visin, Yoshua Bengio
- arxiv: [http://arxiv.org/abs/1608.04980](http://arxiv.org/abs/1608.04980)

**Local Binary Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.06049](http://arxiv.org/abs/1608.06049)

**Densely Connected Convolutional Networks**

![](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)

- intro: Cornell University & Tsinghua University. DenseNet
- arxiv: [http://arxiv.org/abs/1608.06993](http://arxiv.org/abs/1608.06993)
- github: [https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)
- github(Lasagne): [https://github.com/Lasagne/Recipes/tree/master/papers/densenet](https://github.com/Lasagne/Recipes/tree/master/papers/densenet)
- github(Keras): [https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/DenseNet)
- github(Caffe): [https://github.com/liuzhuang13/DenseNetCaffe](https://github.com/liuzhuang13/DenseNetCaffe)
- github(Tensorflow): [https://github.com/YixuanLi/densenet-tensorflow](https://github.com/YixuanLi/densenet-tensorflow)
- github(Keras): [https://github.com/titu1994/DenseNet](https://github.com/titu1994/DenseNet)

**CliqueCNN: Deep Unsupervised Exemplar Learning**

- intro: NIPS 2016
- arxiv: [http://arxiv.org/abs/1608.08792](http://arxiv.org/abs/1608.08792)
- github: [https://github.com/asanakoy/cliquecnn](https://github.com/asanakoy/cliquecnn)

**Convexified Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1609.01000](http://arxiv.org/abs/1609.01000)

**Multi-scale brain networks**

- arxiv: [http://arxiv.org/abs/1608.08828](http://arxiv.org/abs/1608.08828)

**Warped Convolutions: Efficient Invariance to Spatial Transformations**

- arxiv: [http://arxiv.org/abs/1609.04382](http://arxiv.org/abs/1609.04382)

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

**Video Pixel Networks**

- intro: Google DeepMind
- arxiv: [https://arxiv.org/abs/1610.00527](https://arxiv.org/abs/1610.00527)

**Tensor Switching Networks**

- intro: NIPS 2016
- arixiv: [https://arxiv.org/abs/1610.10087](https://arxiv.org/abs/1610.10087)
- github: [https://github.com/coxlab/tsnet](https://github.com/coxlab/tsnet)

**BranchyNet: Fast Inference via Early Exiting from Deep Neural Networks**

- intro: Harvard University
- paper: [http://www.eecs.harvard.edu/~htk/publication/2016-icpr-teerapittayanon-mcdanel-kung.pdf](http://www.eecs.harvard.edu/~htk/publication/2016-icpr-teerapittayanon-mcdanel-kung.pdf)

**Spectral Convolution Networks**

- arxiv: [https://arxiv.org/abs/1611.05378](https://arxiv.org/abs/1611.05378)

**DelugeNets: Deep Networks with Massive and Flexible Cross-layer Information Inflows**

- arxiv: [https://arxiv.org/abs/1611.05552](https://arxiv.org/abs/1611.05552)
- github: [https://github.com/xternalz/DelugeNets](https://github.com/xternalz/DelugeNets)

**Weakly Supervised Cascaded Convolutional Networks**

- arxiv: [https://arxiv.org/abs/1611.08258](https://arxiv.org/abs/1611.08258)

**DeepSetNet: Predicting Sets with Deep Neural Networks**

- intro: multi-class image classification and pedestrian detection
- arxiv: [https://arxiv.org/abs/1611.08998](https://arxiv.org/abs/1611.08998)

## Highway Networks

**Highway Networks**

- intro: ICML 2015 Deep Learning workshop
- intro: shortcut connections with gating functions. These gates are data-dependent and have parameters
- arxiv: [http://arxiv.org/abs/1505.00387](http://arxiv.org/abs/1505.00387)

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

**Semi-Supervised Classification with Graph Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1609.02907](http://arxiv.org/abs/1609.02907)
- github: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

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

# Deep Learning with Traditional Machine Learning Methods

## Bag of Words (BoW)

**Deep Learning Transcends the Bag of Words**

- blog: [http://www.kdnuggets.com/2015/12/deep-learning-outgrows-bag-words-recurrent-neural-networks.html](http://www.kdnuggets.com/2015/12/deep-learning-outgrows-bag-words-recurrent-neural-networks.html)

## Bootstrap

**Training Deep Neural Networks on Noisy Labels with Bootstrapping**

- arxiv: [http://arxiv.org/abs/1412.6596](http://arxiv.org/abs/1412.6596)

## Decision Tree

**Neural Network and Decision Tree**

- blog: [http://rotationsymmetry.github.io/2015/07/18/neural-network-decision-tree/](http://rotationsymmetry.github.io/2015/07/18/neural-network-decision-tree/)

**Decision Forests, Convolutional Networks and the Models in-Between**

- arxiv: [http://arxiv.org/abs/1603.01250](http://arxiv.org/abs/1603.01250)
- notes: [http://blog.csdn.net/stdcoutzyx/article/details/50993124](http://blog.csdn.net/stdcoutzyx/article/details/50993124)

## Dictionary Learning

**Sparse Factorization Layers for Neural Networks with Limited Supervision**

- arxiv: [https://arxiv.org/abs/1612.04468](https://arxiv.org/abs/1612.04468)

## HMM

**Unsupervised Neural Hidden Markov Models**

- intro: EMNLP 2016
- paper: [http://www.isi.edu/natural-language/mt/neural-hmm16.pdf](http://www.isi.edu/natural-language/mt/neural-hmm16.pdf)
- github: [https://github.com/ketranm/neuralHMM](https://github.com/ketranm/neuralHMM)

## SVM

**Large-scale Learning with SVM and Convolutional for Generic Object Categorization**

- paper: [http://yann.lecun.com/exdb/publis/pdf/huang-lecun-06.pdf](http://yann.lecun.com/exdb/publis/pdf/huang-lecun-06.pdf)

**Convolutional Neural Support Vector Machines:Hybrid Visual Pattern Classifiers for Multi-robot Systems**

- paper: [http://people.idsia.ch/~nagi/conferences/idsia/icmla2012.pdf](http://people.idsia.ch/~nagi/conferences/idsia/icmla2012.pdf)

**Deep Learning using Linear Support Vector Machines**

- paper: [http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf)

**Trusting SVM for Piecewise Linear CNNs**

- intro: PL-CNNs
- arxiv: [https://arxiv.org/abs/1611.02185](https://arxiv.org/abs/1611.02185)

## Random Forest

**Towards the effectiveness of Deep Convolutional Neural Network based Fast Random Forest Classifier**

- arxiv: [http://arxiv.org/abs/1609.08864](http://arxiv.org/abs/1609.08864)

## Gaussian Process

**Deep Learning with Gaussian Process**

- blog: [https://amundtveit.com/2016/12/02/deep-learning-with-gaussian-process/](https://amundtveit.com/2016/12/02/deep-learning-with-gaussian-process/)

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

# Deep Learning on Mobile Devices

**Convolutional neural networks on the iPhone with VGGNet**

![](http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/Bookcase.png)

- blog: [http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/](http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/)
- github: [https://github.com/hollance/VGGNet-Metal](https://github.com/hollance/VGGNet-Metal)

**TensorFlow for Mobile Poets**

- blog: [https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/](https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/)

**The Convolutional Neural Network(CNN) for Android**

- intro: CnnForAndroid:A Classification Project using Convolutional Neural Network(CNN) in Android platform。It also support Caffe Model
- github: [https://github.com/zhangqianhui/CnnForAndroid](https://github.com/zhangqianhui/CnnForAndroid)

**Mobilenets: Efficient convolutional neural networks for mobile vision applications**

**TensorFlow on Android**

- blog: [https://www.oreilly.com/learning/tensorflow-on-android](https://www.oreilly.com/learning/tensorflow-on-android)

**Experimenting with TensorFlow on Android**

- part 1: [https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-pt-1-362683b31838#.5gbp2d4st](https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-pt-1-362683b31838#.5gbp2d4st)
- part 2: [https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-part-2-12f3dc294eaf#.2gx3o65f5](https://medium.com/@mgazar/experimenting-with-tensorflow-on-android-part-2-12f3dc294eaf#.2gx3o65f5)
- github: [https://github.com/MostafaGazar/tensorflow](https://github.com/MostafaGazar/tensorflow)

# Embedded Deep Learning

**Embedded Deep Learning with NVIDIA Jetson**

- youtube: [https://www.youtube.com/watch?v=_4tzlXPQWb8](https://www.youtube.com/watch?v=_4tzlXPQWb8)
- mirror: [https://pan.baidu.com/s/1pKCDXkZ](https://pan.baidu.com/s/1pKCDXkZ)

# Deep Learning in Finance

**Deep Learning in Finance**

- arxiv: [http://arxiv.org/abs/1602.06561](http://arxiv.org/abs/1602.06561)

**A Survey of Deep Learning Techniques Applied to Trading**

- blog: [http://gregharris.info/a-survey-of-deep-learning-techniques-applied-to-trading/](http://gregharris.info/a-survey-of-deep-learning-techniques-applied-to-trading/)

**Deep Learning and Long-Term Investing**

- part 1: [http://www.euclidean.com/deep-learning-long-term-investing-1](http://www.euclidean.com/deep-learning-long-term-investing-1)
- part 2: [http://www.euclidean.com/deep-learning-investing-part-2-preprocessing-data](http://www.euclidean.com/deep-learning-investing-part-2-preprocessing-data)

**Deep Learning in Trading**

- youtube: [https://www.youtube.com/watch?v=FoQKCeDuPiY](https://www.youtube.com/watch?v=FoQKCeDuPiY)
- mirror: [https://pan.baidu.com/s/1sltRra9](https://pan.baidu.com/s/1sltRra9)

**Research to Products: Machine & Human Intelligence in Finance**

- intro: Peter Sarlin, Hanken School of Economics - Deep Learning in Finance Summit 2016 #reworkfin
- youtube: [https://www.youtube.com/watch?v=Fd7Cc-KOVXg](https://www.youtube.com/watch?v=Fd7Cc-KOVXg)
- mirror: [https://pan.baidu.com/s/1kVpZKur#list/path=%2F](https://pan.baidu.com/s/1kVpZKur#list/path=%2F)

**eep Neural Networks for Real-time Market Predictions**

- youtube: [https://www.youtube.com/watch?v=Kzz2-wAEK7A](https://www.youtube.com/watch?v=Kzz2-wAEK7A)

**Deep Learning the Stock Market**

- blog: [https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02#.z752rf43u](https://medium.com/@TalPerry/deep-learning-the-stock-market-df853d139e02#.z752rf43u)

# Deep Learning in Speech

**Deep Speech 2: End-to-End Speech Recognition in English and Mandarin**

- intro: Baidu Research, ICML 2016
- arxiv: [https://arxiv.org/abs/1512.02595](https://arxiv.org/abs/1512.02595)
- github(Neon): [https://github.com/NervanaSystems/deepspeech](https://github.com/NervanaSystems/deepspeech)

**End-to-end speech recognition with neon**

- blog: [https://www.nervanasys.com/end-end-speech-recognition-neon/](https://www.nervanasys.com/end-end-speech-recognition-neon/)

## WaveNet

**WaveNet: A Generative Model for Raw Audio**

- homepage: [https://deepmind.com/blog/wavenet-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)
- paper: [https://drive.google.com/file/d/0B3cxcnOkPx9AeWpLVXhkTDJINDQ/view](https://drive.google.com/file/d/0B3cxcnOkPx9AeWpLVXhkTDJINDQ/view)
- mirror: [https://pan.baidu.com/s/1gfmGWaJ](https://pan.baidu.com/s/1gfmGWaJ)
- github(Keras): [https://github.com/usernaamee/keras-wavenet](https://github.com/usernaamee/keras-wavenet)
- github(TensorFlow): [https://github.com/ibab/tensorflow-wavenet](https://github.com/ibab/tensorflow-wavenet)
- github(Chainer): [https://github.com/monthly-hack/chainer-wavenet](https://github.com/monthly-hack/chainer-wavenet)
- github(Theano): [https://github.com/huyouare/WaveNet-Theano](https://github.com/huyouare/WaveNet-Theano)
- github(Keras): [https://github.com/basveeling/wavenet](https://github.com/basveeling/wavenet)
- github: [https://github.com/ritheshkumar95/WaveNet](https://github.com/ritheshkumar95/WaveNet)

**A TensorFlow implementation of DeepMind's WaveNet paper for text generation.**

- github: [https://github.com/Zeta36/tensorflow-tex-wavenet](https://github.com/Zeta36/tensorflow-tex-wavenet)

**Fast Wavenet Generation Algorithm**

- intro: An efficient Wavenet generation implementation
- arxiv: [https://arxiv.org/abs/1611.09482](https://arxiv.org/abs/1611.09482)
- github [https://github.com/tomlepaine/fast-wavenet](https://github.com/tomlepaine/fast-wavenet)

**Speech-to-Text-WaveNet : End-to-end sentence level English speech recognition based on DeepMind's WaveNet and tensorflow**

- github: [https://github.com/buriburisuri/speech-to-text-wavenet](https://github.com/buriburisuri/speech-to-text-wavenet)

**Wav2Letter: an End-to-End ConvNet-based Speech Recognition System**

- arxiv: [http://arxiv.org/abs/1609.03193](http://arxiv.org/abs/1609.03193)

**TristouNet: Triplet Loss for Speaker Turn Embedding**

- arxiv: [https://arxiv.org/abs/1609.04301](https://arxiv.org/abs/1609.04301)
- github: [https://github.com/hbredin/TristouNet](https://github.com/hbredin/TristouNet)

**Speech Recognion and Deep Learning**

- intro: Baidu Research Silicon Valley AI Lab
- slides: [http://cs.stanford.edu/~acoates/ba_dls_speech2016.pdf](http://cs.stanford.edu/~acoates/ba_dls_speech2016.pdf)
- mirror: [https://pan.baidu.com/s/1qYrPkPQ](https://pan.baidu.com/s/1qYrPkPQ)
- github: [https://github.com/baidu-research/ba-dls-deepspeech](https://github.com/baidu-research/ba-dls-deepspeech)

**Robust end-to-end deep audiovisual speech recognition**

- intro: CMU
- github: [https://arxiv.org/abs/1611.06986](https://arxiv.org/abs/1611.06986)

**An Experimental Comparison of Deep Neural Networks for End-to-end Speech Recognition**

- arxiv: [https://arxiv.org/abs/1611.07174](https://arxiv.org/abs/1611.07174)

**Recurrent Deep Stacking Networks for Speech Recognition**

- intro: The Ohio State University
- arxiv: [https://arxiv.org/abs/1612.04675](https://arxiv.org/abs/1612.04675)

# Deep Learning for Sound / Music

## Sound

**Suggesting Sounds for Images from Video Collections**

- intro: ETH Zurich & 2Disney Research
- paper: [https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20161014182443/Suggesting-Sounds-for-Images-from-Video-Collections-Paper.pdf](https://s3-us-west-1.amazonaws.com/disneyresearch/wp-content/uploads/20161014182443/Suggesting-Sounds-for-Images-from-Video-Collections-Paper.pdf)

**Disney AI System Associates Images with Sounds**

- blog: [https://news.developer.nvidia.com/disneys-ai-system-associates-images-with-sounds/](https://news.developer.nvidia.com/disneys-ai-system-associates-images-with-sounds/)

## Music

**Learning Features of Music from Scratch**

- intro: University of Washington. MusicNet
- project page: [http://homes.cs.washington.edu/~thickstn/musicnet.html](http://homes.cs.washington.edu/~thickstn/musicnet.html)
- arxiv: [https://arxiv.org/abs/1611.09827](https://arxiv.org/abs/1611.09827)
- demo: [http://homes.cs.washington.edu/~thickstn/demos.html](http://homes.cs.washington.edu/~thickstn/demos.html)

**DeepBach: a Steerable Model for Bach chorales generation**

- project page: [http://www.flow-machines.com/deepbach-steerable-model-bach-chorales-generation/](http://www.flow-machines.com/deepbach-steerable-model-bach-chorales-generation/)
- arxiv: [https://arxiv.org/abs/1612.01010](https://arxiv.org/abs/1612.01010)
- github: [https://github.com/SonyCSL-Paris/DeepBach](https://github.com/SonyCSL-Paris/DeepBach)
- youtube: [https://www.youtube.com/watch?v=QiBM7-5hA6o](https://www.youtube.com/watch?v=QiBM7-5hA6o)

**Deep Learning for Music**

- blog: [https://amundtveit.com/2016/11/22/deep-learning-for-music/](https://amundtveit.com/2016/11/22/deep-learning-for-music/)

# Deep Learning on Games

**TorchCraft: a Library for Machine Learning Research on Real-Time Strategy Games**

- intro: Connecting Torch to StarCraft
- arxiv: [https://arxiv.org/abs/1611.00625](https://arxiv.org/abs/1611.00625)
- github: [https://github.com/TorchCraft/TorchCraft](https://github.com/TorchCraft/TorchCraft)

**BlizzCon 2016 DeepMind and StarCraft II Deep Learning Panel Transcript**

- part 1: [http://starcraft.blizzplanet.com/blog/comments/blizzcon-2016-deepmind-and-starcraft-ii-deep-learning-panel-transcript](http://starcraft.blizzplanet.com/blog/comments/blizzcon-2016-deepmind-and-starcraft-ii-deep-learning-panel-transcript)
- part 2: [http://starcraft.blizzplanet.com/blog/comments/blizzcon-2016-deepmind-and-starcraft-ii-deep-learning-panel-transcript/2](http://starcraft.blizzplanet.com/blog/comments/blizzcon-2016-deepmind-and-starcraft-ii-deep-learning-panel-transcript/2)

# Deep Learning in Medicine and Biology

**Low Data Drug Discovery with One-shot Learning**

- intro: MIT & Stanford University
- arxiv: [https://arxiv.org/abs/1611.03199](https://arxiv.org/abs/1611.03199)
- homepage: [http://deepchem.io/](http://deepchem.io/)
- github: [https://github.com/deepchem/deepchem](https://github.com/deepchem/deepchem)

**Introduction to Deep Learning in Medicine and Biology**

- blog: [http://a12d.com/deep-learning-biomedicine](http://a12d.com/deep-learning-biomedicine)

**Deep Learning for Alzheimer Diagnostics and Decision Support**

[https://amundtveit.com/2016/11/18/deep-learning-for-alzheimer-diagnostics-and-decision-support/](https://amundtveit.com/2016/11/18/deep-learning-for-alzheimer-diagnostics-and-decision-support/)

**DeepCancer: Detecting Cancer through Gene Expressions via Deep Generative Learning**

- intro: University of Florida
- arxiv: [https://arxiv.org/abs/1612.03211](https://arxiv.org/abs/1612.03211)

**Towards biologically plausible deep learning**

- intro: Yoshua	Bengio, NIPS’2016 Workshops
- slides: [http://www.iro.umontreal.ca/~bengioy/talks/Brains+Bits-NIPS2016Workshop.pptx.pdf](http://www.iro.umontreal.ca/~bengioy/talks/Brains+Bits-NIPS2016Workshop.pptx.pdf)

# Benchmarks

**Deep Learning’s Accuracy**

- blog: [http://deeplearning4j.org/accuracy.html](http://deeplearning4j.org/accuracy.html)

**Benchmarks for popular CNN models**

[https://github.com/jcjohnson/cnn-benchmarks](https://github.com/jcjohnson/cnn-benchmarks)

**Deep Learning Benchmarks**

[http://add-for.com/deep-learning-benchmarks/](http://add-for.com/deep-learning-benchmarks/)

**cudnn-rnn-benchmarks**

- github: [https://github.com/MaximumEntropy/cudnn_rnn_theano_benchmarks](https://github.com/MaximumEntropy/cudnn_rnn_theano_benchmarks)

# Papers

**Reweighted Wake-Sleep**

- paper: [http://arxiv.org/abs/1406.2751](http://arxiv.org/abs/1406.2751)
- github: [https://github.com/jbornschein/reweighted-ws](https://github.com/jbornschein/reweighted-ws)

**Striving for Simplicity: The All Convolutional Net**

- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

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

**A Survey: Time Travel in Deep Learning Space: An Introduction to Deep Learning Models and How Deep Learning Models Evolved from the Initial Ideas**

- arxiv: [http://arxiv.org/abs/1510.04781](http://arxiv.org/abs/1510.04781)

**Towards Open Set Deep Networks**

- arxiv: [http://arxiv.org/abs/1511.06233](http://arxiv.org/abs/1511.06233)
- github: [https://github.com/abhijitbendale/OSDN](https://github.com/abhijitbendale/OSDN)

**Structured Prediction Energy Networks (SPEN)**

- arxiv: [http://arxiv.org/abs/1511.06350](http://arxiv.org/abs/1511.06350)
- github: [https://github.com/davidBelanger/SPEN](https://github.com/davidBelanger/SPEN)

**A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction**

- arxiv: [http://arxiv.org/abs/1512.06293](http://arxiv.org/abs/1512.06293)

**Deep Neural Networks predict Hierarchical Spatio-temporal Cortical Dynamics of Human Visual Object Recognition**

- arxiv: [http://arxiv.org/abs/1601.02970](http://arxiv.org/abs/1601.02970)
- demo: [http://brainmodels.csail.mit.edu/dnn/drawCNN/](http://brainmodels.csail.mit.edu/dnn/drawCNN/)

**A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction**

- arxiv: [http://arxiv.org/abs/1512.06293](http://arxiv.org/abs/1512.06293)

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

- arxiv: [http://arxiv.org/abs/1603.09114](http://arxiv.org/abs/1603.09114)

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

**Convolutional Neural Networks Analyzed via Convolutional Sparse Coding**

**DeepMath - Deep Sequence Models for Premise Selection**

- arxiv: [https://arxiv.org/abs/1606.04442](https://arxiv.org/abs/1606.04442)
- github: [https://github.com/tensorflow/deepmath](https://github.com/tensorflow/deepmath)

- arxiv: [http://arxiv.org/abs/1607.08194](http://arxiv.org/abs/1607.08194)

**TI-POOLING: transformation-invariant pooling for feature learning in Convolutional Neural Networks**

- intro: CVPR 2016
- paper: [http://dlaptev.org/papers/Laptev16_CVPR.pdf](http://dlaptev.org/papers/Laptev16_CVPR.pdf)
- github: [https://github.com/dlaptev/TI-pooling](https://github.com/dlaptev/TI-pooling)

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

**Deep Learning with Separable Convolutions**

**Xception: Deep Learning with Depthwise Separable Convolutions**

- arxiv: [https://arxiv.org/abs/1610.02357](https://arxiv.org/abs/1610.02357)
- code(Keras): [https://keras.io/applications/#xception](https://keras.io/applications/#xception)
- github: [https://gist.github.com/culurciello/554c8e56d3bbaf7c66bf66c6089dc221](https://gist.github.com/culurciello/554c8e56d3bbaf7c66bf66c6089dc221)

**Uncertainty in Deep Learning**

- intro: PhD Thesis. Cambridge Machine Learning Group
- blog: [http://mlg.eng.cam.ac.uk/yarin/blog_2248.html](http://mlg.eng.cam.ac.uk/yarin/blog_2248.html)
- thesis: [http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf](http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf)

**Stochastic Variational Deep Kernel Learning**

- intro: NIPS 2016
- arxiv: [https://arxiv.org/abs/1611.00336](https://arxiv.org/abs/1611.00336)
- code: [https://people.orie.cornell.edu/andrew/code/#SVDKL](https://people.orie.cornell.edu/andrew/code/#SVDKL)

**Deep Convolutional Neural Network Design Patterns**

- arxiv: [https://arxiv.org/abs/1611.00847](https://arxiv.org/abs/1611.00847)
- github: [https://github.com/iPhysicist/CNNDesignPatterns](https://github.com/iPhysicist/CNNDesignPatterns)

**Extensions and Limitations of the Neural GPU**

- arxiv: [https://arxiv.org/abs/1611.00736](https://arxiv.org/abs/1611.00736)
- github: [https://github.com/openai/ecprice-neural-gpu](https://github.com/openai/ecprice-neural-gpu)

**Neural Functional Programming**

- arxiv: [https://arxiv.org/abs/1611.01988](https://arxiv.org/abs/1611.01988)

**Snapshot Ensembles: Train 1, Get M for Free**

- paper: [http://openreview.net/pdf?id=BJYwwY9ll](http://openreview.net/pdf?id=BJYwwY9ll)
- github: [https://github.com/gaohuang/SnapshotEnsemble](https://github.com/gaohuang/SnapshotEnsemble)

**Deep Information Propagation**

- arxiv: [https://arxiv.org/abs/1611.01232](https://arxiv.org/abs/1611.01232)

**Compressed Learning: A Deep Neural Network Approach**

- arxiv: [https://arxiv.org/abs/1610.09615](https://arxiv.org/abs/1610.09615)

**A backward pass through a CNN using a generative model of its activations**

- arxiv: [https://arxiv.org/abs/1611.02767](https://arxiv.org/abs/1611.02767)

**Understanding deep learning requires rethinking generalization**

- intro: MIT & Google Brain & UC Berkeley & Google DeepMind
- arxiv: [https://arxiv.org/abs/1611.03530](https://arxiv.org/abs/1611.03530)

**Local minima in training of deep networks**

- intro: DeepMind
- arxiv: [https://arxiv.org/abs/1611.06310](https://arxiv.org/abs/1611.06310)

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

## One Shot Learning

**Matching Networks for One Shot Learning**

- intro: Google DeepMind
- arxiv: [https://arxiv.org/abs/1606.04080](https://arxiv.org/abs/1606.04080)

## Incremental Learning

**iCaRL: Incremental Classifier and Representation Learning**

- arxiv: [https://arxiv.org/abs/1611.07725](https://arxiv.org/abs/1611.07725)

## Ensemble Deep Learning

**Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles**

- arxiv: [https://arxiv.org/abs/1606.07839](https://arxiv.org/abs/1606.07839)
- youtube: [https://www.youtube.com/watch?v=KjUfMtZjyfg&feature=youtu.be](https://www.youtube.com/watch?v=KjUfMtZjyfg&feature=youtu.be)

**Ensemble Deep Learning**

- blog: [https://amundtveit.com/2016/12/02/ensemble-deep-learning/](https://amundtveit.com/2016/12/02/ensemble-deep-learning/)

## Computer Vision

**End-to-End Integration of a Convolutional Network, Deformable Parts Model and Non-Maximum Suppression**

- arxiv: [http://arxiv.org/abs/1411.5309](http://arxiv.org/abs/1411.5309)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wan_End-to-End_Integration_of_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wan_End-to-End_Integration_of_2015_CVPR_paper.pdf)

**FlowNet: Learning Optical Flow with Convolutional Networks**

- intro: "competitive accuracy at frame rates of 5 to 10 fps"
- project page: [http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/](http://lmb.informatik.uni-freiburg.de/Publications/2015/DFIB15/)
- arxiv: [https://arxiv.org/abs/1504.06852](https://arxiv.org/abs/1504.06852)

**FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks**

- project page: [http://lmb.informatik.uni-freiburg.de/Publications/2016/IMKDB16/](http://lmb.informatik.uni-freiburg.de/Publications/2016/IMKDB16/)
- arxiv: [https://arxiv.org/abs/1612.01925](https://arxiv.org/abs/1612.01925)
- video: [http://lmb.informatik.uni-freiburg.de/Publications/2016/IMKDB16/](http://lmb.informatik.uni-freiburg.de/Publications/2016/IMKDB16/)

**A convnet for non-maximum suppression**

- arxiv: [http://arxiv.org/abs/1511.06437](http://arxiv.org/abs/1511.06437)

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

**Exploiting Depth from Single Monocular Images for Object Detection and Semantic Segmentation**

- intro: IEEE T. Image Processing
- arxiv: [https://arxiv.org/abs/1610.01706](https://arxiv.org/abs/1610.01706)

**Optical Flow Estimation using a Spatial Pyramid Network**

- arxiv: [https://arxiv.org/abs/1611.00850](https://arxiv.org/abs/1611.00850)

**DSAC - Differentiable RANSAC for Camera Localization**

- arxiv: [https://arxiv.org/abs/1611.05705](https://arxiv.org/abs/1611.05705)

### All-In-One Network

**HyperFace: A Deep Multi-task Learning Framework for Face Detection, Landmark Localization, Pose Estimation, and Gender Recognition**

- arxiv: [https://arxiv.org/abs/1603.01249](https://arxiv.org/abs/1603.01249)
- summary: [https://github.com/aleju/papers/blob/master/neural-nets/HyperFace.md](https://github.com/aleju/papers/blob/master/neural-nets/HyperFace.md)

**UberNet: Training a `Universal' Convolutional Neural Network for Low-, Mid-, and High-Level Vision using Diverse Datasets and Limited Memory**

![](http://cvn.ecp.fr/ubernet/res.png)

- arxiv: [http://arxiv.org/abs/1609.02132](http://arxiv.org/abs/1609.02132)
- demo: [http://cvn.ecp.fr/ubernet/](http://cvn.ecp.fr/ubernet/)

**An All-In-One Convolutional Neural Network for Face Analysis**

- intro: simultaneous face detection, face alignment, pose estimation, gender recognition, smile detection, age estimation and face recognition 
- arxiv: [https://arxiv.org/abs/1611.00851](https://arxiv.org/abs/1611.00851)

**MultiNet: Real-time Joint Semantic Reasoning for Autonomous Driving**

- intro: first place on Kitti Road Segmentation. 
joint classification, detection and semantic segmentation via a unified architecture, less than 100 ms to perform all tasks
- arxiv: [https://arxiv.org/abs/1612.07695](https://arxiv.org/abs/1612.07695)
- github: coming soon

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

**Visual Search Server**

![](https://raw.githubusercontent.com/AKSHAYUBHAT/VisualSearchServer/master/appcode/static/alpha3.png)

- intro: "A simple implementation of Visual Search using features extracted from Tensor Flow inception model"
- github: [https://github.com/AKSHAYUBHAT/VisualSearchServer](https://github.com/AKSHAYUBHAT/VisualSearchServer)

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

**Deep CNN - Deep convolution neural network project with TensorFlow**

- github: [https://github.com/tobegit3hub/deep_cnn](https://github.com/tobegit3hub/deep_cnn)

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

**Attention**

- intro: Attention在视觉上的递归模型 / 基于Attention的图片生成 / 基于Attention的图片主题生成 / 基于Attention的字符识别
- blog: [http://www.cosmosshadow.com/ml/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/2016/03/08/Attention.html](http://www.cosmosshadow.com/ml/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/2016/03/08/Attention.html)

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
