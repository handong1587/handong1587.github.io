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

![](https://culurciello.github.io/assets/nets/nin.jpg)

- arxiv: [http://arxiv.org/abs/1312.4400](http://arxiv.org/abs/1312.4400)
- gitxiv: [http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin](http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin)

**Batch-normalized Maxout Network in Network**

- arxiv: [http://arxiv.org/abs/1511.02583](http://arxiv.org/abs/1511.02583)

## GoogLeNet

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

## Inception-V2 / Inception-V3

Inception-V3 = Inception-V2 + BN-auxiliary (fully connected layer of the auxiliary classifier is also batch-normalized, 
not just the convolutions)

**Rethinking the Inception Architecture for Computer Vision**

- intro: "21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network; 
3.5% top-5 error and 17.3% top-1 error With an ensemble of 4 models and multi-crop evaluation."
- arxiv: [http://arxiv.org/abs/1512.00567](http://arxiv.org/abs/1512.00567)
- github(Torch): [https://github.com/Moodstocks/inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch)
- github(TensorFlow): [https://github.com/tensorflow/models/tree/master/inception#how-to-train-from-scratch-in-a-distributed-setting](https://github.com/tensorflow/models/tree/master/inception#how-to-train-from-scratch-in-a-distributed-setting)

**Notes on the TensorFlow Implementation of Inception v3**

[https://pseudoprofound.wordpress.com/2016/08/28/notes-on-the-tensorflow-implementation-of-inception-v3/](https://pseudoprofound.wordpress.com/2016/08/28/notes-on-the-tensorflow-implementation-of-inception-v3/)

## ResNet

![](/assets/dl_resources/ResNet_CVPR2016_BestPaperAward.jpg)

**Deep Residual Learning for Image Recognition**

- arxiv: [http://arxiv.org/abs/1512.03385](http://arxiv.org/abs/1512.03385)
- slides: [http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/ilsvrc15/ilsvrc2015_deep_residual_learning_kaiminghe.pdf)
- gitxiv: [http://gitxiv.com/posts/LgPRdTY3cwPBiMKbm/deep-residual-learning-for-image-recognition](http://gitxiv.com/posts/LgPRdTY3cwPBiMKbm/deep-residual-learning-for-image-recognition)
- github: [https://github.com/KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)
- github: [https://github.com/alrojo/lasagne_residual_network](https://github.com/alrojo/lasagne_residual_network)
- github: [https://github.com/shuokay/resnet](https://github.com/shuokay/resnet)
- github: [https://github.com/gcr/torch-residual-networks](https://github.com/gcr/torch-residual-networks)
- github: [https://github.com/apark263/cfmz](https://github.com/apark263/cfmz)
- github: [https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_msra.py](https://github.com/NervanaSystems/neon/blob/master/examples/cifar10_msra.py)
- github: [https://github.com/facebook/fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
- github: [https://github.com/yasunorikudo/chainer-ResNet](https://github.com/yasunorikudo/chainer-ResNet)
- github: [https://github.com/raghakot/keras-resnet](https://github.com/raghakot/keras-resnet)
- github: [https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)

**Third-party re-implementations**

[https://github.com/KaimingHe/deep-residual-networks#third-party-re-implementations](https://github.com/KaimingHe/deep-residual-networks#third-party-re-implementations)

**Training and investigating Residual Nets**

[http://torch.ch/blog/2016/02/04/resnets.html](http://torch.ch/blog/2016/02/04/resnets.html)

**Highway Networks and Deep Residual Networks**

- blog: [http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html](http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html)

**Interpretating Deep Residual Learning Blocks as Locally Recurrent Connections**

- blog: [https://matrixmashing.wordpress.com/2016/01/29/interpretating-deep-residual-learning-blocks-as-locally-recurrent-connections/](https://matrixmashing.wordpress.com/2016/01/29/interpretating-deep-residual-learning-blocks-as-locally-recurrent-connections/)

**Resnet in Resnet: Generalizing Residual Architectures**

- paper: [http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g](http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g)
- arxiv: [http://arxiv.org/abs/1603.08029](http://arxiv.org/abs/1603.08029)

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

## Inception-V4

**Inception-V4, Inception-Resnet And The Impact Of Residual Connections On Learning (Workshop track - ICLR 2016)**

- intro: "achieve 3.08% top-5 error on the test set of the ImageNet classification (CLS) challenge"
- arxiv: [http://arxiv.org/abs/1602.07261](http://arxiv.org/abs/1602.07261)
- paper: [http://beta.openreview.net/pdf?id=q7kqBkL33f8LEkD3t7X9](http://beta.openreview.net/pdf?id=q7kqBkL33f8LEkD3t7X9)
- github: [https://github.com/lim0606/torch-inception-resnet-v2](https://github.com/lim0606/torch-inception-resnet-v2)

- - -

**Striving for Simplicity: The All Convolutional Net**

- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

**Systematic evaluation of CNN advances on the ImageNet**

- arxiv: [http://arxiv.org/abs/1606.02228](http://arxiv.org/abs/1606.02228)
- github: [https://github.com/ducha-aiki/caffenet-benchmark](https://github.com/ducha-aiki/caffenet-benchmark)

# Deep Learning And Bayesian

**Scalable Bayesian Optimization Using Deep Neural Networks (ICML 2015)**

- paper: [http://jmlr.org/proceedings/papers/v37/snoek15.html](http://jmlr.org/proceedings/papers/v37/snoek15.html)
- arxiv: [http://arxiv.org/abs/1502.05700](http://arxiv.org/abs/1502.05700)
- github: [https://github.com/bshahr/torch-dngo](https://github.com/bshahr/torch-dngo)

**Bayesian Dark Knowledge**

- paper: [http://arxiv.org/abs/1506.04416](http://arxiv.org/abs/1506.04416)
- notes: [Notes on Bayesian Dark Knowledge](https://www.evernote.com/shard/s189/sh/92cc4cbf-285e-4038-af08-c6d9e4aee6ea/d505237e82dc81be9859bc82f3902f9f)

**Memory-based Bayesian Reasoning with Deep Learning(2015.Google DeepMind)**

- slides: [http://blog.shakirm.com/wp-content/uploads/2015/11/CSML_BayesDeep.pdf](http://blog.shakirm.com/wp-content/uploads/2015/11/CSML_BayesDeep.pdf)

**Towards Bayesian Deep Learning: A Survey**

- arxiv: [http://arxiv.org/abs/1604.01662](http://arxiv.org/abs/1604.01662)

**Towards Bayesian Deep Learning: A Framework and Some Existing Methods**

- intro: IEEE Transactions on Knowledge and Data Engineering (TKDE), 2016
- arxiv: [http://arxiv.org/abs/1608.06884](http://arxiv.org/abs/1608.06884)

# Autoencoders

**Auto-Encoding Variational Bayes**

- arxiv: [http://arxiv.org/abs/1312.6114](http://arxiv.org/abs/1312.6114)

**The Potential Energy of an Autoencoder (PAMI 2014)**

- paper: [http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.698.4921&rep=rep1&type=pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.698.4921&rep=rep1&type=pdf)

**Importance Weighted Autoencoders**

- paper: [http://arxiv.org/abs/1509.00519](http://arxiv.org/abs/1509.00519)
- github: [https://github.com/yburda/iwae](https://github.com/yburda/iwae)

**Review of Auto-Encoders(by Piotr Mirowski, Microsoft Bing London, 2014)**

- slides: [https://piotrmirowski.files.wordpress.com/2014/03/piotrmirowski_2014_reviewautoencoders.pdf](https://piotrmirowski.files.wordpress.com/2014/03/piotrmirowski_2014_reviewautoencoders.pdf)
- github: [https://github.com/piotrmirowski/Tutorial_AutoEncoders/](https://github.com/piotrmirowski/Tutorial_AutoEncoders/)

**Stacked What-Where Auto-encoders**

- arxiv: [http://arxiv.org/abs/1506.02351](http://arxiv.org/abs/1506.02351)

**Ladder Variational Autoencoders**
**How to Train Deep Variational Autoencoders and Probabilistic Ladder Networks**

- arxiv:[http://arxiv.org/abs/1602.02282](http://arxiv.org/abs/1602.02282)
- github: [https://github.com/casperkaae/LVAE](https://github.com/casperkaae/LVAE)

**Rank Ordered Autoencoders**

- arxiv: [http://arxiv.org/abs/1605.01749](http://arxiv.org/abs/1605.01749)
- github: [https://github.com/paulbertens/rank-ordered-autoencoder](https://github.com/paulbertens/rank-ordered-autoencoder)

**Decoding Stacked Denoising Autoencoders**

- arxiv: [http://arxiv.org/abs/1605.02832](http://arxiv.org/abs/1605.02832)

**Keras autoencoders (convolutional/fcc)**

- github: [https://github.com/nanopony/keras-convautoencoder](https://github.com/nanopony/keras-convautoencoder)

**Building Autoencoders in Keras**

![](http://blog.keras.io/img/ae/autoencoder_schema.jpg)

- blog: [http://blog.keras.io/building-autoencoders-in-keras.html](http://blog.keras.io/building-autoencoders-in-keras.html)

**Review of auto-encoders**

- intro: Tutorial code for Auto-Encoders, implementing Marc'Aurelio Ranzato's Sparse Encoding Symmetric Machine and 
testing it on the MNIST handwritten digits data.
- paper: [https://github.com/piotrmirowski/Tutorial_AutoEncoders/blob/master/PiotrMirowski_2014_ReviewAutoEncoders.pdf](https://github.com/piotrmirowski/Tutorial_AutoEncoders/blob/master/PiotrMirowski_2014_ReviewAutoEncoders.pdf)
- github: [https://github.com/piotrmirowski/Tutorial_AutoEncoders](https://github.com/piotrmirowski/Tutorial_AutoEncoders)

**Autoencoders: Torch implementations of various types of autoencoders**

- intro: AE / SparseAE / DeepAE / ConvAE / UpconvAE / DenoisingAE / VAE / AdvAE
- github: [https://github.com/Kaixhin/Autoencoders](https://github.com/Kaixhin/Autoencoders)

**Tutorial on Variational Autoencoders**

- arxiv: [http://arxiv.org/abs/1606.05908](http://arxiv.org/abs/1606.05908)
- github: [https://github.com/cdoersch/vae_tutorial](https://github.com/cdoersch/vae_tutorial)

**Variational Autoencoders Explained**

- blog: [http://kvfrans.com/variational-autoencoders-explained/](http://kvfrans.com/variational-autoencoders-explained/)
- github: [https://github.com/kvfrans/variational-autoencoder](https://github.com/kvfrans/variational-autoencoder)

**Introducing Variational Autoencoders (in Prose and Code)**

- blog: [http://blog.fastforwardlabs.com/post/148842796218/introducing-variational-autoencoders-in-prose-and](http://blog.fastforwardlabs.com/post/148842796218/introducing-variational-autoencoders-in-prose-and)

**Under the Hood of the Variational Autoencoder (in Prose and Code)**

- blog: [http://blog.fastforwardlabs.com/post/149329060653/under-the-hood-of-the-variational-autoencoder-in](http://blog.fastforwardlabs.com/post/149329060653/under-the-hood-of-the-variational-autoencoder-in)

**The Unreasonable Confusion of Variational Autoencoders**

- blog: [https://jaan.io/unreasonable-confusion/](https://jaan.io/unreasonable-confusion/)

# Semi-Supervised Learning

**Semi-Supervised Learning with Graphs (Label Propagation)**

- paper: [http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf)
- blog("标签传播算法（Label Propagation）及Python实现"): [http://blog.csdn.net/zouxy09/article/details/49105265](http://blog.csdn.net/zouxy09/article/details/49105265)

**Semi-Supervised Learning with Ladder Networks**

- arxiv: [http://arxiv.org/abs/1507.02672](http://arxiv.org/abs/1507.02672)
- github: [https://github.com/CuriousAI/ladder](https://github.com/CuriousAI/ladder)

**Semi-supervised Feature Transfer: The Practical Benefit of Deep Learning Today?**

- blog: [http://www.kdnuggets.com/2016/07/semi-supervised-feature-transfer-deep-learning.html](http://www.kdnuggets.com/2016/07/semi-supervised-feature-transfer-deep-learning.html)

# Unsupervised Learning

Restricted Boltzmann Machine (RBM), Sparse Coding and Auto-encoder

**On Random Weights and Unsupervised Feature Learning (ICML 2011)**

- paper: [http://www.robotics.stanford.edu/~ang/papers/icml11-RandomWeights.pdf](http://www.robotics.stanford.edu/~ang/papers/icml11-RandomWeights.pdf)

**Unsupervised Learning of Spatiotemporally Coherent Metrics**

- paper: [http://arxiv.org/abs/1412.6056](http://arxiv.org/abs/1412.6056)
- code: [https://github.com/jhjin/flattened-cnn](https://github.com/jhjin/flattened-cnn)

**Unsupervised Visual Representation Learning by Context Prediction (ICCV 2015)**

- homepage: [http://graphics.cs.cmu.edu/projects/deepContext/](http://graphics.cs.cmu.edu/projects/deepContext/)
- arxiv: [http://arxiv.org/abs/1505.05192](http://arxiv.org/abs/1505.05192)
- github: [https://github.com/cdoersch/deepcontext](https://github.com/cdoersch/deepcontext)

**Unsupervised Learning on Neural Network Outputs**

- intro: "use CNN trained on the ImageNet of 1000 classes to the ImageNet of over 20000 classes"
- arxiv: [http://arxiv.org/abs/1506.00990](http://arxiv.org/abs/1506.00990)
- github: [https://github.com/yaolubrain/ULNNO](https://github.com/yaolubrain/ULNNO)

**Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles**

- arxiv: [http://arxiv.org/abs/1603.09246](http://arxiv.org/abs/1603.09246)
- notes: [http://www.inference.vc/notes-on-unsupervised-learning-of-visual-representations-by-solving-jigsaw-puzzles/](http://www.inference.vc/notes-on-unsupervised-learning-of-visual-representations-by-solving-jigsaw-puzzles/)

## PredNet

**Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning (PredNet)**

- arxiv: [http://arxiv.org/abs/1605.08104](http://arxiv.org/abs/1605.08104)

## Clustering

**Joint Unsupervised Learning of Deep Representations and Image Clusters (CVPR 2016)**

- arxiv: [https://arxiv.org/abs/1604.03628](https://arxiv.org/abs/1604.03628)
- github(Torch): [https://github.com/jwyang/joint-unsupervised-learning](https://github.com/jwyang/joint-unsupervised-learning)

**Single-Channel Multi-Speaker Separation using Deep Clustering**

- arxiv: [http://arxiv.org/abs/1607.02173](http://arxiv.org/abs/1607.02173)

### Deep Embedded Clustering (DEC)

**Unsupervised Deep Embedding for Clustering Analysis (ICML 2016)**

- arxiv: [https://arxiv.org/abs/1511.06335](https://arxiv.org/abs/1511.06335)
- github: [https://github.com/piiswrong/dec](https://github.com/piiswrong/dec)

# Transfer Learning

**How transferable are features in deep neural networks? (NIPS 2014)**

- arxiv: [http://arxiv.org/abs/1411.1792](http://arxiv.org/abs/1411.1792)

**Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks**

- paper: [http://research.microsoft.com/pubs/214307/paper.pdf](http://research.microsoft.com/pubs/214307/paper.pdf)

**Simultaneous Deep Transfer Across Domains and Tasks**

![](http://www.eecs.berkeley.edu/~jhoffman/figs/Tzeng_ICCV2015.png)

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

**From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification (ICML 2016)**

- arxiv: [http://arxiv.org/abs/1602.02068](http://arxiv.org/abs/1602.02068)
- github: [https://github.com/gokceneraslan/SparseMax.torch](https://github.com/gokceneraslan/SparseMax.torch)

**CNN-RNN: A Unified Framework for Multi-label Image Classification**

- arxiv: [http://arxiv.org/abs/1604.04573](http://arxiv.org/abs/1604.04573)

**Improving Multi-label Learning with Missing Labels by Structured Semantic Correlations**

- arxiv: [http://arxiv.org/abs/1608.01441](http://arxiv.org/abs/1608.01441)

# Multi-task Learning

**Multitask Learning / Domain Adaptation**

![](http://www.cs.cornell.edu/~kilian/research/multitasklearning/files/pasted-graphic.jpg)

- homepage: [http://www.cs.cornell.edu/~kilian/research/multitasklearning/multitasklearning.html](http://www.cs.cornell.edu/~kilian/research/multitasklearning/multitasklearning.html)

**multi-task learning**

- discussion: [https://github.com/memect/hao/issues/93](https://github.com/memect/hao/issues/93)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

- intro: training a convolutional network to simultaneously classify, 
locate and detect objects in images can boost the classification
accuracy and the detection and localization accuracy of all tasks
- arxiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- github: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

**Learning and Transferring Multi-task Deep Representation for Face Alignment**

- arxiv: [http://arxiv.org/abs/1408.3967](http://arxiv.org/abs/1408.3967)

**Multi-task learning of facial landmarks and expression**

- paper: [http://www.uoguelph.ca/~gwtaylor/publications/gwtaylor_crv2014.pdf](http://www.uoguelph.ca/~gwtaylor/publications/gwtaylor_crv2014.pdf)

**Heterogeneous multi-task learning for human pose estimation with deep convolutional neural network**

- paper: [www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W15/papers/LI_Heterogeneous_Multi-task_Learning_2014_CVPR_paper.pdf](www.cv-foundation.org/openaccess/content_cvpr_workshops_2014/W15/papers/LI_Heterogeneous_Multi-task_Learning_2014_CVPR_paper.pdf)

**Deep Joint Task Learning for Generic Object Extraction(NIPS2014)**

![](http://vision.sysu.edu.cn/vision_sysu/wp-content/uploads/2013/05/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20141019095211.png)

- homepage: [http://vision.sysu.edu.cn/projects/deep-joint-task-learning/](http://vision.sysu.edu.cn/projects/deep-joint-task-learning/)
- paper: [http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf](http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf)
- github: [https://github.com/xiaolonw/nips14_loc_seg_testonly](https://github.com/xiaolonw/nips14_loc_seg_testonly)
- dataset: [http://objectextraction.github.io/](http://objectextraction.github.io/)

**Multi-Task Deep Visual-Semantic Embedding for Video Thumbnail Selection (CVPR 2015)**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Multi-Task_Deep_Visual-Semantic_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Multi-Task_Deep_Visual-Semantic_2015_CVPR_paper.pdf)

**Learning deep representation of multityped objects and tasks**

- arxiv: [http://arxiv.org/abs/1603.01359](http://arxiv.org/abs/1603.01359)

**Cross-stitch Networks for Multi-task Learning**

- arxiv: [http://arxiv.org/abs/1604.03539](http://arxiv.org/abs/1604.03539)

**Multi-Task Learning in Tensorflow (Part 1)**

- blog: [https://jg8610.github.io/Multi-Task/](https://jg8610.github.io/Multi-Task/)

**一箭N雕：多任务深度学习实战**

- intro: 薛云峰 深度学习大讲堂
- blog: [http://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325281&idx=1&sn=97779ff0da06190d6a71d33f23e9dede#rd](http://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325281&idx=1&sn=97779ff0da06190d6a71d33f23e9dede#rd)

# Multi-modal Learning

**Multimodal Deep Learning**

- paper: [http://ai.stanford.edu/~ang/papers/nipsdlufl10-MultimodalDeepLearning.pdf](http://ai.stanford.edu/~ang/papers/nipsdlufl10-MultimodalDeepLearning.pdf)

**Multimodal Convolutional Neural Networks for Matching Image and Sentence**

![](http://mcnn.noahlab.com.hk/mCNN.png)

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

# Debugging Deep Learning

**Some tips for debugging deep learning**

- blog: [http://www.lab41.org/some-tips-for-debugging-in-deep-learning-2/](http://www.lab41.org/some-tips-for-debugging-in-deep-learning-2/)

**Introduction to debugging neural networks**

- blog: [http://russellsstewart.com/notes/0.html](http://russellsstewart.com/notes/0.html)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/4du7gv/introduction_to_debugging_neural_networks](https://www.reddit.com/r/MachineLearning/comments/4du7gv/introduction_to_debugging_neural_networks)

**How to Visualize, Monitor and Debug Neural Network Learning**

- blog: [http://deeplearning4j.org/visualization](http://deeplearning4j.org/visualization)

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

**Adversarial examples in the physical world (Google Brain & OpenAI)**

- author: Alexey Kurakin, Ian Goodfellow, Samy Bengio
- arxiv: [http://arxiv.org/abs/1607.02533](http://arxiv.org/abs/1607.02533)

## DeepFool

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

**Deep Learning Adversarial Examples – Clarifying Misconceptions (By Ian Goodfellow (Google))**

- blog: [http://www.kdnuggets.com/2015/07/deep-learning-adversarial-examples-misconceptions.html](http://www.kdnuggets.com/2015/07/deep-learning-adversarial-examples-misconceptions.html)

**Adversarial Machines: Fooling A.Is (and turn everyone into a Manga)**

- blog: [https://medium.com/@samim/adversarial-machines-998d8362e996#.iv3muefgt](https://medium.com/@samim/adversarial-machines-998d8362e996#.iv3muefgt)

**How to trick a neural network into thinking a panda is a vulture**

- blog: [https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture](https://codewords.recurse.com/issues/five/why-do-neural-networks-think-a-panda-is-a-vulture)

# Deep Learning Networks

**Deeply-supervised Nets (DSN)**

![](http://vcl.ucsd.edu/~sxie/images/dsn/architecture.png)

- arxiv: [http://arxiv.org/abs/1409.5185](http://arxiv.org/abs/1409.5185)
- homepage: [http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/](http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/)
- github: [https://github.com/s9xie/DSN](https://github.com/s9xie/DSN)
- notes: [http://zhangliliang.com/2014/11/02/paper-note-dsn/](http://zhangliliang.com/2014/11/02/paper-note-dsn/)

**Striving for Simplicity: The All Convolutional Net**

- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

**Pointer Networks**

- arxiv: [https://arxiv.org/abs/1506.03134](https://arxiv.org/abs/1506.03134)
- github: [https://github.com/vshallc/PtrNets](https://github.com/vshallc/PtrNets)
- notes: [https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/pointer-networks.md](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/pointer-networks.md)

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

**A Unified Approach for Learning the Parameters of Sum-Product Networks (SPN)**

- intro: "The Sum-Product Network (SPN) is a new type of machine learning model 
with fast exact probabilistic inference over many layers."
- arxiv: [http://arxiv.org/abs/1601.00318](http://arxiv.org/abs/1601.00318)
- homepage: [http://spn.cs.washington.edu/index.shtml](http://spn.cs.washington.edu/index.shtml)
- code: [http://spn.cs.washington.edu/code.shtml](http://spn.cs.washington.edu/code.shtml)

**Recombinator Networks: Learning Coarse-to-Fine Feature Aggregation**

- arxiv: [http://arxiv.org/abs/1511.07356](http://arxiv.org/abs/1511.07356)
- github: [https://github.com/SinaHonari/RCN](https://github.com/SinaHonari/RCN)

**Dynamic Capacity Networks (ICML 2016)**

![](http://mmbiz.qpic.cn/mmbiz/KmXPKA19gW8YYXeWomd4s4ruu7Jmb3wCMIXYPOgr9KIYzckKoiatgcEhedGnZfDZn40BYIoJMZibibxslEb3uicoibw/640?wx_fmt=jpeg&tp=webp&wxfrom=5&wx_lazy=1)

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

- arxiv: [http://arxiv.org/abs/1602.02867](http://arxiv.org/abs/1602.02867)
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

**XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1603.05279](http://arxiv.org/abs/1603.05279)
- github(Torch): [https://github.com/mrastegari/XNOR-Net](https://github.com/mrastegari/XNOR-Net)

**Deeply-Fused Nets**

- arxiv: [http://arxiv.org/abs/1605.07716](http://arxiv.org/abs/1605.07716)

**SNN: Stacked Neural Networks**

- arxiv: [http://arxiv.org/abs/1605.08512](http://arxiv.org/abs/1605.08512)

**Convolutional Neural Fabrics**

- arxiv: [http://arxiv.org/abs/1606.02492](http://arxiv.org/abs/1606.02492)

**Holistic SparseCNN: Forging the Trident of Accuracy, Speed, and Size**

- arxiv: [http://arxiv.org/abs/1608.01409](http://arxiv.org/abs/1608.01409)

**Factorized Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.04337](http://arxiv.org/abs/1608.04337)

**Mollifying Networks**

- author: Caglar Gulcehre, Marcin Moczulski, Francesco Visin, Yoshua Bengio
- arxiv: [http://arxiv.org/abs/1608.04980](http://arxiv.org/abs/1608.04980)

**Local Binary Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.06049](http://arxiv.org/abs/1608.06049)

## DenseNet

**Densely Connected Convolutional Networks**

![](https://cloud.githubusercontent.com/assets/8370623/17981496/fa648b32-6ad1-11e6-9625-02fdd72fdcd3.jpg)

- arxiv: [http://arxiv.org/abs/1608.06993](http://arxiv.org/abs/1608.06993)
- github: [https://github.com/liuzhuang13/DenseNet](https://github.com/liuzhuang13/DenseNet)

**CliqueCNN: Deep Unsupervised Exemplar Learning**

- arxiv: [http://arxiv.org/abs/1608.08792](http://arxiv.org/abs/1608.08792)

## Highway Networks

**Highway Networks**

- arxiv: [http://arxiv.org/abs/1505.00387](http://arxiv.org/abs/1505.00387)
- blog("Highway Networks with TensorFlow"): [https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.71fgztsb6](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.71fgztsb6)

**Very Deep Learning with Highway Networks**

- homepage(papers+code+FAQ): [http://people.idsia.ch/~rupesh/very_deep_learning/](http://people.idsia.ch/~rupesh/very_deep_learning/)

**Training Very Deep Networks (highway networks)**

- arxiv: [http://arxiv.org/abs/1507.06228](http://arxiv.org/abs/1507.06228)

## Spatial Transformer Networks

**Spatial Transformer Networks (NIPS 2015)**

![](https://camo.githubusercontent.com/bb81d6267f2123d59979453526d958a58899bb4f/687474703a2f2f692e696d6775722e636f6d2f4578474456756c2e706e67)

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

![](https://raw.githubusercontent.com/moodstocks/gtsrb.torch/master/resources/traffic-signs.png)

- blog: [http://torch.ch/blog/2015/09/07/spatial_transformers.html](http://torch.ch/blog/2015/09/07/spatial_transformers.html)
- github: [https://github.com/moodstocks/gtsrb.torch](https://github.com/moodstocks/gtsrb.torch)

**Recurrent Spatial Transformer Networks**

- paper: [http://arxiv.org/abs/1509.05329](http://arxiv.org/abs/1509.05329)

# Deep Learning with Traditional Machine Learning Methods

## Cascade

**Compact Convolutional Neural Network Cascade for Face Detection**

- arxiv: [http://arxiv.org/abs/1508.01292](http://arxiv.org/abs/1508.01292)
- github: [https://github.com/Bkmz21/FD-Evaluation](https://github.com/Bkmz21/FD-Evaluation)

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

## SVM

**Large-scale Learning with SVM and Convolutional for Generic Object Categorization**

- paper: [http://yann.lecun.com/exdb/publis/pdf/huang-lecun-06.pdf](http://yann.lecun.com/exdb/publis/pdf/huang-lecun-06.pdf)

**Convolutional Neural Support Vector Machines:Hybrid Visual Pattern Classifiers for Multi-robot Systems**

- paper: [http://people.idsia.ch/~nagi/conferences/idsia/icmla2012.pdf](http://people.idsia.ch/~nagi/conferences/idsia/icmla2012.pdf)

**Deep Learning using Linear Support Vector Machines**

- paper: [http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf](http://deeplearning.net/wp-content/uploads/2013/03/dlsvm.pdf)

# Deep Learning and Robots

**Robot Learning Manipulation Action Plans by "Watching" Unconstrained Videos from the World Wide Web(AAAI 2015)**

- paper: [http://www.umiacs.umd.edu/~yzyang/paper/YouCookMani_CameraReady.pdf](http://www.umiacs.umd.edu/~yzyang/paper/YouCookMani_CameraReady.pdf)
- author page: [http://www.umiacs.umd.edu/~yzyang/](http://www.umiacs.umd.edu/~yzyang/)

**Robots that can adapt like animals (Nature 2014)**

- arxiv: [http://arxiv.org/abs/1407.3501](http://arxiv.org/abs/1407.3501)
- code: [http://pages.isir.upmc.fr/~mouret/code/ite_source_code.tar.gz](http://pages.isir.upmc.fr/~mouret/code/ite_source_code.tar.gz)
- github(for Bayesian optimization): [http://github.com/jbmouret/limbo](http://github.com/jbmouret/limbo)

**End-to-End Training of Deep Visuomotor Policies**

- arxiv: [http://arxiv.org/abs/1504.00702](http://arxiv.org/abs/1504.00702)

**Comment on Open AI’s Efforts to Robot Learning**

- blog: [https://gridworld.wordpress.com/2016/07/28/comment-on-open-ais-efforts-to-robot-learning/](https://gridworld.wordpress.com/2016/07/28/comment-on-open-ais-efforts-to-robot-learning/)

**The Curious Robot: Learning Visual Representations via Physical Interactions**

- arxiv: [http://arxiv.org/abs/1604.01360](http://arxiv.org/abs/1604.01360)

# Deep Learning on Mobile Devices

**Convolutional neural networks on the iPhone with VGGNet**

![](http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/Bookcase.png)

- blog: [http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/](http://matthijshollemans.com/2016/08/30/vggnet-convolutional-neural-network-iphone/)
- github: [https://github.com/hollance/VGGNet-Metal](https://github.com/hollance/VGGNet-Metal)

# Deep Learning in Finance

**Deep Learning in Finance**

- arxiv: [http://arxiv.org/abs/1602.06561](http://arxiv.org/abs/1602.06561)

**A Survey of Deep Learning Techniques Applied to Trading**

- blog: [http://gregharris.info/a-survey-of-deep-learning-techniques-applied-to-trading/](http://gregharris.info/a-survey-of-deep-learning-techniques-applied-to-trading/)

# Applications

**Some like it hot - visual guidance for preference prediction**

- arxiv: [http://arxiv.org/abs/1510.07867](http://arxiv.org/abs/1510.07867)
- demo: [http://howhot.io/](http://howhot.io/)

**Deep Learning Algorithms with Applications to Video Analytics for A Smart City: A Survey**

- arxiv: [http://arxiv.org/abs/1512.03131](http://arxiv.org/abs/1512.03131)

**Camera identification with deep convolutional networks**

- key word: copyright infringement cases, ownership attribution
- arxiv: [http://arxiv.org/abs/1603.01068](http://arxiv.org/abs/1603.01068)

**Build an AI Cat Chaser with Jetson TX1 and Caffe**

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/07/cat1-2.jpeg)

- blog: [https://devblogs.nvidia.com/parallelforall/ai-cat-chaser-jetson-tx1-caffe/](https://devblogs.nvidia.com/parallelforall/ai-cat-chaser-jetson-tx1-caffe/)

**An Analysis of Deep Neural Network Models for Practical Applications**

- arxiv: [http://arxiv.org/abs/1605.07678](http://arxiv.org/abs/1605.07678)

**8 Inspirational Applications of Deep Learning**

- intro: Colorization of Black and White Images, Adding Sounds To Silent Movies, Automatic Machine Translation
Object Classification in Photographs, Automatic Handwriting Generation, Character Text Generation, 
Image Caption Generation, Automatic Game Playing
- blog: [http://machinelearningmastery.com/inspirational-applications-deep-learning/](http://machinelearningmastery.com/inspirational-applications-deep-learning/)

**16 Open Source Deep Learning Models Running as Microservices**

![](http://blog.algorithmia.com/wp-content/uploads/2016/07/deep-learning-model-roundup-1024x536.png)

-intro: Places 365 Classifier, Deep Face Recognition, Real Estate Classifier, Colorful Image Colorization, 
Illustration Tagger, InceptionNet, Parsey McParseface, ArtsyNetworks
- blog: [http://blog.algorithmia.com/2016/07/open-source-deep-learning-algorithm-roundup/](http://blog.algorithmia.com/2016/07/open-source-deep-learning-algorithm-roundup/)

**Makeup like a superstar: Deep Localized Makeup Transfer Network**

- intro: IJCAI 2016
- arxiv: [http://arxiv.org/abs/1604.07102](http://arxiv.org/abs/1604.07102)

**Deep Cascaded Bi-Network for Face Hallucination**

- project page: [http://mmlab.ie.cuhk.edu.hk/projects/CBN.html](http://mmlab.ie.cuhk.edu.hk/projects/CBN.html)
- arxiv: [http://arxiv.org/abs/1607.05046](http://arxiv.org/abs/1607.05046)

**DeepWarp: Photorealistic Image Resynthesis for Gaze Manipulation**

![](http://sites.skoltech.ru/compvision/projects/deepwarp/images/pipeline.svg)

- project page: [http://yaroslav.ganin.net/static/deepwarp/](http://yaroslav.ganin.net/static/deepwarp/)
- arxiv: [http://arxiv.org/abs/1607.07215](http://arxiv.org/abs/1607.07215)

**Autoencoding Blade Runner**

- blog: [https://medium.com/@Terrybroad/autoencoding-blade-runner-88941213abbe#.9kckqg7cq](https://medium.com/@Terrybroad/autoencoding-blade-runner-88941213abbe#.9kckqg7cq)
- github: [https://github.com/terrybroad/Learned-Sim-Autoencoder-For-Video-Frames](https://github.com/terrybroad/Learned-Sim-Autoencoder-For-Video-Frames)

**A guy trained a machine to "watch" Blade Runner. Then things got seriously sci-fi.**

[http://www.vox.com/2016/6/1/11787262/blade-runner-neural-network-encoding](http://www.vox.com/2016/6/1/11787262/blade-runner-neural-network-encoding)

**Deep Convolution Networks for Compression Artifacts Reduction**

![](http://mmlab.ie.cuhk.edu.hk/projects/ARCNN/img/fig1.png)

- intro: ICCV 2015
- project page(code): [http://mmlab.ie.cuhk.edu.hk/projects/ARCNN.html](http://mmlab.ie.cuhk.edu.hk/projects/ARCNN.html)
- arxiv: [http://arxiv.org/abs/1608.02778](http://arxiv.org/abs/1608.02778)

**Deep GDashboard: Visualizing and Understanding Genomic Sequences Using Deep Neural Networks**

- intro: Deep Genomic Dashboard (Deep GDashboard)
- arxiv: [http://arxiv.org/abs/1608.03644](http://arxiv.org/abs/1608.03644)

**Photo Filter Recommendation by Category-Aware Aesthetic Learning**

- intro: Filter Aesthetic Comparison Dataset (FACD): 28,000 filtered images and 42,240 reliable image pairs with aesthetic comparison annotations
- arxiv: [http://arxiv.org/abs/1608.05339](http://arxiv.org/abs/1608.05339)

**Instagram photos reveal predictive markers of depression**

- arxiv: [http://arxiv.org/abs/1608.03282](http://arxiv.org/abs/1608.03282)

**How an Algorithm Learned to Identify Depressed Individuals by Studying Their Instagram Photos**

- review: [https://www.technologyreview.com/s/602208/how-an-algorithm-learned-to-identify-depressed-individuals-by-studying-their-instagram/](https://www.technologyreview.com/s/602208/how-an-algorithm-learned-to-identify-depressed-individuals-by-studying-their-instagram/)

**IM2CAD**

- arxiv: [http://arxiv.org/abs/1608.05137](http://arxiv.org/abs/1608.05137)

## Age Estimation

**Deeply-Learned Feature for Age Estimation**

- paper: [http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7045931&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7045931](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=7045931&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D7045931)

**Age and Gender Classification using Convolutional Neural Networks**

![](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/teaser_a.png)

- paper: [http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/CNN_AgeGenderEstimation.pdf)
- project page: [http://www.openu.ac.il/home/hassner/projects/cnn_agegender/](http://www.openu.ac.il/home/hassner/projects/cnn_agegender/)
- github: [https://github.com/GilLevi/AgeGenderDeepLearning](https://github.com/GilLevi/AgeGenderDeepLearning)

**Recurrent Face Aging**

- paper: [www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Wang_Recurrent_Face_Aging_CVPR_2016_paper.pdf](www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Wang_Recurrent_Face_Aging_CVPR_2016_paper.pdf)

## Emotion / Expression Recognition

**Real-time emotion recognition for gaming using deep convolutional network features**

- paper: [http://arxiv.org/abs/1408.3750v1](http://arxiv.org/abs/1408.3750v1)
- code: [https://github.com/Zebreu/ConvolutionalEmotion](https://github.com/Zebreu/ConvolutionalEmotion)

**Emotion Recognition in the Wild via Convolutional Neural Networks and Mapped Binary Patterns**

![](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/teaser.png)

- project page: [http://www.openu.ac.il/home/hassner/projects/cnn_emotions/](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/)
- papre: [http://www.openu.ac.il/home/hassner/projects/cnn_emotions/LeviHassnerICMI15.pdf](http://www.openu.ac.il/home/hassner/projects/cnn_emotions/LeviHassnerICMI15.pdf)
- github: [https://gist.github.com/GilLevi/54aee1b8b0397721aa4b](https://gist.github.com/GilLevi/54aee1b8b0397721aa4b)

**DeXpression: Deep Convolutional Neural Network for Expression Recognition**

- paper: [http://arxiv.org/abs/1509.05371](http://arxiv.org/abs/1509.05371)

**DEX: Deep EXpectation of apparent age from a single image (ICCV 2015)**

![](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/img/pipeline.png)

- paper: [https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf](https://www.vision.ee.ethz.ch/en/publications/papers/proceedings/eth_biwi_01229.pdf)
- homepage: [https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)

**How Deep Neural Networks Can Improve Emotion Recognition on Video Data**

- intro: ICIP 2016
- arxiv: [http://arxiv.org/abs/1602.07377](http://arxiv.org/abs/1602.07377)

**Peak-Piloted Deep Network for Facial Expression Recognition**

- arxiv: [http://arxiv.org/abs/1607.06997](http://arxiv.org/abs/1607.06997)

**Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution**

- arxiv: [http://arxiv.org/abs/1608.01041](http://arxiv.org/abs/1608.01041)

**A Recursive Framework for Expression Recognition: From Web Images to Deep Models to Game Dataset**

 -arxiv: [http://arxiv.org/abs/1608.01647](http://arxiv.org/abs/1608.01647)

## Attribution Prediction

**PANDA: Pose Aligned Networks for Deep Attribute Modeling (Facebook. CVPR 2014)**

- arxiv: [http://arxiv.org/abs/1311.5591](http://arxiv.org/abs/1311.5591)
- github: [https://github.com/facebook/pose-aligned-deep-networks](https://github.com/facebook/pose-aligned-deep-networks)

**Predicting psychological attributions from face photographs with a deep neural network**

- arxiv: [http://arxiv.org/abs/1512.01289](http://arxiv.org/abs/1512.01289)

**Learning Human Identity from Motion Patterns**

- arxiv: [http://arxiv.org/abs/1511.03908](http://arxiv.org/abs/1511.03908)

## Pose Estimation

**DeepPose: Human Pose Estimation via Deep Neural Networks (CVPR 2014)**

- arxiv: [http://arxiv.org/abs/1312.4659](http://arxiv.org/abs/1312.4659)
- slides: [http://140.122.184.143/paperlinks/Slides/DeepPose_HumanPose_Estimation_via_Deep_Neural_Networks.pptx](http://140.122.184.143/paperlinks/Slides/DeepPose_HumanPose_Estimation_via_Deep_Neural_Networks.pptx)

**Flowing ConvNets for Human Pose Estimation in Videos**

- arxiv: [http://arxiv.org/abs/1506.02897](http://arxiv.org/abs/1506.02897)
- homepage: [http://www.robots.ox.ac.uk/~vgg/software/cnn_heatmap/](http://www.robots.ox.ac.uk/~vgg/software/cnn_heatmap/)
- github: [https://github.com/tpfister/caffe-heatmap](https://github.com/tpfister/caffe-heatmap)

**Structured Feature Learning for Pose Estimation**

- arxiv: [http://arxiv.org/abs/1603.09065](http://arxiv.org/abs/1603.09065)
- homepage: [http://www.ee.cuhk.edu.hk/~xgwang/projectpage_structured_feature_pose.html](http://www.ee.cuhk.edu.hk/~xgwang/projectpage_structured_feature_pose.html)

**Convolutional Pose Machines**

- arxiv: [http://arxiv.org/abs/1602.00134](http://arxiv.org/abs/1602.00134)
- github: [https://github.com/shihenw/convolutional-pose-machines-release](https://github.com/shihenw/convolutional-pose-machines-release)

**Model-based Deep Hand Pose Estimation**

- paper: [http://xingyizhou.xyz/zhou2016model.pdf](http://xingyizhou.xyz/zhou2016model.pdf)
- github: [https://github.com/tenstep/DeepModel](https://github.com/tenstep/DeepModel)

**Stacked Hourglass Networks for Human Pose Estimation**

![](http://www-personal.umich.edu/~alnewell/images/stacked-hg.png)

- homepage: [http://www-personal.umich.edu/~alnewell/pose/](http://www-personal.umich.edu/~alnewell/pose/)
- arxiv: [http://arxiv.org/abs/1603.06937](http://arxiv.org/abs/1603.06937)
- github: [https://github.com/anewell/pose-hg-train](https://github.com/anewell/pose-hg-train)
- demo: [https://github.com/anewell/pose-hg-demo](https://github.com/anewell/pose-hg-demo)

**Chained Predictions Using Convolutional Neural Networks (EECV 2016)**

- keywords: CNN, structured prediction, RNN, human pose estimation
- arxiv: [http://arxiv.org/abs/1605.02346](http://arxiv.org/abs/1605.02346)

**DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model**

- arxiv: [http://arxiv.org/abs/1605.03170](http://arxiv.org/abs/1605.03170)
- github: [https://github.com/eldar/deepcut-cnn](https://github.com/eldar/deepcut-cnn)

## Sentiment Prediction

**From Pixels to Sentiment: Fine-tuning CNNs for Visual Sentiment Prediction**

- arxiv: [http://arxiv.org/abs/1604.03489](http://arxiv.org/abs/1604.03489)
- github: [https://github.com/imatge-upc/sentiment-2016](https://github.com/imatge-upc/sentiment-2016)
- gitxiv: [http://gitxiv.com/posts/ruqRgXdPTHJ77LDEb/from-pixels-to-sentiment-fine-tuning-cnns-for-visual](http://gitxiv.com/posts/ruqRgXdPTHJ77LDEb/from-pixels-to-sentiment-fine-tuning-cnns-for-visual)

**Predict Sentiment From Movie Reviews Using Deep Learning**

- blog: [http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/](http://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/)

**Neural Sentiment Classification with User and Product Attention**

- intro: EMNLP 2016
- paper: [http://www.thunlp.org/~chm/publications/emnlp2016_NSCUPA.pdf](http://www.thunlp.org/~chm/publications/emnlp2016_NSCUPA.pdf)
- github: [https://github.com/thunlp/NSC](https://github.com/thunlp/NSC)

## Place Recognition

**NetVLAD: CNN architecture for weakly supervised place recognition**

![](http://www.di.ens.fr/willow/research/netvlad/images/teaser.png)

- arxiv: [http://arxiv.org/abs/1511.07247](http://arxiv.org/abs/1511.07247)
- homepage: [http://www.di.ens.fr/willow/research/netvlad/](http://www.di.ens.fr/willow/research/netvlad/)

**PlaNet - Photo Geolocation with Convolutional Neural Networks**

![](https://d267cvn3rvuq91.cloudfront.net/i/images/planet.jpg?sw=590&cx=0&cy=0&cw=928&ch=614)

- arxiv: [http://arxiv.org/abs/1602.05314](http://arxiv.org/abs/1602.05314)
- review("Google Unveils Neural Network with “Superhuman” Ability to Determine the Location of Almost Any Image"): [https://www.technologyreview.com/s/600889/google-unveils-neural-network-with-superhuman-ability-to-determine-the-location-of-almost/](https://www.technologyreview.com/s/600889/google-unveils-neural-network-with-superhuman-ability-to-determine-the-location-of-almost/)
- github("City-Recognition: CS231n Project for Winter 2016"): [https://github.com/dmakian/LittlePlaNet](https://github.com/dmakian/LittlePlaNet)
- github: [https://github.com/wulfebw/LittlePlaNet-Models](https://github.com/wulfebw/LittlePlaNet-Models)

### Camera Relocalization

**PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization**

- paper: [http://arxiv.org/abs/1505.07427](http://arxiv.org/abs/1505.07427)
- project page: [http://mi.eng.cam.ac.uk/projects/relocalisation/#results](http://mi.eng.cam.ac.uk/projects/relocalisation/#results)
- github: [https://github.com/alexgkendall/caffe-posenet](https://github.com/alexgkendall/caffe-posenet)

**Modelling Uncertainty in Deep Learning for Camera Relocalization**

- paper: [http://arxiv.org/abs/1509.05909](http://arxiv.org/abs/1509.05909)

## Crowd Counting / Analysis

**Large scale crowd analysis based on convolutional neural network**

- paper: [http://www.sciencedirect.com/science/article/pii/S0031320315001259](http://www.sciencedirect.com/science/article/pii/S0031320315001259)

**Cross-scene Crowd Counting via Deep Convolutional Neural Networks (CVPR 2015)**

- paper: [http://www.ee.cuhk.edu.hk/~xgwang/papers/zhangLWYcvpr15.pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/zhangLWYcvpr15.pdf)

**Single-Image Crowd Counting via Multi-Column Convolutional Neural Network (CVPR 2016)**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf)
- dataset(pwd: p1rv): [http://pan.baidu.com/s/1gfyNBTh](http://pan.baidu.com/s/1gfyNBTh)

### CrowdNet

**CrowdNet: A Deep Convolutional Network for Dense Crowd Counting**

- intro: ACM Multimedia (MM) 2016
- arxiv: [http://arxiv.org/abs/1608.06197](http://arxiv.org/abs/1608.06197)

## Music / Sound Classification

**Explaining Deep Convolutional Neural Networks on Music Classification**

- arxiv: [http://arxiv.org/abs/1607.02444](http://arxiv.org/abs/1607.02444)
- blog: [https://keunwoochoi.wordpress.com/2015/12/09/ismir-2015-lbd-auralisation-of-deep-convolutional-neural-networks-listening-to-learned-features-auralization/](https://keunwoochoi.wordpress.com/2015/12/09/ismir-2015-lbd-auralisation-of-deep-convolutional-neural-networks-listening-to-learned-features-auralization/)
- blog: [https://keunwoochoi.wordpress.com/2016/03/23/what-cnns-see-when-cnns-see-spectrograms/](https://keunwoochoi.wordpress.com/2016/03/23/what-cnns-see-when-cnns-see-spectrograms/)
- github: [https://github.com/keunwoochoi/Auralisation](https://github.com/keunwoochoi/Auralisation)
- audio samples: [https://soundcloud.com/kchoi-research](https://soundcloud.com/kchoi-research)

**Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification**

- arxiv: [http://arxiv.org/abs/1608.04363](http://arxiv.org/abs/1608.04363)
- project page: [http://www.stat.ucla.edu/~yang.lu/project/deepFrame/main.html](http://www.stat.ucla.edu/~yang.lu/project/deepFrame/main.html)

## NSFW Detection / Classification

**Nipple Detection using Convolutional Neural Network**

- reddit: [https://www.reddit.com/over18?dest=https%3A%2F%2Fwww.reddit.com%2Fr%2FMachineLearning%2Fcomments%2F33n77s%2Fandroid_app_nipple_detection_using_convolutional%2F](https://www.reddit.com/over18?dest=https%3A%2F%2Fwww.reddit.com%2Fr%2FMachineLearning%2Fcomments%2F33n77s%2Fandroid_app_nipple_detection_using_convolutional%2F)

**Applying deep learning to classify pornographic images and videos**

- arxiv: [http://arxiv.org/abs/1511.08899](http://arxiv.org/abs/1511.08899)

**MODERATE, FILTER, OR CURATE ADULT CONTENT WITH CLARIFAI’S NSFW MODEL**

- blog: [http://blog.clarifai.com/moderate-filter-or-curate-adult-content-with-clarifais-nsfw-model/#.VzVhM-yECZY](http://blog.clarifai.com/moderate-filter-or-curate-adult-content-with-clarifais-nsfw-model/#.VzVhM-yECZY)

**WHAT CONVOLUTIONAL NEURAL NETWORKS LOOK AT WHEN THEY SEE NUDITY**

- blog: [http://blog.clarifai.com/what-convolutional-neural-networks-see-at-when-they-see-nudity#.VzVh_-yECZY](http://blog.clarifai.com/what-convolutional-neural-networks-see-at-when-they-see-nudity#.VzVh_-yECZY)

## Image Reconstruction / Inpainting

**Context Encoders: Feature Learning by Inpainting (CVPR 2016)**

![](http://www.cs.berkeley.edu/~pathak/context_encoder/resources/result_fig.jpg)

- project page: [http://www.cs.berkeley.edu/~pathak/context_encoder/](http://www.cs.berkeley.edu/~pathak/context_encoder/)
- arxiv: [https://arxiv.org/abs/1604.07379](https://arxiv.org/abs/1604.07379)
- github: [https://github.com/pathak22/context-encoder](https://github.com/pathak22/context-encoder)

**Semantic Image Inpainting with Perceptual and Contextual Losses**

- keywords: Deep Convolutional Generative Adversarial Network (DCGAN)
- arxiv: [http://arxiv.org/abs/1607.07539](http://arxiv.org/abs/1607.07539)

## Image Restoration

**Image Restoration Using Convolutional Auto-encoders with Symmetric Skip Connections**

- arxiv: [http://arxiv.org/abs/1606.08921](http://arxiv.org/abs/1606.08921)

**Image Completion with Deep Learning in TensorFlow**

- blog: [http://bamos.github.io/2016/08/09/deep-completion/](http://bamos.github.io/2016/08/09/deep-completion/)

## Image Super-Resolution

**Image Super-Resolution Using Deep Convolutional Networks (Microsoft Research)**

![](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/img/figure1.png)

- project page: [http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)
- arxiv: [http://arxiv.org/abs/1501.00092](http://arxiv.org/abs/1501.00092)
- training code: [http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_train.zip)
- test code: [http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_v1.zip](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN/SRCNN_v1.zip)

**Learning a Deep Convolutional Network for Image Super-Resolution**

- Baidu-pan: [http://pan.baidu.com/s/1c0k0wRu](http://pan.baidu.com/s/1c0k0wRu)

**Shepard Convolutional Neural Networks**

![](/assets/cnn-materials/comic_bicubic_x3.png)  ![](/assets/cnn-materials/comic_shcnn_x3.png) <br>
Bicubic VS. Shepard CNN

- paper: [https://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf](https://papers.nips.cc/paper/5774-shepard-convolutional-neural-networks.pdf)
- github: [https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/Shepard_CNN](https://github.com/jimmy-ren/vcnn_double-bladed/tree/master/applications/Shepard_CNN)

**Bidirectional Recurrent Convolutional Networks for Multi-Frame Super-Resolution**

- intro: NIPS 2015
- paper: [https://papers.nips.cc/paper/5778-bidirectional-recurrent-convolutional-networks-for-multi-frame-super-resolution](https://papers.nips.cc/paper/5778-bidirectional-recurrent-convolutional-networks-for-multi-frame-super-resolution)

**Deeply-Recursive Convolutional Network for Image Super-Resolution**

- arxiv: [http://arxiv.org/abs/1511.04491](http://arxiv.org/abs/1511.04491)

**Accurate Image Super-Resolution Using Very Deep Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1511.04587](http://arxiv.org/abs/1511.04587)

**Super-Resolution with Deep Convolutional Sufficient Statistics**

- arxiv: [http://arxiv.org/abs/1511.05666](http://arxiv.org/abs/1511.05666)

**Deep Depth Super-Resolution : Learning Depth Super-Resolution using Deep Convolutional Neural Network**

- arxiv: [http://arxiv.org/abs/1607.01977](http://arxiv.org/abs/1607.01977)

**Local- and Holistic- Structure Preserving Image Super Resolution via Deep Joint Component Learning**

- arxiv: [http://arxiv.org/abs/1607.07220](http://arxiv.org/abs/1607.07220)

**End-to-End Image Super-Resolution via Deep and Shallow Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1607.07680](http://arxiv.org/abs/1607.07680)

**Accelerating the Super-Resolution Convolutional Neural Network**

- intro: speed up of more than 40 times with even superior restoration quality, real-time performance on a generic CPU
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html)
- arxiv: [http://arxiv.org/abs/1608.00367](http://arxiv.org/abs/1608.00367)

**srez: Image super-resolution through deep learning**

- github: [https://github.com/david-gpu/srez](https://github.com/david-gpu/srez)

## Image Denoising

**Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising**

- arxiv: [http://arxiv.org/abs/1608.03981](http://arxiv.org/abs/1608.03981)
- github: [https://github.com/cszn/DnCNN](https://github.com/cszn/DnCNN)

**Medical image denoising using convolutional denoising autoencoders**

- arxiv: [http://arxiv.org/abs/1608.04667](http://arxiv.org/abs/1608.04667)

## Image Haze Removal

**DehazeNet: An End-to-End System for Single Image Haze Removal**

- arxiv: [http://arxiv.org/abs/1601.07661](http://arxiv.org/abs/1601.07661)

## Blur Detection and Removal

**Learning to Deblur**

- arxiv: [http://arxiv.org/abs/1406.7444](http://arxiv.org/abs/1406.7444)

**Learning a Convolutional Neural Network for Non-uniform Motion Blur Removal**

- arxiv: [http://arxiv.org/abs/1503.00593](http://arxiv.org/abs/1503.00593)

**End-to-End Learning for Image Burst Deblurring**

- arxiv: [http://arxiv.org/abs/1607.04433](http://arxiv.org/abs/1607.04433)

## Image Compression

**An image compression and encryption scheme based on deep learning**

- arxiv: [http://arxiv.org/abs/1608.05001](http://arxiv.org/abs/1608.05001)

**Full Resolution Image Compression with Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.05148](http://arxiv.org/abs/1608.05148)

## Depth Prediction

**Deeper Depth Prediction with Fully Convolutional Residual Networks**

![](https://camo.githubusercontent.com/eaee1af707b9ad2b2ac3b5be87de82fc590800d2/687474703a2f2f63616d7061722e696e2e74756d2e64652f66696c65732f7275707072656368742f6465707468707265642f696d616765732e6a7067)

- arxiv: [https://arxiv.org/abs/1606.00373](https://arxiv.org/abs/1606.00373)
- github: [https://github.com/iro-cp/FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction)

## Texture Synthesis

**Texture Synthesis Using Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1505.07376](http://arxiv.org/abs/1505.07376)

**Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis**

- arxiv: [http://arxiv.org/abs/1601.04589](http://arxiv.org/abs/1601.04589)

**Texture Networks: Feed-forward Synthesis of Textures and Stylized Images**

![](https://raw.githubusercontent.com/DmitryUlyanov/texture_nets/master/data/readme_pics/all.jpg)

- intro: IMCL 2016
- arxiv: [http://arxiv.org/abs/1603.03417](http://arxiv.org/abs/1603.03417)
- github: [https://github.com/DmitryUlyanov/texture_nets](https://github.com/DmitryUlyanov/texture_nets)

**Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks**

- arxiv: [http://arxiv.org/abs/1604.04382](http://arxiv.org/abs/1604.04382)
- github(Torch): [https://github.com/chuanli11/MGANs](https://github.com/chuanli11/MGANs)

**Generative Adversarial Text to Image Synthesis**

![](https://camo.githubusercontent.com/1925e23b5b6e19efa60f45daa3787f1f4a098ef3/687474703a2f2f692e696d6775722e636f6d2f644e6c32486b5a2e6a7067)

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1605.05396](http://arxiv.org/abs/1605.05396)
- github(Tensorflow): [https://github.com/paarthneekhara/text-to-image](https://github.com/paarthneekhara/text-to-image)

## Image Tagging

**Flexible Image Tagging with Fast0Tag**

![](https://cdn-images-1.medium.com/max/800/1*SsIf1Bhe-G4HmN6DPDogmQ.png)

- blog: [https://gab41.lab41.org/flexible-image-tagging-with-fast0tag-681c6283c9b7](https://gab41.lab41.org/flexible-image-tagging-with-fast0tag-681c6283c9b7)

## Music Tagging

**Automatic tagging using deep convolutional neural networks**

- arxiv: [https://arxiv.org/abs/1606.00298](https://arxiv.org/abs/1606.00298)
- github: [https://github.com/keunwoochoi/music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

# Deep Learning’s Accuracy

- blog: [http://deeplearning4j.org/accuracy.html](http://deeplearning4j.org/accuracy.html)

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

**Deep learning (Nature 2015)**

- author: Yann LeCun, Yoshua Bengio & Geoffrey Hinton
- paper: [http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)

**On the Expressive Power of Deep Learning: A Tensor Analysis**

- paper: [http://arxiv.org/abs/1509.05009](http://arxiv.org/abs/1509.05009)

**Understanding and Predicting Image Memorability at a Large Scale (MIT. ICCV 2015)**

- homepage: [http://memorability.csail.mit.edu/](http://memorability.csail.mit.edu/)
- paper: [https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf](https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf)
- code: [http://memorability.csail.mit.edu/download.html](http://memorability.csail.mit.edu/download.html)
- reviews: [http://petapixel.com/2015/12/18/how-memorable-are-times-top-10-photos-of-2015-to-a-computer/](http://petapixel.com/2015/12/18/how-memorable-are-times-top-10-photos-of-2015-to-a-computer/)

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

**Deep-Spying: Spying using Smartwatch and Deep Learning**

- arxiv: [http://arxiv.org/abs/1512.05616](http://arxiv.org/abs/1512.05616)
- github: [https://github.com/tonybeltramelli/Deep-Spying](https://github.com/tonybeltramelli/Deep-Spying)

**A Mathematical Theory of Deep Convolutional Neural Networks for Feature Extraction**

- arxiv: [http://arxiv.org/abs/1512.06293](http://arxiv.org/abs/1512.06293)

**Understanding Deep Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1601.04920](http://arxiv.org/abs/1601.04920)

**DeepCare: A Deep Dynamic Memory Model for Predictive Medicine**

- arxiv: [http://arxiv.org/abs/1602.00357](http://arxiv.org/abs/1602.00357)

**Exploiting Cyclic Symmetry in Convolutional Neural Networks (ICML 2016)**

![](http://benanne.github.io/images/cyclicroll.png)

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

**Learning Convolutional Neural Networks for Graphs**

- arxiv: [http://arxiv.org/abs/1605.05273](http://arxiv.org/abs/1605.05273)

**Unreasonable Effectiveness of Learning Neural Nets: Accessible States and Robust Ensembles**

- arxiv: [http://arxiv.org/abs/1605.06444](http://arxiv.org/abs/1605.06444)

**FractalNet: Ultra-Deep Neural Networks without Residuals**

![](http://people.cs.uchicago.edu/~larsson/fractalnet/overview.png)

- project: [http://people.cs.uchicago.edu/~larsson/fractalnet/](http://people.cs.uchicago.edu/~larsson/fractalnet/)
- arxiv: [http://arxiv.org/abs/1605.07648](http://arxiv.org/abs/1605.07648)
- github: [https://github.com/gustavla/fractalnet](https://github.com/gustavla/fractalnet)
- github: [https://github.com/edgelord/FractalNet](https://github.com/edgelord/FractalNet)
- github(keras): [https://github.com/snf/keras-fractalnet](https://github.com/snf/keras-fractalnet)

**Newtonian Image Understanding: Unfolding the Dynamics of Objects in Static Images**

![](http://allenai.org/images/projects/plato_newton.png?cb=1466683222538)

- homepage: [http://allenai.org/plato/newtonian-understanding/](http://allenai.org/plato/newtonian-understanding/)
- arxiv: [http://arxiv.org/abs/1511.04048](http://arxiv.org/abs/1511.04048)
- github: [https://github.com/roozbehm/newtonian](https://github.com/roozbehm/newtonian)

**Convolutional Neural Networks Analyzed via Convolutional Sparse Coding**

- arxiv: [http://arxiv.org/abs/1607.08194](http://arxiv.org/abs/1607.08194)

**Recent Advances in Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1512.07108](http://arxiv.org/abs/1512.07108)

**TI-POOLING: transformation-invariant pooling for feature learning in Convolutional Neural Networks**

- intro: CVPR 2016
- paper: [http://dlaptev.org/papers/Laptev16_CVPR.pdf](http://dlaptev.org/papers/Laptev16_CVPR.pdf)
- github: [https://github.com/dlaptev/TI-pooling](https://github.com/dlaptev/TI-pooling)

**Why does deep and cheap learning work so well?**

- arxiv: [http://arxiv.org/abs/1608.08225](http://arxiv.org/abs/1608.08225)

## STDP

**A biological gradient descent for prediction through a combination of STDP and homeostatic plasticity**

- arxiv: [http://arxiv.org/abs/1206.4812](http://arxiv.org/abs/1206.4812)

**An objective function for STDP(Yoshua Bengio)**

- arxiv: [http://arxiv.org/abs/1509.05936](http://arxiv.org/abs/1509.05936)

**Towards a Biologically Plausible Backprop**

- arxiv: [http://arxiv.org/abs/1602.05179](http://arxiv.org/abs/1602.05179)

## Target Propagation

**How Auto-Encoders Could Provide Credit Assignment in Deep Networks via Target Propagation (Yoshua Bengio)**

- arxiv: [http://arxiv.org/abs/1407.7906](http://arxiv.org/abs/1407.7906)

**Difference Target Propagation**

- arxiv: [http://arxiv.org/abs/1412.7525](http://arxiv.org/abs/1412.7525)
- github: [https://github.com/donghyunlee/dtp](https://github.com/donghyunlee/dtp)

## CNN with Computer Vision

**End-to-End Integration of a Convolutional Network, Deformable Parts Model and Non-Maximum Suppression**

- arxiv: [http://arxiv.org/abs/1411.5309](http://arxiv.org/abs/1411.5309)

**A convnet for non-maximum suppression**

- arxiv: [http://arxiv.org/abs/1511.06437](http://arxiv.org/abs/1511.06437)

**A Taxonomy of Deep Convolutional Neural Nets for Computer Vision**

- arxiv: [http://arxiv.org/abs/1601.06615](http://arxiv.org/abs/1601.06615)

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

![](https://raw.githubusercontent.com/Kadenze/siamese_net/master/images/prediction.png)

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

![](https://raw.githubusercontent.com/h2oai/deepwater/master/architecture/overview.png)

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

![](https://raw.githubusercontent.com/Erkaman/regl-cnn/gh-pages/gifs/record_resize.gif)

- intro: TensorFlow, WebGL, [regl](https://github.com/mikolalysenko/regl)
- github: [https://github.com/Erkaman/regl-cnn/](https://github.com/Erkaman/regl-cnn/)
- demo: [https://erkaman.github.io/regl-cnn/src/demo.html](https://erkaman.github.io/regl-cnn/src/demo.html)

## gvnn

**gvnn: Neural Network Library for Geometric Computer Vision**

- arxiv: [http://arxiv.org/abs/1607.07405](http://arxiv.org/abs/1607.07405)
- github: [https://github.com/ankurhanda/gvnn](https://github.com/ankurhanda/gvnn)

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

![](http://machinelearningmastery.com/wp-content/uploads/2016/04/Deep-Learning-Books.jpg)

- blog: [http://machinelearningmastery.com/deep-learning-books/](http://machinelearningmastery.com/deep-learning-books/)

**awesome-very-deep-learning: A curated list of papers and code about very deep neural networks (50+ layers)**

- github: [https://github.com/daviddao/awesome-very-deep-learning](https://github.com/daviddao/awesome-very-deep-learning)

**Deep Learning Resources and Tutorials using Keras and Lasagne**

- github: [https://github.com/Vict0rSch/deep_learning](https://github.com/Vict0rSch/deep_learning)

**Deep Learning: Definition, Resources, Comparison with Machine Learning**

![](http://api.ning.com/files/mRIiJdI0bcyTIJftimMbMYW4VRcz-NC1gCERsgtyu*mNietZRBt5g3AUs06WtU2BigiPWs1MvWCYq6bsuWrNbsG1KBqrcm8b/Capture.PNG)

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

**Deep Learning (by Ian Goodfellow, Aaron Courville and Yoshua Bengio)**

- homepage: [http://www.deeplearningbook.org/](http://www.deeplearningbook.org/)
- website: [http://goodfeli.github.io/dlbook/](http://goodfeli.github.io/dlbook/)
- notes("Deep Learning for Beginners"): [http://randomekek.github.io/deep/deeplearning.html](http://randomekek.github.io/deep/deeplearning.html)

**Fundamentals of Deep Learning: Designing Next-Generation Artificial Intelligence Algorithms (Nikhil Buduma)**

- book review: [http://www.opengardensblog.futuretext.com/archives/2015/08/book-review-fundamentals-of-deep-learning-designing-next-generation-artificial-intelligence-algorithms-by-nikhil-buduma.html](http://www.opengardensblog.futuretext.com/archives/2015/08/book-review-fundamentals-of-deep-learning-designing-next-generation-artificial-intelligence-algorithms-by-nikhil-buduma.html)
- github: [https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book](https://github.com/darksigma/Fundamentals-of-Deep-Learning-Book)

**FIRST CONTACT WITH TENSORFLOW: Get started with with Deep Learning programming (by Jordi Torres)**

[http://www.jorditorres.org/first-contact-with-tensorflow/](http://www.jorditorres.org/first-contact-with-tensorflow/)

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
