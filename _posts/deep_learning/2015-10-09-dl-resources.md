---
layout: post
category: deep_learning
title: Deep Learning Resources
date: 2015-10-09
---

* TOC
{:toc}

# ImageNet

## AlexNet

**ImageNet Classification with Deep Convolutional Neural Networks**

- nips-page: [http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-)
- paper: [http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- slides: [http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf](http://www.image-net.org/challenges/LSVRC/2012/supervision.pdf)

## Network In Network

**Network In Network**

![](https://culurciello.github.io/assets/nets/nin.jpg)

- arxiv: [http://arxiv.org/abs/1312.4400](http://arxiv.org/abs/1312.4400)
- gitxiv: [http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin](http://gitxiv.com/posts/PA98qGuMhsijsJzgX/network-in-network-nin)

**Batch-normalized Maxout Network in Network**

- arxiv: [http://arxiv.org/abs/1511.02583](http://arxiv.org/abs/1511.02583)

## GoogLeNet

**Going Deeper with Convolutions**

- paper: [http://arxiv.org/abs/1409.4842](http://arxiv.org/abs/1409.4842)
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

## Inception-v3

**Rethinking the Inception Architecture for Computer Vision**

- intro: "21.2% top-1 and 5.6% top-5 error for single frame evaluation using a network; 
3.5% top-5 error and 17.3% top-1 error With an ensemble of 4 models and multi-crop evaluation."
- arXiv: [http://arxiv.org/abs/1512.00567](http://arxiv.org/abs/1512.00567)
- github(Torch): [https://github.com/Moodstocks/inception-v3.torch](https://github.com/Moodstocks/inception-v3.torch)
- github(TensorFlow): [https://github.com/tensorflow/models/tree/master/inception#how-to-train-from-scratch-in-a-distributed-setting](https://github.com/tensorflow/models/tree/master/inception#how-to-train-from-scratch-in-a-distributed-setting)

## ResNet

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

**Training and investigating Residual Nets**

[http://torch.ch/blog/2016/02/04/resnets.html](http://torch.ch/blog/2016/02/04/resnets.html)

**Highway Networks and Deep Residual Networks** 

- blog: [http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html](http://yanran.li/peppypapers/2016/01/10/highway-networks-and-deep-residual-networks.html)

**Interpretating Deep Residual Learning Blocks as Locally Recurrent Connections**
 
- blog: [https://matrixmashing.wordpress.com/2016/01/29/interpretating-deep-residual-learning-blocks-as-locally-recurrent-connections/](https://matrixmashing.wordpress.com/2016/01/29/interpretating-deep-residual-learning-blocks-as-locally-recurrent-connections/)

**Resnet in Resnet: Generalizing Residual Architectures**

- paper: [http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g](http://beta.openreview.net/forum?id=lx9l4r36gU2OVPy8Cv9g)
- arxiv: [http://arxiv.org/abs/1603.08029](http://arxiv.org/abs/1603.08029)

**Identity Mappings in Deep Residual Networks (by Kaiming He)**

- arxiv: [http://arxiv.org/abs/1603.05027](http://arxiv.org/abs/1603.05027)
- github: [https://github.com/KaimingHe/resnet-1k-layers](https://github.com/KaimingHe/resnet-1k-layers)
- github: [https://github.com/bazilas/matconvnet-ResNet](https://github.com/bazilas/matconvnet-ResNet)
- github: [https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne](https://github.com/FlorianMuellerklein/Identity-Mapping-ResNet-Lasagne)

**Residual Networks are Exponential Ensembles of Relatively Shallow Networks**

- arxiv: [http://arxiv.org/abs/1605.06431](http://arxiv.org/abs/1605.06431)

**Wide Residual Networks**

- arxiv: [http://arxiv.org/abs/1605.07146](http://arxiv.org/abs/1605.07146)
- github: [https://github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks)
- github: [https://github.com/asmith26/wide_resnets_keras](https://github.com/asmith26/wide_resnets_keras)

## Inception-V4

**Inception-V4, Inception-Resnet And The Impact Of Residual Connections On Learning (Workshop track - ICLR 2016)**

- intro: "achieve 3.08% top-5 error on the test set of the ImageNet classification (CLS) challenge"
- arxiv: [http://arxiv.org/abs/1602.07261](http://arxiv.org/abs/1602.07261)
- paper: [http://beta.openreview.net/pdf?id=q7kqBkL33f8LEkD3t7X9](http://beta.openreview.net/pdf?id=q7kqBkL33f8LEkD3t7X9)

- - -

**Striving for Simplicity: The All Convolutional Net**

- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

**Systematic evaluation of CNN advances on the ImageNet**

- arxiv: [http://arxiv.org/abs/1606.02228](http://arxiv.org/abs/1606.02228)
- github: [https://github.com/ducha-aiki/caffenet-benchmark](https://github.com/ducha-aiki/caffenet-benchmark)

# Tensor

**Tensorizing Neural Networks (TensorNet)**

- paper: [http://arxiv.org/abs/1509.06569v1](http://arxiv.org/abs/1509.06569v1)
- github(Matlab+Theano+Lasagne): [https://github.com/Bihaqo/TensorNet](https://github.com/Bihaqo/TensorNet)
- github(TensorFlow): [https://github.com/timgaripov/TensorNet-TF](https://github.com/timgaripov/TensorNet-TF)

**On the Expressive Power of Deep Learning: A Tensor Analysis**

- paper: [http://arxiv.org/abs/1509.05009](http://arxiv.org/abs/1509.05009)

**Convolutional neural networks with low-rank regularization**

- arxiv: [http://arxiv.org/abs/1511.06067](http://arxiv.org/abs/1511.06067)
- github: [https://github.com/chengtaipu/lowrankcnn](https://github.com/chengtaipu/lowrankcnn)

**Tensor methods for training neural networks**

- homepage: [http://newport.eecs.uci.edu/anandkumar/#home](http://newport.eecs.uci.edu/anandkumar/#home)
- youtube: [https://www.youtube.com/watch?v=B4YvhcGaafw](https://www.youtube.com/watch?v=B4YvhcGaafw)
- slides: [http://newport.eecs.uci.edu/anandkumar/slides/Strata-NY.pdf](http://newport.eecs.uci.edu/anandkumar/slides/Strata-NY.pdf)
- talks: [http://newport.eecs.uci.edu/anandkumar/#talks](http://newport.eecs.uci.edu/anandkumar/#talks)

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

# Autoencoders

**Importance Weighted Autoencoders**

- paper: [http://arxiv.org/abs/1509.00519](http://arxiv.org/abs/1509.00519)
- github: [https://github.com/yburda/iwae](https://github.com/yburda/iwae)

**Review of Auto-Encoders(by Piotr Mirowski, Microsoft Bing London, 2014)**

- slides: [https://piotrmirowski.files.wordpress.com/2014/03/piotrmirowski_2014_reviewautoencoders.pdf](https://piotrmirowski.files.wordpress.com/2014/03/piotrmirowski_2014_reviewautoencoders.pdf)
- github: [https://github.com/piotrmirowski/Tutorial_AutoEncoders/](https://github.com/piotrmirowski/Tutorial_AutoEncoders/)

**Stacked What-Where Auto-encoders**

- arxiv: [http://arxiv.org/abs/1506.02351](http://arxiv.org/abs/1506.02351)

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

# Semi-Supervised Learning

**Semi-Supervised Learning with Graphs (Label Propagation)**

- paper: [http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf](http://pages.cs.wisc.edu/~jerryzhu/pub/thesis.pdf)
- blog("标签传播算法（Label Propagation）及Python实现"): [http://blog.csdn.net/zouxy09/article/details/49105265](http://blog.csdn.net/zouxy09/article/details/49105265)

# Unsupervised Learning

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
- arXiv: [http://arxiv.org/abs/1506.00990](http://arxiv.org/abs/1506.00990)
- github: [https://github.com/yaolubrain/ULNNO](https://github.com/yaolubrain/ULNNO)

**Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles**

- arxiv: [http://arxiv.org/abs/1603.09246](http://arxiv.org/abs/1603.09246)
- notes: [http://www.inference.vc/notes-on-unsupervised-learning-of-visual-representations-by-solving-jigsaw-puzzles/](http://www.inference.vc/notes-on-unsupervised-learning-of-visual-representations-by-solving-jigsaw-puzzles/)

**Joint Unsupervised Learning of Deep Representations and Image Clusters (CVPR 2016)**

- arxiv: [https://arxiv.org/abs/1604.03628](https://arxiv.org/abs/1604.03628)
- github(Torch): [https://github.com/jwyang/joint-unsupervised-learning](https://github.com/jwyang/joint-unsupervised-learning)

**Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning (PredNet)**

- arxiv: [http://arxiv.org/abs/1605.08104](http://arxiv.org/abs/1605.08104)

# Deep Learning Networks

**Deeply-supervised Nets (DSN)**

![](http://vcl.ucsd.edu/~sxie/images/dsn/architecture.png)

- arxiv: [http://arxiv.org/abs/1409.5185](http://arxiv.org/abs/1409.5185)
- homepage: [http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/](http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/)
- github: [https://github.com/s9xie/DSN](https://github.com/s9xie/DSN)
- notes: [http://zhangliliang.com/2014/11/02/paper-note-dsn/](http://zhangliliang.com/2014/11/02/paper-note-dsn/)

**Striving for Simplicity: The All Convolutional Net**

- arxiv: [http://arxiv.org/abs/1412.6806](http://arxiv.org/abs/1412.6806)

**Highway Networks**

- arxiv: [http://arxiv.org/abs/1505.00387](http://arxiv.org/abs/1505.00387)
- blog("Highway Networks with TensorFlow"): [https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.71fgztsb6](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.71fgztsb6)

**Training Very Deep Networks (highway networks)**

- arxiv: [http://arxiv.org/abs/1507.06228](http://arxiv.org/abs/1507.06228)

**Very Deep Learning with Highway Networks**

- homepage(papers, code, FAQ): [http://people.idsia.ch/~rupesh/very_deep_learning/](http://people.idsia.ch/~rupesh/very_deep_learning/)

**Rectified Factor Networks**

- arXiv: [http://arxiv.org/abs/1502.06464](http://arxiv.org/abs/1502.06464)
- github: [https://github.com/untom/librfn](https://github.com/untom/librfn)

**Correlational Neural Networks**

- arxiv: [http://arxiv.org/abs/1504.07225](http://arxiv.org/abs/1504.07225)

**Semi-Supervised Learning with Ladder Networks**

- arxiv: [http://arxiv.org/abs/1507.02672](http://arxiv.org/abs/1507.02672)
- github: [https://github.com/CuriousAI/ladder](https://github.com/CuriousAI/ladder)

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

**Binarized Neural Networks**

- arxiv: [http://arxiv.org/abs/1602.02505](http://arxiv.org/abs/1602.02505)

**BinaryConnect: Training Deep Neural Networks with binary weights during propagations**

- paper: [http://papers.nips.cc/paper/5647-shape-and-illumination-from-shading-using-the-generic-viewpoint-assumption](http://papers.nips.cc/paper/5647-shape-and-illumination-from-shading-using-the-generic-viewpoint-assumption)
- github: [https://github.com/MatthieuCourbariaux/BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect)

**A Theory of Generative ConvNet**

- arxiv: [http://arxiv.org/abs/1602.03264](http://arxiv.org/abs/1602.03264)
- project page: [http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html](http://www.stat.ucla.edu/~ywu/GenerativeConvNet/main.html)

**Value Iteration Networks**

- arxiv: [http://arxiv.org/abs/1602.02867](http://arxiv.org/abs/1602.02867)

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

**Deeply-Fused Nets**

- arxiv: [http://arxiv.org/abs/1605.07716](http://arxiv.org/abs/1605.07716)

**SNN: Stacked Neural Networks**

- arxiv: [http://arxiv.org/abs/1605.08512](http://arxiv.org/abs/1605.08512)

# Deep Learning’s Accuracy

- blog: [http://deeplearning4j.org/accuracy.html](http://deeplearning4j.org/accuracy.html)

# Papers

**Reweighted Wake-Sleep**

- paper: [http://arxiv.org/abs/1406.2751](http://arxiv.org/abs/1406.2751)
- code: [https://github.com/jbornschein/reweighted-ws](https://github.com/jbornschein/reweighted-ws)

**Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks**

- paper: [http://arxiv.org/abs/1502.05336](http://arxiv.org/abs/1502.05336)
- code: [https://github.com/HIPS/Probabilistic-Backpropagation](https://github.com/HIPS/Probabilistic-Backpropagation)

**Deeply-Supervised Nets**

- paper: [http://arxiv.org/abs/1409.5185](http://arxiv.org/abs/1409.5185)
- code: [https://github.com/mbhenaff/spectral-lib](https://github.com/mbhenaff/spectral-lib)

**Deep learning (Nature 2015)**

- author: Yann LeCun, Yoshua Bengio & Geoffrey Hinton
- paper: [http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf](http://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf)

## STDP

**A biological gradient descent for prediction through a combination of STDP and homeostatic plasticity**

- arxiv: [http://arxiv.org/abs/1206.4812](http://arxiv.org/abs/1206.4812)

**An objective function for STDP(Yoshua Bengio)**

- arxiv: [http://arxiv.org/abs/1509.05936](http://arxiv.org/abs/1509.05936)

**Towards a Biologically Plausible Backprop**

- arxiv: [http://arxiv.org/abs/1602.05179](http://arxiv.org/abs/1602.05179)

- - -

**Understanding and Predicting Image Memorability at a Large Scale (MIT. ICCV2015)**

- homepage: [http://memorability.csail.mit.edu/](http://memorability.csail.mit.edu/)
- paper: [https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf](https://people.csail.mit.edu/khosla/papers/iccv2015_khosla.pdf)
- code: [http://memorability.csail.mit.edu/download.html](http://memorability.csail.mit.edu/download.html)
- reviews: [http://petapixel.com/2015/12/18/how-memorable-are-times-top-10-photos-of-2015-to-a-computer/](http://petapixel.com/2015/12/18/how-memorable-are-times-top-10-photos-of-2015-to-a-computer/)

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

**DrMAD: Distilling Reverse-Mode Automatic Differentiation for Optimizing Hyperparameters of Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1601.00917](http://arxiv.org/abs/1601.00917)
- github: [https://github.com/bigaidream-projects/drmad](https://github.com/bigaidream-projects/drmad)

**Do Deep Convolutional Nets Really Need to be Deep (Or Even Convolutional)?**

- arxiv: [http://arxiv.org/abs/1603.05691](http://arxiv.org/abs/1603.05691)
- review: [http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/](http://www.erogol.com/paper-review-deep-convolutional-nets-really-need-deep-even-convolutional/)

**Harnessing Deep Neural Networks with Logic Rules**

 - arxiv: [http://arxiv.org/abs/1603.06318](http://arxiv.org/abs/1603.06318)
 
 **A guide to convolution arithmetic for deep learning**
 
 - arxiv: [http://arxiv.org/abs/1603.07285](http://arxiv.org/abs/1603.07285)
 
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

**An Analysis of Deep Neural Network Models for Practical Applications**

- arxiv: [http://arxiv.org/abs/1605.07678](http://arxiv.org/abs/1605.07678)

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

# Installation

**Setting up a Deep Learning Machine from Scratch (Software): Instructions for setting up the software on your deep learning machine**

- intro: A detailed guide to setting up your machine for deep learning research. 
Includes instructions to install drivers, tools and various deep learning frameworks. 
This was tested on a 64 bit machine with Nvidia Titan X, running Ubuntu 14.04
- github: [https://github.com/saiprashanths/dl-setup](https://github.com/saiprashanths/dl-setup)

**All-in-one Docker image for Deep Learning**

- intro: An all-in-one Docker image for deep learning. 
Contains all the popular DL frameworks (TensorFlow, Theano, Torch, Caffe, etc.)
- github: [https://github.com/saiprashanths/dl-docker](https://github.com/saiprashanths/dl-docker)

# Resources

**Awesome Deep Learning**

- github: [https://github.com/ChristosChristofidis/awesome-deep-learning](https://github.com/ChristosChristofidis/awesome-deep-learning)

**Deep Learning Libraries by Language**

- website: [http://www.teglor.com/b/deep-learning-libraries-language-cm569/](http://www.teglor.com/b/deep-learning-libraries-language-cm569/)

**Deep Learning Resources**

[http://yanirseroussi.com/deep-learning-resources/](http://yanirseroussi.com/deep-learning-resources/)

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

# Tools

**DNNGraph - A deep neural network model generation DSL in Haskell**

- homepage: [http://ajtulloch.github.io/dnngraph/](http://ajtulloch.github.io/dnngraph/)

**Deep playground: an interactive visualization of neural networks, written in typescript using d3.js**

- homepage: [http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.23990&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification](http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.23990&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification)
- github: [https://github.com/tensorflow/playground](https://github.com/tensorflow/playground)

**Neural Network Package**

- intro: This package provides an easy and modular way to build and train simple or complex neural networks using Torch
- github: [https://github.com/torch/nn](https://github.com/torch/nn)

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