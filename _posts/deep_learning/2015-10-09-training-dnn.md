---
layout: post
category: deep_learning
title: Training Deep Neural Networks
date: 2015-10-09
---

# Tutorials

**Popular Training Approaches of DNNs — A Quick Overview**

[https://medium.com/@asjad/popular-training-approaches-of-dnns-a-quick-overview-26ee37ad7e96#.pqyo039bb](https://medium.com/@asjad/popular-training-approaches-of-dnns-a-quick-overview-26ee37ad7e96#.pqyo039bb)

**Optimisation and training techniques for deep learning**

[https://blog.acolyer.org/2017/03/01/optimisation-and-training-techniques-for-deep-learning/](https://blog.acolyer.org/2017/03/01/optimisation-and-training-techniques-for-deep-learning/)

# Activation functions

## ReLU

**Rectified linear units improve restricted boltzmann machines**

- intro: ReLU
- paper: [http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)

**Expressiveness of Rectifier Networks**

- intro: ICML 2016
- intro: This paper studies the expressiveness of ReLU Networks
- arxiv: [https://arxiv.org/abs/1511.05678](https://arxiv.org/abs/1511.05678)

**How can a deep neural network with ReLU activations in its hidden layers approximate any function?**

- quora: [https://www.quora.com/How-can-a-deep-neural-network-with-ReLU-activations-in-its-hidden-layers-approximate-any-function](https://www.quora.com/How-can-a-deep-neural-network-with-ReLU-activations-in-its-hidden-layers-approximate-any-function)

**Understanding Deep Neural Networks with Rectified Linear Units**

- intro: Johns Hopkins University
- arxiv: [https://arxiv.org/abs/1611.01491](https://arxiv.org/abs/1611.01491)

**Learning ReLUs via Gradient Descent**

[https://arxiv.org/abs/1705.04591](https://arxiv.org/abs/1705.04591)

## LReLU

**Rectifier Nonlinearities Improve Neural Network Acoustic Models**

- intro: leaky-ReLU, aka LReLU
- paper: [http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

**Deep Sparse Rectifier Neural Networks**

- paper: [http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf](http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf)

## PReLU

**Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification**

- keywords: PReLU, Caffe "msra" weights initilization
- arxiv: [http://arxiv.org/abs/1502.01852](http://arxiv.org/abs/1502.01852)

**Empirical Evaluation of Rectified Activations in Convolutional Network**

- intro: ReLU / LReLU / PReLU / RReLU
- arxiv: [http://arxiv.org/abs/1505.00853](http://arxiv.org/abs/1505.00853)

## SReLU

**Deep Learning with S-shaped Rectified Linear Activation Units**

- intro:  SReLU
- arxiv: [http://arxiv.org/abs/1512.07030](http://arxiv.org/abs/1512.07030)

**Parametric Activation Pools greatly increase performance and consistency in ConvNets**

- blog: [http://blog.claymcleod.io/2016/02/06/Parametric-Activation-Pools-greatly-increase-performance-and-consistency-in-ConvNets/](http://blog.claymcleod.io/2016/02/06/Parametric-Activation-Pools-greatly-increase-performance-and-consistency-in-ConvNets/)

**From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification**

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1602.02068](http://arxiv.org/abs/1602.02068)
- github: [https://github.com/gokceneraslan/SparseMax.torch](https://github.com/gokceneraslan/SparseMax.torch)
- github: [https://github.com/Unbabel/sparsemax](https://github.com/Unbabel/sparsemax)

**Revise Saturated Activation Functions**

- arxiv: [http://arxiv.org/abs/1602.05980](http://arxiv.org/abs/1602.05980)

**Noisy Activation Functions**

- arxiv: [http://arxiv.org/abs/1603.00391](http://arxiv.org/abs/1603.00391)

## MBA

**Multi-Bias Non-linear Activation in Deep Neural Networks**

- intro: MBA
- arxiv: [https://arxiv.org/abs/1604.00676](https://arxiv.org/abs/1604.00676)

**Learning activation functions from data using cubic spline interpolation**

- arxiv: [http://arxiv.org/abs/1605.05509](http://arxiv.org/abs/1605.05509)
- bitbucket: [https://bitbucket.org/ispamm/spline-nn](https://bitbucket.org/ispamm/spline-nn)

**What is the role of the activation function in a neural network?**

- quora: [https://www.quora.com/What-is-the-role-of-the-activation-function-in-a-neural-network](https://www.quora.com/What-is-the-role-of-the-activation-function-in-a-neural-network)

## Concatenated ReLU (CRelu)

**Understanding and Improving Convolutional Neural Networks via Concatenated Rectified Linear Units**

- intro: ICML 2016
- arxiv: [http://arxiv.org/abs/1603.05201](http://arxiv.org/abs/1603.05201)

**Implement CReLU (Concatenated ReLU)**

- github: [https://github.com/pfnet/chainer/pull/1142](https://github.com/pfnet/chainer/pull/1142)

## GELU

**Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units**

- arxiv: [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)

**Formulating The ReLU**

- blog: [http://www.jefkine.com/general/2016/08/24/formulating-the-relu/](http://www.jefkine.com/general/2016/08/24/formulating-the-relu/)

**Activation Ensembles for Deep Neural Networks**

[https://arxiv.org/abs/1702.07790](https://arxiv.org/abs/1702.07790)

## SELU

**Self-Normalizing Neural Networks**

- intro: SELU
- arxiv: [https://arxiv.org/abs/1706.02515](https://arxiv.org/abs/1706.02515)
- github: [https://github.com/bioinf-jku/SNNs](https://github.com/bioinf-jku/SNNs)
- notes: [https://github.com/kevinzakka/research-paper-notes/blob/master/snn.md](https://github.com/kevinzakka/research-paper-notes/blob/master/snn.md)
- github(Chainer): [https://github.com/musyoku/self-normalizing-networks](https://github.com/musyoku/self-normalizing-networks)

**SELUs (scaled exponential linear units) - Visualized and Histogramed Comparisons among ReLU and Leaky ReLU**

[https://github.com/shaohua0116/Activation-Visualization-Histogram](https://github.com/shaohua0116/Activation-Visualization-Histogram)

**Difference Between Softmax Function and Sigmoid Function**

[http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/](http://dataaspirant.com/2017/03/07/difference-between-softmax-function-and-sigmoid-function/)

**Flexible Rectified Linear Units for Improving Convolutional Neural Networks**

- keywords: flexible rectified linear unit (FReLU)
- arxiv: [https://arxiv.org/abs/1706.08098](https://arxiv.org/abs/1706.08098)

**Be Careful What You Backpropagate: A Case For Linear Output Activations & Gradient Boosting**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1707.04199](https://arxiv.org/abs/1707.04199)

## Series on Initialization of Weights for DNN

**Initialization Of Feedfoward Networks**

- blog: [http://www.jefkine.com/deep/2016/07/27/initialization-of-feedfoward-networks/](http://www.jefkine.com/deep/2016/07/27/initialization-of-feedfoward-networks/)

**Initialization Of Deep Feedfoward Networks**

- blog: [http://www.jefkine.com/deep/2016/08/01/initialization-of-deep-feedfoward-networks/](http://www.jefkine.com/deep/2016/08/01/initialization-of-deep-feedfoward-networks/)

**Initialization Of Deep Networks Case of Rectifiers**

- blog: [http://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/](http://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/)

# Weights Initialization

**An Explanation of Xavier Initialization**

- blog: [http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)

**Random Walk Initialization for Training Very Deep Feedforward Networks**

- arxiv: [http://arxiv.org/abs/1412.6558](http://arxiv.org/abs/1412.6558)

**Deep Neural Networks with Random Gaussian Weights: A Universal Classification Strategy?**

- arxiv: [http://arxiv.org/abs/1504.08291](http://arxiv.org/abs/1504.08291)

**All you need is a good init**

- intro: ICLR 2016
- intro: Layer-sequential unit-variance (LSUV) initialization
- arxiv: [http://arxiv.org/abs/1511.06422](http://arxiv.org/abs/1511.06422)
- github(Caffe): [https://github.com/ducha-aiki/LSUVinit](https://github.com/ducha-aiki/LSUVinit)
- github(Torch): [https://github.com/yobibyte/torch-lsuv](https://github.com/yobibyte/torch-lsuv)
- github: [https://github.com/yobibyte/yobiblog/blob/master/posts/all-you-need-is-a-good-init.md](https://github.com/yobibyte/yobiblog/blob/master/posts/all-you-need-is-a-good-init.md)
- github(Keras): [https://github.com/ducha-aiki/LSUV-keras](https://github.com/ducha-aiki/LSUV-keras)
- review: [http://www.erogol.com/need-good-init/](http://www.erogol.com/need-good-init/)

**All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation**

- intro: CVPR 2017. HIKVision
- arxiv: [https://arxiv.org/abs/1703.01827](https://arxiv.org/abs/1703.01827)

**Data-dependent Initializations of Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1511.06856](http://arxiv.org/abs/1511.06856)
- github: [https://github.com/philkr/magic_init](https://github.com/philkr/magic_init)

**What are good initial weights in a neural network?**

- stackexchange: [http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network](http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network)

**RandomOut: Using a convolutional gradient norm to win The Filter Lottery**

- arxiv: [http://arxiv.org/abs/1602.05931](http://arxiv.org/abs/1602.05931)

**Categorical Reparameterization with Gumbel-Softmax**

- intro: Google Brain & University of Cambridge & Stanford University
- arxiv: [https://arxiv.org/abs/1611.01144](https://arxiv.org/abs/1611.01144)
- github: [https://github.com/ericjang/gumbel-softmax](https://github.com/ericjang/gumbel-softmax)

**On weight initialization in deep neural networks**

- arxiv: [https://arxiv.org/abs/1704.08863](https://arxiv.org/abs/1704.08863)
- github: [https://github.com/sidkk86/weight_initialization](https://github.com/sidkk86/weight_initialization)

## Batch Normalization

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift**

- intro: ImageNet top-5 error: 4.82% 
- keywords: internal covariate shift problem
- arxiv: [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167)
- blog: [https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/](https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/)
- notes: [http://blog.csdn.net/happynear/article/details/44238541](http://blog.csdn.net/happynear/article/details/44238541)

**Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1602.07868](http://arxiv.org/abs/1602.07868)
- github(Lasagne): [https://github.com/TimSalimans/weight_norm](https://github.com/TimSalimans/weight_norm)
- github: [https://github.com/openai/weightnorm](https://github.com/openai/weightnorm)
- notes: [http://www.erogol.com/my-notes-weight-normalization/](http://www.erogol.com/my-notes-weight-normalization/)

**Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks**

- arxiv: [http://arxiv.org/abs/1603.01431](http://arxiv.org/abs/1603.01431)

**Implementing Batch Normalization in Tensorflow**

- blog: [http://r2rt.com/implementing-batch-normalization-in-tensorflow.html](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

**Deriving the Gradient for the Backward Pass of Batch Normalization**

- blog: [https://kevinzakka.github.io/2016/09/14/batch_normalization/](https://kevinzakka.github.io/2016/09/14/batch_normalization/)

**Exploring Normalization in Deep Residual Networks with Concatenated Rectified Linear Units**

- intro: Oculus VR & Facebook & NEC Labs America
- paper: [https://research.fb.com/publications/exploring-normalization-in-deep-residual-networks-with-concatenated-rectified-linear-units/](https://research.fb.com/publications/exploring-normalization-in-deep-residual-networks-with-concatenated-rectified-linear-units/)

**Batch Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models**

- intro: Sergey Ioffe, Google
- arxiv: [https://arxiv.org/abs/1702.03275](https://arxiv.org/abs/1702.03275)

### Backward pass of BN

**Understanding the backward pass through Batch Normalization Layer**

[https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

**Deriving the Gradient for the Backward Pass of Batch Normalization**

[https://kevinzakka.github.io/2016/09/14/batch_normalization/](https://kevinzakka.github.io/2016/09/14/batch_normalization/)

**What does the gradient flowing through batch normalization looks like ?**

[http://cthorey.github.io./backpropagation/](http://cthorey.github.io./backpropagation/)

## Layer Normalization

**Layer Normalization**

- arxiv: [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
- github: [https://github.com/ryankiros/layer-norm](https://github.com/ryankiros/layer-norm)
- github(TensorFlow): [https://github.com/pbhatia243/tf-layer-norm](https://github.com/pbhatia243/tf-layer-norm)
- github: [https://github.com/MycChiu/fast-LayerNorm-TF](https://github.com/MycChiu/fast-LayerNorm-TF)

**Keras GRU with Layer Normalization**

- gist: [https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940](https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940)

**Cosine Normalization: Using Cosine Similarity Instead of Dot Product in Neural Networks**

- arxiv: [https://arxiv.org/abs/1702.05870](https://arxiv.org/abs/1702.05870)

# Loss Function

**The Loss Surfaces of Multilayer Networks**

- arxiv: [http://arxiv.org/abs/1412.0233](http://arxiv.org/abs/1412.0233)

**Direct Loss Minimization for Training Deep Neural Nets**

- arxiv: [http://arxiv.org/abs/1511.06411](http://arxiv.org/abs/1511.06411)

**Nonconvex Loss Functions for Classifiers and Deep Networks**

- blog: [https://casmls.github.io/general/2016/10/27/NonconvexLosses.html](https://casmls.github.io/general/2016/10/27/NonconvexLosses.html)

**Learning Deep Embeddings with Histogram Loss**

- arxiv: [https://arxiv.org/abs/1611.00822](https://arxiv.org/abs/1611.00822)

**Large-Margin Softmax Loss for Convolutional Neural Networks**

- intro: ICML 2016
- intro: Peking University & South China University of Technology & CMU & Shenzhen University
- arxiv: [https://arxiv.org/abs/1612.02295](https://arxiv.org/abs/1612.02295)
- github(Official. Caffe): [https://github.com/wy1iu/LargeMargin_Softmax_Loss](https://github.com/wy1iu/LargeMargin_Softmax_Loss)
- github(MXNet): [https://github.com/luoyetx/mx-lsoftmax](https://github.com/luoyetx/mx-lsoftmax)

**An empirical analysis of the optimization of deep network loss surfaces**

[https://arxiv.org/abs/1612.04010](https://arxiv.org/abs/1612.04010)

**Towards Understanding Generalization of Deep Learning: Perspective of Loss Landscapes**

- intro: Peking University
- arxiv: [https://arxiv.org/abs/1706.10239](https://arxiv.org/abs/1706.10239)

**Hierarchical Softmax**

[http://building-babylon.net/2017/08/01/hierarchical-softmax/](http://building-babylon.net/2017/08/01/hierarchical-softmax/)

**Noisy Softmax: Improving the Generalization Ability of DCNN via Postponing the Early Softmax Saturation**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1708.03769](https://arxiv.org/abs/1708.03769)

# Learning Rate

**No More Pesky Learning Rates**

- intro: Tom Schaul, Sixin Zhang, Yann LeCun
- arxiv: [https://arxiv.org/abs/1206.1106](https://arxiv.org/abs/1206.1106)

**Coupling Adaptive Batch Sizes with Learning Rates**

- intro: Max Planck Institute for Intelligent Systems
- intro: Tensorflow implementation of SGD with Coupled Adaptive Batch Size (CABS)
- arxiv: [https://arxiv.org/abs/1612.05086](https://arxiv.org/abs/1612.05086)
- github: [https://github.com/ProbabilisticNumerics/cabs](https://github.com/ProbabilisticNumerics/cabs)

# Pooling

**Stochastic Pooling for Regularization of Deep Convolutional Neural Networks**

- intro: ICLR 2013. Matthew D. Zeiler, Rob Fergus
- paper: [http://www.matthewzeiler.com/pubs/iclr2013/iclr2013.pdf](http://www.matthewzeiler.com/pubs/iclr2013/iclr2013.pdf)

**Multi-scale Orderless Pooling of Deep Convolutional Activation Features**

- intro: ECCV 2014
- intro: MOP-CNN, orderless VLAD pooling, image classification / instance-level retrieval
- arxiv: [https://arxiv.org/abs/1403.1840](https://arxiv.org/abs/1403.1840)
- paper: [http://web.engr.illinois.edu/~slazebni/publications/yunchao_eccv14_mopcnn.pdf](http://web.engr.illinois.edu/~slazebni/publications/yunchao_eccv14_mopcnn.pdf)

**Fractional Max-Pooling**

- arxiv: [https://arxiv.org/abs/1412.6071](https://arxiv.org/abs/1412.6071)
- notes: [https://gist.github.com/shagunsodhani/ccfe3134f46fd3738aa0](https://gist.github.com/shagunsodhani/ccfe3134f46fd3738aa0)
- github: [https://github.com/torch/nn/issues/371](https://github.com/torch/nn/issues/371)

**TI-POOLING: transformation-invariant pooling for feature learning in Convolutional Neural Networks**

- intro: CVPR 2016
- paper: [http://dlaptev.org/papers/Laptev16_CVPR.pdf](http://dlaptev.org/papers/Laptev16_CVPR.pdf)
- github: [https://github.com/dlaptev/TI-pooling](https://github.com/dlaptev/TI-pooling)

**S3Pool: Pooling with Stochastic Spatial Sampling**

- arxiv: [https://arxiv.org/abs/1611.05138](https://arxiv.org/abs/1611.05138)
- github(Lasagne): [https://github.com/Shuangfei/s3pool](https://github.com/Shuangfei/s3pool)

**Inductive Bias of Deep Convolutional Networks through Pooling Geometry**

- arxiv: [https://arxiv.org/abs/1605.06743](https://arxiv.org/abs/1605.06743)
- github: [https://github.com/HUJI-Deep/inductive-pooling](https://github.com/HUJI-Deep/inductive-pooling)

**Improved Bilinear Pooling with CNNs**

[https://arxiv.org/abs/1707.06772](https://arxiv.org/abs/1707.06772)

**Learning Bag-of-Features Pooling for Deep Convolutional Neural Networks

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1707.08105](https://arxiv.org/abs/1707.08105)
- github: [https://github.com/passalis/cbof](https://github.com/passalis/cbof)

# Batch

**Online Batch Selection for Faster Training of Neural Networks**

- intro: Workshop paper at ICLR 2016
- arxiv: [https://arxiv.org/abs/1511.06343](https://arxiv.org/abs/1511.06343)

**On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima**

- intro: ICLR 2017
- arxiv: [https://arxiv.org/abs/1609.04836](https://arxiv.org/abs/1609.04836)

**Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour**

- intro: Facebook
- keywords: Training with 256 GPUs, minibatches of 8192
- arxiv: [https://arxiv.org/abs/1706.02677](https://arxiv.org/abs/1706.02677)

**Scaling SGD Batch Size to 32K for ImageNet Training**

[https://arxiv.org/abs/1708.03888](https://arxiv.org/abs/1708.03888)

# Optimization Methods

**On Optimization Methods for Deep Learning**

- paper: [http://www.icml-2011.org/papers/210_icmlpaper.pdf](http://www.icml-2011.org/papers/210_icmlpaper.pdf)

**Invariant backpropagation: how to train a transformation-invariant neural network**

- arxiv: [http://arxiv.org/abs/1502.04434](http://arxiv.org/abs/1502.04434)
- github: [https://github.com/sdemyanov/ConvNet](https://github.com/sdemyanov/ConvNet)

**A practical theory for designing very deep convolutional neural network**

- kaggle: [https://www.kaggle.com/c/datasciencebowl/forums/t/13166/happy-lantern-festival-report-and-code/69284](https://www.kaggle.com/c/datasciencebowl/forums/t/13166/happy-lantern-festival-report-and-code/69284)
- paper: [https://kaggle2.blob.core.windows.net/forum-message-attachments/69182/2287/A%20practical%20theory%20for%20designing%20very%20deep%20convolutional%20neural%20networks.pdf?sv=2012-02-12&se=2015-12-05T15%3A40%3A02Z&sr=b&sp=r&sig=kfBQKduA1pDtu837Y9Iqyrp2VYItTV0HCgOeOok9E3E%3D](https://kaggle2.blob.core.windows.net/forum-message-attachments/69182/2287/A%20practical%20theory%20for%20designing%20very%20deep%20convolutional%20neural%20networks.pdf?sv=2012-02-12&se=2015-12-05T15%3A40%3A02Z&sr=b&sp=r&sig=kfBQKduA1pDtu837Y9Iqyrp2VYItTV0HCgOeOok9E3E%3D)
- slides: [http://vdisk.weibo.com/s/3nFsznjLKn](http://vdisk.weibo.com/s/3nFsznjLKn)

**Stochastic Optimization Techniques**

- intro: SGD/Momentum/NAG/Adagrad/RMSProp/Adadelta/Adam/ESGD/Adasecant/vSGD/Rprop
- blog: [http://colinraffel.com/wiki/stochastic_optimization_techniques](http://colinraffel.com/wiki/stochastic_optimization_techniques)

**Alec Radford's animations for optimization algorithms**

[http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html](http://www.denizyuret.com/2015/03/alec-radfords-animations-for.html)

**Faster Asynchronous SGD (FASGD)**

- arxiv: [http://arxiv.org/abs/1601.04033](http://arxiv.org/abs/1601.04033)
- github: [https://github.com/DoctorTeeth/fred](https://github.com/DoctorTeeth/fred)

**An overview of gradient descent optimization algorithms (★★★★★)**

![](http://sebastianruder.com/content/images/2016/01/contours_evaluation_optimizers.gif)

- arxiv: [https://arxiv.org/abs/1609.04747](https://arxiv.org/abs/1609.04747)
- blog: [http://sebastianruder.com/optimizing-gradient-descent/](http://sebastianruder.com/optimizing-gradient-descent/)

**Exploiting the Structure: Stochastic Gradient Methods Using Raw Clusters**

- arxiv: [http://arxiv.org/abs/1602.02151](http://arxiv.org/abs/1602.02151)

**Writing fast asynchronous SGD/AdaGrad with RcppParallel**

- blog: [http://gallery.rcpp.org/articles/rcpp-sgd/](http://gallery.rcpp.org/articles/rcpp-sgd/)

**Quick Explanations Of Optimization Methods**

- blog: [http://jxieeducation.com/2016-07-02/Quick-Explanations-of-Optimization-Methods/](http://jxieeducation.com/2016-07-02/Quick-Explanations-of-Optimization-Methods/)

**Learning to learn by gradient descent by gradient descent**

- intro: Google DeepMind
- arxiv: [https://arxiv.org/abs/1606.04474](https://arxiv.org/abs/1606.04474)
- github: [https://github.com/deepmind/learning-to-learn](https://github.com/deepmind/learning-to-learn)
- github(TensorFlow): [https://github.com/runopti/Learning-To-Learn](https://github.com/runopti/Learning-To-Learn)
- github(PyTorch): [https://github.com/ikostrikov/pytorch-meta-optimizer](https://github.com/ikostrikov/pytorch-meta-optimizer)

**SGDR: Stochastic Gradient Descent with Restarts**

- arxiv: [http://arxiv.org/abs/1608.03983](http://arxiv.org/abs/1608.03983)
- github: [https://github.com/loshchil/SGDR](https://github.com/loshchil/SGDR)

**The zen of gradient descent**

- blog: [http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)

**Big Batch SGD: Automated Inference using Adaptive Batch Sizes**

- arxiv: [https://arxiv.org/abs/1610.05792](https://arxiv.org/abs/1610.05792)

**Improving Stochastic Gradient Descent with Feedback**

- arxiv: [https://arxiv.org/abs/1611.01505](https://arxiv.org/abs/1611.01505)
- github: [https://github.com/jayanthkoushik/sgd-feedback](https://github.com/jayanthkoushik/sgd-feedback)
- github: [https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Eve](https://github.com/tdeboissiere/DeepLearningImplementations/tree/master/Eve)

**Learning Gradient Descent: Better Generalization and Longer Horizons**

- intro: Tsinghua University
- arxiv: [https://arxiv.org/abs/1703.03633](https://arxiv.org/abs/1703.03633)
- github(TensorFlow): [https://github.com/vfleaking/rnnprop](https://github.com/vfleaking/rnnprop)

# Tensor Methods

**Tensorizing Neural Networks**

- intro: TensorNet
- arxiv: [http://arxiv.org/abs/1509.06569](http://arxiv.org/abs/1509.06569)
- github(Matlab+Theano+Lasagne): [https://github.com/Bihaqo/TensorNet](https://github.com/Bihaqo/TensorNet)
- github(TensorFlow): [https://github.com/timgaripov/TensorNet-TF](https://github.com/timgaripov/TensorNet-TF)

**Tensor methods for training neural networks**

- homepage: [http://newport.eecs.uci.edu/anandkumar/#home](http://newport.eecs.uci.edu/anandkumar/#home)
- youtube: [https://www.youtube.com/watch?v=B4YvhcGaafw](https://www.youtube.com/watch?v=B4YvhcGaafw)
- slides: [http://newport.eecs.uci.edu/anandkumar/slides/Strata-NY.pdf](http://newport.eecs.uci.edu/anandkumar/slides/Strata-NY.pdf)
- talks: [http://newport.eecs.uci.edu/anandkumar/#talks](http://newport.eecs.uci.edu/anandkumar/#talks)

# Regularization

**DisturbLabel: Regularizing CNN on the Loss Layer**

- intro:  University of California & MSR 2016
- intro: "an extremely simple algorithm which randomly replaces a part of labels as incorrect values in each iteration"
- paper: [http://research.microsoft.com/en-us/um/people/jingdw/pubs/cvpr16-disturblabel.pdf](http://research.microsoft.com/en-us/um/people/jingdw/pubs/cvpr16-disturblabel.pdf)

**Robust Convolutional Neural Networks under Adversarial Noise**

- intro:  ICLR 2016
- arxiv: [http://arxiv.org/abs/1511.06306](http://arxiv.org/abs/1511.06306)

**Adding Gradient Noise Improves Learning for Very Deep Networks**

- intro:  ICLR 2016
- arxiv: [http://arxiv.org/abs/1511.06807](http://arxiv.org/abs/1511.06807)

**Stochastic Function Norm Regularization of Deep Networks**

- arxiv: [http://arxiv.org/abs/1605.09085](http://arxiv.org/abs/1605.09085)
- github: [https://github.com/AmalRT/DNN_Reg](https://github.com/AmalRT/DNN_Reg)

**SoftTarget Regularization: An Effective Technique to Reduce Over-Fitting in Neural Networks**

- arxiv: [http://arxiv.org/abs/1609.06693](http://arxiv.org/abs/1609.06693)

**Regularizing neural networks by penalizing confident predictions**

- intro: Gabriel Pereyra, George Tucker, Lukasz Kaiser, Geoffrey Hinton [Google Brain
- dropbox: [https://www.dropbox.com/s/8kqf4v2c9lbnvar/BayLearn%202016%20(gjt).pdf?dl=0](https://www.dropbox.com/s/8kqf4v2c9lbnvar/BayLearn%202016%20(gjt).pdf?dl=0)
- mirror: [https://pan.baidu.com/s/1kUUtxdl](https://pan.baidu.com/s/1kUUtxdl)

**Automatic Node Selection for Deep Neural Networks using Group Lasso Regularization**

- arxiv: [https://arxiv.org/abs/1611.05527](https://arxiv.org/abs/1611.05527)

**Regularization in deep learning**

- blog: [https://medium.com/@cristina_scheau/regularization-in-deep-learning-f649a45d6e0#.py327hkuv](https://medium.com/@cristina_scheau/regularization-in-deep-learning-f649a45d6e0#.py327hkuv)
- github: [https://github.com/cscheau/Examples/blob/master/iris_l1_l2.py](https://github.com/cscheau/Examples/blob/master/iris_l1_l2.py)

## Dropout

**Improving neural networks by preventing co-adaptation of feature detectors**

- intro: Dropout
- arxiv: [http://arxiv.org/abs/1207.0580](http://arxiv.org/abs/1207.0580)

**Dropout: A Simple Way to Prevent Neural Networks from Overfitting**

- paper: [https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)

**Fast dropout training**

- paper: [http://jmlr.org/proceedings/papers/v28/wang13a.pdf](http://jmlr.org/proceedings/papers/v28/wang13a.pdf)
- github: [https://github.com/sidaw/fastdropout](https://github.com/sidaw/fastdropout)

**Dropout as data augmentation**

- paper: [http://arxiv.org/abs/1506.08700](http://arxiv.org/abs/1506.08700)
- notes: [https://www.evernote.com/shard/s189/sh/ef0c3302-21a4-40d7-b8b4-1c65b8ebb1c9/24ff553fcfb70a27d61ff003df75b5a9](https://www.evernote.com/shard/s189/sh/ef0c3302-21a4-40d7-b8b4-1c65b8ebb1c9/24ff553fcfb70a27d61ff003df75b5a9)

**A Theoretically Grounded Application of Dropout in Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1512.05287](http://arxiv.org/abs/1512.05287)
- github: [https://github.com/yaringal/BayesianRNN](https://github.com/yaringal/BayesianRNN)

**Improved Dropout for Shallow and Deep Learning**

- arxiv: [http://arxiv.org/abs/1602.02220](http://arxiv.org/abs/1602.02220)

**Dropout Regularization in Deep Learning Models With Keras**

- blog: [http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/](http://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/)

**Dropout with Expectation-linear Regularization**

- arxiv: [http://arxiv.org/abs/1609.08017](http://arxiv.org/abs/1609.08017)

**Dropout with Theano**

- blog: [http://rishy.github.io/ml/2016/10/12/dropout-with-theano/](http://rishy.github.io/ml/2016/10/12/dropout-with-theano/)
- ipn: [http://nbviewer.jupyter.org/github/rishy/rishy.github.io/blob/master/ipy_notebooks/Dropout-Theano.ipynb](http://nbviewer.jupyter.org/github/rishy/rishy.github.io/blob/master/ipy_notebooks/Dropout-Theano.ipynb)

**Information Dropout: learning optimal representations through noise**

- arxiv: [https://arxiv.org/abs/1611.01353](https://arxiv.org/abs/1611.01353)

**Recent Developments in Dropout**

- blog: [https://casmls.github.io/general/2016/11/11/dropout.html](https://casmls.github.io/general/2016/11/11/dropout.html)

**Generalized Dropout**

- arxiv: [https://arxiv.org/abs/1611.06791](https://arxiv.org/abs/1611.06791)

**Analysis of Dropout**

- blog: [https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/](https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/)

**Variational Dropout Sparsifies Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1701.05369](https://arxiv.org/abs/1701.05369)

**Learning Deep Networks from Noisy Labels with Dropout Regularization**

- intro: 2016 IEEE 16th International Conference on Data Mining
- arxiv: [https://arxiv.org/abs/1705.03419](https://arxiv.org/abs/1705.03419)

**Concrete Dropout**

- intro: University of Cambridge
- arxiv: [https://arxiv.org/abs/1705.07832](https://arxiv.org/abs/1705.07832)
- github: [https://github.com/yaringal/ConcreteDropout](https://github.com/yaringal/ConcreteDropout)

**Analysis of dropout learning regarded as ensemble learning**

- intro: Nihon University
- arxiv: [https://arxiv.org/abs/1706.06859](https://arxiv.org/abs/1706.06859)

## DropConnect

**Regularization of Neural Networks using DropConnect**

- homepage: [http://cs.nyu.edu/~wanli/dropc/](http://cs.nyu.edu/~wanli/dropc/)
- gitxiv: [http://gitxiv.com/posts/rJucpiQiDhQ7HkZoX/regularization-of-neural-networks-using-dropconnect](http://gitxiv.com/posts/rJucpiQiDhQ7HkZoX/regularization-of-neural-networks-using-dropconnect)
- github: [https://github.com/iassael/torch-dropconnect](https://github.com/iassael/torch-dropconnect)

**Regularizing neural networks with dropout and with DropConnect**

- blog: [http://fastml.com/regularizing-neural-networks-with-dropout-and-with-dropconnect/](http://fastml.com/regularizing-neural-networks-with-dropout-and-with-dropconnect/)

## DropNeuron

**DropNeuron: Simplifying the Structure of Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1606.07326](http://arxiv.org/abs/1606.07326)
- github: [https://github.com/panweihit/DropNeuron](https://github.com/panweihit/DropNeuron)

## Maxout

**Maxout Networks**

- intro: ICML 2013
- intro: "its output is the max of a set of inputs, a natural companion to dropout"
- project page: [http://www-etud.iro.umontreal.ca/~goodfeli/maxout.html](http://www-etud.iro.umontreal.ca/~goodfeli/maxout.html)
- arxiv: [https://arxiv.org/abs/1302.4389](https://arxiv.org/abs/1302.4389)
- github: [https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/maxout.py](https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/models/maxout.py)

**Improving Deep Neural Networks with Probabilistic Maxout Units**

- arxiv: [https://arxiv.org/abs/1312.6116](https://arxiv.org/abs/1312.6116)

## Swapout

**Swapout: Learning an ensemble of deep architectures**

- arxiv: [https://arxiv.org/abs/1605.06465](https://arxiv.org/abs/1605.06465)
- blog: [https://gab41.lab41.org/lab41-reading-group-swapout-learning-an-ensemble-of-deep-architectures-e67d2b822f8a#.9r2s4c58n](https://gab41.lab41.org/lab41-reading-group-swapout-learning-an-ensemble-of-deep-architectures-e67d2b822f8a#.9r2s4c58n)

## Whiteout

**Whiteout: Gaussian Adaptive Regularization Noise in Deep Neural Networks**

- intro: University of Notre Dame & University of Science and Technology of China
- arxiv: [https://arxiv.org/abs/1612.01490](https://arxiv.org/abs/1612.01490)

# Gradient Descent

**RMSProp: Divide the gradient by a running average of its recent magnitude**

![](/assets/train-dnn/rmsprop.jpg)

- intro: it was not proposed in a paper, in fact it was just introduced in a slide in Geoffrey Hinton's Coursera class 
- slides: [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

**Adam: A Method for Stochastic Optimization**

- arxiv: [http://arxiv.org/abs/1412.6980](http://arxiv.org/abs/1412.6980)

**Fitting a model via closed-form equations vs. Gradient Descent vs Stochastic Gradient Descent vs Mini-Batch Learning. What is the difference?(Normal Equations vs. GD vs. SGD vs. MB-GD)**

[http://sebastianraschka.com/faq/docs/closed-form-vs-gd.html](http://sebastianraschka.com/faq/docs/closed-form-vs-gd.html)

**An Introduction to Gradient Descent in Python**

- blog: [http://tillbergmann.com/blog/articles/python-gradient-descent.html](http://tillbergmann.com/blog/articles/python-gradient-descent.html)

**Train faster, generalize better: Stability of stochastic gradient descent**

- arxiv: [http://arxiv.org/abs/1509.01240](http://arxiv.org/abs/1509.01240)

**A Variational Analysis of Stochastic Gradient Algorithms**

- arxiv: [http://arxiv.org/abs/1602.02666](http://arxiv.org/abs/1602.02666)

**The vanishing gradient problem: Oh no — an obstacle to deep learning!**

- blog: [https://medium.com/a-year-of-artificial-intelligence/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b#.50hu5vwa8](https://medium.com/a-year-of-artificial-intelligence/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b#.50hu5vwa8)

**Gradient Descent For Machine Learning**

- blog: [http://machinelearningmastery.com/gradient-descent-for-machine-learning/](http://machinelearningmastery.com/gradient-descent-for-machine-learning/)

**Revisiting Distributed Synchronous SGD**

- arxiv: [http://arxiv.org/abs/1604.00981](http://arxiv.org/abs/1604.00981)

**Convergence rate of gradient descent**

- blog: [https://building-babylon.net/2016/06/23/convergence-rate-of-gradient-descent/](https://building-babylon.net/2016/06/23/convergence-rate-of-gradient-descent/)

**A Robust Adaptive Stochastic Gradient Method for Deep Learning**

- intro: IJCNN 2017 Accepted Paper, An extension of paper, "ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient"
- intro: Universite de Montreal & University of Oxford
- arxiv: [https://arxiv.org/abs/1703.00788](https://arxiv.org/abs/1703.00788)

**Gentle Introduction to the Adam Optimization Algorithm for Deep Learning**

- blog: [http://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/](http://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

## AdaGrad

**Adaptive Subgradient Methods for Online Learning and Stochastic Optimization**

- paper: [http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

**ADADELTA: An Adaptive Learning Rate Method**

- arxiv: [http://arxiv.org/abs/1212.5701](http://arxiv.org/abs/1212.5701)

## Momentum

**On the importance of initialization and momentum in deep learning**

- intro:  NAG: Nesterov
- paper: [http://www.cs.toronto.edu/~fritz/absps/momentum.pdf](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
- paper: [http://jmlr.org/proceedings/papers/v28/sutskever13.pdf](http://jmlr.org/proceedings/papers/v28/sutskever13.pdf)

**YellowFin and the Art of Momentum Tuning**

- intro: Stanford University
- intro: auto-tuning momentum SGD optimizer
- project page: [http://cs.stanford.edu/~zjian/project/YellowFin/](http://cs.stanford.edu/~zjian/project/YellowFin/)
- arxiv: [https://arxiv.org/abs/1706.03471](https://arxiv.org/abs/1706.03471)
- github(TensorFlow): [https://github.com/JianGoForIt/YellowFin](https://github.com/JianGoForIt/YellowFin)
[https://github.com/JianGoForIt/YellowFin_Pytorch](https://github.com/JianGoForIt/YellowFin_Pytorch)

# Backpropagation

**Relay Backpropagation for Effective Learning of Deep Convolutional Neural Networks**

- intro: ECCV 2016. first place of ILSVRC 2015 Scene Classification Challenge
- arxiv: [https://arxiv.org/abs/1512.05830](https://arxiv.org/abs/1512.05830)
- paper: [http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2016-ECCV-RelayBP.pdf](http://www.cis.pku.edu.cn/faculty/vision/zlin/Publications/2016-ECCV-RelayBP.pdf)

**Top-down Neural Attention by Excitation Backprop**

![](http://cs-people.bu.edu/jmzhang/images/screen%20shot%202016-08-19%20at%2035847%20pm.jpg?crc=3911895888)

- intro: ECCV, 2016 (oral)
- projpage: [http://cs-people.bu.edu/jmzhang/excitationbp.html](http://cs-people.bu.edu/jmzhang/excitationbp.html)
- arxiv: [http://arxiv.org/abs/1608.00507](http://arxiv.org/abs/1608.00507)
- paper: [http://cs-people.bu.edu/jmzhang/EB/ExcitationBackprop.pdf](http://cs-people.bu.edu/jmzhang/EB/ExcitationBackprop.pdf)
- github: [https://github.com/jimmie33/Caffe-ExcitationBP](https://github.com/jimmie33/Caffe-ExcitationBP)

**Towards a Biologically Plausible Backprop**

- arxiv: [http://arxiv.org/abs/1602.05179](http://arxiv.org/abs/1602.05179)
- github: [https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop](https://github.com/bscellier/Towards-a-Biologically-Plausible-Backprop)

**Sampled Backpropagation: Training Deep and Wide Neural Networks on Large Scale, User Generated Content Using Label Sampling**

- blog: [https://medium.com/@karl1980.lab41/sampled-backpropagation-27ac58d5c51c#.xnbhyxtou](https://medium.com/@karl1980.lab41/sampled-backpropagation-27ac58d5c51c#.xnbhyxtou)

**The Reversible Residual Network: Backpropagation Without Storing Activations**

- intro: CoRR 2017. University of Toronto
- arxiv: [https://arxiv.org/abs/1707.04585](https://arxiv.org/abs/1707.04585)
- github: [https://github.com/renmengye/revnet-public](https://github.com/renmengye/revnet-public)

# Accelerate Training

**Neural Networks with Few Multiplications**

- intro:  ICLR 2016
- arxiv: [https://arxiv.org/abs/1510.03009](https://arxiv.org/abs/1510.03009)

**Acceleration of Deep Neural Network Training with Resistive Cross-Point Devices**

- arxiv: [http://arxiv.org/abs/1603.07341](http://arxiv.org/abs/1603.07341)

**Deep Q-Networks for Accelerating the Training of Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1606.01467](http://arxiv.org/abs/1606.01467)
- github: [https://github.com/bigaidream-projects/qan](https://github.com/bigaidream-projects/qan)

**Omnivore: An Optimizer for Multi-device Deep Learning on CPUs and GPUs**

- arxiv: [http://arxiv.org/abs/1606.04487](http://arxiv.org/abs/1606.04487)

## Parallelism

**One weird trick for parallelizing convolutional neural networks**

- author: Alex Krizhevsky
- arxiv: [http://arxiv.org/abs/1404.5997](http://arxiv.org/abs/1404.5997)

**8-Bit Approximations for Parallelism in Deep Learning (ICLR 2016)**

- arxiv: [http://arxiv.org/abs/1511.04561](http://arxiv.org/abs/1511.04561)

# Handling Datasets

## Data Augmentation

**DataAugmentation ver1.0: Image data augmentation tool for training of image recognition algorithm**

- github: [https://github.com/takmin/DataAugmentation](https://github.com/takmin/DataAugmentation)

**Caffe-Data-Augmentation: a branc caffe with feature of Data Augmentation using a configurable stochastic combination of 7 data augmentation techniques**

- github: [https://github.com/ShaharKatz/Caffe-Data-Augmentation](https://github.com/ShaharKatz/Caffe-Data-Augmentation)

**Image Augmentation for Deep Learning With Keras**

- blog: [http://machinelearningmastery.com/image-augmentation-deep-learning-keras/](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

**What you need to know about data augmentation for machine learning**

- intro: keras Imagegenerator
- blog: [https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/](https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/)

**HZPROC: torch data augmentation toolbox (supports affine transform)**

- github: [https://github.com/zhanghang1989/hzproc](https://github.com/zhanghang1989/hzproc)

**AGA: Attribute Guided Augmentation**

- intro: one-shot recognition
- arxiv: [https://arxiv.org/abs/1612.02559](https://arxiv.org/abs/1612.02559)

**Accelerating Deep Learning with Multiprocess Image Augmentation in Keras**

- blog: [http://blog.stratospark.com/multiprocess-image-augmentation-keras.html](http://blog.stratospark.com/multiprocess-image-augmentation-keras.html)
- github: [https://github.com/stratospark/keras-multiprocess-image-data-generator](https://github.com/stratospark/keras-multiprocess-image-data-generator)

**Comprehensive Data Augmentation and Sampling for Pytorch**

- github: [https://github.com/ncullen93/torchsample](https://github.com/ncullen93/torchsample)

**Image augmentation for machine learning experiments.**

[https://github.com/aleju/imgaug](https://github.com/aleju/imgaug)

**Google/inception's data augmentation: scale and aspect ratio augmentation**

[https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua#L130](https://github.com/facebook/fb.resnet.torch/blob/master/datasets/transforms.lua#L130)

**Caffe Augmentation Extension**

- intro: Data Augmentation for Caffe
- github: [https://github.com/twtygqyy/caffe-augmentation](https://github.com/twtygqyy/caffe-augmentation)

## Imbalanced Datasets

**Investigation on handling Structured & Imbalanced Datasets with Deep Learning**

- intro: smote resampling, cost sensitive learning
- blog: [https://www.analyticsvidhya.com/blog/2016/10/investigation-on-handling-structured-imbalanced-datasets-with-deep-learning/](https://www.analyticsvidhya.com/blog/2016/10/investigation-on-handling-structured-imbalanced-datasets-with-deep-learning/)

# Low Numerical Precision

**Training deep neural networks with low precision multiplications**

- intro: ICLR 2015
- intro: Maxout networks, 10-bit activations, 12-bit parameter updates
- arxiv: [http://arxiv.org/abs/1412.7024](http://arxiv.org/abs/1412.7024)
- github: [https://github.com/MatthieuCourbariaux/deep-learning-multipliers](https://github.com/MatthieuCourbariaux/deep-learning-multipliers)

**Deep Learning with Limited Numerical Precision**

- intro: ICML 2015
- arxiv: [http://arxiv.org/abs/1502.02551](http://arxiv.org/abs/1502.02551)

**BinaryConnect: Training Deep Neural Networks with binary weights during propagations**

- paper: [http://papers.nips.cc/paper/5647-shape-and-illumination-from-shading-using-the-generic-viewpoint-assumption](http://papers.nips.cc/paper/5647-shape-and-illumination-from-shading-using-the-generic-viewpoint-assumption)
- github: [https://github.com/MatthieuCourbariaux/BinaryConnect](https://github.com/MatthieuCourbariaux/BinaryConnect)

**Binarized Neural Networks**

- arxiv: [http://arxiv.org/abs/1602.02505](http://arxiv.org/abs/1602.02505)

**BinaryNet: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1**

**Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1**

- arxiv: [http://arxiv.org/abs/1602.02830](http://arxiv.org/abs/1602.02830)
- github: [https://github.com/MatthieuCourbariaux/BinaryNet](https://github.com/MatthieuCourbariaux/BinaryNet)
- github: [https://github.com/codekansas/tinier-nn](https://github.com/codekansas/tinier-nn)

**Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations**

- arxiv: [http://arxiv.org/abs/1609.07061](http://arxiv.org/abs/1609.07061)

# Adversarial Training

**Learning from Simulated and Unsupervised Images through Adversarial Training**

- intro: CVPR 2017 oral, best paper award. Apple Inc.
- arxiv: [https://arxiv.org/abs/1612.07828](https://arxiv.org/abs/1612.07828)

# Papers

**Understanding the difficulty of training deep feed forward neural networks**

- intro: Xavier initialization
- paper: [http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

**Domain-Adversarial Training of Neural Networks**

- arxiv: [https://arxiv.org/abs/1505.07818](https://arxiv.org/abs/1505.07818)
- paper: [http://jmlr.org/papers/v17/15-239.html](http://jmlr.org/papers/v17/15-239.html)
- github: [https://github.com/pumpikano/tf-dann](https://github.com/pumpikano/tf-dann)

**Scalable and Sustainable Deep Learning via Randomized Hashing**

- arxiv: [http://arxiv.org/abs/1602.08194](http://arxiv.org/abs/1602.08194)

**Training Deep Nets with Sublinear Memory Cost**

- arxiv: [https://arxiv.org/abs/1604.06174](https://arxiv.org/abs/1604.06174)
- github: [https://github.com/dmlc/mxnet-memonger](https://github.com/dmlc/mxnet-memonger)
- github: [https://github.com/Bihaqo/tf-memonger](https://github.com/Bihaqo/tf-memonger)

**Improving the Robustness of Deep Neural Networks via Stability Training**

- arxiv: [http://arxiv.org/abs/1604.04326](http://arxiv.org/abs/1604.04326)

**Faster Training of Very Deep Networks Via p-Norm Gates**

- arxiv: [http://arxiv.org/abs/1608.03639](http://arxiv.org/abs/1608.03639)

**Fast Training of Convolutional Neural Networks via Kernel Rescaling**

- arxiv: [https://arxiv.org/abs/1610.03623](https://arxiv.org/abs/1610.03623)

**FreezeOut: Accelerate Training by Progressively Freezing Layers**

- arxiv: [https://arxiv.org/abs/1706.04983](https://arxiv.org/abs/1706.04983)
- github: [https://github.com/ajbrock/FreezeOut](https://github.com/ajbrock/FreezeOut)

**Normalized Gradient with Adaptive Stepsize Method for Deep Neural Network Training**

- intro: CMU & The University of Iowa
- arxiv: [https://arxiv.org/abs/1707.04822](https://arxiv.org/abs/1707.04822)

**Image Quality Assessment Guided Deep Neural Networks Training**

[https://arxiv.org/abs/1708.03880](https://arxiv.org/abs/1708.03880)

**An Effective Training Method For Deep Convolutional Neural Network**

- intro: Beijing Institute of Technology & Tsinghua University
- arxiv: [https://arxiv.org/abs/1708.01666](https://arxiv.org/abs/1708.01666)

**On the Importance of Consistency in Training Deep Neural Networks**

- intro: University of Maryland & Arizona State University
- arxiv: [https://arxiv.org/abs/1708.00631](https://arxiv.org/abs/1708.00631)

# Tools

**pastalog: Simple, realtime visualization of neural network training performance**

![](/assets/train-dnn/pastalog-main-big.gif)

- github: [https://github.com/rewonc/pastalog](https://github.com/rewonc/pastalog)

**torch-pastalog: A Torch interface for pastalog - simple, realtime visualization of neural network training performance**

- github: [https://github.com/Kaixhin/torch-pastalog](https://github.com/Kaixhin/torch-pastalog)

# Blogs

**Important nuances to train deep learning models**

[http://www.erogol.com/important-nuances-train-deep-learning-models/](http://www.erogol.com/important-nuances-train-deep-learning-models/)

**Train your deep model faster and sharper — two novel techniques**

[https://hackernoon.com/training-your-deep-model-faster-and-sharper-e85076c3b047](https://hackernoon.com/training-your-deep-model-faster-and-sharper-e85076c3b047)
