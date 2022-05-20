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

# Papers

**SNIPER: Efficient Multi-Scale Training**

[https://arxiv.org/abs/1805.09300](https://arxiv.org/abs/1805.09300)

**RePr: Improved Training of Convolutional Filters**

[https://arxiv.org/abs/1811.07275](https://arxiv.org/abs/1811.07275)

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

**Training Better CNNs Requires to Rethink ReLU**

[https://arxiv.org/abs/1709.06247](https://arxiv.org/abs/1709.06247)

**Deep Learning using Rectified Linear Units (ReLU)**

- intro: Adamson University
- arxiv: [https://arxiv.org/abs/1803.08375](https://arxiv.org/abs/1803.08375)
- github: [https://github.com/AFAgarap/relu-classifier](https://github.com/AFAgarap/relu-classifier)

**Stochastic Gradient Descent Optimizes Over-parameterized Deep ReLU Networks**

- intro: University of California, Los Angeles
- arxiv: [https://arxiv.org/abs/1811.08888](https://arxiv.org/abs/1811.08888)

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

## EraseReLU

**EraseReLU: A Simple Way to Ease the Training of Deep Convolution Neural Networks**

[https://arxiv.org/abs/1709.07634](https://arxiv.org/abs/1709.07634)

## Swish

**Swish: a Self-Gated Activation Function**

**Searching for Activation Functions**

- intro: Google Brain
- arxiv: [https://arxiv.org/abs/1710.05941](https://arxiv.org/abs/1710.05941)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/77gcrv/d_swish_is_not_performing_very_well/](https://www.reddit.com/r/MachineLearning/comments/77gcrv/d_swish_is_not_performing_very_well/)

**Deep Learning with Data Dependent Implicit Activation Function**

[https://arxiv.org/abs/1802.00168](https://arxiv.org/abs/1802.00168)

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

**Dynamical Isometry and a Mean Field Theory of CNNs: How to Train 10,000-Layer Vanilla Convolutional Neural Networks**

- intro: ICML 2018. Google Brain
- arxiv: [https://arxiv.org/abs/1806.05393](https://arxiv.org/abs/1806.05393)

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

**Revisiting Batch Normalization For Practical Domain Adaptation**

- intro: Peking University & TuSimple & SenseTime
- intro: Pattern Recognition
- keywords: Adaptive Batch Normalization (AdaBN)
- arxiv: [https://arxiv.org/abs/1603.04779](https://arxiv.org/abs/1603.04779)

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

**Comparison of Batch Normalization and Weight Normalization Algorithms for the Large-scale Image Classification**

[https://arxiv.org/abs/1709.08145](https://arxiv.org/abs/1709.08145)

**In-Place Activated BatchNorm for Memory-Optimized Training of DNNs**

- intro: Mapillary Research
- arxiv: [https://arxiv.org/abs/1712.02616](https://arxiv.org/abs/1712.02616)
- github: [https://github.com/mapillary/inplace_abn](https://github.com/mapillary/inplace_abn)

**Batch Kalman Normalization: Towards Training Deep Neural Networks with Micro-Batches**

[https://arxiv.org/abs/1802.03133](https://arxiv.org/abs/1802.03133)

**Decorrelated Batch Normalization**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1804.08450](https://arxiv.org/abs/1804.08450)
- github: [https://github.com/umich-vl/DecorrelatedBN](https://github.com/umich-vl/DecorrelatedBN)

**Understanding Batch Normalization**

[https://arxiv.org/abs/1806.02375](https://arxiv.org/abs/1806.02375)

**Implementing Synchronized Multi-GPU Batch Normalization**

[http://hangzh.com/PyTorch-Encoding/notes/syncbn.html](http://hangzh.com/PyTorch-Encoding/notes/syncbn.html)

**Restructuring Batch Normalization to Accelerate CNN Training**

[https://arxiv.org/abs/1807.01702](https://arxiv.org/abs/1807.01702)

**Intro to optimization in deep learning: Busting the myth about batch normalization**

- blog: [https://blog.paperspace.com/busting-the-myths-about-batch-normalization/](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/)

**Understanding Regularization in Batch Normalization**

[https://arxiv.org/abs/1809.00846](https://arxiv.org/abs/1809.00846)

**How Does Batch Normalization Help Optimization?**

- intro: NeurIPS 2018. MIT
- arxiv: [https://arxiv.org/abs/1805.11604](https://arxiv.org/abs/1805.11604)
- video: [https://www.youtube.com/watch?v=ZOabsYbmBRM](https://www.youtube.com/watch?v=ZOabsYbmBRM)

**Cross-Iteration Batch Normalization**

[https://arxiv.org/abs/2002.05712](https://arxiv.org/abs/2002.05712)

**Extended Batch Normalization**

- intro: Chinese Academy of Sciences
- arxiv: [https://arxiv.org/abs/2003.05569](https://arxiv.org/abs/2003.05569)

**Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization**

- intro: ICLR 2020 Poster
- keywords: Moving Average Batch Normalization
- openreview: [https://openreview.net/forum?id=SkgGjRVKDS](https://openreview.net/forum?id=SkgGjRVKDS)
- github(official, Pytorch): [https://github.com/megvii-model/MABN](https://github.com/megvii-model/MABN)

**Rethinking “Batch” in BatchNorm**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/2105.07576](https://arxiv.org/abs/2105.07576)

**Delving into the Estimation Shift of Batch Normalization in a Network**

- intro: CVPR 2022
- arxiv: [https://arxiv.org/abs/2203.10778](https://arxiv.org/abs/2203.10778)
- gtihub: [https://github.com/huangleiBuaa/XBNBlock](https://github.com/huangleiBuaa/XBNBlock)

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

**Differentiable Learning-to-Normalize via Switchable Normalization**

- arxiv: [https://arxiv.org/abs/1806.10779](https://arxiv.org/abs/1806.10779)
- github: [https://github.com/switchablenorms/Switchable-Normalization](https://github.com/switchablenorms/Switchable-Normalization)

## Group Normalization

**Group Normalization**

- intro: ECCV 2018 Best Paper Award Honorable Mention
- intro: Facebook AI Research (FAIR)
- arxiv: [https://arxiv.org/abs/1803.08494](https://arxiv.org/abs/1803.08494)

## Batch-Instance Normalization

**Batch-Instance Normalization for Adaptively Style-Invariant Neural Networks**

[https://arxiv.org/abs/1805.07925](https://arxiv.org/abs/1805.07925)

**Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1807.09441](https://arxiv.org/abs/1807.09441)
- github(official, Pytorch): [https://github.com/XingangPan/IBN-Net](https://github.com/XingangPan/IBN-Net)

## Dynamic Normalization

**Dynamic Normalization**

[https://arxiv.org/abs/2101.06073](https://arxiv.org/abs/2101.06073)

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
- github: [https://github.com/luoyetx/mx-lsoftmax](https://github.com/luoyetx/mx-lsoftmax)
- github: [https://github.com/tpys/face-recognition-caffe2](https://github.com/tpys/face-recognition-caffe2)
- github: [https://github.com/jihunchoi/lsoftmax-pytorch](https://github.com/jihunchoi/lsoftmax-pytorch)

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

**DropMax: Adaptive Stochastic Softmax**

- intro: UNIST & Postech & KAIST
- arxiv: [https://arxiv.org/abs/1712.07834](https://arxiv.org/abs/1712.07834)

**Rethinking Feature Distribution for Loss Functions in Image Classification**

- intro: CVPR 2018 spotlight
- arxiv: [https://arxiv.org/abs/1803.02988](https://arxiv.org/abs/1803.02988)

**Ensemble Soft-Margin Softmax Loss for Image Classification**

- intro: IJCAI 2018
- arxiv: [https://arxiv.org/abs/1805.03922](https://arxiv.org/abs/1805.03922)

**Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels**

- intro: Cornell University
- arxiv: [https://arxiv.org/abs/1805.07836](https://arxiv.org/abs/1805.07836)

# Learning Rates

**No More Pesky Learning Rates**

- intro: Tom Schaul, Sixin Zhang, Yann LeCun
- arxiv: [https://arxiv.org/abs/1206.1106](https://arxiv.org/abs/1206.1106)

**Coupling Adaptive Batch Sizes with Learning Rates**

- intro: Max Planck Institute for Intelligent Systems
- intro: Tensorflow implementation of SGD with Coupled Adaptive Batch Size (CABS)
- arxiv: [https://arxiv.org/abs/1612.05086](https://arxiv.org/abs/1612.05086)
- github: [https://github.com/ProbabilisticNumerics/cabs](https://github.com/ProbabilisticNumerics/cabs)

**Super-Convergence: Very Fast Training of Residual Networks Using Large Learning Rates**

[https://arxiv.org/abs/1708.07120](https://arxiv.org/abs/1708.07120)

**Improving the way we work with learning rate.**

[https://medium.com/@bushaev/improving-the-way-we-work-with-learning-rate-5e99554f163b](https://medium.com/@bushaev/improving-the-way-we-work-with-learning-rate-5e99554f163b)

**WNGrad: Learn the Learning Rate in Gradient Descent**

- intro: University of Texas at Austin & Facebook AI Research
- arxiv: [https://arxiv.org/abs/1803.02865](https://arxiv.org/abs/1803.02865)

**Learning with Random Learning Rates**

- intro: Facebook AI Research & Universite Paris Sud
- keywords: All Learning Rates At Once (Alrao)
- project page: [https://leonardblier.github.io/alrao/](https://leonardblier.github.io/alrao/)
- arxiv: [https://arxiv.org/abs/1810.01322](https://arxiv.org/abs/1810.01322)
- github(PyTorch, official): [https://github.com/leonardblier/alrao](https://github.com/leonardblier/alrao)

**Learning Rate Dropout**

- intro: 1Xiamen University & Columbia University
- arxiv: [https://arxiv.org/abs/1912.00144](https://arxiv.org/abs/1912.00144)

# Convolution Filters

**Non-linear Convolution Filters for CNN-based Learning**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.07038](https://arxiv.org/abs/1708.07038)

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

**A new kind of pooling layer for faster and sharper convergence**

- blog: [https://medium.com/@singlasahil14/a-new-kind-of-pooling-layer-for-faster-and-sharper-convergence-1043c756a221](https://medium.com/@singlasahil14/a-new-kind-of-pooling-layer-for-faster-and-sharper-convergence-1043c756a221)
- github: [https://github.com/singlasahil14/sortpool2d](https://github.com/singlasahil14/sortpool2d)

**Statistically Motivated Second Order Pooling**

[https://arxiv.org/abs/1801.07492](https://arxiv.org/abs/1801.07492)

**Detail-Preserving Pooling in Deep Networks**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1804.04076](https://arxiv.org/abs/1804.04076)

# Mini-Batch

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

**Large Batch Training of Convolutional Networks**

[https://arxiv.org/abs/1708.03888](https://arxiv.org/abs/1708.03888)

**ImageNet Training in 24 Minutes**

[https://arxiv.org/abs/1709.05011](https://arxiv.org/abs/1709.05011)

**Don't Decay the Learning Rate, Increase the Batch Size**

- intro: Google Brain
- arxiv: [https://arxiv.org/abs/1711.00489](https://arxiv.org/abs/1711.00489)

**Extremely Large Minibatch SGD: Training ResNet-50 on ImageNet in 15 Minutes**

- intro: NIPS 2017 Workshop: Deep Learning at Supercomputer Scale
- arxiv: [https://arxiv.org/abs/1711.04325](https://arxiv.org/abs/1711.04325)

**AdaBatch: Adaptive Batch Sizes for Training Deep Neural Networks**

- intro: UC Berkeley & NVIDIA
- arxiv: [https://arxiv.org/abs/1712.02029](https://arxiv.org/abs/1712.02029)

**Hessian-based Analysis of Large Batch Training and Robustness to Adversaries**

- intro: UC Berkeley & University of Texas
- arxiv: [https://arxiv.org/abs/1802.08241](https://arxiv.org/abs/1802.08241)

**Large Batch Training of Convolutional Networks with Layer-wise Adaptive Rate Scaling**

- keywords: large batch, LARS, adaptive rate scaling
- openreview: [https://openreview.net/forum?id=rJ4uaX2aW](https://openreview.net/forum?id=rJ4uaX2aW)

**Revisiting Small Batch Training for Deep Neural Networks**

[https://arxiv.org/abs/1804.07612](https://arxiv.org/abs/1804.07612)

**Second-order Optimization Method for Large Mini-batch: Training ResNet-50 on ImageNet in 35 Epochs**

[https://arxiv.org/abs/1811.12019](https://arxiv.org/abs/1811.12019)

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

- intro: ICLR 2017
- keywords: cosine annealing strategy
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

**Optimization Algorithms**

- blog: [https://3dbabove.com/2017/11/14/optimizationalgorithms/](https://3dbabove.com/2017/11/14/optimizationalgorithms/)
- github: [https://github.com//ManuelGonzalezRivero/3dbabove](https://github.com//ManuelGonzalezRivero/3dbabove)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/7ehxky/d_optimization_algorithms_math_and_code/](https://www.reddit.com/r/MachineLearning/comments/7ehxky/d_optimization_algorithms_math_and_code/)

**Gradient Normalization & Depth Based Decay For Deep Learning**

- intro: Columbia University
- arxiv: [https://arxiv.org/abs/1712.03607](https://arxiv.org/abs/1712.03607)

**Neumann Optimizer: A Practical Optimization Algorithm for Deep Neural Networks**

- intro: Google Research
- arxiv: [https://arxiv.org/abs/1712.03298](https://arxiv.org/abs/1712.03298)

**Optimization for Deep Learning Highlights in 2017**

[http://ruder.io/deep-learning-optimization-2017/index.html](http://ruder.io/deep-learning-optimization-2017/index.html)

**Gradients explode - Deep Networks are shallow - ResNet explained**

- intro: CMU & UC Berkeley
- arxiv: [https://arxiv.org/abs/1712.05577](https://arxiv.org/abs/1712.05577)

**A Sufficient Condition for Convergences of Adam and RMSProp**

[https://arxiv.org/abs/1811.09358](https://arxiv.org/abs/1811.09358)

## Adam

**Adam: A Method for Stochastic Optimization**

- intro: ICLR 2015
- arxiv: [http://arxiv.org/abs/1412.6980](http://arxiv.org/abs/1412.6980)

**Fixing Weight Decay Regularization in Adam**

- intro: University of Freiburg
- arxiv: [https://arxiv.org/abs/1711.05101](https://arxiv.org/abs/1711.05101)
- github: [https://github.com/loshchil/AdamW-and-SGDW](https://github.com/loshchil/AdamW-and-SGDW)
- github: [https://github.com/fastai/fastai/pull/46/files](https://github.com/fastai/fastai/pull/46/files)

**On the Convergence of Adam and Beyond**

- intro: ICLR 2018 best paper award. CMU & IBM Research
- paper: [https://openreview.net/pdf?id=ryQu7f-RZ](https://openreview.net/pdf?id=ryQu7f-RZ)
- openreview: [https://openreview.net/forum?id=ryQu7f-RZ](https://openreview.net/forum?id=ryQu7f-RZ)

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

**LDMNet: Low Dimensional Manifold Regularized Neural Networks**

[https://arxiv.org/abs/1711.06246](https://arxiv.org/abs/1711.06246)

**Learning Sparse Neural Networks through L0 Regularization**

- intro: University of Amsterdam & OpenAI
- arxiv: [https://arxiv.org/abs/1712.01312](https://arxiv.org/abs/1712.01312)

**Regularization and Optimization strategies in Deep Convolutional Neural Network**

[https://arxiv.org/abs/1712.04711](https://arxiv.org/abs/1712.04711)

**Regularizing Deep Networks by Modeling and Predicting Label Structure**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1804.02009](https://arxiv.org/abs/1804.02009)

**Adversarial Noise Layer: Regularize Neural Network By Adding Noise**

- intro: Peking University & ‡University of Electronic Science and Technology of China & Australian National University
- arxiv: [https://arxiv.org/abs/1805.08000](https://arxiv.org/abs/1805.08000)
- github: [https://github.com/youzhonghui/ANL](https://github.com/youzhonghui/ANL)

**Deep Bilevel Learning**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1809.01465](https://arxiv.org/abs/1809.01465)

**Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?**

- intro: NIPS 2018
- arxiv: [https://arxiv.org/abs/1810.09102](https://arxiv.org/abs/1810.09102)

**Gradient-Coherent Strong Regularization for Deep Neural Networks**

[https://arxiv.org/abs/1811.08056](https://arxiv.org/abs/1811.08056)

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

**An Analysis of Dropout for Matrix Factorization**

[https://arxiv.org/abs/1710.03487](https://arxiv.org/abs/1710.03487)

**Analysis of Dropout in Online Learning**

[https://arxiv.org/abs/1711.03343](https://arxiv.org/abs/1711.03343)

**Regularization of Deep Neural Networks with Spectral Dropout**

[https://arxiv.org/abs/1711.08591](https://arxiv.org/abs/1711.08591)

**Data Dropout in Arbitrary Basis for Deep Network Regularization**

[https://arxiv.org/abs/1712.00891](https://arxiv.org/abs/1712.00891)

**A New Angle on L2 Regularization**

- intro: An explorable explanation on the phenomenon of adversarial examples in linear classification and its relation to L2 regularization
- blog: [https://thomas-tanay.github.io/post--L2-regularization/](https://thomas-tanay.github.io/post--L2-regularization/)
- arxiv: [https://arxiv.org/abs/1806.11186](https://arxiv.org/abs/1806.11186)

**Dropout is a special case of the stochastic delta rule: faster and more accurate deep learning**

- intro: Rutgers University
- arxiv: [https://arxiv.org/abs/1808.03578](https://arxiv.org/abs/1808.03578)
- github: [https://github.com/noahfl/densenet-sdr/](https://github.com/noahfl/densenet-sdr/)

**Data Dropout: Optimizing Training Data for Convolutional Neural Networks**

[https://arxiv.org/abs/1809.00193](https://arxiv.org/abs/1809.00193)

**DropFilter: Dropout for Convolutions**

[https://arxiv.org/abs/1810.09849](https://arxiv.org/abs/1810.09849)

**DropFilter: A Novel Regularization Method for Learning Convolutional Neural Networks**

[https://arxiv.org/abs/1811.06783](https://arxiv.org/abs/1811.06783)

**Targeted Dropout**

- intro: Google Brain & FOR.ai & University of Oxford
- paper: [https://openreview.net/pdf?id=HkghWScuoQ](https://openreview.net/pdf?id=HkghWScuoQ)
- github: [https://github.com/for-ai/TD](https://github.com/for-ai/TD)

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

## DropBlock

**DropBlock: A regularization method for convolutional networks**

- intro: NIPS 2018
- arxiv: [https://arxiv.org/abs/1810.12890](https://arxiv.org/abs/1810.12890)

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

**ShakeDrop regularization**

[https://arxiv.org/abs/1802.02375](https://arxiv.org/abs/1802.02375)

**Shakeout: A New Approach to Regularized Deep Neural Network Training**

- intro: T-PAMI 2018
- arxiv: [https://arxiv.org/abs/1904.06593](https://arxiv.org/abs/1904.06593)

# Gradient Descent

**RMSProp: Divide the gradient by a running average of its recent magnitude**

![](/assets/train-dnn/rmsprop.jpg)

- intro: it was not proposed in a paper, in fact it was just introduced in a slide in Geoffrey Hinton's Coursera class 
- slides: [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

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

**Accelerating Stochastic Gradient Descent**

[https://arxiv.org/abs/1704.08227](https://arxiv.org/abs/1704.08227)

**Gentle Introduction to the Adam Optimization Algorithm for Deep Learning**

- blog: [http://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/](http://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)

**Understanding Generalization and Stochastic Gradient Descent**

**A Bayesian Perspective on Generalization and Stochastic Gradient Descent**

- intro: Google Brain
- arxiv: [https://arxiv.org/abs/1710.06451](https://arxiv.org/abs/1710.06451)

**Accelerated Gradient Descent Escapes Saddle Points Faster than Gradient Descent**

- intro: UC Berkeley & Microsoft Research, India
- arxiv: [https://arxiv.org/abs/1711.10456](https://arxiv.org/abs/1711.10456)

**Improving Generalization Performance by Switching from Adam to SGD**

[https://arxiv.org/abs/1712.07628](https://arxiv.org/abs/1712.07628)

**Laplacian Smoothing Gradient Descent**

- intro: UCLA
- arxiv: [https://arxiv.org/abs/1806.06317](https://arxiv.org/abs/1806.06317)

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

**meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting**

- intro: ICML 2017
- arxiv: [https://arxiv.org/abs/1706.06197](https://arxiv.org/abs/1706.06197)
- github: [https://github.com//jklj077/meProp](https://github.com//jklj077/meProp)

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

**Improving Deep Learning using Generic Data Augmentation**

- intro: University of Cape Town
- arxiv: [https://arxiv.org/abs/1708.06020](https://arxiv.org/abs/1708.06020)
- github: [https://github.com/webstorms/AugmentedDatasets](https://github.com/webstorms/AugmentedDatasets)

**Augmentor: An Image Augmentation Library for Machine Learning**

- arxiv: [https://arxiv.org/abs/1708.04680](https://arxiv.org/abs/1708.04680)
- github: [https://github.com/mdbloice/Augmentor](https://github.com/mdbloice/Augmentor)

**Automatic Dataset Augmentation**

- project page: [https://auto-da.github.io/](https://auto-da.github.io/)
- arxiv: [https://arxiv.org/abs/1708.08201](https://arxiv.org/abs/1708.08201)

**Learning to Compose Domain-Specific Transformations for Data Augmentation**

[https://arxiv.org/abs/1709.01643](https://arxiv.org/abs/1709.01643)

**Data Augmentation in Classification using GAN**

[https://arxiv.org/abs/1711.00648](https://arxiv.org/abs/1711.00648)

**Data Augmentation Generative Adversarial Networks**

[https://arxiv.org/abs/1711.04340](https://arxiv.org/abs/1711.04340)

**Random Erasing Data Augmentation**

- arxiv: [https://arxiv.org/abs/1708.04896](https://arxiv.org/abs/1708.04896)
- github: [https://github.com/zhunzhong07/Random-Erasing](https://github.com/zhunzhong07/Random-Erasing)

**Context Augmentation for Convolutional Neural Networks**

[https://arxiv.org/abs/1712.01653](https://arxiv.org/abs/1712.01653)

**The Effectiveness of Data Augmentation in Image Classification using Deep Learning**

[https://arxiv.org/abs/1712.04621](https://arxiv.org/abs/1712.04621)

**MentorNet: Regularizing Very Deep Neural Networks on Corrupted Labels**

- intro: Google Inc & Stanford University
- arxiv: [https://arxiv.org/abs/1712.05055](https://arxiv.org/abs/1712.05055)

**mixup: Beyond Empirical Risk Minimization**

- intro: MIT & FAIR
- arxiv: [https://arxiv.org/abs/1710.09412](https://arxiv.org/abs/1710.09412)
- github: [https://github.com//leehomyc/mixup_pytorch](https://github.com//leehomyc/mixup_pytorch)
- github: [https://github.com//unsky/mixup](https://github.com//unsky/mixup)

**mixup: Data-Dependent Data Augmentation**

[http://www.inference.vc/mixup-data-dependent-data-augmentation/](http://www.inference.vc/mixup-data-dependent-data-augmentation/)

**Data Augmentation by Pairing Samples for Images Classification**

- intro: IBM Research - Tokyo
- arxiv: [https://arxiv.org/abs/1801.02929](https://arxiv.org/abs/1801.02929)

**Feature Space Transfer for Data Augmentation**

- keywords: eATure TransfEr Network (FATTEN)
- arxiv: [https://arxiv.org/abs/1801.04356](https://arxiv.org/abs/1801.04356)

**Visual Data Augmentation through Learning**

[https://arxiv.org/abs/1801.06665](https://arxiv.org/abs/1801.06665)

**Data Augmentation Generative Adversarial Networks**

- arxiv: [https://arxiv.org/abs/1711.04340](https://arxiv.org/abs/1711.04340)
- github: [https://github.com/AntreasAntoniou/DAGAN](https://github.com/AntreasAntoniou/DAGAN)

**BAGAN: Data Augmentation with Balancing GAN**

[https://arxiv.org/abs/1803.09655](https://arxiv.org/abs/1803.09655)

**Parallel Grid Pooling for Data Augmentation**

- intro: The University of Tokyo & NTT Communications Science Laboratories
- arxiv: [https://arxiv.org/abs/1803.11370](https://arxiv.org/abs/1803.11370)
- github(Chainer): [https://github.com/akitotakeki/pgp-chainer](https://github.com/akitotakeki/pgp-chainer)

**AutoAugment: Learning Augmentation Policies from Data**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1805.09501](https://arxiv.org/abs/1805.09501)
- github: [https://github.com/DeepVoltaire/AutoAugment](https://github.com/DeepVoltaire/AutoAugment)

**Improved Mixed-Example Data Augmentation**

[https://arxiv.org/abs/1805.11272](https://arxiv.org/abs/1805.11272)

**Data augmentation instead of explicit regularization**

[https://arxiv.org/abs/1806.03852](https://arxiv.org/abs/1806.03852)

**Data Augmentation using Random Image Cropping and Patching for Deep CNNs**

- intro: An extended version of a proceeding of ACML2018
- keywords: random image cropping and patching (RICAP)
- arxiv: [https://arxiv.org/abs/1811.09030](https://arxiv.org/abs/1811.09030)

**GANsfer Learning: Combining labelled and unlabelled data for GAN based data augmentat**

[https://arxiv.org/abs/1811.10669](https://arxiv.org/abs/1811.10669)

**Adversarial Learning of General Transformations for Data Augmentation**

- intro: Ecole de Technologie Sup ´ erieure & Element AI
- arxiv: [https://arxiv.org/abs/1909.09801](https://arxiv.org/abs/1909.09801)

**Implicit Semantic Data Augmentation for Deep Networks**

- intro: NeurIPS 2019
- arxiv: [https://arxiv.org/abs/1909.12220](https://arxiv.org/abs/1909.12220)
- github(official): [https://github.com/blackfeather-wang/ISDA-for-Deep-Networks](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks)

**Data Augmentation Revisited: Rethinking the Distribution Gap between Clean and Augmented Data**

[https://arxiv.org/abs/1909.09148](https://arxiv.org/abs/1909.09148)

**AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty**

- intro: ICLR 2020
- intro: Google & Deepmind
- arxiv: [https://arxiv.org/abs/1912.02781](https://arxiv.org/abs/1912.02781)
- github: [https://github.com/google-research/augmix](https://github.com/google-research/augmix)

**GridMask Data Augmentation**

[https://arxiv.org/abs/2001.04086](https://arxiv.org/abs/2001.04086)

**On Feature Normalization and Data Augmentation**

- intro: Cornell University & Cornell Tech & ASAPP Inc. & Facebook AI
- keywords: MoEx (Moment Exchange)
- arxiv: [https://arxiv.org/abs/2002.11102](https://arxiv.org/abs/2002.11102)
- github: [https://github.com/Boyiliee/MoEx](https://github.com/Boyiliee/MoEx)

**DADA: Differentiable Automatic Data Augmentation**

[https://arxiv.org/abs/2003.03780](https://arxiv.org/abs/2003.03780)

**Negative Data Augmentation**

- intro: ICLR 2021
- intro: Stanford University & Samsung Research America
- arxiv: [https://arxiv.org/abs/2102.05113](https://arxiv.org/abs/2102.05113)

## Imbalanced Datasets

**Investigation on handling Structured & Imbalanced Datasets with Deep Learning**

- intro: smote resampling, cost sensitive learning
- blog: [https://www.analyticsvidhya.com/blog/2016/10/investigation-on-handling-structured-imbalanced-datasets-with-deep-learning/](https://www.analyticsvidhya.com/blog/2016/10/investigation-on-handling-structured-imbalanced-datasets-with-deep-learning/)

**A systematic study of the class imbalance problem in convolutional neural networks**

- intro: Duke University & Royal Institute of Technology (KTH)
- arxiv: [https://arxiv.org/abs/1710.05381](https://arxiv.org/abs/1710.05381)

**Class Rectification Hard Mining for Imbalanced Deep Learning**

[https://arxiv.org/abs/1712.03162](https://arxiv.org/abs/1712.03162)

**Bridging the Gap: Simultaneous Fine Tuning for Data Re-Balancing**

- arxiv: [https://arxiv.org/abs/1801.02548](https://arxiv.org/abs/1801.02548)
- github: [https://github.com/JohnMcKay/dataImbalance](https://github.com/JohnMcKay/dataImbalance)

**Imbalanced Deep Learning by Minority Class Incremental Rectification**

- intro: TPAMI
- arxiv: [https://arxiv.org/abs/1804.10851](https://arxiv.org/abs/1804.10851)

**Pseudo-Feature Generation for Imbalanced Data Analysis in Deep Learning**

- intro: National Institute of Information and Communications Technology, Tokyo Japan
- arxiv: [https://arxiv.org/abs/1807.06538](https://arxiv.org/abs/1807.06538)
- slides: [https://www.slideshare.net/TomohikoKonno/pseudofeature-generation-for-imbalanced-data-analysis-in-deep-learning-tomohiko-105318569](https://www.slideshare.net/TomohikoKonno/pseudofeature-generation-for-imbalanced-data-analysis-in-deep-learning-tomohiko-105318569)

**Max-margin Class Imbalanced Learning with Gaussian Affinity**

[https://arxiv.org/abs/1901.07711](https://arxiv.org/abs/1901.07711)

**Dynamic Curriculum Learning for Imbalanced Data Classification**

- intro: ICCV 2019
- intro: SenseTime
- arxiv: [https://arxiv.org/abs/1901.06783](https://arxiv.org/abs/1901.06783)

**Class Rectification Hard Mining for Imbalanced Deep Learning**

- intro: ICCV 2017
- paper: [https://www.eecs.qmul.ac.uk/~sgg/papers/DongEtAl_ICCV2017.pdf](https://www.eecs.qmul.ac.uk/~sgg/papers/DongEtAl_ICCV2017.pdf)

## Noisy / Unlabelled Data

**Data Distillation: Towards Omni-Supervised Learning**

- intro: Facebook AI Research (FAIR)
- arxiv: [https://arxiv.org/abs/1712.04440](https://arxiv.org/abs/1712.04440)

**Learning From Noisy Singly-labeled Data**

- intro: University of Illinois Urbana Champaign & CMU & Caltech & Amazon AI
- arxiv: [https://arxiv.org/abs/1712.04577](https://arxiv.org/abs/1712.04577)

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

# Distributed Training

**Large Scale Distributed Systems for Training Neural Networks**

- intro: By Jeff Dean & Oriol Vinyals, Google. NIPS 2015.
- slides: [https://media.nips.cc/Conferences/2015/tutorialslides/Jeff-Oriol-NIPS-Tutorial-2015.pdf](https://media.nips.cc/Conferences/2015/tutorialslides/Jeff-Oriol-NIPS-Tutorial-2015.pdf)
- video: [http://research.microsoft.com/apps/video/default.aspx?id=259564&l=i](http://research.microsoft.com/apps/video/default.aspx?id=259564&l=i)
- mirror: [http://pan.baidu.com/s/1mgXV0hU](http://pan.baidu.com/s/1mgXV0hU)

**Large Scale Distributed Deep Networks**

- intro: distributed CPU training, data parallelism, model parallelism
- paper: [http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf](http://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf)
- slides: [http://admis.fudan.edu.cn/~yfhuang/files/LSDDN_slide.pdf](http://admis.fudan.edu.cn/~yfhuang/files/LSDDN_slide.pdf)

**Implementation of a Practical Distributed Calculation System with Browsers and JavaScript, and Application to Distributed Deep Learning**

- project page: [http://mil-tokyo.github.io/](http://mil-tokyo.github.io/)
- arxiv: [https://arxiv.org/abs/1503.05743](https://arxiv.org/abs/1503.05743)

**SparkNet: Training Deep Networks in Spark**

- arxiv: [http://arxiv.org/abs/1511.06051](http://arxiv.org/abs/1511.06051)
- github: [https://github.com/amplab/SparkNet](https://github.com/amplab/SparkNet)
- blog: [http://www.kdnuggets.com/2015/12/spark-deep-learning-training-with-sparknet.html](http://www.kdnuggets.com/2015/12/spark-deep-learning-training-with-sparknet.html)

**A Scalable Implementation of Deep Learning on Spark**

- intro: Alexander Ulanov
- slides: [http://www.slideshare.net/AlexanderUlanov1/a-scalable-implementation-of-deep-learning-on-spark-alexander-ulanov](http://www.slideshare.net/AlexanderUlanov1/a-scalable-implementation-of-deep-learning-on-spark-alexander-ulanov)
- mirror: [http://pan.baidu.com/s/1jHiNW5C](http://pan.baidu.com/s/1jHiNW5C)

**TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**

- arxiv: [http://arxiv.org/abs/1603.04467](http://arxiv.org/abs/1603.04467)
- gitxiv: [http://gitxiv.com/posts/57kjddp3AWt4y5K4h/tensorflow-large-scale-machine-learning-on-heterogeneous](http://gitxiv.com/posts/57kjddp3AWt4y5K4h/tensorflow-large-scale-machine-learning-on-heterogeneous)

**Distributed Supervised Learning using Neural Networks**

- intro: Ph.D. thesis
- arxiv: [http://arxiv.org/abs/1607.06364](http://arxiv.org/abs/1607.06364)

**Distributed Training of Deep Neuronal Networks: Theoretical and Practical Limits of Parallel Scalability**

- arxiv: [http://arxiv.org/abs/1609.06870](http://arxiv.org/abs/1609.06870)

**How to scale distributed deep learning?**

- intro: Extended version of paper accepted at ML Sys 2016 (at NIPS 2016)
- arxiv: [https://arxiv.org/abs/1611.04581](https://arxiv.org/abs/1611.04581)

**Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training**

- intro: Tsinghua University & Stanford University
- comments: we find 99.9% of the gradient exchange in distributed SGD is redundant; we reduce the communication bandwidth by two orders of magnitude without losing accuracy
- keywords: momentum correction, local gradient clipping, momentum factor masking, and warm-up training
- arxiv: [https://arxiv.org/abs/1712.01887](https://arxiv.org/abs/1712.01887)

**Distributed learning of CNNs on heterogeneous CPU/GPU architectures**

[https://arxiv.org/abs/1712.02546](https://arxiv.org/abs/1712.02546)

**Integrated Model and Data Parallelism in Training Neural Networks**

- intro: UC Berkeley & Lawrence Berkeley National Laboratory
- arxiv: [https://arxiv.org/abs/1712.04432](https://arxiv.org/abs/1712.04432)

**Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training**

- intro: ICLR 2018
- intro: we find 99.9% of the gradient exchange in distributed SGD is redundant; we reduce the communication bandwidth by two orders of magnitude without losing accuracy
- arxiv: [https://arxiv.org/abs/1712.01887](https://arxiv.org/abs/1712.01887)

**RedSync : Reducing Synchronization Traffic for Distributed Deep Learning**

[https://arxiv.org/abs/1808.04357](https://arxiv.org/abs/1808.04357)

## Projects

**Theano-MPI: a Theano-based Distributed Training Framework**

- arxiv: [https://arxiv.org/abs/1605.08325](https://arxiv.org/abs/1605.08325)
- github: [https://github.com/uoguelph-mlrg/Theano-MPI](https://github.com/uoguelph-mlrg/Theano-MPI)

**CaffeOnSpark: Open Sourced for Distributed Deep Learning on Big Data Clusters**

- intro: Yahoo Big ML Team
- blog: [http://yahoohadoop.tumblr.com/post/139916563586/caffeonspark-open-sourced-for-distributed-deep](http://yahoohadoop.tumblr.com/post/139916563586/caffeonspark-open-sourced-for-distributed-deep)
- github: [https://github.com/yahoo/CaffeOnSpark](https://github.com/yahoo/CaffeOnSpark)
- youtube: [https://www.youtube.com/watch?v=bqj7nML-aHk](https://www.youtube.com/watch?v=bqj7nML-aHk)

**Tunnel: Data Driven Framework for Distributed Computing in Torch 7**

- github: [https://github.com/zhangxiangxiao/tunnel](https://github.com/zhangxiangxiao/tunnel)

**Distributed deep learning with Keras and Apache Spark**

- project page: [http://joerihermans.com/work/distributed-keras/](http://joerihermans.com/work/distributed-keras/)
- github: [https://github.com/JoeriHermans/dist-keras](https://github.com/JoeriHermans/dist-keras)

**BigDL: Distributed Deep learning Library for Apache Spark**

- github: [https://github.com/intel-analytics/BigDL](https://github.com/intel-analytics/BigDL)

## Videos

**A Scalable Implementation of Deep Learning on Spark**

- youtube: [https://www.youtube.com/watch?v=pNYBBhuK8yU](https://www.youtube.com/watch?v=pNYBBhuK8yU)
- mirror: [http://pan.baidu.com/s/1mhzF1uK](http://pan.baidu.com/s/1mhzF1uK)

**Distributed TensorFlow on Spark: Scaling Google's Deep Learning Library (Spark Summit)**

- youtube: [https://www.youtube.com/watch?v=-QtcP3yRqyM](https://www.youtube.com/watch?v=-QtcP3yRqyM)
- mirror: [http://pan.baidu.com/s/1mgOR1GG](http://pan.baidu.com/s/1mgOR1GG)

**Deep Recurrent Neural Networks for Sequence Learning in Spark (Spark Summit)**

- youtube: [https://www.youtube.com/watch?v=mUuqLcl8Jog](https://www.youtube.com/watch?v=mUuqLcl8Jog)
- mirror: [http://pan.baidu.com/s/1sklHTPr](http://pan.baidu.com/s/1sklHTPr)

**Distributed deep learning on Spark**

- author: Alexander Ulanov July 12, 2016
- intro: Alexander Ulanov offers an overview of tools and frameworks that have been proposed for performing deep learning on Spark.
- video: [https://www.oreilly.com/learning/distributed-deep-learning-on-spark](https://www.oreilly.com/learning/distributed-deep-learning-on-spark)

## Blogs

**Distributed Deep Learning Reads**

[https://github.com//tmulc18/DistributedDeepLearningReads](https://github.com//tmulc18/DistributedDeepLearningReads)

**Hadoop, Spark, Deep Learning Mesh on Single GPU Cluster**

[http://www.nextplatform.com/2016/02/24/hadoop-spark-deep-learning-mesh-on-single-gpu-cluster/](http://www.nextplatform.com/2016/02/24/hadoop-spark-deep-learning-mesh-on-single-gpu-cluster/)

**The Unreasonable Effectiveness of Deep Learning on Spark**

[https://databricks.com/blog/2016/04/01/unreasonable-effectiveness-of-deep-learning-on-spark.html](https://databricks.com/blog/2016/04/01/unreasonable-effectiveness-of-deep-learning-on-spark.html)

**Distributed Deep Learning with Caffe Using a MapR Cluster**

![](https://www.mapr.com/sites/default/files/spark-driver.jpg)

[https://www.mapr.com/blog/distributed-deep-learning-caffe-using-mapr-cluster](https://www.mapr.com/blog/distributed-deep-learning-caffe-using-mapr-cluster)

**Deep Learning with Apache Spark and TensorFlow**

[https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html](https://databricks.com/blog/2016/01/25/deep-learning-with-apache-spark-and-tensorflow.html)

**Deeplearning4j on Spark**

[http://deeplearning4j.org/spark](http://deeplearning4j.org/spark)

**Distributed Deep Learning, Part 1: An Introduction to Distributed Training of Neural Networks**

- blog: [http://engineering.skymind.io/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks](http://engineering.skymind.io/distributed-deep-learning-part-1-an-introduction-to-distributed-training-of-neural-networks)

**GPU Acceleration in Databricks: Speeding Up Deep Learning on Apache Spark**

[https://databricks.com/blog/2016/10/27/gpu-acceleration-in-databricks.html](https://databricks.com/blog/2016/10/27/gpu-acceleration-in-databricks.html)

**Distributed Deep Learning with Apache Spark and Keras**

[https://db-blog.web.cern.ch/blog/joeri-hermans/2017-01-distributed-deep-learning-apache-spark-and-keras](https://db-blog.web.cern.ch/blog/joeri-hermans/2017-01-distributed-deep-learning-apache-spark-and-keras)

# Adversarial Training

**Learning from Simulated and Unsupervised Images through Adversarial Training**

- intro: CVPR 2017 oral, best paper award. Apple Inc.
- arxiv: [https://arxiv.org/abs/1612.07828](https://arxiv.org/abs/1612.07828)

**The Robust Manifold Defense: Adversarial Training using Generative Models**

[https://arxiv.org/abs/1712.09196](https://arxiv.org/abs/1712.09196)

**DeepDefense: Training Deep Neural Networks with Improved Robustness**

[https://arxiv.org/abs/1803.00404](https://arxiv.org/abs/1803.00404)

**Gradient Adversarial Training of Neural Networks**

- intro: Magic Leap
- arxiv: [https://arxiv.org/abs/1806.08028](https://arxiv.org/abs/1806.08028)

**Gray-box Adversarial Training**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1808.01753](https://arxiv.org/abs/1808.01753)

**Universal Adversarial Training**

[https://arxiv.org/abs/1811.11304](https://arxiv.org/abs/1811.11304)

**MEAL: Multi-Model Ensemble via Adversarial Learning**

- intro: AAAI 2019
- intro: Fudan University & University of Illinois at Urbana-Champaign
- arxiv: [https://arxiv.org/abs/1812.02425](https://arxiv.org/abs/1812.02425)
- github(official): [https://github.com/AaronHeee/MEAL](https://github.com/AaronHeee/MEAL)

**Regularized Ensembles and Transferability in Adversarial Learning**

[https://arxiv.org/abs/1812.01821](https://arxiv.org/abs/1812.01821)

**Feature denoising for improving adversarial robustness**

- intro: Johns Hopkins University & Facebook AI Research
- intro: ranked first in Competition on Adversarial Attacks and Defenses (CAAD) 2018
- arxiv: [https://arxiv.org/abs/1812.03411](https://arxiv.org/abs/1812.03411)
- github: [https://github.com/facebookresearch/ImageNet-Adversarial-Training](https://github.com/facebookresearch/ImageNet-Adversarial-Training)

**Second Rethinking of Network Pruning in the Adversarial Setting**

[https://arxiv.org/abs/1903.12561](https://arxiv.org/abs/1903.12561)

**Interpreting Adversarially Trained Convolutional Neural Networks**

- intro: ICML 2019
- arxiv: [https://arxiv.org/abs/1905.09797](https://arxiv.org/abs/1905.09797)

**On Stabilizing Generative Adversarial Training with Noise**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1906.04612](https://arxiv.org/abs/1906.04612)

**Adversarial Learning with Margin-based Triplet Embedding Regularization**

- intro: ICCV 2019
- intro: BUPT
- arxiv: [https://arxiv.org/abs/1909.09481](https://arxiv.org/abs/1909.09481)
- github: [https://github.com/zhongyy/Adversarial_MTER](https://github.com/zhongyy/Adversarial_MTER)

**Bag of Tricks for Adversarial Training**

- intro: Tsinghua University
- arxiv: [https://arxiv.org/abs/2010.00467](https://arxiv.org/abs/2010.00467)

# Low-Precision Training

**Mixed Precision Training**

- intro: ICLR 2018
- arxiv: [https://arxiv.org/abs/1710.03740](https://arxiv.org/abs/1710.03740)

**High-Accuracy Low-Precision Training**

- intro: Cornell University & Stanford University
- arxiv: [https://arxiv.org/abs/1803.03383](https://arxiv.org/abs/1803.03383)

# Incremental Training

**ClickBAIT: Click-based Accelerated Incremental Training of Convolutional Neural Networks**

- arxiv: [https://arxiv.org/abs/1709.05021](https://arxiv.org/abs/1709.05021)
- dataset: [http://clickbait.crossmobile.info/](http://clickbait.crossmobile.info/)

**ClickBAIT-v2: Training an Object Detector in Real-Time**

[https://arxiv.org/abs/1803.10358](https://arxiv.org/abs/1803.10358)

**Class-incremental Learning via Deep Model Consolidation**

- intro: University of Southern California & Arizona State University & Samsung Research America
- arxiv: [https://arxiv.org/abs/1903.07864](https://arxiv.org/abs/1903.07864)

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

**Solving internal covariate shift in deep learning with linked neurons**

- intro: Universitat de Barcelona
- arxiv: [https://arxiv.org/abs/1712.02609](https://arxiv.org/abs/1712.02609)
- github: [https://github.com/blauigris/linked_neurons](https://github.com/blauigris/linked_neurons)

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
