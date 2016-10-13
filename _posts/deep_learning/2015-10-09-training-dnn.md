---
layout: post
category: deep_learning
title: Training Deep Neural Networks
date: 2015-10-09
---

# Tutorials

**Popular Training Approaches of DNNs — A Quick Overview**

[https://medium.com/@asjad/popular-training-approaches-of-dnns-a-quick-overview-26ee37ad7e96#.pqyo039bb](https://medium.com/@asjad/popular-training-approaches-of-dnns-a-quick-overview-26ee37ad7e96#.pqyo039bb)

# Activation functions

## ReLU

**Rectified linear units improve restricted boltzmann machines**

- intro: ReLU
- paper: [http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)

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

- arxiv: [http://arxiv.org/abs/1603.05201](http://arxiv.org/abs/1603.05201)

## GELU

**Bridging Nonlinearities and Stochastic Regularizers with Gaussian Error Linear Units**

- arxiv: [https://arxiv.org/abs/1606.08415](https://arxiv.org/abs/1606.08415)

**Formulating The ReLU**

- blog: [http://www.jefkine.com/general/2016/08/24/formulating-the-relu/](http://www.jefkine.com/general/2016/08/24/formulating-the-relu/)

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

**Data-dependent Initializations of Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1511.06856](http://arxiv.org/abs/1511.06856)
- github: [https://github.com/philkr/magic_init](https://github.com/philkr/magic_init)

**What are good initial weights in a neural network?**

- stackexchange: [http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network](http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network)

**RandomOut: Using a convolutional gradient norm to win The Filter Lottery**

- arxiv: [http://arxiv.org/abs/1602.05931](http://arxiv.org/abs/1602.05931)

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
- notes: [http://www.erogol.com/my-notes-weight-normalization/](http://www.erogol.com/my-notes-weight-normalization/)

**Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks**

- arxiv: [http://arxiv.org/abs/1603.01431](http://arxiv.org/abs/1603.01431)

**Understanding the backward pass through Batch Normalization Layer**

![](https://kratzert.github.io/images/bn_backpass/BNcircuit.png)

- blog: [https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html)

**Implementing Batch Normalization in Tensorflow**

- blog: [http://r2rt.com/implementing-batch-normalization-in-tensorflow.html](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html)

**Deriving the Gradient for the Backward Pass of Batch Normalization**

- blog: [https://kevinzakka.github.io/2016/09/14/batch_normalization/](https://kevinzakka.github.io/2016/09/14/batch_normalization/)

## Layer Normalization

**Layer Normalization**

- arxiv: [https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)
- github: [https://github.com/ryankiros/layer-norm](https://github.com/ryankiros/layer-norm)
- github(TensorFlow): [https://github.com/pbhatia243/tf-layer-norm](https://github.com/pbhatia243/tf-layer-norm)

**Keras GRU with Layer Normalization**

- gist: [https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940](https://gist.github.com/udibr/7f46e790c9e342d75dcbd9b1deb9d940)

# Loss Function

**The Loss Surfaces of Multilayer Networks**

- arxiv: [http://arxiv.org/abs/1412.0233](http://arxiv.org/abs/1412.0233)

**Direct Loss Minimization for Training Deep Neural Nets**

- arxiv: [http://arxiv.org/abs/1511.06411](http://arxiv.org/abs/1511.06411)

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

- blog: [http://sebastianruder.com/optimizing-gradient-descent/](http://sebastianruder.com/optimizing-gradient-descent/)

**Exploiting the Structure: Stochastic Gradient Methods Using Raw Clusters**

- arxiv: [http://arxiv.org/abs/1602.02151](http://arxiv.org/abs/1602.02151)

**Writing fast asynchronous SGD/AdaGrad with RcppParallel**

- blog: [http://gallery.rcpp.org/articles/rcpp-sgd/](http://gallery.rcpp.org/articles/rcpp-sgd/)

**Quick Explanations Of Optimization Methods**

- blog: [http://jxieeducation.com/2016-07-02/Quick-Explanations-of-Optimization-Methods/](http://jxieeducation.com/2016-07-02/Quick-Explanations-of-Optimization-Methods/)

**SGDR: Stochastic Gradient Descent with Restarts**

- arxiv: [http://arxiv.org/abs/1608.03983](http://arxiv.org/abs/1608.03983)
- github: [https://github.com/loshchil/SGDR](https://github.com/loshchil/SGDR)

**The zen of gradient descent**

- blog: [http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html](http://blog.mrtz.org/2013/09/07/the-zen-of-gradient-descent.html)

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

# Gradient Descent

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

# Backpropagation

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

# Image Data Augmentation

**DataAugmentation ver1.0: Image data augmentation tool for training of image recognition algorithm**

- github: [https://github.com/takmin/DataAugmentation](https://github.com/takmin/DataAugmentation)

**Caffe-Data-Augmentation: a branc caffe with feature of Data Augmentation using a configurable stochastic combination of 7 data augmentation techniques**

- github: [https://github.com/ShaharKatz/Caffe-Data-Augmentation](https://github.com/ShaharKatz/Caffe-Data-Augmentation)

**Image Augmentation for Deep Learning With Keras**

- blog: [http://machinelearningmastery.com/image-augmentation-deep-learning-keras/](http://machinelearningmastery.com/image-augmentation-deep-learning-keras/)

**What you need to know about data augmentation for machine learning**

- intro: keras Imagegenerator
- blog: [https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/](https://cartesianfaith.com/2016/10/06/what-you-need-to-know-about-data-augmentation-for-machine-learning/)

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

**Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations**

- arxiv: [http://arxiv.org/abs/1609.07061](http://arxiv.org/abs/1609.07061)

# Papers

**Understanding the difficulty of training deep feed forward neural networks**

- intro: Xavier initialization
- paper: [http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)

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

# Tools

**pastalog: Simple, realtime visualization of neural network training performance**

![](/assets/train-dnn/pastalog-main-big.gif)

- github: [https://github.com/rewonc/pastalog](https://github.com/rewonc/pastalog)

**torch-pastalog: A Torch interface for pastalog - simple, realtime visualization of neural network training performance**

- github: [https://github.com/Kaixhin/torch-pastalog](https://github.com/Kaixhin/torch-pastalog)

# Blogs

**Important nuances to train deep learning models**

[http://www.erogol.com/important-nuances-train-deep-learning-models/](http://www.erogol.com/important-nuances-train-deep-learning-models/)
