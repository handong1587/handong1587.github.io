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

**Rectified linear units improve restricted boltzmann machines (ReLU)**

- paper: [http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf](http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_NairH10.pdf)

**Rectifier Nonlinearities Improve Neural Network Acoustic Models (leaky-ReLU, aka LReLU)**

- paper: [http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf](http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf)

**Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification (PReLU)**

- keywords: PReLU, Caffe "msra" weights initilization
- arXiv: [http://arxiv.org/abs/1502.01852](http://arxiv.org/abs/1502.01852)

**Empirical Evaluation of Rectified Activations in Convolutional Network (ReLU/LReLU/PReLU/RReLU)**

- arXiv: [http://arxiv.org/abs/1505.00853](http://arxiv.org/abs/1505.00853)

**Deep Learning with S-shaped Rectified Linear Activation Units (SReLU)**

- arxiv: [http://arxiv.org/abs/1512.07030](http://arxiv.org/abs/1512.07030)

**Parametric Activation Pools greatly increase performance and consistency in ConvNets**

- blog: [http://blog.claymcleod.io/2016/02/06/Parametric-Activation-Pools-greatly-increase-performance-and-consistency-in-ConvNets/](http://blog.claymcleod.io/2016/02/06/Parametric-Activation-Pools-greatly-increase-performance-and-consistency-in-ConvNets/)

**Noisy Activation Functions**

- arxiv: [http://arxiv.org/abs/1603.00391](http://arxiv.org/abs/1603.00391)

# Weights Initialization

**An Explanation of Xavier Initialization**

- blog: [http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)

**Deep Neural Networks with Random Gaussian Weights: A Universal Classification Strategy?**

- arxiv: [http://arxiv.org/abs/1504.08291](http://arxiv.org/abs/1504.08291)

**All you need is a good init**

- arxiv: [http://arxiv.org/abs/1511.06422](http://arxiv.org/abs/1511.06422)
- github: [https://github.com/ducha-aiki/LSUVinit](https://github.com/ducha-aiki/LSUVinit)

**What are good initial weights in a neural network?**

- stackexchange: [http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network](http://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network)

**RandomOut: Using a convolutional gradient norm to win The Filter Lottery**

- arxiv: [http://arxiv.org/abs/1602.05931](http://arxiv.org/abs/1602.05931)

## Batch Normalization

**Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(ImageNet top-5 error: 4.82%)**

- arXiv: [http://arxiv.org/abs/1502.03167](http://arxiv.org/abs/1502.03167)
- blog: [https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/](https://standardfrancis.wordpress.com/2015/04/16/batch-normalization/)
- notes: [http://blog.csdn.net/happynear/article/details/44238541](http://blog.csdn.net/happynear/article/details/44238541)

**Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1602.07868](http://arxiv.org/abs/1602.07868)
- github(Lasagne): [https://github.com/TimSalimans/weight_norm](https://github.com/TimSalimans/weight_norm)
- notes: [http://www.erogol.com/my-notes-weight-normalization/](http://www.erogol.com/my-notes-weight-normalization/)

**Normalization Propagation: A Parametric Technique for Removing Internal Covariate Shift in Deep Networks**

- arxiv: [http://arxiv.org/abs/1603.01431](http://arxiv.org/abs/1603.01431)

# Loss Function

**The Loss Surfaces of Multilayer Networks**

- arxiv: [http://arxiv.org/abs/1412.0233](http://arxiv.org/abs/1412.0233)

# Optimization Methods

**On Optimization Methods for Deep Learning**

- paper: [http://www.icml-2011.org/papers/210_icmlpaper.pdf](http://www.icml-2011.org/papers/210_icmlpaper.pdf)

**On the importance of initialization and momentum in deep learning**

- paper: [http://jmlr.org/proceedings/papers/v28/sutskever13.pdf](http://jmlr.org/proceedings/papers/v28/sutskever13.pdf)

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

# Regularization

**DisturbLabel: Regularizing CNN on the Loss Layer [University of California & MSR] (2016)**

- intro: "an extremely simple algorithm which randomly replaces a part of labels as incorrect values in each iteration"
- paper: [http://research.microsoft.com/en-us/um/people/jingdw/pubs/cvpr16-disturblabel.pdf](http://research.microsoft.com/en-us/um/people/jingdw/pubs/cvpr16-disturblabel.pdf)

## Dropout

**Improving neural networks by preventing co-adaptation of feature detectors (Dropout)**

- arxiv: [http://arxiv.org/abs/1207.0580](http://arxiv.org/abs/1207.0580)

**Regularization of Neural Networks using DropConnect**

- homepage: [http://cs.nyu.edu/~wanli/dropc/](http://cs.nyu.edu/~wanli/dropc/)
- gitxiv: [http://gitxiv.com/posts/rJucpiQiDhQ7HkZoX/regularization-of-neural-networks-using-dropconnect](http://gitxiv.com/posts/rJucpiQiDhQ7HkZoX/regularization-of-neural-networks-using-dropconnect)
- github: [https://github.com/iassael/torch-dropconnect](https://github.com/iassael/torch-dropconnect)

**Regularizing neural networks with dropout and with DropConnect**

- blog: [http://fastml.com/regularizing-neural-networks-with-dropout-and-with-dropconnect/](http://fastml.com/regularizing-neural-networks-with-dropout-and-with-dropconnect/)

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

# Gradient Descent

**Fitting a model via closed-form equations vs. Gradient Descent vs Stochastic Gradient Descent vs Mini-Batch Learning. What is the difference?(Normal Equations vs. GD vs. SGD vs. MB-GD)**

[http://sebastianraschka.com/faq/docs/closed-form-vs-gd.html](http://sebastianraschka.com/faq/docs/closed-form-vs-gd.html)

**An Introduction to Gradient Descent in Python**

- blog: [http://tillbergmann.com/blog/articles/python-gradient-descent.html](http://tillbergmann.com/blog/articles/python-gradient-descent.html)

**A Variational Analysis of Stochastic Gradient Algorithms**

- arxiv: [http://arxiv.org/abs/1602.02666](http://arxiv.org/abs/1602.02666)

**The vanishing gradient problem: Oh no — an obstacle to deep learning!**

- blog: [https://medium.com/a-year-of-artificial-intelligence/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b#.50hu5vwa8](https://medium.com/a-year-of-artificial-intelligence/rohan-4-the-vanishing-gradient-problem-ec68f76ffb9b#.50hu5vwa8)

**Gradient Descent For Machine Learning**

[http://machinelearningmastery.com/gradient-descent-for-machine-learning/](http://machinelearningmastery.com/gradient-descent-for-machine-learning/)

# Accelerate Training

**Acceleration of Deep Neural Network Training with Resistive Cross-Point Devices**

- arxiv: [http://arxiv.org/abs/1603.07341](http://arxiv.org/abs/1603.07341)

# Image Data Augmentation

**DataAugmentation ver1.0: Image data augmentation tool for training of image recognition algorithm**

- github: [https://github.com/takmin/DataAugmentation](https://github.com/takmin/DataAugmentation)

**Caffe-Data-Augmentation: a branc caffe with feature of Data Augmentation using a configurable stochastic combination of 7 data augmentation techniques**

- github: [https://github.com/ShaharKatz/Caffe-Data-Augmentation](https://github.com/ShaharKatz/Caffe-Data-Augmentation)

# Papers

**Scalable and Sustainable Deep Learning via Randomized Hashing**

- arxiv: [http://arxiv.org/abs/1602.08194](http://arxiv.org/abs/1602.08194)

# Tools

**pastalog: Simple, realtime visualization of neural network training performance**

![](/assets/train-dnn/pastalog-main-big.gif)

- github: [https://github.com/rewonc/pastalog](https://github.com/rewonc/pastalog)