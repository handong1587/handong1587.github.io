---
layout: post
category: deep_learning
title: Accelerate Convolutional Neural Networks
date: 2015-10-09
---

# Papers

**High-Performance Neural Networks for Visual Object Classification**

- intro: "reduced network parameters by randomly removing connections before training"
- arxiv: [http://arxiv.org/abs/1102.0183](http://arxiv.org/abs/1102.0183)

**Predicting Parameters in Deep Learning**

- intro: "decomposed the weighting matrix into two low-rank matrices"
- arxiv: [http://arxiv.org/abs/1306.0543](http://arxiv.org/abs/1306.0543)

**Neurons vs Weights Pruning in Artificial Neural Networks**

- paper: [http://journals.ru.lv/index.php/ETR/article/view/166](http://journals.ru.lv/index.php/ETR/article/view/166)

**Exploiting Linear Structure Within Convolutional Networks for Efﬁcient Evaluation**

- intro: "presented a series of low-rank decomposition designs for convolutional kernels. 
singular value decomposition was adopted for the matrix factorization"
- paper: [http://papers.nips.cc/paper/5544-exploiting-linear-structure-within-convolutional-networks-for-efficient-evaluation.pdf](http://papers.nips.cc/paper/5544-exploiting-linear-structure-within-convolutional-networks-for-efficient-evaluation.pdf)

**Efficient and accurate approximations of nonlinear convolutional networks**

- intro: "considered the subsequent nonlinear units while learning the low-rank decomposition"
- arxiv: [http://arxiv.org/abs/1411.4229](http://arxiv.org/abs/1411.4229)

**Flattened Convolutional Neural Networks for Feedforward Acceleration(ICLR 2015)**

- arXiv: [http://arxiv.org/abs/1412.5474](http://arxiv.org/abs/1412.5474)
- github: [https://github.com/jhjin/flattened-cnn](https://github.com/jhjin/flattened-cnn)

**Compressing Deep Convolutional Networks using Vector Quantization**

- intro: "this paper showed that vector quantization had a clear advantage 
over matrix factorization methods in compressing fully-connected layers."
- arxiv: [http://arxiv.org/abs/1412.6115](http://arxiv.org/abs/1412.6115)

**Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition**

- intro: "a low-rank CPdecomposition was adopted to 
transform a convolutional layer into multiple layers of lower complexity"
- arxiv: [http://arxiv.org/abs/1412.6553](http://arxiv.org/abs/1412.6553)

**Deep Fried Convnets**

- intro: "fully-connected layers were replaced by a single “Fastfood” layer for end-to-end training with convolutional layers"
- arxiv: [http://arxiv.org/abs/1412.7149](http://arxiv.org/abs/1412.7149)

**Distilling the Knowledge in a Neural Network (by Geoffrey Hinton, Oriol Vinyals, Jeff Dean)**

- intro: "trained a distilled model to mimic the response of a larger and well-trained network"
- arxiv: [http://arxiv.org/abs/1503.02531](http://arxiv.org/abs/1503.02531)

**Compressing Neural Networks with the Hashing Trick**

- intro: "randomly grouped connection weights into hash buckets, and then fine-tuned network parameters with back-propagation"
- arxiv: [http://arxiv.org/abs/1504.04788](http://arxiv.org/abs/1504.04788)

**Accelerating Very Deep Convolutional Networks for Classification and Detection**

- intro: "considered the subsequent nonlinear units while learning the low-rank decomposition"
- arxiv: [http://arxiv.org/abs/1505.06798](http://arxiv.org/abs/1505.06798)

**Fast ConvNets Using Group-wise Brain Damage**

- intro: "applied group-wise pruning to the convolutional tensor 
to decompose it into the multiplications of thinned dense matrices"
- arxiv: [http://arxiv.org/abs/1506.02515](http://arxiv.org/abs/1506.02515)

**Learning both Weights and Connections for Efficient Neural Networks**

- arxiv: [http://arxiv.org/abs/1506.02626](http://arxiv.org/abs/1506.02626)

**Data-free parameter pruning for Deep Neural Networks**

- intro: "proposed to remove redundant neurons instead of network connections"
- arXiv: [http://arxiv.org/abs/1507.06149](http://arxiv.org/abs/1507.06149)

**Fast Algorithms for Convolutional Neural Networks**

- intro: "2.6x as fast as Caffe when comparing CPU implementations"
- arXiv: [http://arxiv.org/abs/1509.09308](http://arxiv.org/abs/1509.09308)
- discussion: [https://github.com/soumith/convnet-benchmarks/issues/59#issuecomment-150111895](https://github.com/soumith/convnet-benchmarks/issues/59#issuecomment-150111895)

**Tensorizing Neural Networks**

- arXiv: [http://arxiv.org/abs/1509.06569](http://arxiv.org/abs/1509.06569)

**A Deep Neural Network Compression Pipeline: Pruning, Quantization, Huffman Encoding(ICLR 2016)**

- intro: "reduced the size of AlexNet by 35x from 240MB to 6.9MB, the size of VGG16 by 49x from 552MB to 11.3MB, with no loss of accuracy"
- arXiv: [http://arxiv.org/abs/1510.00149](http://arxiv.org/abs/1510.00149)

**ZNN - A Fast and Scalable Algorithm for Training 3D Convolutional Networks on Multi-Core and Many-Core Shared Memory Machines**

- arXiv: [http://arxiv.org/abs/1510.06706](http://arxiv.org/abs/1510.06706)
- github: [https://github.com/seung-lab/znn-release](https://github.com/seung-lab/znn-release)

**Reducing the Training Time of Neural Networks by Partitioning**

- arXiv: [http://arxiv.org/abs/1511.02954](http://arxiv.org/abs/1511.02954)

**Convolutional neural networks with low-rank regularization**

- arxiv: [http://arxiv.org/abs/1511.06067](http://arxiv.org/abs/1511.06067)

**Quantized Convolutional Neural Networks for Mobile Devices (Q-CNN)**

- intro: "Extensive experiments on the ILSVRC-12 benchmark demonstrate 
4 ∼ 6× speed-up and 15 ∼ 20× compression with merely one percentage loss of classification accuracy"
- arxiv: [http://arxiv.org/abs/1512.06473](http://arxiv.org/abs/1512.06473)

**Convolutional Tables Ensemble: classification in microseconds**

- arxiv: [http://arxiv.org/abs/1602.04489](http://arxiv.org/abs/1602.04489)

**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <1MB model size**

- arxiv: [http://arxiv.org/abs/1602.07360](http://arxiv.org/abs/1602.07360)
- github: [https://github.com/DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet)
- note: [https://www.evernote.com/shard/s146/sh/108eea91-349b-48ba-b7eb-7ac8f548bee9/5171dc6b1088fba05a4e317f7f5d32a3](https://www.evernote.com/shard/s146/sh/108eea91-349b-48ba-b7eb-7ac8f548bee9/5171dc6b1088fba05a4e317f7f5d32a3)

**Convolutional Neural Networks using Logarithmic Data Representation**

- arxiv: [http://arxiv.org/abs/1603.01025](http://arxiv.org/abs/1603.01025)

**DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices**

- paper: [http://niclane.org/pubs/deepx_ipsn.pdf](http://niclane.org/pubs/deepx_ipsn.pdf)

# Codes

**Accelerate Convolutional Neural Networks**

- intro: "This tool aims to accelerate the test-time computation and decrease number of parameters of deep CNNs."
- github: [https://github.com/dmlc/mxnet/tree/master/tools/accnn](https://github.com/dmlc/mxnet/tree/master/tools/accnn)

**OptNet - reducing memory usage in torch neural networks**

- github: [https://github.com/fmassa/optimize-net](https://github.com/fmassa/optimize-net)

**NNPACK: Acceleration package for neural networks on multi-core CPUs**

![](https://camo.githubusercontent.com/376828536285f7a1a4f054aaae998e805023f489/68747470733a2f2f6d6172617479737a637a612e6769746875622e696f2f4e4e5041434b2f4e4e5041434b2e706e67)

- github: [https://github.com/Maratyszcza/NNPACK](https://github.com/Maratyszcza/NNPACK)
- comments(Yann LeCun): [https://www.facebook.com/yann.lecun/posts/10153459577707143](https://www.facebook.com/yann.lecun/posts/10153459577707143)