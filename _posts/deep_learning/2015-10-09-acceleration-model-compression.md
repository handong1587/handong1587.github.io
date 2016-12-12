---
layout: post
category: deep_learning
title: Acceleration and Model Compression
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

**cuDNN: Efficient Primitives for Deep Learning**

- arxiv: [https://arxiv.org/abs/1410.0759](https://arxiv.org/abs/1410.0759)
- download: [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)

**Efficient and accurate approximations of nonlinear convolutional networks**

- intro: "considered the subsequent nonlinear units while learning the low-rank decomposition"
- arxiv: [http://arxiv.org/abs/1411.4229](http://arxiv.org/abs/1411.4229)

**Convolutional Neural Networks at Constrained Time Cost**

- arxiv: [https://arxiv.org/abs/1412.1710](https://arxiv.org/abs/1412.1710)

**Flattened Convolutional Neural Networks for Feedforward Acceleration**

- intro: ICLR 2015
- arxiv: [http://arxiv.org/abs/1412.5474](http://arxiv.org/abs/1412.5474)
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

**Fast Convolutional Nets With fbfft: A GPU Performance Evaluation**

- intro: Facebook. ICLR 2015
- arxiv: [http://arxiv.org/abs/1412.7580](http://arxiv.org/abs/1412.7580)
- github: [http://facebook.github.io/fbcunn/fbcunn/](http://facebook.github.io/fbcunn/fbcunn/)

**Distilling the Knowledge in a Neural Network**

- author: Geoffrey Hinton, Oriol Vinyals, Jeff Dean
- intro: "trained a distilled model to mimic the response of a larger and well-trained network"
- comments: "Soft targets are a VERY good regulizer! Also trains much faster (soft targets enrich gradients)" -- Jeff Dean in CS231n talk
- arxiv: [http://arxiv.org/abs/1503.02531](http://arxiv.org/abs/1503.02531)
- blog: [http://fastml.com/geoff-hintons-dark-knowledge/](http://fastml.com/geoff-hintons-dark-knowledge/)
- notes: [https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/distilling-the-knowledge-in-a-nn.md](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/distilling-the-knowledge-in-a-nn.md)

**Caffe con Troll: Shallow Ideas to Speed Up Deep Learning**

- intro: a fully compatible end-to-end version of the popular framework Caffe with rebuilt internals
- arxiv: [http://arxiv.org/abs/1504.04343](http://arxiv.org/abs/1504.04343)

**Compressing Neural Networks with the Hashing Trick**

![](http://www.cse.wustl.edu/~wenlinchen/project/HashedNets/hashednets.png)

- intro: HashedNets. ICML 2015
- intro: "randomly grouped connection weights into hash buckets, and then fine-tuned network parameters with back-propagation"
- project page: [http://www.cse.wustl.edu/~wenlinchen/project/HashedNets/index.html](http://www.cse.wustl.edu/~wenlinchen/project/HashedNets/index.html)
- arxiv: [http://arxiv.org/abs/1504.04788](http://arxiv.org/abs/1504.04788)
- code: [http://www.cse.wustl.edu/~wenlinchen/project/HashedNets/HashedNets.zip](http://www.cse.wustl.edu/~wenlinchen/project/HashedNets/HashedNets.zip)

**PerforatedCNNs: Acceleration through Elimination of Redundant Convolutions**

- intro: NIPS 2016
- arxiv: [https://arxiv.org/abs/1504.08362](https://arxiv.org/abs/1504.08362)
- github: [https://github.com/mfigurnov/perforated-cnn-matconvnet](https://github.com/mfigurnov/perforated-cnn-matconvnet)
- github: [https://github.com/mfigurnov/perforated-cnn-caffe](https://github.com/mfigurnov/perforated-cnn-caffe)

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
- arxiv: [http://arxiv.org/abs/1507.06149](http://arxiv.org/abs/1507.06149)

**Fast Algorithms for Convolutional Neural Networks**

- intro: "2.6x as fast as Caffe when comparing CPU implementations"
- arxiv: [http://arxiv.org/abs/1509.09308](http://arxiv.org/abs/1509.09308)
- github: [https://github.com/andravin/wincnn](https://github.com/andravin/wincnn)
- discussion: [https://github.com/soumith/convnet-benchmarks/issues/59#issuecomment-150111895](https://github.com/soumith/convnet-benchmarks/issues/59#issuecomment-150111895)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/3nocg5/fast_algorithms_for_convolutional_neural_networks/?](https://www.reddit.com/r/MachineLearning/comments/3nocg5/fast_algorithms_for_convolutional_neural_networks/?)

**Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding**

- intro: ICLR 2016 Best Paper
- intro: "reduced the size of AlexNet by 35x from 240MB to 6.9MB, the size of VGG16 by 49x from 552MB to 11.3MB, with no loss of accuracy"
- arxiv: [http://arxiv.org/abs/1510.00149](http://arxiv.org/abs/1510.00149)
- video: [http://videolectures.net/iclr2016_han_deep_compression/](http://videolectures.net/iclr2016_han_deep_compression/)

**ZNN - A Fast and Scalable Algorithm for Training 3D Convolutional Networks on Multi-Core and Many-Core Shared Memory Machines**

- arxiv: [http://arxiv.org/abs/1510.06706](http://arxiv.org/abs/1510.06706)
- github: [https://github.com/seung-lab/znn-release](https://github.com/seung-lab/znn-release)

**Reducing the Training Time of Neural Networks by Partitioning**

- arxiv: [http://arxiv.org/abs/1511.02954](http://arxiv.org/abs/1511.02954)

**Convolutional neural networks with low-rank regularization**

- arxiv: [http://arxiv.org/abs/1511.06067](http://arxiv.org/abs/1511.06067)
- github: [https://github.com/chengtaipu/lowrankcnn](https://github.com/chengtaipu/lowrankcnn)

**CNNdroid: Open Source Library for GPU-Accelerated Execution of Trained Deep Convolutional Neural Networks on Android**

- arxiv: [https://arxiv.org/abs/1511.07376](https://arxiv.org/abs/1511.07376)
- paper: [http://dl.acm.org/authorize.cfm?key=N14731](http://dl.acm.org/authorize.cfm?key=N14731)
- slides: [http://sharif.edu/~matin/pub/2016_mm_slides.pdf](http://sharif.edu/~matin/pub/2016_mm_slides.pdf)
- github: [https://github.com/ENCP/CNNdroid](https://github.com/ENCP/CNNdroid)

**Quantized Convolutional Neural Networks for Mobile Devices**

- intro: Q-CNN
- intro: "Extensive experiments on the ILSVRC-12 benchmark demonstrate 
4 ∼ 6× speed-up and 15 ∼ 20× compression with merely one percentage loss of classification accuracy"
- arxiv: [http://arxiv.org/abs/1512.06473](http://arxiv.org/abs/1512.06473)
- github: [https://github.com/jiaxiang-wu/quantized-cnn](https://github.com/jiaxiang-wu/quantized-cnn)

**EIE: Efficient Inference Engine on Compressed Deep Neural Network**

- intro: ISCA 2016
- arxiv: [http://arxiv.org/abs/1602.01528](http://arxiv.org/abs/1602.01528)
- slides: [http://on-demand.gputechconf.com/gtc/2016/presentation/s6561-song-han-deep-compression.pdf](http://on-demand.gputechconf.com/gtc/2016/presentation/s6561-song-han-deep-compression.pdf)
- slides: [http://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf](http://web.stanford.edu/class/ee380/Abstracts/160106-slides.pdf)

**Convolutional Tables Ensemble: classification in microseconds**

- arxiv: [http://arxiv.org/abs/1602.04489](http://arxiv.org/abs/1602.04489)

**SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size**

- intro: DeepScale & UC Berkeley
- arxiv: [http://arxiv.org/abs/1602.07360](http://arxiv.org/abs/1602.07360)
- github: [https://github.com/DeepScale/SqueezeNet](https://github.com/DeepScale/SqueezeNet)
- homepage: [http://songhan.github.io/SqueezeNet-Deep-Compression/](http://songhan.github.io/SqueezeNet-Deep-Compression/)
- github: [https://github.com/songhan/SqueezeNet-Deep-Compression](https://github.com/songhan/SqueezeNet-Deep-Compression)
- note: [https://www.evernote.com/shard/s146/sh/108eea91-349b-48ba-b7eb-7ac8f548bee9/5171dc6b1088fba05a4e317f7f5d32a3](https://www.evernote.com/shard/s146/sh/108eea91-349b-48ba-b7eb-7ac8f548bee9/5171dc6b1088fba05a4e317f7f5d32a3)
- github(Keras): [https://github.com/DT42/squeezenet_demo](https://github.com/DT42/squeezenet_demo)

**Lab41 Reading Group: SqueezeNet**

[https://medium.com/m/global-identity?redirectUrl=https://gab41.lab41.org/lab41-reading-group-squeezenet-9b9d1d754c75](https://medium.com/m/global-identity?redirectUrl=https://gab41.lab41.org/lab41-reading-group-squeezenet-9b9d1d754c75)

**Convolutional Neural Networks using Logarithmic Data Representation**

- arxiv: [http://arxiv.org/abs/1603.01025](http://arxiv.org/abs/1603.01025)

**XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1603.05279](http://arxiv.org/abs/1603.05279)
- github(Torch): [https://github.com/mrastegari/XNOR-Net](https://github.com/mrastegari/XNOR-Net)

**DeepX: A Software Accelerator for Low-Power Deep Learning Inference on Mobile Devices**

- paper: [http://niclane.org/pubs/deepx_ipsn.pdf](http://niclane.org/pubs/deepx_ipsn.pdf)

**Hardware-oriented Approximation of Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1604.03168](http://arxiv.org/abs/1604.03168)
- homepage: [http://ristretto.lepsucd.com/](http://ristretto.lepsucd.com/)
- github("Ristretto: Caffe-based approximation of convolutional neural networks"): [https://github.com/pmgysel/caffe](https://github.com/pmgysel/caffe)

**Deep Neural Networks Under Stress**

- intro: ICIP 2016
- arxiv: [http://arxiv.org/abs/1605.03498](http://arxiv.org/abs/1605.03498)
- github: [https://github.com/MicaelCarvalho/DNNsUnderStress](https://github.com/MicaelCarvalho/DNNsUnderStress)

**ASP Vision: Optically Computing the First Layer of Convolutional Neural Networks using Angle Sensitive Pixels**

- arxiv: [http://arxiv.org/abs/1605.03621](http://arxiv.org/abs/1605.03621)

**Deep Roots: Improving CNN Efficiency with Hierarchical Filter Groups**

- intro: "for ResNet 50, our model has 40% fewer parameters, 45% fewer floating point operations, and is 31% (12%) faster on a CPU (GPU).
For the deeper ResNet 200 our model has 25% fewer floating point operations and 44% fewer parameters, 
while maintaining state-of-the-art accuracy. For GoogLeNet, our model has 7% fewer parameters and is 21% (16%) faster on a CPU (GPU)."
- arxiv: [https://arxiv.org/abs/1605.06489](https://arxiv.org/abs/1605.06489)

**Functional Hashing for Compressing Neural Networks**

- intro: FunHashNN
- arxiv: [http://arxiv.org/abs/1605.06560](http://arxiv.org/abs/1605.06560)

**Ristretto: Hardware-Oriented Approximation of Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1605.06402](http://arxiv.org/abs/1605.06402)

**YodaNN: An Ultra-Low Power Convolutional Neural Network Accelerator Based on Binary Weights**

- arxiv: [https://arxiv.org/abs/1606.05487](https://arxiv.org/abs/1606.05487)

**Learning Structured Sparsity in Deep Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.03665](http://arxiv.org/abs/1608.03665)

**Dynamic Network Surgery for Efficient DNNs**

- intro: compress the number of parameters in LeNet-5 and AlexNet by a factor of 108× and 17.7× respectively
- arxiv: [http://arxiv.org/abs/1608.04493](http://arxiv.org/abs/1608.04493)

**Scalable Compression of Deep Neural Networks**

- intro: ACM Multimedia 2016
- arxiv: [http://arxiv.org/abs/1608.07365](http://arxiv.org/abs/1608.07365)

**Pruning Filters for Efficient ConvNets**

- arxiv: [http://arxiv.org/abs/1608.08710](http://arxiv.org/abs/1608.08710)

**Accelerating Deep Convolutional Networks using low-precision and sparsity**

- intro: Intel Labs
- arxiv: [https://arxiv.org/abs/1610.00324](https://arxiv.org/abs/1610.00324)

**Fixed-point Factorized Networks**

- arxiv: [https://arxiv.org/abs/1611.01972](https://arxiv.org/abs/1611.01972)

**Ultimate tensorization: compressing convolutional and FC layers alike**

- intro: NIPS 2016 workshop: Learning with Tensors: Why Now and How?
- arxiv: [https://arxiv.org/abs/1611.03214](https://arxiv.org/abs/1611.03214)
- github: [https://github.com/timgaripov/TensorNet-TF](https://github.com/timgaripov/TensorNet-TF)

**Designing Energy-Efficient Convolutional Neural Networks using Energy-Aware Pruning**

- intro: "the energy consumption of AlexNet and GoogLeNet are reduced by 3.7x and 1.6x, respectively, with less than 1% top-5 accuracy loss"
- arxiv: [https://arxiv.org/abs/1611.05128](https://arxiv.org/abs/1611.05128)

**Net-Trim: A Layer-wise Convex Pruning of Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1611.05162](https://arxiv.org/abs/1611.05162)

**LCNN: Lookup-based Convolutional Neural Network**

- intro: "Our fastest LCNN offers 37.6x speed up over AlexNet while maintaining 44.3% top-1 accuracy."
- arxiv: [https://arxiv.org/abs/1611.06473](https://arxiv.org/abs/1611.06473)

**Deep Tensor Convolution on Multicores**

- intro: present the first practical CPU implementation of tensor convolution optimized for deep networks of small kernels
- arxiv: [https://arxiv.org/abs/1611.06565](https://arxiv.org/abs/1611.06565)

**Training Sparse Neural Networks**

- arxiv: [https://arxiv.org/abs/1611.06694](https://arxiv.org/abs/1611.06694)

**FINN: A Framework for Fast, Scalable Binarized Neural Network Inference**

- intro: Xilinx Research Labs & Norwegian University of Science and Technology & University of Sydney
- keywords: FPGA
- paper: [http://www.idi.ntnu.no/~yamanu/2017-fpga-finn-preprint.pdf](http://www.idi.ntnu.no/~yamanu/2017-fpga-finn-preprint.pdf)

**Deep Learning with INT8 Optimization on Xilinx Devices**

- intro: "Xilinx's integrated DSP architecture can achieve 1.75X solution-level performance 
at INT8 deep learning operations than other FPGA DSP architectures"
- paper: [https://www.xilinx.com/support/documentation/white_papers/wp486-deep-learning-int8.pdf](https://www.xilinx.com/support/documentation/white_papers/wp486-deep-learning-int8.pdf)

**Parameter Compression of Recurrent Neural Networks and Degredation of Short-term Memory**

- arxiv: [https://arxiv.org/abs/1612.00891](https://arxiv.org/abs/1612.00891)

# Projects

**Accelerate Convolutional Neural Networks**

- intro: "This tool aims to accelerate the test-time computation and decrease number of parameters of deep CNNs."
- github: [https://github.com/dmlc/mxnet/tree/master/tools/accnn](https://github.com/dmlc/mxnet/tree/master/tools/accnn)

## OptNet

**OptNet - reducing memory usage in torch neural networks**

- github: [https://github.com/fmassa/optimize-net](https://github.com/fmassa/optimize-net)

**NNPACK: Acceleration package for neural networks on multi-core CPUs**

![](https://camo.githubusercontent.com/376828536285f7a1a4f054aaae998e805023f489/68747470733a2f2f6d6172617479737a637a612e6769746875622e696f2f4e4e5041434b2f4e4e5041434b2e706e67)

- github: [https://github.com/Maratyszcza/NNPACK](https://github.com/Maratyszcza/NNPACK)
- comments(Yann LeCun): [https://www.facebook.com/yann.lecun/posts/10153459577707143](https://www.facebook.com/yann.lecun/posts/10153459577707143)

**Deep Compression on AlexNet**

- github: [https://github.com/songhan/Deep-Compression-AlexNet](https://github.com/songhan/Deep-Compression-AlexNet)

**Tiny Darknet**

- github: [http://pjreddie.com/darknet/tiny-darknet/](http://pjreddie.com/darknet/tiny-darknet/)

**CACU: Calculate deep convolution neurAl network on Cell Unit**

- github: [https://github.com/luhaofang/CACU](https://github.com/luhaofang/CACU)

# Blogs

**Neural Networks Are Impressively Good At Compression**

[https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/](https://probablydance.com/2016/04/30/neural-networks-are-impressively-good-at-compression/)

**“Mobile friendly” deep convolutional neural networks**

- part 1: [https://medium.com/@sidd_reddy/mobile-friendly-deep-convolutional-neural-networks-part-1-331120ad40f9#.uy64ladvz](https://medium.com/@sidd_reddy/mobile-friendly-deep-convolutional-neural-networks-part-1-331120ad40f9#.uy64ladvz)
- part 2: [https://medium.com/@sidd_reddy/mobile-friendly-deep-convolutional-neural-networks-part-2-making-deep-nets-shallow-701b2fbd3ca9#.u58fkuak3](https://medium.com/@sidd_reddy/mobile-friendly-deep-convolutional-neural-networks-part-2-making-deep-nets-shallow-701b2fbd3ca9#.u58fkuak3)

**Lab41 Reading Group: Deep Compression**

- blog: [https://gab41.lab41.org/lab41-reading-group-deep-compression-9c36064fb209#.hbqzn8wfu](https://gab41.lab41.org/lab41-reading-group-deep-compression-9c36064fb209#.hbqzn8wfu)

**Accelerating Machine Learning**

![](http://www.linleygroup.com/mpr/h/2016/11561/U26_F4v2.png)

- blog: [http://www.linleygroup.com/mpr/article.php?id=11561](http://www.linleygroup.com/mpr/article.php?id=11561)

**Compressing and regularizing deep neural networks**

[https://www.oreilly.com/ideas/compressing-and-regularizing-deep-neural-networks](https://www.oreilly.com/ideas/compressing-and-regularizing-deep-neural-networks)

# Videos

**Deep compression and EIE: Deep learning model compression, design space exploration and hardware acceleration**

- youtube: [https://www.youtube.com/watch?v=baZOmGSSUAg](https://www.youtube.com/watch?v=baZOmGSSUAg)

**Deep Compression, DSD Training and EIE: Deep Neural Network Model Compression, Regularization and Hardware Acceleration**

[http://research.microsoft.com/apps/video/default.aspx?id=266664](http://research.microsoft.com/apps/video/default.aspx?id=266664)

**Tailoring Convolutional Neural Networks for Low-Cost, Low-Power Implementation**

- intro: tutorial at the May 2015 Embedded Vision Summit
- youtube: [https://www.youtube.com/watch?v=xACJBACStaU](https://www.youtube.com/watch?v=xACJBACStaU)
