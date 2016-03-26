---
layout: post
category: deep_learning
title: Object Segmentation
date: 2015-10-09
---

**Deep Joint Task Learning for Generic Object Extraction(NIPS2014)**

![](http://vision.sysu.edu.cn/vision_sysu/wp-content/uploads/2013/05/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20141019095211.png)

- homepage: [http://vision.sysu.edu.cn/projects/deep-joint-task-learning/](http://vision.sysu.edu.cn/projects/deep-joint-task-learning/)
- paper: [http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf](http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf)
- code: [https://github.com/xiaolonw/nips14_loc_seg_testonly](https://github.com/xiaolonw/nips14_loc_seg_testonly)
- dataset: [http://objectextraction.github.io/](http://objectextraction.github.io/)

**Fully Convolutional Networks for Semantic Segmentation**

- keywords: deconvolutional layer, crop layer
- arxiv: [http://arxiv.org/abs/1411.4038](http://arxiv.org/abs/1411.4038)
- slides: [https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc](https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf)
- github: [https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)
- notes: [http://zhangliliang.com/2014/11/28/paper-note-fcn-segment/](http://zhangliliang.com/2014/11/28/paper-note-fcn-segment/)

**Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs("DeepLab")**

- intro: "adopted a more simplistic approach for maintaining resolution by removing
the stride in the layers of FullConvNet, wherever possible. 
Following this, the FullConvNet predicted output is modeled as 
a unary term for Conditional Random Field (CRF) constructed over 
the image grid at its original resolution. 
With labelling smoothness constraint enforced through pair-wise terms, 
the per-pixel classification task is modeled as a CRF inference problem."
- arXiv: [http://arxiv.org/abs/1412.7062](http://arxiv.org/abs/1412.7062)
- github: [https://bitbucket.org/deeplab/deeplab-public/](https://bitbucket.org/deeplab/deeplab-public/)

**Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation("DeepLab")**

- arXiv: [http://arxiv.org/abs/1502.02734](http://arxiv.org/abs/1502.02734)
- bitbucket: [https://bitbucket.org/deeplab/deeplab-public/](https://bitbucket.org/deeplab/deeplab-public/)

**Hypercolumns for object segmentation and fine-grained localization (CVPR 2015)**

- paper: [http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathCVPR2015.pdf](http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathCVPR2015.pdf)

**Conditional Random Fields as Recurrent Neural Networks(ICCV2015. Oxford/Stanford/Baidu)**

![](http://www.robots.ox.ac.uk/~szheng/Res_CRFRNN/CRFasRNN.jpg)

- intro: "proposed a better approach where the CRF
constructed on image is modeled as a Recurrent Neural Network (RNN). 
By modeling the CRF as an RNN, it can be integrated as a part of any Deep Convolutinal Net 
making the system efficient at both semantic feature extraction
and fine-grained structure prediction. 
This enables the end-to-end training of the entire FullConvNet + RNN system
using the stochastic gradient descent (SGD) algorithm to obtain fine pixel-level segmentation."
- arXiv: [http://arxiv.org/abs/1502.03240](http://arxiv.org/abs/1502.03240)
- homepage: [http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html](http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html)
- github: [https://github.com/torrvision/crfasrnn](https://github.com/torrvision/crfasrnn)
- demo: [http://www.robots.ox.ac.uk/~szheng/crfasrnndemo](http://www.robots.ox.ac.uk/~szheng/crfasrnndemo)

**Learning to Segment Object Candidates**

- arxiv: [http://arxiv.org/abs/1506.06204](http://arxiv.org/abs/1506.06204)

**Proposal-free Network for Instance-level Object Segmentation**

- paper: [http://arxiv.org/abs/1509.02636](http://arxiv.org/abs/1509.02636)

**Semantic Image Segmentation via Deep Parsing Network**

![](http://personal.ie.cuhk.edu.hk/~lz013/projects/dpn/intro.png)

- paper: [http://arxiv.org/abs/1509.02634](http://arxiv.org/abs/1509.02634)
- homepage: [http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html](http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html)

**Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation(NIPS 2015)**

![](http://cvlab.postech.ac.kr/research/decouplednet/images/overall.png)

- paper: [http://arxiv.org/abs/1506.04924](http://arxiv.org/abs/1506.04924)
- project[paper+code]: [http://cvlab.postech.ac.kr/research/decouplednet/](http://cvlab.postech.ac.kr/research/decouplednet/)
- github: [https://github.com/HyeonwooNoh/DecoupledNet](https://github.com/HyeonwooNoh/DecoupledNet)

**Learning Deconvolution Network for Semantic Segmentation**

- arXiv: [http://arxiv.org/abs/1505.04366](http://arxiv.org/abs/1505.04366)

**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling**

- arXiv: [http://arxiv.org/abs/1505.07293](http://arxiv.org/abs/1505.07293)
- github: [https://github.com/alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)

**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**

![](http://mi.eng.cam.ac.uk/projects/segnet/images/segnet.png)

- homepage: [http://mi.eng.cam.ac.uk/projects/segnet/](http://mi.eng.cam.ac.uk/projects/segnet/)
- arXiv: [http://arxiv.org/abs/1511.00561](http://arxiv.org/abs/1511.00561)
- github: [https://github.com/alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
- tutorial: [http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)

**SegNet: Pixel-Wise Semantic Labelling Using a Deep Networks**

- youtube: [https://www.youtube.com/watch?v=xfNYAly1iXo](https://www.youtube.com/watch?v=xfNYAly1iXo)
- video: [http://pan.baidu.com/s/1gdUzDlD](http://pan.baidu.com/s/1gdUzDlD)

**Recurrent Instance Segmentation**

![](http://romera-paredes.com/wp-content/uploads/2015/12/RIS.png)
	
- arXiv: [http://arxiv.org/abs/1511.08250](http://arxiv.org/abs/1511.08250)
- homepage: [http://romera-paredes.com/recurrent-instance-segmentation](http://romera-paredes.com/recurrent-instance-segmentation)

**Instance-aware Semantic Segmentation via Multi-task Network Cascades**

- intro: "1st-place winner of MS COCO 2015 segmentation competition"
- arxiv: [http://arxiv.org/abs/1512.04412](http://arxiv.org/abs/1512.04412)

**Semantic Object Parsing with Graph LSTM**

- arxiv: [http://arxiv.org/abs/1603.07063](http://arxiv.org/abs/1603.07063)