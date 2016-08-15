---
layout: post
category: deep_learning
title: Segmentation
date: 2015-10-09
---

# Papers

**Deep Joint Task Learning for Generic Object Extraction (NIPS2014)**

![](http://vision.sysu.edu.cn/vision_sysu/wp-content/uploads/2013/05/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20141019095211.png)

- homepage: [http://vision.sysu.edu.cn/projects/deep-joint-task-learning/](http://vision.sysu.edu.cn/projects/deep-joint-task-learning/)
- paper: [http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf](http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf)
- code: [https://github.com/xiaolonw/nips14_loc_seg_testonly](https://github.com/xiaolonw/nips14_loc_seg_testonly)
- dataset: [http://objectextraction.github.io/](http://objectextraction.github.io/)

## U-Net

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

![](https://raw.githubusercontent.com/orobix/retina-unet/master/test/test_Original_GroundTruth_Prediction_3.png)

- project page: [http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- arxiv: [http://arxiv.org/abs/1505.04597](http://arxiv.org/abs/1505.04597)
- code+data: [http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz)
- github: [https://github.com/orobix/retina-unet](https://github.com/orobix/retina-unet)
- notes: [http://zongwei.leanote.com/post/Pa](http://zongwei.leanote.com/post/Pa)

**Segmentation from Natural Language Expressions**

![](https://camo.githubusercontent.com/b3ad4ad6d83ceb6bf6b8f8346bc545ac6ae1fba1/687474703a2f2f7777772e656563732e6265726b656c65792e6564752f253745726f6e6768616e672f70726f6a656374732f746578745f6f626a7365672f746578745f6f626a7365675f64656d6f2e6a7067)

- homepage: [http://ronghanghu.com/text_objseg/](http://ronghanghu.com/text_objseg/)
- arxiv: [http://arxiv.org/abs/1603.06180](http://arxiv.org/abs/1603.06180)
- github: [https://github.com/ronghanghu/text_objseg](https://github.com/ronghanghu/text_objseg)

**Semantic Object Parsing with Graph LSTM**

- arxiv: [http://arxiv.org/abs/1603.07063](http://arxiv.org/abs/1603.07063)

# Instance Segmentation

**Simultaneous Detection and Segmentation (ECCV 2014)**

- author: Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik
- arxiv: [http://arxiv.org/abs/1407.1808](http://arxiv.org/abs/1407.1808)
- github(Matlab): [https://github.com/bharath272/sds_eccv2014](https://github.com/bharath272/sds_eccv2014)

**Proposal-free Network for Instance-level Object Segmentation**

- paper: [http://arxiv.org/abs/1509.02636](http://arxiv.org/abs/1509.02636)

**Hypercolumns for object segmentation and fine-grained localization (CVPR 2015)**

- paper: [http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathCVPR2015.pdf](http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathCVPR2015.pdf)
- github("SDS using hypercolumns"): [https://github.com/bharath272/sds](https://github.com/bharath272/sds)

**Learning to decompose for object detection and instance segmentation (ICLR 2016 Workshop)**

- intro: CNN / RNN, MNIST, KITTI 
- arxiv: [http://arxiv.org/abs/1511.06449](http://arxiv.org/abs/1511.06449)

**Recurrent Instance Segmentation**

![](http://romera-paredes.com/wp-content/uploads/2015/12/RIS.png)
	
- arxiv: [http://arxiv.org/abs/1511.08250](http://arxiv.org/abs/1511.08250)
- homepage: [http://romera-paredes.com/recurrent-instance-segmentation](http://romera-paredes.com/recurrent-instance-segmentation)

**Instance-aware Semantic Segmentation via Multi-task Network Cascades**

![](https://raw.githubusercontent.com/daijifeng001/MNC/master/data/readme_img/example.png)

- intro: "1st-place winner of MS COCO 2015 segmentation competition"
- keywords: RoI warping
- arxiv: [http://arxiv.org/abs/1512.04412](http://arxiv.org/abs/1512.04412)
- github: [https://github.com/daijifeng001/MNC](https://github.com/daijifeng001/MNC)

**Learning to Refine Object Segments**

- intro: Facebook AI Research (FAIR)
- arxiv: [http://arxiv.org/abs/1603.08695](http://arxiv.org/abs/1603.08695)

**Bridging Category-level and Instance-level Semantic Image Segmentation**

- arxiv: [http://arxiv.org/abs/1605.06885](http://arxiv.org/abs/1605.06885)

## DeepCut

**DeepCut: Object Segmentation from Bounding Box Annotations using Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1605.07866](http://arxiv.org/abs/1605.07866)

**End-to-End Instance Segmentation and Counting with Recurrent Attention**

- arxiv: [http://arxiv.org/abs/1605.09410](http://arxiv.org/abs/1605.09410)

# Semantic Segmentation

**Fully Convolutional Networks for Semantic Segmentation**

- keywords: deconvolutional layer, crop layer
- intro: An interesting idea in this work is that a simple interpolation filter is employed for deconvolution and 
only the CNN part of the network is fine-tuned to learn deconvolution indirectly.
- arxiv: [http://arxiv.org/abs/1411.4038](http://arxiv.org/abs/1411.4038)
- arxiv(PAMI 2016): [http://arxiv.org/abs/1605.06211](http://arxiv.org/abs/1605.06211)
- slides: [https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc](https://docs.google.com/presentation/d/1VeWFMpZ8XN7OC3URZP4WdXvOGYckoFWGVN7hApoXVnc)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-pixels.pdf)
- github: [https://github.com/shelhamer/fcn.berkeleyvision.org](https://github.com/shelhamer/fcn.berkeleyvision.org)
- github: [https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn](https://github.com/BVLC/caffe/wiki/Model-Zoo#fcn)
- github: [https://github.com/MarvinTeichmann/tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn)
- notes: [http://zhangliliang.com/2014/11/28/paper-note-fcn-segment/](http://zhangliliang.com/2014/11/28/paper-note-fcn-segment/)

**From Image-level to Pixel-level Labeling with Convolutional Networks (CVPR 2015)**

- intro: "Weakly Supervised Semantic Segmentation with Convolutional Networks"
- intro: performs semantic segmentation based only on image-level annotations in a multiple instance learning framework
- arxiv: [http://arxiv.org/abs/1411.6228](http://arxiv.org/abs/1411.6228)
- paper: [http://ronan.collobert.com/pub/matos/2015_semisupsemseg_cvpr.pdf](http://ronan.collobert.com/pub/matos/2015_semisupsemseg_cvpr.pdf)

## DeepLab

**Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs (DeepLab)**

- intro: "adopted a more simplistic approach for maintaining resolution by removing
the stride in the layers of FullConvNet, wherever possible. 
Following this, the FullConvNet predicted output is modeled as 
a unary term for Conditional Random Field (CRF) constructed over 
the image grid at its original resolution. 
With labelling smoothness constraint enforced through pair-wise terms, 
the per-pixel classification task is modeled as a CRF inference problem."
- arxiv: [http://arxiv.org/abs/1412.7062](http://arxiv.org/abs/1412.7062)
- github: [https://bitbucket.org/deeplab/deeplab-public/](https://bitbucket.org/deeplab/deeplab-public/)

**Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation (DeepLab)**

- arxiv: [http://arxiv.org/abs/1502.02734](http://arxiv.org/abs/1502.02734)
- bitbucket: [https://bitbucket.org/deeplab/deeplab-public/](https://bitbucket.org/deeplab/deeplab-public/)

**Conditional Random Fields as Recurrent Neural Networks (ICCV2015. Oxford / Stanford / Baidu)**

![](http://www.robots.ox.ac.uk/~szheng/Res_CRFRNN/CRFasRNN.jpg)

- intro: "proposed a better approach where the CRF
constructed on image is modeled as a Recurrent Neural Network (RNN). 
By modeling the CRF as an RNN, it can be integrated as a part of any Deep Convolutinal Net 
making the system efficient at both semantic feature extraction
and fine-grained structure prediction. 
This enables the end-to-end training of the entire FullConvNet + RNN system
using the stochastic gradient descent (SGD) algorithm to obtain fine pixel-level segmentation."
- arxiv: [http://arxiv.org/abs/1502.03240](http://arxiv.org/abs/1502.03240)
- homepage: [http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html](http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html)
- github: [https://github.com/torrvision/crfasrnn](https://github.com/torrvision/crfasrnn)
- demo: [http://www.robots.ox.ac.uk/~szheng/crfasrnndemo](http://www.robots.ox.ac.uk/~szheng/crfasrnndemo)
- github: [https://github.com/martinkersner/train-CRF-RNN](https://github.com/martinkersner/train-CRF-RNN)

## BoxSup

**BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation**

- arxiv: [http://arxiv.org/abs/1503.01640](http://arxiv.org/abs/1503.01640)

## DeconvNet

**Learning Deconvolution Network for Semantic Segmentation (DeconvNet. ICCV 2015)**

![](http://cvlab.postech.ac.kr/research/deconvnet/images/overall.png)

- intro: two-stage training: train the network with easy examples first and 
fine-tune the trained network with more challenging examples later
- project page: [http://cvlab.postech.ac.kr/research/deconvnet/](http://cvlab.postech.ac.kr/research/deconvnet/)
- arxiv: [http://arxiv.org/abs/1505.04366](http://arxiv.org/abs/1505.04366)
- slides: [http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w06-deconvnet.pdf](http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w06-deconvnet.pdf)
- gitxiv: [http://gitxiv.com/posts/9tpJKNTYksN5eWcHz/learning-deconvolution-network-for-semantic-segmentation](http://gitxiv.com/posts/9tpJKNTYksN5eWcHz/learning-deconvolution-network-for-semantic-segmentation)
- github: [https://github.com/HyeonwooNoh/DeconvNet](https://github.com/HyeonwooNoh/DeconvNet)
- github: [https://github.com/HyeonwooNoh/caffe](https://github.com/HyeonwooNoh/caffe)

## SegNet

**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Robust Semantic Pixel-Wise Labelling**

- arxiv: [http://arxiv.org/abs/1505.07293](http://arxiv.org/abs/1505.07293)
- github: [https://github.com/alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)

**SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation**

![](http://mi.eng.cam.ac.uk/projects/segnet/images/segnet.png)

- homepage: [http://mi.eng.cam.ac.uk/projects/segnet/](http://mi.eng.cam.ac.uk/projects/segnet/)
- arxiv: [http://arxiv.org/abs/1511.00561](http://arxiv.org/abs/1511.00561)
- github: [https://github.com/alexgkendall/caffe-segnet](https://github.com/alexgkendall/caffe-segnet)
- tutorial: [http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)

**SegNet: Pixel-Wise Semantic Labelling Using a Deep Networks**

- youtube: [https://www.youtube.com/watch?v=xfNYAly1iXo](https://www.youtube.com/watch?v=xfNYAly1iXo)
- mirror: [http://pan.baidu.com/s/1gdUzDlD](http://pan.baidu.com/s/1gdUzDlD)

**Getting Started with SegNet**

- blog: [http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html](http://mi.eng.cam.ac.uk/projects/segnet/tutorial.html)
- github: [https://github.com/alexgkendall/SegNet-Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)

## DecoupledNet

**Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation (NIPS 2015)**

![](http://cvlab.postech.ac.kr/research/decouplednet/images/overall.png)

- project(paper+code): [http://cvlab.postech.ac.kr/research/decouplednet/](http://cvlab.postech.ac.kr/research/decouplednet/)
- arxiv: [http://arxiv.org/abs/1506.04924](http://arxiv.org/abs/1506.04924)
- github: [https://github.com/HyeonwooNoh/DecoupledNet](https://github.com/HyeonwooNoh/DecoupledNet)

**Semantic Image Segmentation via Deep Parsing Network**

![](http://personal.ie.cuhk.edu.hk/~lz013/projects/dpn/intro.png)

- homepage: [http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html](http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html)
- paper: [http://arxiv.org/abs/1509.02634](http://arxiv.org/abs/1509.02634)

**Multi-Scale Context Aggregation by Dilated Convolutions**

![](http://vladlen.info/wp-content/uploads/2016/02/dilated-convolutions1-894x263.png)

- homepage: [http://vladlen.info/publications/multi-scale-context-aggregation-by-dilated-convolutions/](http://vladlen.info/publications/multi-scale-context-aggregation-by-dilated-convolutions/)
- arxiv: [http://arxiv.org/abs/1511.07122](http://arxiv.org/abs/1511.07122)
- github: [https://github.com/fyu/dilation](https://github.com/fyu/dilation)
- notes: [http://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/](http://www.inference.vc/dilated-convolutions-and-kronecker-factorisation/)

## TransferNet

**Learning Transferrable Knowledge for Semantic Segmentation with Deep Convolutional Neural Network**

![](http://cvlab.postech.ac.kr/research/transfernet/images/architecture.png)

- project page: [http://cvlab.postech.ac.kr/research/transfernet/](http://cvlab.postech.ac.kr/research/transfernet/)
- arxiv: [http://arxiv.org/abs/1512.07928](http://arxiv.org/abs/1512.07928)
- github: [https://github.com/maga33/TransferNet](https://github.com/maga33/TransferNet)

**Combining the Best of Convolutional Layers and Recurrent Layers: A Hybrid Network for Semantic Segmentation**

- arxiv: [http://arxiv.org/abs/1603.04871](http://arxiv.org/abs/1603.04871)

## ScribbleSup

**ScribbleSup: Scribble-Supervised Convolutional Networks for Semantic Segmentation**

![](http://research.microsoft.com/en-us/um/people/jifdai/downloads/scribble_sup/Figures/scribble_example_12.jpg)

- project page: [http://research.microsoft.com/en-us/um/people/jifdai/downloads/scribble_sup/](http://research.microsoft.com/en-us/um/people/jifdai/downloads/scribble_sup/)
- arxiv: [http://arxiv.org/abs/1604.05144](http://arxiv.org/abs/1604.05144)

**Natural Scene Image Segmentation Based on Multi-Layer Feature Extraction**

- arxiv: [http://arxiv.org/abs/1605.07586](http://arxiv.org/abs/1605.07586)

**Convolutional Random Walk Networks for Semantic Image Segmentation**

- arxiv: [http://arxiv.org/abs/1605.07681](http://arxiv.org/abs/1605.07681)

## ENet

**ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation**

- arxiv: [http://arxiv.org/abs/1606.02147](http://arxiv.org/abs/1606.02147)
- github: [https://github.com/e-lab/ENet-training](https://github.com/e-lab/ENet-training)
- blog: [http://culurciello.github.io/tech/2016/06/20/training-enet.html](http://culurciello.github.io/tech/2016/06/20/training-enet.html)

**Fully Convolutional Networks for Dense Semantic Labelling of High-Resolution Aerial Imagery**

- arxiv: [http://arxiv.org/abs/1606.02585](http://arxiv.org/abs/1606.02585)

**Deep Learning Markov Random Field for Semantic Segmentation**

- arxiv: [http://arxiv.org/abs/1606.07230](http://arxiv.org/abs/1606.07230)

**Region-based semantic segmentation with end-to-end training (ECCV 2016 camera-ready)**

- arxiv: [http://arxiv.org/abs/1607.07671](http://arxiv.org/abs/1607.07671)

# Scene Labeling/Parsing

**Indoor Semantic Segmentation using depth information**

- arxiv: [http://arxiv.org/abs/1301.3572](http://arxiv.org/abs/1301.3572)

**Recurrent Convolutional Neural Networks for Scene Parsing**

- arxiv: [http://arxiv.org/abs/1306.2795](http://arxiv.org/abs/1306.2795)
- slides: [http://people.ee.duke.edu/~lcarin/Yizhe8.14.2015.pdf](http://people.ee.duke.edu/~lcarin/Yizhe8.14.2015.pdf)

**Learning hierarchical features for scene labeling**

- intro: "Their approach comprised of densely computing multi-scale CNN features
for each pixel and aggregating them over image regions upon which they are classified.
However, their methodstill required the post-processing step of generating over-segmented regions, 
like superpixels, for obtaining the final segmentation result. 
Additionally, the CNNs used for multi-scale feature learning were 
not very deep with only three convolution layers."
- paper: [http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)

**Multi-modal unsupervised feature learning for rgb-d scene labeling**

- paper: [http://www3.ntu.edu.sg/home/wanggang/WangECCV2014.pdf](http://www3.ntu.edu.sg/home/wanggang/WangECCV2014.pdf)

**Scene Labeling with LSTM Recurrent Neural Networks**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf)

**Attend, Infer, Repeat: Fast Scene Understanding with Generative Models**

- arxiv: [http://arxiv.org/abs/1603.08575](http://arxiv.org/abs/1603.08575)
- notes: [http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16](http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16)

**"Semantic Segmentation for Scene Understanding: Algorithms and Implementations" tutorial (2016 Embedded Vision Summit)**

- youtube: [https://www.youtube.com/watch?v=pQ318oCGJGY](https://www.youtube.com/watch?v=pQ318oCGJGY)

# Segmentation From Video

**Fast object segmentation in unconstrained video**

- project page: [http://calvin.inf.ed.ac.uk/software/fast-video-segmentation/](http://calvin.inf.ed.ac.uk/software/fast-video-segmentation/)
- paper: [http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/papazoglouICCV2013-camera-ready.pdf](http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/papazoglouICCV2013-camera-ready.pdf)

**Object Detection, Tracking, and Motion Segmentation for Object-level Video Segmentation**

- arxiv: [http://arxiv.org/abs/1608.03066](http://arxiv.org/abs/1608.03066)

## Benchmarks

**Semantic Understanding of Urban Street Scenes: Benchmark Suite**

[https://www.cityscapes-dataset.com/benchmarks/](https://www.cityscapes-dataset.com/benchmarks/)

## Challenges

**Large-scale Scene Understanding Challenge**

![](http://lsun.cs.princeton.edu/img/overview_4crop.jpg)

- homepage: [http://lsun.cs.princeton.edu/](http://lsun.cs.princeton.edu/)

# Blogs

**Deep Learning for Natural Image Segmentation Priors**

[http://cs.brown.edu/courses/csci2951-t/finals/ghope/](http://cs.brown.edu/courses/csci2951-t/finals/ghope/)
