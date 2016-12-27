---
layout: post
category: deep_learning
title: Segmentation
date: 2015-10-09
---

# Papers

**Deep Joint Task Learning for Generic Object Extraction**

![](http://vision.sysu.edu.cn/vision_sysu/wp-content/uploads/2013/05/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20141019095211.png)

- intro: NIPS 2014
- homepage: [http://vision.sysu.edu.cn/projects/deep-joint-task-learning/](http://vision.sysu.edu.cn/projects/deep-joint-task-learning/)
- paper: [http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf](http://ss.sysu.edu.cn/~ll/files/NIPS2014_JointTask.pdf)
- github: [https://github.com/xiaolonw/nips14_loc_seg_testonly](https://github.com/xiaolonw/nips14_loc_seg_testonly)
- dataset: [http://objectextraction.github.io/](http://objectextraction.github.io/)

**Highly Efficient Forward and Backward Propagation of Convolutional Neural Networks for Pixelwise Classification**

- arxiv: [https://arxiv.org/abs/1412.4526](https://arxiv.org/abs/1412.4526)
- code(Caffe): [https://dl.dropboxusercontent.com/u/6448899/caffe.zip](https://dl.dropboxusercontent.com/u/6448899/caffe.zip)
- author page: [http://www.ee.cuhk.edu.hk/~hsli/](http://www.ee.cuhk.edu.hk/~hsli/)

## U-Net

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

![](https://raw.githubusercontent.com/orobix/retina-unet/master/test/test_Original_GroundTruth_Prediction_3.png)

- project page: [http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- arxiv: [http://arxiv.org/abs/1505.04597](http://arxiv.org/abs/1505.04597)
- code+data: [http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-release-2015-10-02.tar.gz)
- github: [https://github.com/orobix/retina-unet](https://github.com/orobix/retina-unet)
- notes: [http://zongwei.leanote.com/post/Pa](http://zongwei.leanote.com/post/Pa)

**Segmentation from Natural Language Expressions**

![](http://ronghanghu.com/wp-content/uploads/text_objseg_method-768x331.jpg)

- intro: ECCV 2016
- project page: [http://ronghanghu.com/text_objseg/](http://ronghanghu.com/text_objseg/)
- arxiv: [http://arxiv.org/abs/1603.06180](http://arxiv.org/abs/1603.06180)
- github(TensorFlow): [https://github.com/ronghanghu/text_objseg](https://github.com/ronghanghu/text_objseg)
- gtihub(Caffe): [https://github.com/Seth-Park/text_objseg_caffe](https://github.com/Seth-Park/text_objseg_caffe)

**Semantic Object Parsing with Graph LSTM**

- arxiv: [http://arxiv.org/abs/1603.07063](http://arxiv.org/abs/1603.07063)

**Fine Hand Segmentation using Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1608.07454](http://arxiv.org/abs/1608.07454)

**Feedback Neural Network for Weakly Supervised Geo-Semantic Segmentation**

- intro: Facebook Connectivity Lab & Facebook Core Data Science & University of Illinois
- arxiv: [https://arxiv.org/abs/1612.02766](https://arxiv.org/abs/1612.02766)

**FusionNet: A deep fully residual convolutional neural network for image segmentation in connectomics**

- arxiv: [https://arxiv.org/abs/1612.05360](https://arxiv.org/abs/1612.05360)

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

**From Image-level to Pixel-level Labeling with Convolutional Networks**

- intro: CVPR 2015
- intro: "Weakly Supervised Semantic Segmentation with Convolutional Networks"
- intro: performs semantic segmentation based only on image-level annotations in a multiple instance learning framework
- arxiv: [http://arxiv.org/abs/1411.6228](http://arxiv.org/abs/1411.6228)
- paper: [http://ronan.collobert.com/pub/matos/2015_semisupsemseg_cvpr.pdf](http://ronan.collobert.com/pub/matos/2015_semisupsemseg_cvpr.pdf)

**Feedforward semantic segmentation with zoom-out features**

- intro: CVPR 2015. Toyota Technological Institute at Chicago
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf)
- bitbuckt: [https://bitbucket.org/m_mostajabi/zoom-out-release](https://bitbucket.org/m_mostajabi/zoom-out-release)
- video: [https://www.youtube.com/watch?v=HvgvX1LXQa8](https://www.youtube.com/watch?v=HvgvX1LXQa8)

## DeepLab

**Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs (DeepLab)**

- intro: ICLR 2015
- arxiv: [http://arxiv.org/abs/1412.7062](http://arxiv.org/abs/1412.7062)
- github: [https://bitbucket.org/deeplab/deeplab-public/](https://bitbucket.org/deeplab/deeplab-public/)
- github: [https://github.com/TheLegendAli/DeepLab-Context](https://github.com/TheLegendAli/DeepLab-Context)

**Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation (DeepLab)**

- arxiv: [http://arxiv.org/abs/1502.02734](http://arxiv.org/abs/1502.02734)
- bitbucket: [https://bitbucket.org/deeplab/deeplab-public/](https://bitbucket.org/deeplab/deeplab-public/)
- github: [https://github.com/TheLegendAli/DeepLab-Context](https://github.com/TheLegendAli/DeepLab-Context)

## DeepLab v2

**DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs**

![](http://liangchiehchen.com/fig/deeplab_intro.jpg)

- intro: 79.7% mIOU in the test set, PASCAL VOC-2012 semantic image segmentation task
- intro: Updated version of our previous ICLR 2015 paper
- project page: [http://liangchiehchen.com/projects/DeepLab.html](http://liangchiehchen.com/projects/DeepLab.html)
- arxiv: [https://arxiv.org/abs/1606.00915](https://arxiv.org/abs/1606.00915)
- bitbucket: [https://bitbucket.org/aquariusjay/deeplab-public-ver2](https://bitbucket.org/aquariusjay/deeplab-public-ver2)

## CRF-RNN

**Conditional Random Fields as Recurrent Neural Networks**

![](http://www.robots.ox.ac.uk/~szheng/Res_CRFRNN/CRFasRNN.jpg)

- intro: ICCV 2015. Oxford / Stanford / Baidu
- project page: [http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html](http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html)
- arxiv: [http://arxiv.org/abs/1502.03240](http://arxiv.org/abs/1502.03240)
- github: [https://github.com/torrvision/crfasrnn](https://github.com/torrvision/crfasrnn)
- demo: [http://www.robots.ox.ac.uk/~szheng/crfasrnndemo](http://www.robots.ox.ac.uk/~szheng/crfasrnndemo)
- github: [https://github.com/martinkersner/train-CRF-RNN](https://github.com/martinkersner/train-CRF-RNN)

## BoxSup

**BoxSup: Exploiting Bounding Boxes to Supervise Convolutional Networks for Semantic Segmentation**

- arxiv: [http://arxiv.org/abs/1503.01640](http://arxiv.org/abs/1503.01640)

**Efficient piecewise training of deep structured models for semantic segmentation**

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1504.01013](http://arxiv.org/abs/1504.01013)

## DeconvNet

**Learning Deconvolution Network for Semantic Segmentation (DeconvNet)**

![](http://cvlab.postech.ac.kr/research/deconvnet/images/overall.png)

- intro: ICLR 2016
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

## ParseNet

**ParseNet: Looking Wider to See Better**

- intro:ICLR 2016
- arxiv: [http://arxiv.org/abs/1506.04579](http://arxiv.org/abs/1506.04579)
- github: [https://github.com/weiliu89/caffe/tree/fcn](https://github.com/weiliu89/caffe/tree/fcn)
- caffe model zoo: [https://github.com/BVLC/caffe/wiki/Model-Zoo#parsenet-looking-wider-to-see-better](https://github.com/BVLC/caffe/wiki/Model-Zoo#parsenet-looking-wider-to-see-better)

## DecoupledNet

**Decoupled Deep Neural Network for Semi-supervised Semantic Segmentation**

![](http://cvlab.postech.ac.kr/research/decouplednet/images/overall.png)

- intro: ICLR 2016
- project(paper+code): [http://cvlab.postech.ac.kr/research/decouplednet/](http://cvlab.postech.ac.kr/research/decouplednet/)
- arxiv: [http://arxiv.org/abs/1506.04924](http://arxiv.org/abs/1506.04924)
- github: [https://github.com/HyeonwooNoh/DecoupledNet](https://github.com/HyeonwooNoh/DecoupledNet)

**Semantic Image Segmentation via Deep Parsing Network**

![](http://personal.ie.cuhk.edu.hk/~lz013/projects/dpn/intro.png)

- homepage: [http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html](http://personal.ie.cuhk.edu.hk/~lz013/projects/DPN.html)
- paper: [http://arxiv.org/abs/1509.02634](http://arxiv.org/abs/1509.02634)

**Multi-Scale Context Aggregation by Dilated Convolutions**

![](http://vladlen.info/wp-content/uploads/2016/02/dilated-convolutions1-894x263.png)

- intro: ICLR 2016. Dilated Convolution for Semantic Image Segmentation
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

**Seed, Expand and Constrain: Three Principles for Weakly-Supervised Image Segmentation**

![](https://cloud.githubusercontent.com/assets/460828/19045346/9b3bd058-8998-11e6-93f2-4c667fb7a1e8.png)

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1603.06098](https://arxiv.org/abs/1603.06098)
- github: [https://github.com/kolesman/SEC](https://github.com/kolesman/SEC)

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

**Region-based semantic segmentation with end-to-end training**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.07671](http://arxiv.org/abs/1607.07671)

**Built-in Foreground/Background Prior for Weakly-Supervised Semantic Segmentation**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1609.00446](http://arxiv.org/abs/1609.00446)

## PixelNet

**PixelNet: Towards a General Pixel-level Architecture**

- intro: semantic segmentation, edge detection
- arxiv: [http://arxiv.org/abs/1609.06694](http://arxiv.org/abs/1609.06694)

**Semantic Segmentation of Earth Observation Data Using Multimodal and Multi-scale Deep Networks**

- arxiv: [http://arxiv.org/abs/1609.06846](http://arxiv.org/abs/1609.06846)

**Deep Structured Features for Semantic Segmentation**

- arxiv: [http://arxiv.org/abs/1609.07916](http://arxiv.org/abs/1609.07916)

**CNN-aware Binary Map for General Semantic Segmentation**

- intro: ICIP 2016 Best Paper / Student Paper Finalist
- arxiv: [https://arxiv.org/abs/1609.09220](https://arxiv.org/abs/1609.09220)

**Efficient Convolutional Neural Network with Binary Quantization Layer**

- arxiv: [https://arxiv.org/abs/1611.06764](https://arxiv.org/abs/1611.06764)

**Mixed context networks for semantic segmentation**

- intro: Hikvision Research Institute
- arxiv: [https://arxiv.org/abs/1610.05854](https://arxiv.org/abs/1610.05854)

**High-Resolution Semantic Labeling with Convolutional Neural Networks**

- arxiv: [https://arxiv.org/abs/1611.01962](https://arxiv.org/abs/1611.01962)

## RefineNet

**RefineNet: Multi-Path Refinement Networks with Identity Mappings for High-Resolution Semantic Segmentation**

**RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation**

- intro: IoU 83.4% on PASCAL VOC 2012
- arxiv: [https://arxiv.org/abs/1611.06612](https://arxiv.org/abs/1611.06612)
- github: [https://github.com/guosheng/refinenet](https://github.com/guosheng/refinenet)
- leaderboard: [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_Multipath-RefineNet-Res152](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6#KEY_Multipath-RefineNet-Res152)

**Full-Resolution Residual Networks for Semantic Segmentation in Street Scenes**

- arxiv: [https://arxiv.org/abs/1611.08323](https://arxiv.org/abs/1611.08323)

**Semantic Segmentation using Adversarial Networks**

- intro: Facebook AI Research & INRIA. NIPS Workshop on Adversarial Training, Dec 2016, Barcelona, Spain
- arxiv: [https://arxiv.org/abs/1611.08408](https://arxiv.org/abs/1611.08408)

**Improving Fully Convolution Network for Semantic Segmentation**

- arxiv: [https://arxiv.org/abs/1611.08986](https://arxiv.org/abs/1611.08986)

**The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation**

- intro: Montreal Institute for Learning Algorithms & Ecole Polytechnique de Montreal
- arxiv: [https://arxiv.org/abs/1611.09326](https://arxiv.org/abs/1611.09326)
- github: [https://github.com/SimJeg/FC-DenseNet](https://github.com/SimJeg/FC-DenseNet)

**Training Bit Fully Convolutional Network for Fast Semantic Segmentation**

- intro: Megvii
- arxiv: [https://arxiv.org/abs/1612.00212](https://arxiv.org/abs/1612.00212)

**Classification With an Edge: Improving Semantic Image Segmentation with Boundary Detection**

- intro: "an end-to-end trainable deep convolutional neural network (DCNN) for semantic segmentation 
with built-in awareness of semantically meaningful boundaries. "
- arxiv: [https://arxiv.org/abs/1612.01337](https://arxiv.org/abs/1612.01337)

**Diverse Sampling for Self-Supervised Learning of Semantic Segmentation**

- arxiv: [https://arxiv.org/abs/1612.01991](https://arxiv.org/abs/1612.01991)

**Mining Pixels: Weakly Supervised Semantic Segmentation Using Image Labels**

- intro: Nankai University & University of Oxford & NUS
- arxiv: [https://arxiv.org/abs/1612.02101](https://arxiv.org/abs/1612.02101)

**FCNs in the Wild: Pixel-level Adversarial and Constraint-based Adaptation**

- arxiv: [https://arxiv.org/abs/1612.02649](https://arxiv.org/abs/1612.02649)

# Instance Segmentation

**Simultaneous Detection and Segmentation**

- intro: ECCV 2014
- author: Bharath Hariharan, Pablo Arbelaez, Ross Girshick, Jitendra Malik
- arxiv: [http://arxiv.org/abs/1407.1808](http://arxiv.org/abs/1407.1808)
- github(Matlab): [https://github.com/bharath272/sds_eccv2014](https://github.com/bharath272/sds_eccv2014)

**Convolutional Feature Masking for Joint Object and Stuff Segmentation**

- intro: CVPR 2015
- keywords: masking layers
- arxiv: [https://arxiv.org/abs/1412.1283](https://arxiv.org/abs/1412.1283)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dai_Convolutional_Feature_Masking_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Dai_Convolutional_Feature_Masking_2015_CVPR_paper.pdf)

**Proposal-free Network for Instance-level Object Segmentation**

- paper: [http://arxiv.org/abs/1509.02636](http://arxiv.org/abs/1509.02636)

**Hypercolumns for object segmentation and fine-grained localization**

- intro: CVPR 2015
- paper: [http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathCVPR2015.pdf](http://www.cs.berkeley.edu/~bharath2/pubs/pdfs/BharathCVPR2015.pdf)
- github("SDS using hypercolumns"): [https://github.com/bharath272/sds](https://github.com/bharath272/sds)

**Learning to decompose for object detection and instance segmentation**

- intro: ICLR 2016 Workshop
- keyword: CNN / RNN, MNIST, KITTI
- arxiv: [http://arxiv.org/abs/1511.06449](http://arxiv.org/abs/1511.06449)

**Recurrent Instance Segmentation**

![](http://romera-paredes.com/wp-content/uploads/2015/12/RIS.png)

- homepage: [http://romera-paredes.com/recurrent-instance-segmentation](http://romera-paredes.com/recurrent-instance-segmentation)
- arxiv: [http://arxiv.org/abs/1511.08250](http://arxiv.org/abs/1511.08250)

**Instance-aware Semantic Segmentation via Multi-task Network Cascades**

![](https://raw.githubusercontent.com/daijifeng001/MNC/master/data/readme_img/example.png)

- intro: CVPR 2016 oral. 1st-place winner of MS COCO 2015 segmentation competition
- keywords: RoI warping layer
- arxiv: [http://arxiv.org/abs/1512.04412](http://arxiv.org/abs/1512.04412)
- github: [https://github.com/daijifeng001/MNC](https://github.com/daijifeng001/MNC)

## DeepMask

**Learning to Segment Object Candidates**

- intro: Facebook AI Research (FAIR). learning segmentation proposals
- arxiv: [http://arxiv.org/abs/1506.06204](http://arxiv.org/abs/1506.06204)
- github: [https://github.com/facebookresearch/deepmask](https://github.com/facebookresearch/deepmask)
- github: [https://github.com/abbypa/NNProject_DeepMask](https://github.com/abbypa/NNProject_DeepMask)

## SharpMask

**Learning to Refine Object Segments**

- intro: ECCV 2016. Facebook AI Research (FAIR)
- intro: an extension of DeepMask which generates higher-fidelity masks using an additional top-down refinement step.
- arxiv: [http://arxiv.org/abs/1603.08695](http://arxiv.org/abs/1603.08695)
- github: [https://github.com/facebookresearch/deepmask](https://github.com/facebookresearch/deepmask)

**Instance-sensitive Fully Convolutional Networks**

- intro: ECCV 2016. instance segment proposal
- arxiv: [http://arxiv.org/abs/1603.08678](http://arxiv.org/abs/1603.08678)

**Amodal Instance Segmentation**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1604.08202](http://arxiv.org/abs/1604.08202)

**Bridging Category-level and Instance-level Semantic Image Segmentation**

- arxiv: [http://arxiv.org/abs/1605.06885](http://arxiv.org/abs/1605.06885)

**Recurrent Instance Segmentation**

- youtube: [https://www.youtube.com/watch?v=l_WD2OWOqBk](https://www.youtube.com/watch?v=l_WD2OWOqBk)

**Bottom-up Instance Segmentation using Deep Higher-Order CRFs**

- intro: BMVC 2016
- arxiv: [http://arxiv.org/abs/1609.02583](http://arxiv.org/abs/1609.02583)

## DeepCut

**DeepCut: Object Segmentation from Bounding Box Annotations using Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1605.07866](http://arxiv.org/abs/1605.07866)

**End-to-End Instance Segmentation and Counting with Recurrent Attention**

- intro: ReInspect
- arxiv: [http://arxiv.org/abs/1605.09410](http://arxiv.org/abs/1605.09410)

## TA-FCN

**Translation-aware Fully Convolutional Instance Segmentation**

**Fully Convolutional Instance-aware Semantic Segmentation**

- intro: winning entry of COCO segmentation challenge 2016
- arxiv: [https://arxiv.org/abs/1611.07709](https://arxiv.org/abs/1611.07709)
- github: [https://github.com/daijifeng001/TA-FCN](https://github.com/daijifeng001/TA-FCN)
- slides: [https://onedrive.live.com/?cid=f371d9563727b96f&id=F371D9563727B96F%2197213&authkey=%21AEYOyOirjIutSVk](https://onedrive.live.com/?cid=f371d9563727b96f&id=F371D9563727B96F%2197213&authkey=%21AEYOyOirjIutSVk)

**InstanceCut: from Edges to Instances with MultiCut**

- arxiv: [https://arxiv.org/abs/1611.08272](https://arxiv.org/abs/1611.08272)

**Deep Watershed Transform for Instance Segmentation**

- arxiv: [https://arxiv.org/abs/1611.08303](https://arxiv.org/abs/1611.08303)

**Object Detection Free Instance Segmentation With Labeling Transformations**

- arxiv: [https://arxiv.org/abs/1611.08991](https://arxiv.org/abs/1611.08991)

**Shape-aware Instance Segmentation**

- arxiv: [https://arxiv.org/abs/1612.03129](https://arxiv.org/abs/1612.03129)

# Scene Labeling/Parsing

**Indoor Semantic Segmentation using depth information**

- arxiv: [http://arxiv.org/abs/1301.3572](http://arxiv.org/abs/1301.3572)

**Recurrent Convolutional Neural Networks for Scene Parsing**

- arxiv: [http://arxiv.org/abs/1306.2795](http://arxiv.org/abs/1306.2795)
- slides: [http://people.ee.duke.edu/~lcarin/Yizhe8.14.2015.pdf](http://people.ee.duke.edu/~lcarin/Yizhe8.14.2015.pdf)
- github: [https://github.com/NP-coder/CLPS1520Project](https://github.com/NP-coder/CLPS1520Project)
- github: [https://github.com/rkargon/Scene-Labeling](https://github.com/rkargon/Scene-Labeling)

**Learning hierarchical features for scene labeling**

- paper: [http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf](http://yann.lecun.com/exdb/publis/pdf/farabet-pami-13.pdf)

**Multi-modal unsupervised feature learning for rgb-d scene labeling**

- intro: ECCV 2014
- paper: [http://www3.ntu.edu.sg/home/wanggang/WangECCV2014.pdf](http://www3.ntu.edu.sg/home/wanggang/WangECCV2014.pdf)

**Scene Labeling with LSTM Recurrent Neural Networks**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Byeon_Scene_Labeling_With_2015_CVPR_paper.pdf)

**Attend, Infer, Repeat: Fast Scene Understanding with Generative Models**

- arxiv: [http://arxiv.org/abs/1603.08575](http://arxiv.org/abs/1603.08575)
- notes: [http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16](http://www.shortscience.org/paper?bibtexKey=journals/corr/EslamiHWTKH16)

**"Semantic Segmentation for Scene Understanding: Algorithms and Implementations" tutorial**

- intro: 2016 Embedded Vision Summit
- youtube: [https://www.youtube.com/watch?v=pQ318oCGJGY](https://www.youtube.com/watch?v=pQ318oCGJGY)

**Semantic Understanding of Scenes through the ADE20K Dataset**

- arxiv: [https://arxiv.org/abs/1608.05442](https://arxiv.org/abs/1608.05442)

## MPF-RNN

**Multi-Path Feedback Recurrent Neural Network for Scene Parsing**

- arxiv: [http://arxiv.org/abs/1608.07706](http://arxiv.org/abs/1608.07706)

**Scene Labeling using Recurrent Neural Networks with Explicit Long Range Contextual Dependency**

- arxiv: [https://arxiv.org/abs/1611.07485](https://arxiv.org/abs/1611.07485)

**Pyramid Scene Parsing Network**

- intro: mIoU score as 85.4% on PASCAL VOC 2012 and 80.2% on Cityscapes, 
ranked 1st place in ImageNet Scene Parsing Challenge 2016
- project page: [http://appsrv.cse.cuhk.edu.hk/~hszhao/projects/pspnet/index.html](http://appsrv.cse.cuhk.edu.hk/~hszhao/projects/pspnet/index.html)
- arxiv: [https://arxiv.org/abs/1612.01105](https://arxiv.org/abs/1612.01105)
- slides: [http://image-net.org/challenges/talks/2016/SenseCUSceneParsing.pdf](http://image-net.org/challenges/talks/2016/SenseCUSceneParsing.pdf)
- github: [https://github.com/hszhao/PSPNet](https://github.com/hszhao/PSPNet)

## Benchmarks

**Semantic Understanding of Urban Street Scenes: Benchmark Suite**

[https://www.cityscapes-dataset.com/benchmarks/](https://www.cityscapes-dataset.com/benchmarks/)

## Challenges

**Large-scale Scene Understanding Challenge**

![](http://lsun.cs.princeton.edu/img/overview_4crop.jpg)

- homepage: [http://lsun.cs.princeton.edu/](http://lsun.cs.princeton.edu/)

**Places2 Challenge**

[http://places2.csail.mit.edu/challenge.html](http://places2.csail.mit.edu/challenge.html)

## Datasets

**ADE20K**

- intro: train: 20,120 images, val: 2000 images
- homepage: [http://groups.csail.mit.edu/vision/datasets/ADE20K/](http://groups.csail.mit.edu/vision/datasets/ADE20K/)

# Segmentation From Video

**Fast object segmentation in unconstrained video**

- project page: [http://calvin.inf.ed.ac.uk/software/fast-video-segmentation/](http://calvin.inf.ed.ac.uk/software/fast-video-segmentation/)
- paper: [http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/papazoglouICCV2013-camera-ready.pdf](http://calvin.inf.ed.ac.uk/wp-content/uploads/Publications/papazoglouICCV2013-camera-ready.pdf)

**Recurrent Fully Convolutional Networks for Video Segmentation**

- arxiv: [https://arxiv.org/abs/1606.00487](https://arxiv.org/abs/1606.00487)

**Object Detection, Tracking, and Motion Segmentation for Object-level Video Segmentation**

- arxiv: [http://arxiv.org/abs/1608.03066](http://arxiv.org/abs/1608.03066)

**Clockwork Convnets for Video Semantic Segmentation**

- intro: evaluated on the Youtube-Objects, NYUD, and Cityscapes video datasets
- arxiv: [http://arxiv.org/abs/1608.03609](http://arxiv.org/abs/1608.03609)
- github: [https://github.com/shelhamer/clockwork-fcn](https://github.com/shelhamer/clockwork-fcn)

**STFCN: Spatio-Temporal FCN for Semantic Video Segmentation**

- arxiv: [http://arxiv.org/abs/1608.05971](http://arxiv.org/abs/1608.05971)

**One-Shot Video Object Segmentation**

- intro: OSVOS
- arxiv: [https://arxiv.org/abs/1611.05198](https://arxiv.org/abs/1611.05198)

**Convolutional Gated Recurrent Networks for Video Segmentation**

- arxiv: [https://arxiv.org/abs/1611.05435](https://arxiv.org/abs/1611.05435)

**One-Shot Video Object Segmentation**

- arxiv: [https://arxiv.org/abs/1611.05198](https://arxiv.org/abs/1611.05198)

**Learning Video Object Segmentation from Static Images**

- arxiv: [https://arxiv.org/abs/1612.02646](https://arxiv.org/abs/1612.02646)

# Leaderboard

**Segmentation Results: VOC2012 BETA: Competition "comp6" (train on own data)**

[http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6)

# Blogs

**Deep Learning for Natural Image Segmentation Priors**

[http://cs.brown.edu/courses/csci2951-t/finals/ghope/](http://cs.brown.edu/courses/csci2951-t/finals/ghope/)

**Image Segmentation Using DIGITS 5**

[https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/](https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/)

**Image Segmentation with Tensorflow using CNNs and Conditional Random Fields**

[http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/)

# Talks

**Deep learning for image segmentation**

- intro: PyData Warsaw - Mateusz Opala & Michał Jamroż
- youtube: [https://www.youtube.com/watch?v=W6r_a5crqGI](https://www.youtube.com/watch?v=W6r_a5crqGI)
