---
layout: post
category: deep_learning
title: Object Detection
date: 2015-10-09
---

| Method           | backbone      | test size | VOC2007 | VOC2010 | VOC2012 | ILSVRC 2013 | MSCOCO 2015                     | Speed                          |
| :------------:   | :-----:       | :-----:   | :-----: | :-----: | :-----: | :---------: | :---------:                     | :---------:                    |
| OverFeat         |               |           |         |         |         | 24.3%       |                                 |                                |
| R-CNN            | AlexNet       |           | 58.5%   | 53.7%   | 53.3%   | 31.4%       |                                 |                                |
| R-CNN            | VGG16         |           | 66.0%   |         |         |             |                                 |                                |
| SPP_net          | ZF-5          |           | 54.2%   |         |         | 31.84%      |                                 |                                |
| DeepID-Net       |               |           | 64.1%   |         |         | 50.3%       |                                 |                                |
| NoC              | 73.3%         |           | 68.8%   |         |         |             |                                 |                                |
| Fast-RCNN        | VGG16         |           | 70.0%   | 68.8%   | 68.4%   |             | 19.7%(@[0.5-0.95]), 35.9%(@0.5) |                                |
| MR-CNN           | 78.2%         |           | 73.9%   |         |         |             |                                 |                                |
| Faster-RCNN      | VGG16         |           | 78.8%   |         | 75.9%   |             | 21.9%(@[0.5-0.95]), 42.7%(@0.5) | 198ms                          |
| Faster-RCNN      | ResNet101     |           | 85.6%   |         | 83.8%   |             | 37.4%(@[0.5-0.95]), 59.0%(@0.5) |                                |
| YOLO             |               |           | 63.4%   |         | 57.9%   |             |                                 | 45 fps                         |
| YOLO VGG-16      |               |           | 66.4%   |         |         |             |                                 | 21 fps                         |
| YOLOv2           |               | 448x448   | 78.6%   |         | 73.4%   |             | 21.6%(@[0.5-0.95]), 44.0%(@0.5) | 40 fps                         |
| SSD              | VGG16         | 300x300   | 77.2%   |         | 75.8%   |             | 25.1%(@[0.5-0.95]), 43.1%(@0.5) | 46 fps                         |
| SSD              | VGG16         | 512x512   | 79.8%   |         | 78.5%   |             | 28.8%(@[0.5-0.95]), 48.5%(@0.5) | 19 fps                         |
| SSD              | ResNet101     | 300x300   |         |         |         |             | 28.0%(@[0.5-0.95])              | 16 fps                         |
| SSD              | ResNet101     | 512x512   |         |         |         |             | 31.2%(@[0.5-0.95])              | 8 fps                          |
| DSSD             | ResNet101     | 300x300   |         |         |         |             | 28.0%(@[0.5-0.95])              | 8 fps                          |
| DSSD             | ResNet101     | 500x500   |         |         |         |             | 33.2%(@[0.5-0.95])              | 6 fps                          |
| ION              |               |           | 79.2%   |         | 76.4%   |             |                                 |                                |
| CRAFT            |               |           | 75.7%   |         | 71.3%   | 48.5%       |                                 |                                |
| OHEM             |               |           | 78.9%   |         | 76.3%   |             | 25.5%(@[0.5-0.95]), 45.9%(@0.5) |                                |
| R-FCN            | ResNet50      |           | 77.4%   |         |         |             |                                 | 0.12sec(K40), 0.09sec(TitianX) |
| R-FCN            | ResNet101     |           | 79.5%   |         |         |             |                                 | 0.17sec(K40), 0.12sec(TitianX) |
| R-FCN(ms train)  | ResNet101     |           | 83.6%   |         | 82.0%   |             | 31.5%(@[0.5-0.95]), 53.2%(@0.5) |                                |
| PVANet 9.0       |               |           | 84.9%   |         | 84.2%   |             |                                 | 750ms(CPU), 46ms(TitianX)      |
| RetinaNet        | ResNet101-FPN |           |         |         |         |             |                                 |                                |
| Light-Head R-CNN | Xception\*    | 800/1200  |         |         |         |             | 31.5%@[0.5:0.95]                | 95 fps                         |
| Light-Head R-CNN | Xception\*    | 700/1100  |         |         |         |             | 30.7%@[0.5:0.95]                | 102 fps                        |

# Papers

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- github: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

**Scalable Object Detection using Deep Neural Networks**

- intro: first MultiBox. Train a CNN to predict Region of Interest.
- arxiv: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)
- blog: [https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html](https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html)

**Scalable, High-Quality Object Detection**

- intro: second MultiBox
- arxiv: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

- intro: ECCV 2014 / TPAMI 2015
- keywords: SPP-Net
- arxiv: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

- intro: PAMI 2016
- intro: an extension of R-CNN. box pre-training, cascade on region proposals, deformation layers and context representations
- project page: [http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html](http://www.ee.cuhk.edu.hk/%CB%9Cwlouyang/projects/imagenetDeepId/index.html)
- arxiv: [http://arxiv.org/abs/1412.5661](http://arxiv.org/abs/1412.5661)

**Object Detectors Emerge in Deep Scene CNNs**

- intro: ICLR 2015
- arxiv: [http://arxiv.org/abs/1412.6856](http://arxiv.org/abs/1412.6856)
- paper: [https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf](https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf)
- paper: [https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf](https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf)
- slides: [http://places.csail.mit.edu/slide_iclr2015.pdf](http://places.csail.mit.edu/slide_iclr2015.pdf)

**segDeepM: Exploiting Segmentation and Context in Deep Neural Networks for Object Detection**

- intro: CVPR 2015
- project(code+data): [https://www.cs.toronto.edu/~yukun/segdeepm.html](https://www.cs.toronto.edu/~yukun/segdeepm.html)
- arxiv: [https://arxiv.org/abs/1502.04275](https://arxiv.org/abs/1502.04275)
- github: [https://github.com/YknZhu/segDeepM](https://github.com/YknZhu/segDeepM)

**Object Detection Networks on Convolutional Feature Maps**

- intro: TPAMI 2015
- keywords: NoC
- arxiv: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- github: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

**DeepBox: Learning Objectness with Convolutional Networks**

- keywords: DeepBox
- arxiv: [http://arxiv.org/abs/1505.02146](http://arxiv.org/abs/1505.02146)
- github: [https://github.com/weichengkuo/DeepBox](https://github.com/weichengkuo/DeepBox)

**Object detection via a multi-region & semantic segmentation-aware CNN model**

- intro: ICCV 2015
- keywords: MR-CNN
- arxiv: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- github: [https://github.com/gidariss/mrcnn-object-detection](https://github.com/gidariss/mrcnn-object-detection)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

**AttentionNet: Aggregating Weak Directions for Accurate Object Detection**

- intro: ICCV 2015
- intro: state-of-the-art performance of 65% (AP) on PASCAL VOC 2007/2012 human detection task
- arxiv: [http://arxiv.org/abs/1506.07704](http://arxiv.org/abs/1506.07704)
- slides: [https://www.robots.ox.ac.uk/~vgg/rg/slides/AttentionNet.pdf](https://www.robots.ox.ac.uk/~vgg/rg/slides/AttentionNet.pdf)
- slides: [http://image-net.org/challenges/talks/lunit-kaist-slide.pdf](http://image-net.org/challenges/talks/lunit-kaist-slide.pdf)

## DenseBox

**DenseBox: Unifying Landmark Localization with End to End Object Detection**

- arxiv: [http://arxiv.org/abs/1509.04874](http://arxiv.org/abs/1509.04874)
- demo: [http://pan.baidu.com/s/1mgoWWsS](http://pan.baidu.com/s/1mgoWWsS)
- KITTI result: [http://www.cvlibs.net/datasets/kitti/eval_object.php](http://www.cvlibs.net/datasets/kitti/eval_object.php)

**Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks**

- intro: "0.8s per image on a Titan X GPU (excluding proposal generation) without two-stage bounding-box regression
and 1.15s per image with it".
- keywords: Inside-Outside Net (ION)
- arxiv: [http://arxiv.org/abs/1512.04143](http://arxiv.org/abs/1512.04143)
- slides: [http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf](http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf)
- coco-leaderboard: [http://mscoco.org/dataset/#detections-leaderboard](http://mscoco.org/dataset/#detections-leaderboard)

**Adaptive Object Detection Using Adjacency and Zoom Prediction**

- intro: CVPR 2016. AZ-Net
- arxiv: [http://arxiv.org/abs/1512.07711](http://arxiv.org/abs/1512.07711)
- github: [https://github.com/luyongxi/az-net](https://github.com/luyongxi/az-net)
- youtube: [https://www.youtube.com/watch?v=YmFtuNwxaNM](https://www.youtube.com/watch?v=YmFtuNwxaNM)

**G-CNN: an Iterative Grid Based Object Detector**

- arxiv: [http://arxiv.org/abs/1512.07729](http://arxiv.org/abs/1512.07729)

**We don't need no bounding-boxes: Training object class detectors using only human verification**

- arxiv: [http://arxiv.org/abs/1602.08405](http://arxiv.org/abs/1602.08405)

**HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection**

- arxiv: [http://arxiv.org/abs/1604.00600](http://arxiv.org/abs/1604.00600)

**A MultiPath Network for Object Detection**

- intro: BMVC 2016. Facebook AI Research (FAIR)
- arxiv: [http://arxiv.org/abs/1604.02135](http://arxiv.org/abs/1604.02135)
- github: [https://github.com/facebookresearch/multipathnet](https://github.com/facebookresearch/multipathnet)

**CRAFT Objects from Images**

- intro: CVPR 2016. Cascade Region-proposal-network And FasT-rcnn. an extension of Faster R-CNN
- project page: [http://byangderek.github.io/projects/craft.html](http://byangderek.github.io/projects/craft.html)
- arxiv: [https://arxiv.org/abs/1604.03239](https://arxiv.org/abs/1604.03239)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_CRAFT_Objects_From_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_CRAFT_Objects_From_CVPR_2016_paper.pdf)
- github: [https://github.com/byangderek/CRAFT](https://github.com/byangderek/CRAFT)

## OHEM

**Training Region-based Object Detectors with Online Hard Example Mining**

- intro: CVPR 2016 Oral. Online hard example mining (OHEM)
- arxiv: [http://arxiv.org/abs/1604.03540](http://arxiv.org/abs/1604.03540)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shrivastava_Training_Region-Based_Object_CVPR_2016_paper.pdf)
- github(Official): [https://github.com/abhi2610/ohem](https://github.com/abhi2610/ohem)
- author page: [http://abhinav-shrivastava.info/](http://abhinav-shrivastava.info/)

**S-OHEM: Stratified Online Hard Example Mining for Object Detection**

[https://arxiv.org/abs/1705.02233](https://arxiv.org/abs/1705.02233)

- - -

**Exploit All the Layers: Fast and Accurate CNN Object Detector with Scale Dependent Pooling and Cascaded Rejection Classifiers**

- intro: CVPR 2016
- keywords: scale-dependent pooling  (SDP), cascaded rejection classifiers (CRC)
- paper: [http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf](http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf)

## R-FCN

**R-FCN: Object Detection via Region-based Fully Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)
- github: [https://github.com/daijifeng001/R-FCN](https://github.com/daijifeng001/R-FCN)
- github(MXNet): [https://github.com/msracver/Deformable-ConvNets/tree/master/rfcn](https://github.com/msracver/Deformable-ConvNets/tree/master/rfcn)
- github: [https://github.com/Orpine/py-R-FCN](https://github.com/Orpine/py-R-FCN)
- github: [https://github.com/PureDiors/pytorch_RFCN](https://github.com/PureDiors/pytorch_RFCN)
- github: [https://github.com/bharatsingh430/py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU)
- github: [https://github.com/xdever/RFCN-tensorflow](https://github.com/xdever/RFCN-tensorflow)

**R-FCN-3000 at 30fps: Decoupling Detection and Classification**

[https://arxiv.org/abs/1712.01802](https://arxiv.org/abs/1712.01802)

**Recycle deep features for better object detection**

- arxiv: [http://arxiv.org/abs/1607.05066](http://arxiv.org/abs/1607.05066)

**A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection**

- intro: ECCV 2016
- intro: 640×480: 15 fps, 960×720: 8 fps
- keywords: MS-CNN
- arxiv: [http://arxiv.org/abs/1607.07155](http://arxiv.org/abs/1607.07155)
- github: [https://github.com/zhaoweicai/mscnn](https://github.com/zhaoweicai/mscnn)
- poster: [http://www.eccv2016.org/files/posters/P-2B-38.pdf](http://www.eccv2016.org/files/posters/P-2B-38.pdf)

**Multi-stage Object Detection with Group Recursive Learning**

- intro: VOC2007: 78.6%, VOC2012: 74.9%
- arxiv: [http://arxiv.org/abs/1608.05159](http://arxiv.org/abs/1608.05159)

**Subcategory-aware Convolutional Neural Networks for Object Proposals and Detection**

- intro: WACV 2017. SubCNN
- arxiv: [http://arxiv.org/abs/1604.04693](http://arxiv.org/abs/1604.04693)
- github: [https://github.com/tanshen/SubCNN](https://github.com/tanshen/SubCNN)

**PVANet: Lightweight Deep Neural Networks for Real-time Object Detection**

- intro: Presented at NIPS 2016 Workshop on Efficient Methods for Deep Neural Networks (EMDNN). 
Continuation of [arXiv:1608.08021](https://arxiv.org/abs/1608.08021)
- arxiv: [https://arxiv.org/abs/1611.08588](https://arxiv.org/abs/1611.08588)
- github: [https://github.com/sanghoon/pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn)
- leaderboard(PVANet 9.0): [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4)

**Gated Bi-directional CNN for Object Detection**

- intro: The Chinese University of Hong Kong & Sensetime Group Limited
- keywords: GBD-Net
- paper: [http://link.springer.com/chapter/10.1007/978-3-319-46478-7_22](http://link.springer.com/chapter/10.1007/978-3-319-46478-7_22)
- mirror: [https://pan.baidu.com/s/1dFohO7v](https://pan.baidu.com/s/1dFohO7v)

**Crafting GBD-Net for Object Detection**

- intro: winner of the ImageNet object detection challenge of 2016. CUImage and CUVideo
- intro: gated bi-directional CNN (GBD-Net)
- arxiv: [https://arxiv.org/abs/1610.02579](https://arxiv.org/abs/1610.02579)
- github: [https://github.com/craftGBD/craftGBD](https://github.com/craftGBD/craftGBD)

**StuffNet: Using 'Stuff' to Improve Object Detection**

- arxiv: [https://arxiv.org/abs/1610.05861](https://arxiv.org/abs/1610.05861)

**Generalized Haar Filter based Deep Networks for Real-Time Object Detection in Traffic Scene**

- arxiv: [https://arxiv.org/abs/1610.09609](https://arxiv.org/abs/1610.09609)

**Hierarchical Object Detection with Deep Reinforcement Learning**

- intro: Deep Reinforcement Learning Workshop (NIPS 2016)
- project page: [https://imatge-upc.github.io/detection-2016-nipsws/](https://imatge-upc.github.io/detection-2016-nipsws/)
- arxiv: [https://arxiv.org/abs/1611.03718](https://arxiv.org/abs/1611.03718)
- slides: [http://www.slideshare.net/xavigiro/hierarchical-object-detection-with-deep-reinforcement-learning](http://www.slideshare.net/xavigiro/hierarchical-object-detection-with-deep-reinforcement-learning)
- github: [https://github.com/imatge-upc/detection-2016-nipsws](https://github.com/imatge-upc/detection-2016-nipsws)
- blog: [http://jorditorres.org/nips/](http://jorditorres.org/nips/)

**Learning to detect and localize many objects from few examples**

- arxiv: [https://arxiv.org/abs/1611.05664](https://arxiv.org/abs/1611.05664)

**Speed/accuracy trade-offs for modern convolutional object detectors**

- intro: CVPR 2017. Google Research
- arxiv: [https://arxiv.org/abs/1611.10012](https://arxiv.org/abs/1611.10012)

**SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving**

- arxiv: [https://arxiv.org/abs/1612.01051](https://arxiv.org/abs/1612.01051)
- github: [https://github.com/BichenWuUCB/squeezeDet](https://github.com/BichenWuUCB/squeezeDet)
- github: [https://github.com/fregu856/2D_detection](https://github.com/fregu856/2D_detection)

## Feature Pyramid Network (FPN)

**Feature Pyramid Networks for Object Detection**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)

**Dynamic Feature Pyramid Networks for Object Detection**

- intro: Zhejiang University & Noah’s Ark Lab & Westlake University
- arxiv: [https://arxiv.org/abs/2012.00779](https://arxiv.org/abs/2012.00779)

**Implicit Feature Pyramid Network for Object Detection**

- intro: MEGVII Technology
- arxiv: [https://arxiv.org/abs/2012.13563](https://arxiv.org/abs/2012.13563)

**You Should Look at All Objects**

- intro: ECCV 2022
- intro: The University of Hong Kong & Bytedance & University of Rochester
- arxiv: [https://arxiv.org/abs/2207.07889](https://arxiv.org/abs/2207.07889)
- github: [https://github.com/CharlesPikachu/YSLAO](https://github.com/CharlesPikachu/YSLAO)

**Action-Driven Object Detection with Top-Down Visual Attentions**

- arxiv: [https://arxiv.org/abs/1612.06704](https://arxiv.org/abs/1612.06704)

**Beyond Skip Connections: Top-Down Modulation for Object Detection**

- intro: CMU & UC Berkeley & Google Research
- arxiv: [https://arxiv.org/abs/1612.06851](https://arxiv.org/abs/1612.06851)

**Wide-Residual-Inception Networks for Real-time Object Detection**

- intro: Inha University
- arxiv: [https://arxiv.org/abs/1702.01243](https://arxiv.org/abs/1702.01243)

**Attentional Network for Visual Object Detection**

- intro: University of Maryland & Mitsubishi Electric Research Laboratories
- arxiv: [https://arxiv.org/abs/1702.01478](https://arxiv.org/abs/1702.01478)

**Learning Chained Deep Features and Classifiers for Cascade in Object Detection**

- keykwords: CC-Net
- intro: chained cascade network (CC-Net). 81.1% mAP on PASCAL VOC 2007
- arxiv: [https://arxiv.org/abs/1702.07054](https://arxiv.org/abs/1702.07054)

**DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling**

- intro: ICCV 2017 (poster)
- arxiv: [https://arxiv.org/abs/1703.10295](https://arxiv.org/abs/1703.10295)

**Discriminative Bimodal Networks for Visual Localization and Detection with Natural Language Queries**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1704.03944](https://arxiv.org/abs/1704.03944)

**Spatial Memory for Context Reasoning in Object Detection**

- arxiv: [https://arxiv.org/abs/1704.04224](https://arxiv.org/abs/1704.04224)

**Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection**

[https://arxiv.org/abs/1704.05775](https://arxiv.org/abs/1704.05775)

**LCDet: Low-Complexity Fully-Convolutional Neural Networks for Object Detection in Embedded Systems**

- intro: Embedded Vision Workshop in CVPR. UC San Diego & Qualcomm Inc
- arxiv: [https://arxiv.org/abs/1705.05922](https://arxiv.org/abs/1705.05922)

**Point Linking Network for Object Detection**

- intro: Point Linking Network (PLN)
- arxiv: [https://arxiv.org/abs/1706.03646](https://arxiv.org/abs/1706.03646)

**Perceptual Generative Adversarial Networks for Small Object Detection**

[https://arxiv.org/abs/1706.05274](https://arxiv.org/abs/1706.05274)

**Few-shot Object Detection**

[https://arxiv.org/abs/1706.08249](https://arxiv.org/abs/1706.08249)

**Yes-Net: An effective Detector Based on Global Information**

[https://arxiv.org/abs/1706.09180](https://arxiv.org/abs/1706.09180)

**Towards lightweight convolutional neural networks for object detection**

[https://arxiv.org/abs/1707.01395](https://arxiv.org/abs/1707.01395)

**RON: Reverse Connection with Objectness Prior Networks for Object Detection**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1707.01691](https://arxiv.org/abs/1707.01691)
- github: [https://github.com/taokong/RON](https://github.com/taokong/RON)

**Deformable Part-based Fully Convolutional Network for Object Detection**

- intro: BMVC 2017 (oral). Sorbonne Universités & CEDRIC
- arxiv: [https://arxiv.org/abs/1707.06175](https://arxiv.org/abs/1707.06175)

**Adaptive Feeding: Achieving Fast and Accurate Detections by Adaptively Combining Object Detectors**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1707.06399](https://arxiv.org/abs/1707.06399)

**Recurrent Scale Approximation for Object Detection in CNN**

- intro: ICCV 2017
- keywords: Recurrent Scale Approximation (RSA)
- arxiv: [https://arxiv.org/abs/1707.09531](https://arxiv.org/abs/1707.09531)
- github: [https://github.com/sciencefans/RSA-for-object-detection](https://github.com/sciencefans/RSA-for-object-detection)

**DSOD: Learning Deeply Supervised Object Detectors from Scratch**

![](https://user-images.githubusercontent.com/3794909/28934967-718c9302-78b5-11e7-89ee-8b514e53e23c.png)

- intro: ICCV 2017. Fudan University & Tsinghua University & Intel Labs China
- arxiv: [https://arxiv.org/abs/1708.01241](https://arxiv.org/abs/1708.01241)
- github: [https://github.com/szq0214/DSOD](https://github.com/szq0214/DSOD)

**Object Detection from Scratch with Deep Supervision**

[https://arxiv.org/abs/1809.09294](https://arxiv.org/abs/1809.09294)

**CoupleNet: Coupling Global Structure with Local Parts for Object Detection**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.02863](https://arxiv.org/abs/1708.02863)

**Incremental Learning of Object Detectors without Catastrophic Forgetting**

- intro: ICCV 2017. Inria
- arxiv: [https://arxiv.org/abs/1708.06977](https://arxiv.org/abs/1708.06977)

**Zoom Out-and-In Network with Map Attention Decision for Region Proposal and Object Detection**

[https://arxiv.org/abs/1709.04347](https://arxiv.org/abs/1709.04347)

**StairNet: Top-Down Semantic Aggregation for Accurate One Shot Detection**

[https://arxiv.org/abs/1709.05788](https://arxiv.org/abs/1709.05788)

**Dynamic Zoom-in Network for Fast Object Detection in Large Images**

[https://arxiv.org/abs/1711.05187](https://arxiv.org/abs/1711.05187)

**Zero-Annotation Object Detection with Web Knowledge Transfer**

- intro: NTU, Singapore & Amazon
- keywords: multi-instance multi-label domain adaption learning framework
- arxiv: [https://arxiv.org/abs/1711.05954](https://arxiv.org/abs/1711.05954)

**MegDet: A Large Mini-Batch Object Detector**

- intro: Peking University & Tsinghua University & Megvii Inc
- arxiv: [https://arxiv.org/abs/1711.07240](https://arxiv.org/abs/1711.07240)

**Receptive Field Block Net for Accurate and Fast Object Detection**

- intro: RFBNet
- arxiv: [https://arxiv.org/abs/1711.07767](https://arxiv.org/abs/1711.07767)
- github: [https://github.com//ruinmessi/RFBNet](https://github.com//ruinmessi/RFBNet)

**An Analysis of Scale Invariance in Object Detection - SNIP**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1711.08189](https://arxiv.org/abs/1711.08189)
- github: [https://github.com/bharatsingh430/snip](https://github.com/bharatsingh430/snip)

**Feature Selective Networks for Object Detection**

[https://arxiv.org/abs/1711.08879](https://arxiv.org/abs/1711.08879)

**Learning a Rotation Invariant Detector with Rotatable Bounding Box**

- arxiv: [https://arxiv.org/abs/1711.09405](https://arxiv.org/abs/1711.09405)
- github(official, Caffe): [https://github.com/liulei01/DRBox](https://github.com/liulei01/DRBox)

**Scalable Object Detection for Stylized Objects**

- intro: Microsoft AI & Research Munich
- arxiv: [https://arxiv.org/abs/1711.09822](https://arxiv.org/abs/1711.09822)

**Learning Object Detectors from Scratch with Gated Recurrent Feature Pyramids**

- arxiv: [https://arxiv.org/abs/1712.00886](https://arxiv.org/abs/1712.00886)
- github: [https://github.com/szq0214/GRP-DSOD](https://github.com/szq0214/GRP-DSOD)

**Deep Regionlets for Object Detection**

- keywords: region selection network, gating network
- arxiv: [https://arxiv.org/abs/1712.02408](https://arxiv.org/abs/1712.02408)

**Training and Testing Object Detectors with Virtual Images**

- intro: IEEE/CAA Journal of Automatica Sinica
- arxiv: [https://arxiv.org/abs/1712.08470](https://arxiv.org/abs/1712.08470)

**Large-Scale Object Discovery and Detector Adaptation from Unlabeled Video**

- keywords: object mining, object tracking, unsupervised object discovery by appearance-based clustering, self-supervised detector adaptation
- arxiv: [https://arxiv.org/abs/1712.08832](https://arxiv.org/abs/1712.08832)

**Spot the Difference by Object Detection**

- intro: Tsinghua University & JD Group
- arxiv: [https://arxiv.org/abs/1801.01051](https://arxiv.org/abs/1801.01051)

**Localization-Aware Active Learning for Object Detection**

- arxiv: [https://arxiv.org/abs/1801.05124](https://arxiv.org/abs/1801.05124)

**Object Detection with Mask-based Feature Encoding**

[https://arxiv.org/abs/1802.03934](https://arxiv.org/abs/1802.03934)

**LSTD: A Low-Shot Transfer Detector for Object Detection**

- intro: AAAI 2018
- arxiv: [https://arxiv.org/abs/1803.01529](https://arxiv.org/abs/1803.01529)

**Pseudo Mask Augmented Object Detection**

[https://arxiv.org/abs/1803.05858](https://arxiv.org/abs/1803.05858)

**Revisiting RCNN: On Awakening the Classification Power of Faster RCNN**

- intro: ECCV 2018
- keywords: DCR V1
- arxiv: [https://arxiv.org/abs/1803.06799](https://arxiv.org/abs/1803.06799)
- github(official, MXNet): [https://github.com/bowenc0221/Decoupled-Classification-Refinement](https://github.com/bowenc0221/Decoupled-Classification-Refinement)

**Decoupled Classification Refinement: Hard False Positive Suppression for Object Detection**

- keywords: DCR V2
- arxiv: [https://arxiv.org/abs/1810.04002](https://arxiv.org/abs/1810.04002)
- github(official, MXNet): [https://github.com/bowenc0221/Decoupled-Classification-Refinement](https://github.com/bowenc0221/Decoupled-Classification-Refinement)

**Learning Region Features for Object Detection**

- intro: Peking University & MSRA
- arxiv: [https://arxiv.org/abs/1803.07066](https://arxiv.org/abs/1803.07066)

**Object Detection for Comics using Manga109 Annotations**

- intro: University of Tokyo & National Institute of Informatics, Japan
- arxiv: [https://arxiv.org/abs/1803.08670](https://arxiv.org/abs/1803.08670)

**Task-Driven Super Resolution: Object Detection in Low-resolution Images**

[https://arxiv.org/abs/1803.11316](https://arxiv.org/abs/1803.11316)

**Transferring Common-Sense Knowledge for Object Detection**

[https://arxiv.org/abs/1804.01077](https://arxiv.org/abs/1804.01077)

**Multi-scale Location-aware Kernel Representation for Object Detection**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1804.00428](https://arxiv.org/abs/1804.00428)
- github: [https://github.com/Hwang64/MLKP](https://github.com/Hwang64/MLKP)

**Loss Rank Mining: A General Hard Example Mining Method for Real-time Detectors**

- intro: National University of Defense Technology
- arxiv: [https://arxiv.org/abs/1804.04606](https://arxiv.org/abs/1804.04606)

**DetNet: A Backbone network for Object Detection**

- intro: Tsinghua University & Megvii Inc
- arxiv: [https://arxiv.org/abs/1804.06215](https://arxiv.org/abs/1804.06215)

**AdvDetPatch: Attacking Object Detectors with Adversarial Patches**

[https://arxiv.org/abs/1806.02299](https://arxiv.org/abs/1806.02299)

**Attacking Object Detectors via Imperceptible Patches on Background**

[https://arxiv.org/abs/1809.05966](https://arxiv.org/abs/1809.05966)

**Physical Adversarial Examples for Object Detectors**

- intro: WOOT 2018
- arxiv: [https://arxiv.org/abs/1807.07769](https://arxiv.org/abs/1807.07769)

**Object detection at 200 Frames Per Second**

- intro: United Technologies Research Center-Ireland
- arxiv: [https://arxiv.org/abs/1805.06361](https://arxiv.org/abs/1805.06361)

**Object Detection using Domain Randomization and Generative Adversarial Refinement of Synthetic Images**

- intro: CVPR 2018 Deep Vision Workshop
- arxiv: [https://arxiv.org/abs/1805.11778](https://arxiv.org/abs/1805.11778)

**SNIPER: Efficient Multi-Scale Training**

- intro: University of Maryland
- keywords: SNIPER (Scale Normalization for Image Pyramid with Efficient Resampling)
- arxiv: [https://arxiv.org/abs/1805.09300](https://arxiv.org/abs/1805.09300)
- github: [https://github.com/mahyarnajibi/SNIPER](https://github.com/mahyarnajibi/SNIPER)

**Soft Sampling for Robust Object Detection**

[https://arxiv.org/abs/1806.06986](https://arxiv.org/abs/1806.06986)

**MetaAnchor: Learning to Detect Objects with Customized Anchors**

- intro: Megvii Inc (Face++) & Fudan University
- arxiv: [https://arxiv.org/abs/1807.00980](https://arxiv.org/abs/1807.00980)

**Localization Recall Precision (LRP): A New Performance Metric for Object Detection**

- intro: ECCV 2018. Middle East Technical University
- arxiv: [https://arxiv.org/abs/1807.01696](https://arxiv.org/abs/1807.01696)
- github: [https://github.com/cancam/LRP](https://github.com/cancam/LRP)

**Pooling Pyramid Network for Object Detection**

- intro: Google AI Perception
- arxiv: [https://arxiv.org/abs/1807.03284](https://arxiv.org/abs/1807.03284)

**Modeling Visual Context is Key to Augmenting Object Detection Datasets**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1807.07428](https://arxiv.org/abs/1807.07428)

**Acquisition of Localization Confidence for Accurate Object Detection**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1807.11590](https://arxiv.org/abs/1807.11590)
- gihtub: [https://github.com/vacancy/PreciseRoIPooling](https://github.com/vacancy/PreciseRoIPooling)

**CornerNet: Detecting Objects as Paired Keypoints**

- intro: ECCV 2018
- keywords: IoU-Net, PreciseRoIPooling
- arxiv: [https://arxiv.org/abs/1808.01244](https://arxiv.org/abs/1808.01244)
- github: [https://github.com/umich-vl/CornerNet](https://github.com/umich-vl/CornerNet)

**Unsupervised Hard Example Mining from Videos for Improved Object Detection**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1808.04285](https://arxiv.org/abs/1808.04285)

**SAN: Learning Relationship between Convolutional Features for Multi-Scale Object Detection**

[https://arxiv.org/abs/1808.04974](https://arxiv.org/abs/1808.04974)

**A Survey of Modern Object Detection Literature using Deep Learning**

[https://arxiv.org/abs/1808.07256](https://arxiv.org/abs/1808.07256)

**Tiny-DSOD: Lightweight Object Detection for Resource-Restricted Usages**

- intro: BMVC 2018
- arxiv: [https://arxiv.org/abs/1807.11013](https://arxiv.org/abs/1807.11013)
- github: [https://github.com/lyxok1/Tiny-DSOD](https://github.com/lyxok1/Tiny-DSOD)

**Deep Feature Pyramid Reconfiguration for Object Detection**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1808.07993](https://arxiv.org/abs/1808.07993)

**MDCN: Multi-Scale, Deep Inception Convolutional Neural Networks for Efficient Object Detection**

- intro: ICPR 2018
- arxiv: [https://arxiv.org/abs/1809.01791](https://arxiv.org/abs/1809.01791)

**Recent Advances in Object Detection in the Age of Deep Convolutional Neural Networks**

[https://arxiv.org/abs/1809.03193](https://arxiv.org/abs/1809.03193)

**Deep Learning for Generic Object Detection: A Survey**

[https://arxiv.org/abs/1809.02165](https://arxiv.org/abs/1809.02165)

**Training Confidence-Calibrated Classifier for Detecting Out-of-Distribution Samples**

- intro: ICLR 2018
- arxiv: [https://github.com/alinlab/Confident_classifier](https://github.com/alinlab/Confident_classifier)

**Fast and accurate object detection in high resolution 4K and 8K video using GPUs**

- intro: Best Paper Finalist at IEEE High Performance Extreme Computing Conference (HPEC) 2018
- intro: Carnegie Mellon University
- arxiv: [https://arxiv.org/abs/1810.10551](https://arxiv.org/abs/1810.10551)

**Hybrid Knowledge Routed Modules for Large-scale Object Detection**

- intro: NIPS 2018
- arxiv: [https://arxiv.org/abs/1810.12681](https://arxiv.org/abs/1810.12681)
- github(official, PyTorch): [https://github.com/chanyn/HKRM](https://github.com/chanyn/HKRM)

**BAN: Focusing on Boundary Context for Object Detection**

[https://arxiv.org/abs/1811.05243](https://arxiv.org/abs/1811.05243)

**R2CNN++: Multi-Dimensional Attention Based Rotation Invariant Detector with Robust Anchor Strategy**

- arxiv: [https://arxiv.org/abs/1811.07126](https://arxiv.org/abs/1811.07126)
- github: [https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow](https://github.com/DetectionTeamUCAS/R2CNN-Plus-Plus_Tensorflow)

**DeRPN: Taking a further step toward more general object detection**

- intro: AAAI 2019
- intro: South China University of Technology
- ariv: [https://arxiv.org/abs/1811.06700](https://arxiv.org/abs/1811.06700)
- github: [https://github.com/HCIILAB/DeRPN](https://github.com/HCIILAB/DeRPN)

**Fast Efficient Object Detection Using Selective Attention**

[https://arxiv.org/abs/1811.07502](https://arxiv.org/abs/1811.07502)

**Sampling Techniques for Large-Scale Object Detection from Sparsely Annotated Objects**

[https://arxiv.org/abs/1811.10862](https://arxiv.org/abs/1811.10862)

**Efficient Coarse-to-Fine Non-Local Module for the Detection of Small Objects**

[https://arxiv.org/abs/1811.12152](https://arxiv.org/abs/1811.12152)

**Deep Regionlets: Blended Representation and Deep Learning for Generic Object Detection**

[https://arxiv.org/abs/1811.11318](https://arxiv.org/abs/1811.11318)

**Transferable Adversarial Attacks for Image and Video Object Detection**

[https://arxiv.org/abs/1811.12641](https://arxiv.org/abs/1811.12641)

**Anchor Box Optimization for Object Detection**

- intro: University of Illinois at Urbana-Champaign & Microsoft Research
- arxiv: [https://arxiv.org/abs/1812.00469](https://arxiv.org/abs/1812.00469)

**AutoFocus: Efficient Multi-Scale Inference**

- intro: University of Maryland
- arxiv: [https://arxiv.org/abs/1812.01600](https://arxiv.org/abs/1812.01600)

**Few-shot Object Detection via Feature Reweighting**

[https://arxiv.org/abs/1812.01866](https://arxiv.org/abs/1812.01866)

**Practical Adversarial Attack Against Object Detector**

[https://arxiv.org/abs/1812.10217](https://arxiv.org/abs/1812.10217)

**Scale-Aware Trident Networks for Object Detection**

- intro: University of Chinese Academy of Sciences & TuSimple
- arxiv: [https://arxiv.org/abs/1901.01892](https://arxiv.org/abs/1901.01892)
- github: [https://github.com/TuSimple/simpledet](https://github.com/TuSimple/simpledet)

**Region Proposal by Guided Anchoring**

- intro: CVPR 2019
- intro: CUHK - SenseTime Joint Lab & Amazon Rekognition & Nanyang Technological University
- arxiv: [https://arxiv.org/abs/1901.03278](https://arxiv.org/abs/1901.03278)

**Bottom-up Object Detection by Grouping Extreme and Center Points**

- keywords: ExtremeNet
- arxiv: [https://arxiv.org/abs/1901.08043](https://arxiv.org/abs/1901.08043)
- github: [https://github.com/xingyizhou/ExtremeNet](https://github.com/xingyizhou/ExtremeNet)

**Bag of Freebies for Training Object Detection Neural Networks**

- intro: Amazon Web Services
- arxiv: [https://arxiv.org/abs/1902.04103](https://arxiv.org/abs/1902.04103)

**Augmentation for small object detection**

[https://arxiv.org/abs/1902.07296](https://arxiv.org/abs/1902.07296)

**Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1902.09630](https://arxiv.org/abs/1902.09630)

**SimpleDet: A Simple and Versatile Distributed Framework for Object Detection and Instance Recognition**

- intro: TuSimple
- arxiv: [https://arxiv.org/abs/1903.05831](https://arxiv.org/abs/1903.05831)
- github: [https://github.com/tusimple/simpledet](https://github.com/tusimple/simpledet)

**BayesOD: A Bayesian Approach for Uncertainty Estimation in Deep Object Detectors**

- intro: University of Toronto
- arxiv: [https://arxiv.org/abs/1903.03838](https://arxiv.org/abs/1903.03838)

**DetNAS: Neural Architecture Search on Object Detection**

- intro: Chinese Academy of Sciences & Megvii Inc
- arxiv: [https://arxiv.org/abs/1903.10979](https://arxiv.org/abs/1903.10979)

**ThunderNet: Towards Real-time Generic Object Detection**

[https://arxiv.org/abs/1903.11752](https://arxiv.org/abs/1903.11752)

**Feature Intertwiner for Object Detection**

- intro: ICLR 2019
- intro: CUHK & SenseTime & The University of Sydney
- arxiv: [https://arxiv.org/abs/1903.11851](https://arxiv.org/abs/1903.11851)

**Improving Object Detection with Inverted Attention**

[https://arxiv.org/abs/1903.12255](https://arxiv.org/abs/1903.12255)

**What Object Should I Use? - Task Driven Object Detection**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1904.03000](https://arxiv.org/abs/1904.03000)

**Towards Universal Object Detection by Domain Attention**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1904.04402](https://arxiv.org/abs/1904.04402)

**Prime Sample Attention in Object Detection**

[https://arxiv.org/abs/1904.04821](https://arxiv.org/abs/1904.04821)

**BAOD: Budget-Aware Object Detection**

[https://arxiv.org/abs/1904.05443](https://arxiv.org/abs/1904.05443)

**An Analysis of Pre-Training on Object Detection**

- intro: University of Maryland
- arxiv: [https://arxiv.org/abs/1904.05871](https://arxiv.org/abs/1904.05871)

**DuBox: No-Prior Box Objection Detection via Residual Dual Scale Detectors**

- intro: Baidu Inc.
- arxiv: [https://arxiv.org/abs/1904.06883](https://arxiv.org/abs/1904.06883)

**NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection**

- intro: CVPR 2019
- intro: Google Brain
- arxiv: [https://arxiv.org/abs/1904.07392](https://arxiv.org/abs/1904.07392)

**Objects as Points**

![](https://raw.githubusercontent.com/xingyizhou/CenterNet/master/readme/fig2.png)

- intro: Object detection, 3D detection, and pose estimation using center point detection
- arxiv: [https://arxiv.org/abs/1904.07850](https://arxiv.org/abs/1904.07850)
- github: [https://github.com/xingyizhou/CenterNet](https://github.com/xingyizhou/CenterNet)

**MultiTask-CenterNet (MCN): Efficient and Diverse Multitask Learning using an Anchor Free Approach**

- intro: ICCV 2021
- intro: ZF Friedrichshafen AG, Artificial Intelligence Lab
- arxiv: [https://arxiv.org/abs/2108.05060](https://arxiv.org/abs/2108.05060)

**CenterNet: Object Detection with Keypoint Triplets**

**CenterNet: Keypoint Triplets for Object Detection**

- arxiv: [https://arxiv.org/abs/1904.08189](https://arxiv.org/abs/1904.08189)
- github: [https://github.com/Duankaiwen/CenterNet](https://github.com/Duankaiwen/CenterNet)

**CornerNet-Lite: Efficient Keypoint Based Object Detection**

- intro: Princeton University
- arxiv: [https://arxiv.org/abs/1904.08900](https://arxiv.org/abs/1904.08900)
- github: [https://github.com/princeton-vl/CornerNet-Lite](https://github.com/princeton-vl/CornerNet-Lite)

**CenterNet++ for Object Detection**

- arxiv: [https://arxiv.org/abs/2204.08394](https://arxiv.org/abs/2204.08394)
- github; [https://github.com/Duankaiwen/PyCenterNet](https://github.com/Duankaiwen/PyCenterNet)

**Automated Focal Loss for Image based Object Detection**

[https://arxiv.org/abs/1904.09048](https://arxiv.org/abs/1904.09048)

**Exploring Object Relation in Mean Teacher for Cross-Domain Detection**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1904.11245](https://arxiv.org/abs/1904.11245)

**An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection**

- intro: CVPR 2019 CEFRL Workshop
- arxiv: [https://arxiv.org/abs/1904.09730](https://arxiv.org/abs/1904.09730)

**RepPoints: Point Set Representation for Object Detection**

- intro: ICCV 2019
- intro: Peking University & Tsinghua University & Microsoft Research Asia
- arxiv: [https://arxiv.org/abs/1904.11490](https://arxiv.org/abs/1904.11490)
- github: [https://github.com/microsoft/RepPoints](https://github.com/microsoft/RepPoints)

**Dense RepPoints: Representing Visual Objects with Dense Point Sets**

- intro: Peking University & CUHK & Zhejiang University & Shanghai Jiao Tong University & University of Toronto & MSRA
- arxiv: [https://arxiv.org/abs/1912.11473](https://arxiv.org/abs/1912.11473)
- github(official, mmdetection): [https://github.com/justimyhxu/Dense-RepPoints](https://github.com/justimyhxu/Dense-RepPoints)

**RepPoints V2: Verification Meets Regression for Object Detection**

- intro: Microsoft Research Asia & Peking University
- arxiv: [https://arxiv.org/abs/2007.08508](https://arxiv.org/abs/2007.08508)
- github(official, mmdetection): [https://github.com/Scalsol/RepPointsV2](https://github.com/Scalsol/RepPointsV2)

**Object Detection in 20 Years: A Survey**

[https://arxiv.org/abs/1905.05055](https://arxiv.org/abs/1905.05055)

**Light-Weight RetinaNet for Object Detection**

[https://arxiv.org/abs/1905.10011](https://arxiv.org/abs/1905.10011)

**Learning Data Augmentation Strategies for Object Detection**

- intro: Google Research, Brain Team
- arxiv: [https://arxiv.org/abs/1906.11172](https://arxiv.org/abs/1906.11172)
- github: [https://github.com/tensorflow/tpu/tree/master/models/official/detection](https://github.com/tensorflow/tpu/tree/master/models/official/detection)

**Towards Adversarially Robust Object Detection**

- intro: ICCV 2019
- intro: Baidu Research, Sunnyvale USA
- arxiv: [https://arxiv.org/abs/1907.10310](https://arxiv.org/abs/1907.10310)

**Multi-adversarial Faster-RCNN for Unrestricted Object Detection**

- intro: ICCV 2019
- arxiv: [https://arxiv.org/abs/1907.10343](https://arxiv.org/abs/1907.10343)

**Object as Distribution**

- intro: NeurIPS 2019
- intro: MIT
- arxiv: [https://arxiv.org/abs/1907.12929](https://arxiv.org/abs/1907.12929)

**Detecting 11K Classes: Large Scale Object Detection without Fine-Grained Bounding Boxes**

- intro: ICCV 2019
- arxiv: [https://arxiv.org/abs/1908.05217](https://arxiv.org/abs/1908.05217)

**R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object**

- arxiv: [https://arxiv.org/abs/1908.05612](https://arxiv.org/abs/1908.05612)
- github: [https://github.com/Thinklab-SJTU/R3Det_Tensorflow](https://github.com/Thinklab-SJTU/R3Det_Tensorflow)

**SCRDet++: Detecting Small, Cluttered and Rotated Objects via Instance-Level Feature Denoising and Rotation Loss Smoothing**

- project page: [https://yangxue0827.github.io/SCRDet++.html](https://yangxue0827.github.io/SCRDet++.html)
- arxiv: [https://arxiv.org/abs/2004.13316](https://arxiv.org/abs/2004.13316)

**Relation Distillation Networks for Video Object Detection**

- intro: ICCV 2019
- arxiv: [https://arxiv.org/abs/1908.09511](https://arxiv.org/abs/1908.09511)

**Imbalance Problems in Object Detection: A Review**

- arxiv: [https://arxiv.org/abs/1909.00169](https://arxiv.org/abs/1909.00169)
- github: [https://github.com/kemaloksuz/ObjectDetectionImbalance](https://github.com/kemaloksuz/ObjectDetectionImbalance)

**FreeAnchor: Learning to Match Anchors for Visual Object Detection**

- intro: NeurIPS 2019
- arxiv: [https://arxiv.org/abs/1909.02466](https://arxiv.org/abs/1909.02466)

**Efficient Neural Architecture Transformation Search in Channel-Level for Object Detection**

[https://arxiv.org/abs/1909.02293](https://arxiv.org/abs/1909.02293)

**Self-Training and Adversarial Background Regularization for Unsupervised Domain Adaptive One-Stage Object Detection**

- intro: ICCV 2019 oral
- arxiv: [https://arxiv.org/abs/1909.00597](https://arxiv.org/abs/1909.00597)

**CBNet: A Novel Composite Backbone Network Architecture for Object Detection**

- intro: AAAI 2020
- keywords: Composite Backbone Network (CBNet)
- arxiv: [https://arxiv.org/abs/1909.03625](https://arxiv.org/abs/1909.03625)
- paper: [https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuY.1833.pdf](https://aaai.org/Papers/AAAI/2020GB/AAAI-LiuY.1833.pdf)
- github(Caffe2): [https://github.com/PKUbahuangliuhe/CBNet](https://github.com/PKUbahuangliuhe/CBNet)
- github(mmdetection): [https://github.com/VDIGPKU/CBNet](https://github.com/VDIGPKU/CBNet)

**CBNetV2: A Composite Backbone Network Architecture for Object Detection**

- arxiv: [https://arxiv.org/abs/2107.00420](https://arxiv.org/abs/2107.00420)
- github: [https://github.com/VDIGPKU/CBNetV2](https://github.com/VDIGPKU/CBNetV2)

**A System-Level Solution for Low-Power Object Detection**

- intro: ICCV 2019 Low-Power Computer Vision Workshop
- arxiv: [https://arxiv.org/abs/1909.10964](https://arxiv.org/abs/1909.10964)

**Anchor Loss: Modulating Loss Scale based on Prediction Difficulty**

- intro: ICCV 2019 oral
- arxiv: [https://arxiv.org/abs/1909.11155](https://arxiv.org/abs/1909.11155)
- github(Pytorch): [https://github.com/slryou41/AnchorLoss](https://github.com/slryou41/AnchorLoss)

**Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression**

- intro: AAAI 2020
- arxiv: [https://arxiv.org/abs/1911.08287](https://arxiv.org/abs/1911.08287)
- github: [https://github.com/Zzh-tju/DIoU](https://github.com/Zzh-tju/DIoU)
- github: [https://github.com/Zzh-tju/CIoU](https://github.com/Zzh-tju/CIoU)
- github: [https://github.com/Zzh-tju/DIoU-darknet](https://github.com/Zzh-tju/DIoU-darknet)

**Curriculum Self-Paced Learning for Cross-Domain Object Detection**

[https://arxiv.org/abs/1911.06849](https://arxiv.org/abs/1911.06849)

**Multiple Anchor Learning for Visual Object Detection**

[https://arxiv.org/abs/1912.02252](https://arxiv.org/abs/1912.02252)

**MnasFPN: Learning Latency-aware Pyramid Architecture for Object Detection on Mobile Devices**

- intro: Google AI & Google Brain
- arxiv: [https://arxiv.org/abs/1912.01106](https://arxiv.org/abs/1912.01106)

**AugFPN: Improving Multi-scale Feature Learning for Object Detection**

- intro: CVPR 2020
- intro: CASIA & Horizon Robotics
- arxiv: [https://arxiv.org/abs/1912.05384](https://arxiv.org/abs/1912.05384)
- github(official, mmdetection): [https://github.com/Gus-Guo/AugFPN](https://github.com/Gus-Guo/AugFPN)

**Object Detection as a Positive-Unlabeled Problem**

[https://arxiv.org/abs/2002.04672](https://arxiv.org/abs/2002.04672)

**Universal-RCNN: Universal Object Detector via Transferable Graph R-CNN**

- intro: AAAI 2020
- intro: Huawei Noah’s Ark Lab & South China University of Technology & Sun Yat-Sen University
- arxiv: [https://arxiv.org/abs/2002.07417](https://arxiv.org/abs/2002.07417)

**BiDet: An Efficient Binarized Object Detector**

- intro: CVPR 2020
- arxiv: [https://arxiv.org/abs/2003.03961](https://arxiv.org/abs/2003.03961)
- github: [https://github.com/ZiweiWangTHU/BiDet](https://github.com/ZiweiWangTHU/BiDet)

**Revisiting the Sibling Head in Object Detector**

- intro: CVPR 2020 & Method of Champion of OpenImage Challenge 2019, detection track
- intro: SenseTime X-Lab & CUHK
- keywords: task-aware spatial disentanglement (TSD)
- arxiv: [https://arxiv.org/abs/2003.07540](https://arxiv.org/abs/2003.07540)
- github: [https://github.com/Sense-X/TSD](https://github.com/Sense-X/TSD)

**Extended Feature Pyramid Network for Small Object Detection**

[https://arxiv.org/abs/2003.07021](https://arxiv.org/abs/2003.07021)

**SaccadeNet: A Fast and Accurate Object Detector**

- intro: University of Maryland & Wormpex AI Research
- arxiv: [https://arxiv.org/abs/2003.12125](https://arxiv.org/abs/2003.12125)

**Scale-Equalizing Pyramid Convolution for Object Detection**

- intro: CVPR 2020
- intro: SenseTime Research
- arxiv: [https://arxiv.org/abs/2005.03101](https://arxiv.org/abs/2005.03101)
- github: [https://github.com/jshilong/SEPC](https://github.com/jshilong/SEPC)

**Dynamic Refinement Network for Oriented and Densely Packed Object Detection**

- intro: CVPR 2020 oral
- keywords: SKU110K-R
- arxiv: [https://arxiv.org/abs/2005.09973](https://arxiv.org/abs/2005.09973)
- github: [https://github.com/Anymake/DRN_CVPR2020](https://github.com/Anymake/DRN_CVPR2020)

**Robust Object Detection under Occlusion with Context-Aware CompositionalNets**

- intro: CVPR 2020
- intro: Johns Hopkins University
- arxiv: [https://arxiv.org/abs/2005.11643](https://arxiv.org/abs/2005.11643)

**DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution**

- intro: Johns Hopkins University & Google Research
- intro: COCO test-dev 54.7% box AP
- arxiv: [https://arxiv.org/abs/2006.02334](https://arxiv.org/abs/2006.02334)
- github(official, mmdetection): [https://github.com/joe-siyuan-qiao/DetectoRS](https://github.com/joe-siyuan-qiao/DetectoRS)

**Learning a Unified Sample Weighting Network for Object Detection**

- intro: CVPR 2020
- arxiv: [https://arxiv.org/abs/2006.06568](https://arxiv.org/abs/2006.06568)
- github: [https://github.com/caiqi/sample-weighting-network](https://github.com/caiqi/sample-weighting-network)

**2nd Place Solution for Waymo Open Dataset Challenge -- 2D Object Detection**

- intro: Horizon Robotics Inc.
- arxiv: [https://arxiv.org/abs/2006.15507](https://arxiv.org/abs/2006.15507)

**Domain Adaptive Object Detection via Asymmetric Tri-way Faster-RCNN**

- intro: ECCV 2020
- arxiv: [https://arxiv.org/abs/2007.01571](https://arxiv.org/abs/2007.01571)

**AQD: Towards Accurate Quantized Object Detection**

- intro: South China University of Technology & University of Adelaide & Monash University
- arxiv: [https://arxiv.org/abs/2007.06919](https://arxiv.org/abs/2007.06919)
- github: [https://github.com/blueardour/model-quantization](https://github.com/blueardour/model-quantization)

**Probabilistic Anchor Assignment with IoU Prediction for Object Detection**

- intro: ECCV 2020
- arxiv: [https://arxiv.org/abs/2007.08103](https://arxiv.org/abs/2007.08103)
- github: [https://github.com/kkhoot/PAA](https://github.com/kkhoot/PAA)

**BorderDet: Border Feature for Dense Object Detection**

- intro: ECCV 2020 oral
- arxiv: [https://arxiv.org/abs/2007.11056](https://arxiv.org/abs/2007.11056)
- github: [https://github.com/Megvii-BaseDetection/BorderDet](https://github.com/Megvii-BaseDetection/BorderDet)

**Quantum-soft QUBO Suppression for Accurate Object Detection**

- intro: ECCV 2020
- arxiv: [https://arxiv.org/abs/2007.13992](https://arxiv.org/abs/2007.13992)

**VarifocalNet: An IoU-aware Dense Object Detector**

- intro: Queensland University of Technology & University of Queensland
- arxiv: [https://arxiv.org/abs/2008.13367](https://arxiv.org/abs/2008.13367)
- github: [https://github.com/hyz-xmaster/VarifocalNet](https://github.com/hyz-xmaster/VarifocalNet)

**The 1st Tiny Object Detection Challenge:Methods and Results**

- intro: ECCV2020 Workshop on Real-world Computer Vision from Inputs with Limited Quality (RLQ) and Tiny Object Detection Challenge
- arxiv: [https://arxiv.org/abs/2009.07506](https://arxiv.org/abs/2009.07506)

**MimicDet: Bridging the Gap Between One-Stage and Two-Stage Object Detection**

- intro: ECCV 2020
- intro: SenseTime & CUHK
- arxiv: [https://arxiv.org/abs/2009.11528](https://arxiv.org/abs/2009.11528)

**SEA: Bridging the Gap Between One- and Two-stage Detector Distillation via SEmantic-aware Alignment**

- intro: The Chinese University of Hong Kong & SmartMore
- arxiv: [https://arxiv.org/abs/2203.00862](https://arxiv.org/abs/2203.00862)

**A Ranking-based, Balanced Loss Function Unifying Classification and Localisation in Object Detection**

- intro: NeurIPS 2020 spotlight
- intro: Middle East Technical University
- keywords: average Localization-Recall-Precision (aLRP)
- arxiv: [https://arxiv.org/abs/2009.13592](https://arxiv.org/abs/2009.13592)
- github(official, Pytorch): [https://github.com/kemaloksuz/aLRPLoss](https://github.com/kemaloksuz/aLRPLoss)

**Effective Fusion Factor in FPN for Tiny Object Detection**

- intro: WACV 2021
- arxiv: [https://arxiv.org/abs/2011.02298](https://arxiv.org/abs/2011.02298)

**Bi-Dimensional Feature Alignment for Cross-Domain Object Detection**

- intro: ECCV 2020 TASK-CV Workshop
- arxiv: [https://arxiv.org/abs/2011.07205](https://arxiv.org/abs/2011.07205)

**Rethinking Transformer-based Set Prediction for Object Detection**

- intro: Carnegie Mellon University
- arxiv: [https://arxiv.org/abs/2011.10881](https://arxiv.org/abs/2011.10881)

**Unsupervised Object Detection with LiDAR Clues**

- intro: SenseTime & USTC & CASIA & CAS
- arxiv: [https://arxiv.org/abs/2011.12953](https://arxiv.org/abs/2011.12953)

**Self-EMD: Self-Supervised Object Detection without ImageNet**

- intro: MEGVII Technology
- arxiv: [https://arxiv.org/abs/2011.13677](https://arxiv.org/abs/2011.13677)

**End-to-End Object Detection with Fully Convolutional Network**

- intro: Megvii Technology & Xi’an Jiaotong University
- keywords: Prediction-aware One- To-One (POTO) label assignment, 3D Max Filtering (3DMF)
- arxiv: [https://arxiv.org/abs/2012.03544](https://arxiv.org/abs/2012.03544)
- github: [https://github.com/Megvii-BaseDetection/DeFCN](https://github.com/Megvii-BaseDetection/DeFCN)

**Fine-Grained Dynamic Head for Object Detection**

- intro: NeurIPS 2020
- arxiv: [https://arxiv.org/abs/2012.03519](https://arxiv.org/abs/2012.03519)
- github: [https://github.com/StevenGrove/DynamicHead](https://github.com/StevenGrove/DynamicHead)

**Focal and Efficient IOU Loss for Accurate Bounding Box Regression**

- intro: South China University of Technology & 2Horizon Robotics & Chinese Academy of Sciences
- arxiv: [https://arxiv.org/abs/2101.08158](https://arxiv.org/abs/2101.08158)

**Scale Normalized Image Pyramids with AutoFocus for Object Detection**

- intro: T-PAMI 2021
- arxiv: [https://arxiv.org/abs/2102.05646](https://arxiv.org/abs/2102.05646)
- github: [https://github.com/mahyarnajibi/SNIPER](https://github.com/mahyarnajibi/SNIPER)

**DetCo: Unsupervised Contrastive Learning for Object Detection**

- intro: The University of Hong Kong & Huawei Noah’s Ark Lab & Wuhan University & Nanjing University & Chinese University of Hong Kong
- arxiv: [https://arxiv.org/abs/2102.04803](https://arxiv.org/abs/2102.04803)
- github: [https://github.com/xieenze/DetCo](https://github.com/xieenze/DetCo)
- github: [https://github.com/open-mmlab/OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup)

**RMOPP: Robust Multi-Objective Post-Processing for Effective Object Detection**

[https://arxiv.org/abs/2102.04582](https://arxiv.org/abs/2102.04582)

**Instance Localization for Self-supervised Detection Pretraining**

- intro: Chinese University of Hong Kong & Microsoft Research Asia
- arxiv: [https://arxiv.org/abs/2102.08318](https://arxiv.org/abs/2102.08318)

**Localization Distillation for Object Detection**

- arxiv: [https://arxiv.org/abs/2102.12252](https://arxiv.org/abs/2102.12252)
- github: [https://github.com/HikariTJU/LD](https://github.com/HikariTJU/LD)

**General Instance Distillation for Object Detection**

- intro: CVPR 2021
- arxiv: [https://arxiv.org/abs/2103.02340](https://arxiv.org/abs/2103.02340)

**Towards Open World Object Detection**

- intro: CVPR 2021 oral
- arxiv: [https://arxiv.org/abs/2103.02603](https://arxiv.org/abs/2103.02603)
- github: [https://github.com/JosephKJ/OWOD](https://github.com/JosephKJ/OWOD)

**Data Augmentation for Object Detection via Differentiable Neural Rendering**

- arxiv: [https://arxiv.org/abs/2103.02852](https://arxiv.org/abs/2103.02852)
- github: [https://github.com/Guanghan/DANR](https://github.com/Guanghan/DANR)

**Revisiting the Loss Weight Adjustment in Object Detection**

- intro: University of Science and Technology of China & University of Michigan
- arxiv: [https://arxiv.org/abs/2103.09488](https://arxiv.org/abs/2103.09488)
- github: [https://github.com/ywx-hub/ALWA](https://github.com/ywx-hub/ALWA)

**You Only Look One-level Feature**

- intro: CVPR 2021
- arxiv: [https://arxiv.org/abs/2103.09460](https://arxiv.org/abs/2103.09460)
- github: [https://github.com/megvii-model/YOLOF](https://github.com/megvii-model/YOLOF)

**Optimization for Oriented Object Detection via Representation Invariance Loss**

- arxiv: [https://arxiv.org/abs/2103.11636](https://arxiv.org/abs/2103.11636)
- github: [https://github.com/ming71/RIDet](https://github.com/ming71/RIDet)

**Dynamic Anchor Learning for Arbitrary-Oriented Object Detection**

- intro: AAAI 2021
- arxiv: [https://arxiv.org/abs/2012.04150](https://arxiv.org/abs/2012.04150)
- github: [https://github.com/ming71/DAL](https://github.com/ming71/DAL)

**Control Distance IoU and Control Distance IoU Loss Function for Better Bounding Box Regression**

[https://arxiv.org/abs/2103.11696](https://arxiv.org/abs/2103.11696)

**OTA: Optimal Transport Assignment for Object Detection**

- intro: CVPR 2021
- arxiv: [https://arxiv.org/abs/2103.14259](https://arxiv.org/abs/2103.14259)
- github: [https://github.com/Megvii-BaseDetection/OTA](https://github.com/Megvii-BaseDetection/OTA)

**Distilling Object Detectors via Decoupled Features**

- intro: CVPR 2021
- arxiv: [https://arxiv.org/abs/2103.14475](https://arxiv.org/abs/2103.14475)
- github: [https://github.com/ggjy/DeFeat.pytorch](https://github.com/ggjy/DeFeat.pytorch)

**Distilling a Powerful Student Model via Online Knowledge Distillation**

- arxiv: [https://arxiv.org/abs/2103.14473](https://arxiv.org/abs/2103.14473)
- github: [https://github.com/SJLeo/FFSD](https://github.com/SJLeo/FFSD)

**IQDet: Instance-wise Quality Distribution Sampling for Object Detection**

- intro: CVPR 2021
- intro: Megvii Technology
- arxiv: [https://arxiv.org/abs/2104.06936](https://arxiv.org/abs/2104.06936)

**You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection**

- intro: Huazhong University of Science & Technology, Horizon Robotics
- arxiv: [https://arxiv.org/abs/2106.00666](https://arxiv.org/abs/2106.00666)
- github: [https://github.com/hustvl/YOLOS](https://github.com/hustvl/YOLOS)

**Augmenting Anchors by the Detector Itself**

[https://arxiv.org/abs/2105.14086](https://arxiv.org/abs/2105.14086)

**Rethinking Training from Scratch for Object Detection**

- intro: Zhejiang University
- arxiv: [https://arxiv.org/abs/2106.03112](https://arxiv.org/abs/2106.03112)

**Dynamic Head: Unifying Object Detection Heads with Attentions**

- intro: CVPR 2021
- intro: Microsoft
- arxiv: [https://arxiv.org/abs/2106.08322](https://arxiv.org/abs/2106.08322)
- github: [https://github.com/microsoft/DynamicHead](https://github.com/microsoft/DynamicHead)

**Disentangle Your Dense Object Detector**

- intro: ACM MM 2021
- arxiv: [https://arxiv.org/abs/2107.02963](https://arxiv.org/abs/2107.02963)
- github: [https://github.com/zehuichen123/DDOD](https://github.com/zehuichen123/DDOD)

**Improving Object Detection by Label Assignment Distillation**

- arxiv: [https://arxiv.org/abs/2108.10520](https://arxiv.org/abs/2108.10520)
- github: [https://github.com/cybercore-co-ltd/CoLAD_paper](https://github.com/cybercore-co-ltd/CoLAD_paper)

**Progressive Hard-case Mining across Pyramid Levels in Object Detection**

- intro:  Baidu Inc.
- arxiv: [https://arxiv.org/abs/2109.07217](https://arxiv.org/abs/2109.07217)
- github: [https://github.com/zimoqingfeng/UMOP](https://github.com/zimoqingfeng/UMOP)

**Multi-Scale Aligned Distillation for Low-Resolution Detection**

- intro: CVPR 2021
- intro: The Chinese University of Hong Kong & Adobe Research & SmartMore
- arxiv: [https://arxiv.org/abs/2109.06875](https://arxiv.org/abs/2109.06875)
- github: [https://github.com/dvlab-research/MSAD](https://github.com/dvlab-research/MSAD)

**Pix2seq: A Language Modeling Framework for Object Detection**

- intro: Google Research, Brain Team
- arxiv: [https://arxiv.org/abs/2109.10852](https://arxiv.org/abs/2109.10852)

**Mixed Supervised Object Detection by Transferring Mask Prior and Semantic Similarity**

- intro: NeurIPS 2021
- intro: Shanghai Jiao Tong University
- arxiv: [https://arxiv.org/abs/2110.14191](https://arxiv.org/abs/2110.14191)
- github: [https://github.com/bcmi/TraMaS-Weak-Shot-Object-Detection](https://github.com/bcmi/TraMaS-Weak-Shot-Object-Detection)

**Bootstrap Your Object Detector via Mixed Training**

- intro: NeurIPS 2021 Spotlight
- intro: Huazhong University of Science and Technology & Xi’an Jiaotong University & Microsoft Research Asia
- keywords: MixTraining
- arxiv: [https://arxiv.org/abs/2111.03056](https://arxiv.org/abs/2111.03056)
- github: [https://github.com/MendelXu/MixTraining](https://github.com/MendelXu/MixTraining)

**PP-PicoDet: A Better Real-Time Object Detector on Mobile Devices**

- intro: Baidu Inc.
- arxiv: [https://arxiv.org/abs/2111.00902](https://arxiv.org/abs/2111.00902)
- github: [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

**Toward Minimal Misalignment at Minimal Cost in One-Stage and Anchor-Free Object Detection**

[https://arxiv.org/abs/2112.08902](https://arxiv.org/abs/2112.08902)

**GiraffeDet: A Heavy-Neck Paradigm for Object Detection**

[https://arxiv.org/abs/2202.04256](https://arxiv.org/abs/2202.04256)

**A Dual Weighting Label Assignment Scheme for Object Detection**

- intro: CVPR 2022
- arxiv: [https://arxiv.org/abs/2203.09730](https://arxiv.org/abs/2203.09730)
- github: [https://github.com/strongwolf/DW](https://github.com/strongwolf/DW)

**QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection**

- intro: CVPR 2022
- arxiv: [https://arxiv.org/abs/2103.09136](https://arxiv.org/abs/2103.09136)
- github: [https://github.com/ChenhongyiYang/QueryDet-PyTorch](https://github.com/ChenhongyiYang/QueryDet-PyTorch)

# Two-Stage Object Detection

## R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation**

- intro: R-CNN
- arxiv: [http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- supp: [http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf](http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf)
- slides: [http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf](http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- github: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- notes: [http://zhangliliang.com/2014/07/23/paper-note-rcnn/](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)
- caffe-pr("Make R-CNN the Caffe detection example"): [https://github.com/BVLC/caffe/pull/482](https://github.com/BVLC/caffe/pull/482) 

## Fast R-CNN

**Fast R-CNN**

- arxiv: [http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- github: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- github(COCO-branch): [https://github.com/rbgirshick/fast-rcnn/tree/coco](https://github.com/rbgirshick/fast-rcnn/tree/coco)
- webcam demo: [https://github.com/rbgirshick/fast-rcnn/pull/29](https://github.com/rbgirshick/fast-rcnn/pull/29)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/](http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/)
- notes: [http://blog.csdn.net/linj_m/article/details/48930179](http://blog.csdn.net/linj_m/article/details/48930179)
- github("Fast R-CNN in MXNet"): [https://github.com/precedenceguo/mx-rcnn](https://github.com/precedenceguo/mx-rcnn)
- github: [https://github.com/mahyarnajibi/fast-rcnn-torch](https://github.com/mahyarnajibi/fast-rcnn-torch)
- github: [https://github.com/apple2373/chainer-simple-fast-rnn](https://github.com/apple2373/chainer-simple-fast-rnn)
- github: [https://github.com/zplizzi/tensorflow-fast-rcnn](https://github.com/zplizzi/tensorflow-fast-rcnn)

**A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1704.03414](https://arxiv.org/abs/1704.03414)
- paper: [http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf](http://abhinavsh.info/papers/pdfs/adversarial_object_detection.pdf)
- github(Caffe): [https://github.com/xiaolonw/adversarial-frcnn](https://github.com/xiaolonw/adversarial-frcnn)

## Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

- intro: NIPS 2015
- arxiv: [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- gitxiv: [http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region](http://www.gitxiv.com/posts/8pfpcvefDYn2gSgXk/faster-r-cnn-towards-real-time-object-detection-with-region)
- slides: [http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf](http://web.cs.hacettepe.edu.tr/~aykut/classes/spring2016/bil722/slides/w05-FasterR-CNN.pdf)
- github(official, Matlab): [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- github: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- github(MXNet): [https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn](https://github.com/msracver/Deformable-ConvNets/tree/master/faster_rcnn)
- github: [https://github.com//jwyang/faster-rcnn.pytorch](https://github.com//jwyang/faster-rcnn.pytorch)
- github: [https://github.com/mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn)
- github: [https://github.com/andreaskoepf/faster-rcnn.torch](https://github.com/andreaskoepf/faster-rcnn.torch)
- github: [https://github.com/ruotianluo/Faster-RCNN-Densecap-torch](https://github.com/ruotianluo/Faster-RCNN-Densecap-torch)
- github: [https://github.com/smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)
- github: [https://github.com/CharlesShang/TFFRCNN](https://github.com/CharlesShang/TFFRCNN)
- github(C++ demo): [https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus](https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus)
- github: [https://github.com/yhenon/keras-frcnn](https://github.com/yhenon/keras-frcnn)
- github: [https://github.com/Eniac-Xie/faster-rcnn-resnet](https://github.com/Eniac-Xie/faster-rcnn-resnet)
- github(C++): [https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev](https://github.com/D-X-Y/caffe-faster-rcnn/tree/dev)

**R-CNN minus R**

- intro: BMVC 2015
- arxiv: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)

**Faster R-CNN in MXNet with distributed implementation and data parallelization**

- github: [https://github.com/dmlc/mxnet/tree/master/example/rcnn](https://github.com/dmlc/mxnet/tree/master/example/rcnn)

**Contextual Priming and Feedback for Faster R-CNN**

- intro: ECCV 2016. Carnegie Mellon University
- paper: [http://abhinavsh.info/context_priming_feedback.pdf](http://abhinavsh.info/context_priming_feedback.pdf)
- poster: [http://www.eccv2016.org/files/posters/P-1A-20.pdf](http://www.eccv2016.org/files/posters/P-1A-20.pdf)

**An Implementation of Faster RCNN with Study for Region Sampling**

- intro: Technical Report, 3 pages. CMU
- arxiv: [https://arxiv.org/abs/1702.02138](https://arxiv.org/abs/1702.02138)
- github: [https://github.com/endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)

**Interpretable R-CNN**

- intro: North Carolina State University & Alibaba
- keywords: AND-OR Graph (AOG)
- arxiv: [https://arxiv.org/abs/1711.05226](https://arxiv.org/abs/1711.05226)

**Light-Head R-CNN: In Defense of Two-Stage Object Detector**

- intro: Tsinghua University & Megvii Inc
- arxiv: [https://arxiv.org/abs/1711.07264](https://arxiv.org/abs/1711.07264)
- github(official, Tensorflow): [https://github.com/zengarden/light_head_rcnn](https://github.com/zengarden/light_head_rcnn)
- github: [https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784](https://github.com/terrychenism/Deformable-ConvNets/blob/master/rfcn/symbols/resnet_v1_101_rfcn_light.py#L784)

**Cascade R-CNN: Delving into High Quality Object Detection**

- intro: CVPR 2018. UC San Diego
- arxiv: [https://arxiv.org/abs/1712.00726](https://arxiv.org/abs/1712.00726)
- github(Caffe, official): [https://github.com/zhaoweicai/cascade-rcnn](https://github.com/zhaoweicai/cascade-rcnn)

**Cascade R-CNN: High Quality Object Detection and Instance Segmentation**

 -arxiv: [https://arxiv.org/abs/1906.09756](https://arxiv.org/abs/1906.09756)
- github(Caffe, official): [https://github.com/zhaoweicai/cascade-rcnn](https://github.com/zhaoweicai/cascade-rcnn)
- github(official): [https://github.com/zhaoweicai/Detectron-Cascade-RCNN](https://github.com/zhaoweicai/Detectron-Cascade-RCNN)

**Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution**

- intro: NeurIPS 2019 spotlight
- arxiv: [https://arxiv.org/abs/1909.06720](https://arxiv.org/abs/1909.06720)
- github: [https://github.com/thangvubk/Cascade-RPN](https://github.com/thangvubk/Cascade-RPN)

**SMC Faster R-CNN: Toward a scene-specialized multi-object detector**

[https://arxiv.org/abs/1706.10217](https://arxiv.org/abs/1706.10217)

**Domain Adaptive Faster R-CNN for Object Detection in the Wild**

- intro: CVPR 2018. ETH Zurich & ESAT/PSI
- arxiv: [https://arxiv.org/abs/1803.03243](https://arxiv.org/abs/1803.03243)
- github(official. Caffe): [https://github.com/yuhuayc/da-faster-rcnn](https://github.com/yuhuayc/da-faster-rcnn)

**Robust Physical Adversarial Attack on Faster R-CNN Object Detector**

[https://arxiv.org/abs/1804.05810](https://arxiv.org/abs/1804.05810)

**Auto-Context R-CNN**

- intro: Rejected by ECCV18
- arxiv: [https://arxiv.org/abs/1807.02842](https://arxiv.org/abs/1807.02842)

**Grid R-CNN**

- intro: CVPR 2019
- intro: SenseTime
- arxiv: [https://arxiv.org/abs/1811.12030](https://arxiv.org/abs/1811.12030)

**Grid R-CNN Plus: Faster and Better**

- intro: SenseTime Research & CUHK & Beihang University
- arxiv: [https://arxiv.org/abs/1906.05688](https://arxiv.org/abs/1906.05688)
- github: [https://github.com/STVIR/Grid-R-CNN](https://github.com/STVIR/Grid-R-CNN)

**Few-shot Adaptive Faster R-CNN**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1903.09372](https://arxiv.org/abs/1903.09372)

**Libra R-CNN: Towards Balanced Learning for Object Detection**

- intro: CVPR 2019
- arxiv: [https://arxiv.org/abs/1904.02701](https://arxiv.org/abs/1904.02701)

**Rethinking Classification and Localization in R-CNN**

- intro: Northeastern University & Microsoft
- arxiv: [https://arxiv.org/abs/1904.06493](https://arxiv.org/abs/1904.06493)

**Reprojection R-CNN: A Fast and Accurate Object Detector for 360° Images**

- intro: Peking University
- arxiv: [https://arxiv.org/abs/1907.11830](https://arxiv.org/abs/1907.11830)

**Rethinking Classification and Localization for Cascade R-CNN**

- intro: BMVC 2019
- arxiv: [https://arxiv.org/abs/1907.11914](https://arxiv.org/abs/1907.11914)

**IoU-uniform R-CNN: Breaking Through the Limitations of RPN**

- arxiv: [https://arxiv.org/abs/1912.05190](https://arxiv.org/abs/1912.05190)
- github(mmdetection): [https://github.com/zl1994/IoU-Uniform-R-CNN](https://github.com/zl1994/IoU-Uniform-R-CNN)

**Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training**

- arxiv: [https://arxiv.org/abs/2004.06002](https://arxiv.org/abs/2004.06002)
- github: [https://github.com/hkzhang95/DynamicRCNN](https://github.com/hkzhang95/DynamicRCNN)

**Delving into the Imbalance of Positive Proposals in Two-stage Object Detection**

- intro: Waseda University & Tencent AI Lab & Nanjing University of Science and Technology
- arxiv: [https://arxiv.org/abs/2005.11472](https://arxiv.org/abs/2005.11472)

**Hierarchical Context Embedding for Region-based Object Detection**

- intro: ECCV 2020
- intro: Nanjing University & Megvii Technology
- arxiv: [https://arxiv.org/abs/2008.01338](https://arxiv.org/abs/2008.01338)

**Sparse R-CNN: End-to-End Object Detection with Learnable Proposals**

- intro: CVPR 2021
- intro: The University of Hong Kong & Tongji University & ByteDance AI Lab 4University of California
- arxiv: [https://arxiv.org/abs/2011.12450](https://arxiv.org/abs/2011.12450)
- github: [https://github.com/PeizeSun/SparseR-CNN](https://github.com/PeizeSun/SparseR-CNN)

**Dynamic Sparse R-CNN**

- intro: CVPR 2022
- arxiv: [https://arxiv.org/abs/2205.02101](https://arxiv.org/abs/2205.02101)

**Featurized Query R-CNN**

- intro: Huazhong University of Science & Technology & Horizon Robotics
- arxiv: [https://arxiv.org/abs/2206.06258](https://arxiv.org/abs/2206.06258)
- github: [https://github.com/hustvl/Featurized-QueryRCNN](https://github.com/hustvl/Featurized-QueryRCNN)

**Augmenting Proposals by the Detector Itself**

- intro: Tsinghua University & Alibaba Group
- arxiv: [https://arxiv.org/abs/2101.11789](https://arxiv.org/abs/2101.11789)

**Probabilistic two-stage detection**

- intro: UT Austin & Intel Labs
- arxiv: [https://arxiv.org/abs/2103.07461](https://arxiv.org/abs/2103.07461)
- github: [https://github.com/xingyizhou/CenterNet2](https://github.com/xingyizhou/CenterNet2)

# Single-Shot Object Detection

## YOLO

**You Only Look Once: Unified, Real-Time Object Detection**

![](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)

- arxiv: [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)
- github: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- blog: [https://pjreddie.com/publications/yolo/](https://pjreddie.com/publications/yolo/)
- slides: [https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p](https://docs.google.com/presentation/d/1aeRvtKG21KHdD5lg6Hgyhx5rPq_ZOsGjG5rJ1HP7BbA/pub?start=false&loop=false&delayms=3000&slide=id.p)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/](https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/)
- github: [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- github: [https://github.com/xingwangsfu/caffe-yolo](https://github.com/xingwangsfu/caffe-yolo)
- github: [https://github.com/frankzhangrui/Darknet-Yolo](https://github.com/frankzhangrui/Darknet-Yolo)
- github: [https://github.com/BriSkyHekun/py-darknet-yolo](https://github.com/BriSkyHekun/py-darknet-yolo)
- github: [https://github.com/tommy-qichang/yolo.torch](https://github.com/tommy-qichang/yolo.torch)
- github: [https://github.com/frischzenger/yolo-windows](https://github.com/frischzenger/yolo-windows)
- github: [https://github.com/AlexeyAB/yolo-windows](https://github.com/AlexeyAB/yolo-windows)
- github: [https://github.com/nilboy/tensorflow-yolo](https://github.com/nilboy/tensorflow-yolo)

**darkflow - translate darknet to tensorflow. Load trained weights, retrain/fine-tune them using tensorflow, export constant graph def to C++**

- blog: [https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp](https://thtrieu.github.io/notes/yolo-tensorflow-graph-buffer-cpp)
- github: [https://github.com/thtrieu/darkflow](https://github.com/thtrieu/darkflow)

**Start Training YOLO with Our Own Data**

![](http://guanghan.info/blog/en/wp-content/uploads/2015/12/images-40.jpg)

- intro: train with customized data and class numbers/labels. Linux / Windows version for darknet.
- blog: [http://guanghan.info/blog/en/my-works/train-yolo/](http://guanghan.info/blog/en/my-works/train-yolo/)
- github: [https://github.com/Guanghan/darknet](https://github.com/Guanghan/darknet)

**YOLO: Core ML versus MPSNNGraph**

- intro: Tiny YOLO for iOS implemented using CoreML but also using the new MPS graph API.
- blog: [http://machinethink.net/blog/yolo-coreml-versus-mps-graph/](http://machinethink.net/blog/yolo-coreml-versus-mps-graph/)
- github: [https://github.com/hollance/YOLO-CoreML-MPSNNGraph](https://github.com/hollance/YOLO-CoreML-MPSNNGraph)

**TensorFlow YOLO object detection on Android**

- intro: Real-time object detection on Android using the YOLO network with TensorFlow
- github: [https://github.com/natanielruiz/android-yolo](https://github.com/natanielruiz/android-yolo)

**Computer Vision in iOS – Object Detection**

- blog: [https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/](https://sriraghu.com/2017/07/12/computer-vision-in-ios-object-detection/)
- github:[https://github.com/r4ghu/iOS-CoreML-Yolo](https://github.com/r4ghu/iOS-CoreML-Yolo)

## YOLOv2

**YOLO9000: Better, Faster, Stronger**

- arxiv: [https://arxiv.org/abs/1612.08242](https://arxiv.org/abs/1612.08242)
- code: [http://pjreddie.com/yolo9000/](http://pjreddie.com/yolo9000/)
- github(Chainer): [https://github.com/leetenki/YOLOv2](https://github.com/leetenki/YOLOv2)
- github(Keras): [https://github.com/allanzelener/YAD2K](https://github.com/allanzelener/YAD2K)
- github(PyTorch): [https://github.com/longcw/yolo2-pytorch](https://github.com/longcw/yolo2-pytorch)
- github(Tensorflow): [https://github.com/hizhangp/yolo_tensorflow](https://github.com/hizhangp/yolo_tensorflow)
- github(Windows): [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- github: [https://github.com/choasUp/caffe-yolo9000](https://github.com/choasUp/caffe-yolo9000)
- github: [https://github.com/philipperemy/yolo-9000](https://github.com/philipperemy/yolo-9000)

**darknet_scripts**

- intro: Auxilary scripts to work with (YOLO) darknet deep learning famework. AKA -> How to generate YOLO anchors?
- github: [https://github.com/Jumabek/darknet_scripts](https://github.com/Jumabek/darknet_scripts)

**Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2**

- github: [https://github.com/AlexeyAB/Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)

**LightNet: Bringing pjreddie's DarkNet out of the shadows**

[https://github.com//explosion/lightnet](https://github.com//explosion/lightnet)

**YOLO v2 Bounding Box Tool**

- intro: Bounding box labeler tool to generate the training data in the format YOLO v2 requires.
- github: [https://github.com/Cartucho/yolo-boundingbox-labeler-GUI](https://github.com/Cartucho/yolo-boundingbox-labeler-GUI)

## YOLOv3

**YOLOv3: An Incremental Improvement**

- project page: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)
- paper: [https://pjreddie.com/media/files/papers/YOLOv3.pdf](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
- arxiv: [https://arxiv.org/abs/1804.02767](https://arxiv.org/abs/1804.02767)
- githb: [https://github.com/DeNA/PyTorch_YOLOv3](https://github.com/DeNA/PyTorch_YOLOv3)
- github: [https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)

**Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving**

[https://arxiv.org/abs/1904.04620](https://arxiv.org/abs/1904.04620)

**YOLO-LITE: A Real-Time Object Detection Algorithm Optimized for Non-GPU Computers**

[https://arxiv.org/abs/1811.05588](https://arxiv.org/abs/1811.05588)

**Spiking-YOLO: Spiking Neural Network for Real-time Object Detection**

[https://arxiv.org/abs/1903.06530](https://arxiv.org/abs/1903.06530)

**YOLO Nano: a Highly Compact You Only Look Once Convolutional Neural Network for Object Detection**

[https://arxiv.org/abs/1910.01271](https://arxiv.org/abs/1910.01271)

**REQ-YOLO: A Resource-Aware, Efficient Quantization Framework for Object Detection on FPGAs**

[https://arxiv.org/abs/1909.13396](https://arxiv.org/abs/1909.13396)

**Poly-YOLO: higher speed, more precise detection and instance segmentation for YOLOv3**

- intro: TPAMI
- arxiv: [https://arxiv.org/abs/2005.13243](https://arxiv.org/abs/2005.13243)
- gitlab: [https://gitlab.com/irafm-ai/poly-yolo](https://gitlab.com/irafm-ai/poly-yolo)

## YOLOv4

**YOLOv4: Optimal Speed and Accuracy of Object Detection**

- keywords: Weighted-Residual-Connections (WRC), Cross-Stage-Partial-connections (CSP), Cross mini-Batch Normalization (CmBN), Self-adversarial-training (SAT), Mish-activation
- arxiv: [https://arxiv.org/abs/2004.10934](https://arxiv.org/abs/2004.10934)
- github: [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
- github: [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)

**YOLOX: Exceeding YOLO Series in 2021**

- intro: Megvii Technology
- arxiv: [https://arxiv.org/abs/2107.08430](https://arxiv.org/abs/2107.08430)
- github: [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

**PP-YOLO: An Effective and Efficient Implementation of Object Detector**

- intro: Baidu Inc.
- arxiv: [https://arxiv.org/abs/2007.12099](https://arxiv.org/abs/2007.12099)
- github: [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

## YOLOv7

**YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors**

- arxiv: [https://arxiv.org/abs/2207.02696](https://arxiv.org/abs/2207.02696)
- github: [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)

**Real-time Object Detection for Streaming Perception**

- intro: CVPR 2022 oral
- arxiv: [https://arxiv.org/abs/2203.12338](https://arxiv.org/abs/2203.12338)
- github: [https://github.com/yancie-yjr/StreamYOLO](https://github.com/yancie-yjr/StreamYOLO)

## SSD

**SSD: Single Shot MultiBox Detector**

![](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

- intro: ECCV 2016 Oral
- arxiv: [http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- paper: [http://www.cs.unc.edu/~wliu/papers/ssd.pdf](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)
- slides: [http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf](http://www.cs.unc.edu/%7Ewliu/papers/ssd_eccv2016_slide.pdf)
- github(Official): [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
- video: [http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973](http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973)
- github: [https://github.com/zhreshold/mxnet-ssd](https://github.com/zhreshold/mxnet-ssd)
- github: [https://github.com/zhreshold/mxnet-ssd.cpp](https://github.com/zhreshold/mxnet-ssd.cpp)
- github: [https://github.com/rykov8/ssd_keras](https://github.com/rykov8/ssd_keras)
- github: [https://github.com/balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow)
- github: [https://github.com/amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
- github(Caffe): [https://github.com/chuanqi305/MobileNet-SSD](https://github.com/chuanqi305/MobileNet-SSD)

**What's the diffience in performance between this new code you pushed and the previous code? #327**

[https://github.com/weiliu89/caffe/issues/327](https://github.com/weiliu89/caffe/issues/327)

**DSSD : Deconvolutional Single Shot Detector**

- intro: UNC Chapel Hill & Amazon Inc
- arxiv: [https://arxiv.org/abs/1701.06659](https://arxiv.org/abs/1701.06659)
- github: [https://github.com/chengyangfu/caffe/tree/dssd](https://github.com/chengyangfu/caffe/tree/dssd)
- github: [https://github.com/MTCloudVision/mxnet-dssd](https://github.com/MTCloudVision/mxnet-dssd)
- demo: [http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4](http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4)

**Enhancement of SSD by concatenating feature maps for object detection**

- intro: rainbow SSD (R-SSD)
- arxiv: [https://arxiv.org/abs/1705.09587](https://arxiv.org/abs/1705.09587)

**Context-aware Single-Shot Detector**

- keywords: CSSD, DiCSSD, DeCSSD, effective receptive fields (ERFs),  theoretical receptive fields (TRFs)
- arxiv: [https://arxiv.org/abs/1707.08682](https://arxiv.org/abs/1707.08682)

**Feature-Fused SSD: Fast Detection for Small Objects**

[https://arxiv.org/abs/1709.05054](https://arxiv.org/abs/1709.05054)

**FSSD: Feature Fusion Single Shot Multibox Detector**

[https://arxiv.org/abs/1712.00960](https://arxiv.org/abs/1712.00960)

**Weaving Multi-scale Context for Single Shot Detector**

- intro: WeaveNet
- keywords: fuse multi-scale information
- arxiv: [https://arxiv.org/abs/1712.03149](https://arxiv.org/abs/1712.03149)

**Extend the shallow part of Single Shot MultiBox Detector via Convolutional Neural Network**

- keywords: ESSD
- arxiv: [https://arxiv.org/abs/1801.05918](https://arxiv.org/abs/1801.05918)

**Tiny SSD: A Tiny Single-shot Detection Deep Convolutional Neural Network for Real-time Embedded Object Detection**

[https://arxiv.org/abs/1802.06488](https://arxiv.org/abs/1802.06488)

**MDSSD: Multi-scale Deconvolutional Single Shot Detector for small objects**

- intro: Zhengzhou University
- arxiv: [https://arxiv.org/abs/1805.07009](https://arxiv.org/abs/1805.07009)

**Accurate Single Stage Detector Using Recurrent Rolling Convolution**

- intro: CVPR 2017. SenseTime
- keywords: Recurrent Rolling Convolution (RRC)
- arxiv: [https://arxiv.org/abs/1704.05776](https://arxiv.org/abs/1704.05776)
- github: [https://github.com/xiaohaoChen/rrc_detection](https://github.com/xiaohaoChen/rrc_detection)

**Residual Features and Unified Prediction Network for Single Stage Detection**

[https://arxiv.org/abs/1707.05031](https://arxiv.org/abs/1707.05031)

## RetinaNet

**Focal Loss for Dense Object Detection**

- intro: ICCV 2017 Best student paper award. Facebook AI Research
- keywords: RetinaNet
- arxiv: [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

**Cascade RetinaNet: Maintaining Consistency for Single-Stage Object Detection**

- intro: BMVC 2019
- keywords: Cas-RetinaNet, Feature Consistency Module
- arxiv: [https://arxiv.org/abs/1907.06881](https://arxiv.org/abs/1907.06881)

**Focal Loss Dense Detector for Vehicle Surveillance**

[https://arxiv.org/abs/1803.01114](https://arxiv.org/abs/1803.01114)

**Single-Shot Refinement Neural Network for Object Detection**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1711.06897](https://arxiv.org/abs/1711.06897)
- github: [https://github.com/sfzhang15/RefineDet](https://github.com/sfzhang15/RefineDet)
- github: [https://github.com/MTCloudVision/RefineDet-Mxnet](https://github.com/MTCloudVision/RefineDet-Mxnet)

**Single-Shot Bidirectional Pyramid Networks for High-Quality Object Detection**

- intro: Singapore Management University & Zhejiang University
- arxiv: [https://arxiv.org/abs/1803.08208](https://arxiv.org/abs/1803.08208)

**Dual Refinement Network for Single-Shot Object Detection**

[https://arxiv.org/abs/1807.08638](https://arxiv.org/abs/1807.08638)

**ScratchDet:Exploring to Train Single-Shot Object Detectors from Scratch**

- arxiv: [https://arxiv.org/abs/1810.08425](https://arxiv.org/abs/1810.08425)
- github: [https://github.com/KimSoybean/ScratchDet](https://github.com/KimSoybean/ScratchDethttps://github.com/KimSoybean/ScratchDet)

**Gradient Harmonized Single-stage Detector**

- intro: AAAI 2019 Oral
- arxiv: [https://arxiv.org/abs/1811.05181](https://arxiv.org/abs/1811.05181)
- gihtub(official): [https://github.com/libuyu/GHM_Detection](https://github.com/libuyu/GHM_Detection)

**M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network**

- intro: AAAI 2019
- arxiv: [https://arxiv.org/abs/1811.04533](https://arxiv.org/abs/1811.04533)
- github: [https://github.com/qijiezhao/M2Det](https://github.com/qijiezhao/M2Det)

**Multi-layer Pruning Framework for Compressing Single Shot MultiBox Detector**

- intro: WACV 2019
- arxiv: [https://arxiv.org/abs/1811.08342](https://arxiv.org/abs/1811.08342)

**Consistent Optimization for Single-Shot Object Detection**

- arxiv: [https://arxiv.org/abs/1901.06563](https://arxiv.org/abs/1901.06563)
- blog: [https://zhuanlan.zhihu.com/p/55416312](https://zhuanlan.zhihu.com/p/55416312)

**A Single-shot Object Detector with Feature Aggragation and Enhancement**

[https://arxiv.org/abs/1902.02923](https://arxiv.org/abs/1902.02923)

**Towards Accurate One-Stage Object Detection with AP-Loss**

- intro: CVPR 2019
- intro: Shanghai Jiao Tong University & Intel Labs & Malaysia Multimedia University & Tencent YouTu Lab & Peking University
- keywords: Average-Precision loss (AP-loss)
- arxiv: {https://arxiv.org/abs/1904.06373}(https://arxiv.org/abs/1904.06373)

**AP-Loss for Accurate One-Stage Object Detection**

- intro: IEEE TPAMI
- arxiv: [https://arxiv.org/abs/2008.07294](https://arxiv.org/abs/2008.07294)
- github: [https://github.com/cccorn/AP-loss](https://github.com/cccorn/AP-loss)

**Searching Parameterized AP Loss for Object Detection**

- intro: NeurIPS 2021
- intro: 1Tsinghua University & Zhejiang University & SenseTime Research & Shanghai Jiao Tong University & Beijing Academy of Artificial Intelligence
- arxiv: [https://arxiv.org/abs/2112.05138](https://arxiv.org/abs/2112.05138)
- github: [https://github.com/fundamentalvision/Parameterized-AP-Loss](https://github.com/fundamentalvision/Parameterized-AP-Loss)

**Efficient Featurized Image Pyramid Network for Single Shot Detector**

- intro: CVPR 2019
- paper: [http://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Efficient_Featurized_Image_Pyramid_Network_for_Single_Shot_Detector_CVPR_2019_paper.pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Pang_Efficient_Featurized_Image_Pyramid_Network_for_Single_Shot_Detector_CVPR_2019_paper.pdf)
- github: [https://github.com/vaesl/LFIP](https://github.com/vaesl/LFIP)

**DR Loss: Improving Object Detection by Distributional Ranking**

- intro: Alibaba Group
- arxiv: [https://arxiv.org/abs/1907.10156](https://arxiv.org/abs/1907.10156)

**HAR-Net: Joint Learning of Hybrid Attention for Single-stage Object Detection**

[https://arxiv.org/abs/1904.11141](https://arxiv.org/abs/1904.11141)

**Propose-and-Attend Single Shot Detector**

[https://arxiv.org/abs/1907.12736](https://arxiv.org/abs/1907.12736)

**Revisiting Feature Alignment for One-stage Object Detection**

- intro: University of Chinese Academy of Sciences & TuSimple
- keywords: AlignDet, RoIConv
- arxiv: [https://arxiv.org/abs/1908.01570](https://arxiv.org/abs/1908.01570)

**IoU-balanced Loss Functions for Single-stage Object Detection**

- intro: HUST
- arxiv: [https://arxiv.org/abs/1908.05641](https://arxiv.org/abs/1908.05641)

**PosNeg-Balanced Anchors with Aligned Features for Single-Shot Object Detection**

- intro: Chinese Academy of Sciences & University of Chinese Academy of Sciences
- keywords: Anchor Promotion Module (APM), Feature Alignment Module (FAM)
- arxiv: [https://arxiv.org/abs/1908.03295](https://arxiv.org/abs/1908.03295)

**R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object**

[https://arxiv.org/abs/1908.05612](https://arxiv.org/abs/1908.05612)

**Hierarchical Shot Detector**

- intro: ICCV 2019
- keywords: reg-offset-cls (ROC) module
- paper: [http://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Hierarchical_Shot_Detector_ICCV_2019_paper.pdf](http://openaccess.thecvf.com/content_ICCV_2019/papers/Cao_Hierarchical_Shot_Detector_ICCV_2019_paper.pdf)
- github(official, Pytorch): [https://github.com/JialeCao001/HSD](https://github.com/JialeCao001/HSD)

**Learning from Noisy Anchors for One-stage Object Detection**

[https://arxiv.org/abs/1912.05086](https://arxiv.org/abs/1912.05086)

**Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection**

- intro: Nanjing University of Science and Technology & Momenta & Nanjing University & Microsoft Research & Tsinghua University
- arxiv: [https://arxiv.org/abs/2006.04388](https://arxiv.org/abs/2006.04388)
- github: [https://github.com/implus/GFocal](https://github.com/implus/GFocal)

**Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection**

- intro: Nanjing University of Science and Technology & Momenta & Nanjing University & Tsinghua University
- arxiv: [https://arxiv.org/abs/2011.12885](https://arxiv.org/abs/2011.12885)
- github: [https://github.com/implus/GFocalV2](https://github.com/implus/GFocalV2)

**Single-Shot Two-Pronged Detector with Rectified IoU Loss**

- intro: ACM MM 2020
- arxiv: [https://arxiv.org/abs/2008.03511](https://arxiv.org/abs/2008.03511)

**OneNet: Towards End-to-End One-Stage Object Detection**

- intro: The University of Hong Kong & ByteDance AI Lab
- arxiv: [https://arxiv.org/abs/2012.05780](https://arxiv.org/abs/2012.05780)
- github: [https://github.com/PeizeSun/OneNet](https://github.com/PeizeSun/OneNet)

**TOOD: Task-aligned One-stage Object Detection**

- intro: ICCV 2021 Oral
- intro: Intellifusion Inc. & Meituan Inc. & ByteDance Inc. & Malong LLC & Alibaba Group
- arxiv: [https://arxiv.org/abs/2108.07755](https://arxiv.org/abs/2108.07755)
- github: [https://github.com/fcjian/TOOD](https://github.com/fcjian/TOOD)

**Rethinking the Aligned and Misaligned Features in One-stage Object Detection**

[https://arxiv.org/abs/2108.12176](https://arxiv.org/abs/2108.12176)

# Anchor-free

**Feature Selective Anchor-Free Module for Single-Shot Object Detection**

- intro: CVPR 2019
- keywords: feature selective anchor-free (FSAF) module
- arxiv: [https://arxiv.org/abs/1903.00621](https://arxiv.org/abs/1903.00621)

**FCOS: Fully Convolutional One-Stage Object Detection**

- intro: The University of Adelaide
- keywords: anchor-free
- arxiv: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355)
- github: [https://github.com/tianzhi0549/FCOS/](https://github.com/tianzhi0549/FCOS/)

**FoveaBox: Beyond Anchor-based Object Detector**

- intro: Tsinghua University & BNRist & ByteDance AI Lab & University of Pennsylvania
- arxiv: [https://arxiv.org/abs/1904.03797](https://arxiv.org/abs/1904.03797)
- github(official, mmdetection): [https://github.com/taokong/FoveaBox](https://github.com/taokong/FoveaBox)

**IMMVP: An Efficient Daytime and Nighttime On-Road Object Detector**

[https://arxiv.org/abs/1910.06573](https://arxiv.org/abs/1910.06573)

**EfficientDet: Scalable and Efficient Object Detection**

- intro: CVPR 2020
- arxiv: [https://arxiv.org/abs/1911.09070](https://arxiv.org/abs/1911.09070)
- github: [https://github.com/google/automl/tree/master/efficientdet](https://github.com/google/automl/tree/master/efficientdet)
- github: [https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

**Domain Adaptation for Object Detection via Style Consistency**

- intro: BMVC 2019
- arxiv: [https://arxiv.org/abs/1911.10033](https://arxiv.org/abs/1911.10033)

**Soft Anchor-Point Object Detection**

- intro: ECCV 2020
- intro: Carnegie Mellon University
- keywords: Soft Anchor-Point Detector (SAPD)
- arxiv: [https://arxiv.org/abs/1911.12448](https://arxiv.org/abs/1911.12448)

**IPG-Net: Image Pyramid Guidance Network for Object Detection**

[https://arxiv.org/abs/1912.00632](https://arxiv.org/abs/1912.00632)

**Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection**

- arxiv: [https://arxiv.org/abs/1912.02424](https://arxiv.org/abs/1912.02424)
- github: [https://github.com/sfzhang15/ATSS](https://github.com/sfzhang15/ATSS)

**Localization Uncertainty Estimation for Anchor-Free Object Detection**

- keywords: Gaussian-FCOS
- arxiv: [https://arxiv.org/abs/2006.15607](https://arxiv.org/abs/2006.15607)

**Corner Proposal Network for Anchor-free, Two-stage Object Detection**

- intro: ECCV 2020
- arxiv: [https://arxiv.org/abs/2007.13816](https://arxiv.org/abs/2007.13816)
- github: [https://github.com/Duankaiwen/CPNDet](https://github.com/Duankaiwen/CPNDet)

**Dive Deeper Into Box for Object Detection**

- intro: ECCV 2020
- keywords: DDBNet, anchor free
- arxiv: [https://arxiv.org/abs/2007.14350](https://arxiv.org/abs/2007.14350)

**Corner Proposal Network for Anchor-free, Two-stage Object Detection**

- intro: ECCV 2020
- arxiv: [https://arxiv.org/abs/2007.13816](https://arxiv.org/abs/2007.13816)
- github: [https://github.com/Duankaiwen/CPNDet](https://github.com/Duankaiwen/CPNDet)

**Reducing Label Noise in Anchor-Free Object Detection**

- intro: BMVC 2020
- arxiv: [https://arxiv.org/abs/2008.01167](https://arxiv.org/abs/2008.01167)
- github: [https://github.com/nerminsamet/ppdet](https://github.com/nerminsamet/ppdet)

**Balance-Oriented Focal Loss with Linear Scheduling for Anchor Free Object Detection**

[https://arxiv.org/abs/2012.13763](https://arxiv.org/abs/2012.13763)

**PAFNet: An Efficient Anchor-Free Object Detector Guidance**

- intro: Baidu Inc.
- github: [https://arxiv.org/abs/2104.13534](https://arxiv.org/abs/2104.13534)
- arxiv: [https://github.com/PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

**Pseudo-IoU: Improving Label Assignment in Anchor-Free Object Detection**

- intro: CVPR 2021 Workshop
- intro: UIUC & MIT-IBM Watson AI Lab & IBM T.J. Watson Research Center & NVIDIA & University of Oregon & Picsart AI Research (PAIR)
- arxiv: [https://arxiv.org/abs/2104.14082](https://arxiv.org/abs/2104.14082)

**ObjectBox: From Centers to Boxes for Anchor-Free Object Detection**

- intro: ECCV 2022 Oral
- intro: Ingenuity Labs Research Institute & Queen’s University
- arxiv: [https://arxiv.org/abs/2207.06985](https://arxiv.org/abs/2207.06985)
- github: [https://github.com/MohsenZand/ObjectBox](https://github.com/MohsenZand/ObjectBox)

# Transformers

**End-to-End Object Detection with Transformers**

- intro: Facebook AI
- keywords: DEtection TRansformer (DETR)
- arxiv: [https://arxiv.org/abs/2005.12872](https://arxiv.org/abs/2005.12872)
- github: [https://github.com/facebookresearch/detr](https://github.com/facebookresearch/detr)

**Deformable DETR: Deformable Transformers for End-to-End Object Detection**

- intro: SenseTime Research & USTC & CUHK
- arxiv: [https://arxiv.org/abs/2010.04159](https://arxiv.org/abs/2010.04159)
- github: [https://github.com/fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)

**RelationNet++: Bridging Visual Representations for Object Detection via Transformer Decoder**

- intro: NeurIPS2020 Spotlight
- intro: CAS & MSRA
- arxiv: [https://arxiv.org/abs/2010.15831](https://arxiv.org/abs/2010.15831)
- github:[https://github.com/microsoft/RelationNet2](https://github.com/microsoft/RelationNet2)

**UP-DETR: Unsupervised Pre-training for Object Detection with Transformers**

- intro: South China University of Technology & Tencent Wechat AI
- arxiv: [https://arxiv.org/abs/2011.09094](https://arxiv.org/abs/2011.09094)

**Conditional DETR for Fast Training Convergence**

- intro: ICCV 2021
- intro: University of Science and Technology of China & Peking University & Microsoft Research Asia
- arxiv: [https://arxiv.org/abs/2108.06152](https://arxiv.org/abs/2108.06152)
- github: [https://github.com/Atten4Vis/ConditionalDETR](https://github.com/Atten4Vis/ConditionalDETR)

**End-to-End Object Detection with Adaptive Clustering Transformer**

- intro: Peking University & The Chinese University of Hong Kong
- arxiv: [https://arxiv.org/abs/2011.09315](https://arxiv.org/abs/2011.09315)

**Toward Transformer-Based Object Detection**

- intro: Pinterest
- keywords: ViT-FRCNN
- arxiv: [https://arxiv.org/abs/2012.09958](https://arxiv.org/abs/2012.09958)

**Efficient DETR: Improving End-to-End Object Detector with Dense Prior**

- intro: Megvii Technology
- arxiv: [https://arxiv.org/abs/2104.01318](https://arxiv.org/abs/2104.01318)

**Anchor DETR: Query Design for Transformer-Based Detector**

- intro: MEGVII Technology
- arxiv: [https://arxiv.org/abs/2109.07107](https://arxiv.org/abs/2109.07107)
- gihtub: [https://github.com/megvii-model/AnchorDETR](https://github.com/megvii-model/AnchorDETR)

**DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR**

- arxiv: [https://arxiv.org/abs/2201.12329](https://arxiv.org/abs/2201.12329)
- github: [https://github.com/SlongLiu/DAB-DETR](https://github.com/SlongLiu/DAB-DETR)

**DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection**

- arxiv: [https://arxiv.org/abs/2203.03605](https://arxiv.org/abs/2203.03605)
- github: [https://github.com/IDEACVR/DINO](https://github.com/IDEACVR/DINO)

**Oriented Object Detection with Transformer**

- intro: University at Buffalo & Beihang University & Baidu Inc
- arxiv: [https://arxiv.org/abs/2106.03146](https://arxiv.org/abs/2106.03146)

**ViDT: An Efficient and Effective Fully Transformer-based Object Detector**

- intro: NAVER AI Lab & Google Research & University of California at Merced
- arxiv: [https://arxiv.org/abs/2110.03921](https://arxiv.org/abs/2110.03921)
- github: [https://github.com/naver-ai/vidt](https://github.com/naver-ai/vidt)

**An Extendable, Efficient and Effective Transformer-based Object Detector**

- arxiv: [https://arxiv.org/abs/2204.07962](https://arxiv.org/abs/2204.07962)
- github: [https://github.com/naver-ai/vidt](https://github.com/naver-ai/vidt)

**Omni-DETR: Omni-Supervised Object Detection with Transformers**

- intro: CVPR 2022
- arxiv: [https://arxiv.org/abs/2203.16089](https://arxiv.org/abs/2203.16089)

**Accelerating DETR Convergence via Semantic-Aligned Matching**

- intro: CVPR 2022
- arxiv: [https://arxiv.org/abs/2203.06883](https://arxiv.org/abs/2203.06883)
- github: [https://github.com/ZhangGongjie/SAM-DETR](https://github.com/ZhangGongjie/SAM-DETR)

**AdaMixer: A Fast-Converging Query-Based Object Detector**

- intro: CVPR 2022 oral
- intro: Nanjing University, MYbank Ant Group
- arxiv: [https://arxiv.org/abs/2203.16507](https://arxiv.org/abs/2203.16507)
- github: [https://github.com/MCG-NJU/AdaMixer](https://github.com/MCG-NJU/AdaMixer)

**Exploring Plain Vision Transformer Backbones for Object Detection**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/2203.16527](https://arxiv.org/abs/2203.16527)

**Efficient Decoder-free Object Detection with Transformers**

- intro:  Tencent Youtu Lab & Zhejiang University
- arxiv: [https://arxiv.org/abs/2206.06829](https://arxiv.org/abs/2206.06829)
- github: [https://github.com/Pealing/DFFT](https://github.com/Pealing/DFFT)

# Non-Maximum Suppression (NMS)

**End-to-End Integration of a Convolutional Network, Deformable Parts Model and Non-Maximum Suppression**

- intro: CVPR 2015
- arxiv: [http://arxiv.org/abs/1411.5309](http://arxiv.org/abs/1411.5309)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wan_End-to-End_Integration_of_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wan_End-to-End_Integration_of_2015_CVPR_paper.pdf)

**A convnet for non-maximum suppression**

- arxiv: [http://arxiv.org/abs/1511.06437](http://arxiv.org/abs/1511.06437)

**Improving Object Detection With One Line of Code**

**Soft-NMS -- Improving Object Detection With One Line of Code**

- intro: ICCV 2017. University of Maryland
- keywords: Soft-NMS
- arxiv: [https://arxiv.org/abs/1704.04503](https://arxiv.org/abs/1704.04503)
- github: [https://github.com/bharatsingh430/soft-nms](https://github.com/bharatsingh430/soft-nms)

**Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection**

- intro: CMU & Megvii Inc. (Face++)
- arxiv: [https://arxiv.org/abs/1809.08545](https://arxiv.org/abs/1809.08545)
- github: [https://github.com/yihui-he/softer-NMS](https://github.com/yihui-he/softer-NMS)

**Learning non-maximum suppression**

- intro: CVPR 2017
- project page: [https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/learning-nms/](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/learning-nms/)
- arxiv: [https://arxiv.org/abs/1705.02950](https://arxiv.org/abs/1705.02950)
- github: [https://github.com/hosang/gossipnet](https://github.com/hosang/gossipnet)

**Relation Networks for Object Detection**

- intro: CVPR 2018 oral
- arxiv: [https://arxiv.org/abs/1711.11575](https://arxiv.org/abs/1711.11575)
- github(official, MXNet): [https://github.com/msracver/Relation-Networks-for-Object-Detection](https://github.com/msracver/Relation-Networks-for-Object-Detection)

**Learning Pairwise Relationship for Multi-object Detection in Crowded Scenes**

- keywords: Pairwise-NMS
- arxiv: [https://arxiv.org/abs/1901.03796](https://arxiv.org/abs/1901.03796)

**Daedalus: Breaking Non-Maximum Suppression in Object Detection via Adversarial Examples**

[https://arxiv.org/abs/1902.02067](https://arxiv.org/abs/1902.02067)

**NMS by Representative Region: Towards Crowded Pedestrian Detection by Proposal Pairing**

- intro: CVPR 2020
- intro: Waseda University & Tencent AI Lab
- arxiv: [https://arxiv.org/abs/2003.12729](https://arxiv.org/abs/2003.12729)

**Hashing-based Non-Maximum Suppression for Crowded Object Detection**

- intro: Microsoft
- arxiv: [https://arxiv.org/abs/2005.11426](https://arxiv.org/abs/2005.11426)
- github: [https://github.com/microsoft/hnms](https://github.com/microsoft/hnms)

**Visibility Guided NMS: Efficient Boosting of Amodal Object Detection in Crowded Traffic Scenes**

- intro: NeurIPS 2019, Machine Learning for Autonomous Driving Workshop
- intro: Mercedes-Benz AG, R&D & University of Jena
- keywords: Visibility Guided NMS (vg-NMS)
- arxiv: [https://arxiv.org/abs/2006.08547](https://arxiv.org/abs/2006.08547)

**Determinantal Point Process as an alternative to NMS**

[https://arxiv.org/abs/2008.11451](https://arxiv.org/abs/2008.11451)

**Ref-NMS: Breaking Proposal Bottlenecks in Two-Stage Referring Expression Grounding**

- intro: Zhejiang University & Nanyang Technological University & Tencent AI Lab & Columbia University
- arxiv: [https://arxiv.org/abs/2009.01449](https://arxiv.org/abs/2009.01449)

# NMS-free

**Object Detection Made Simpler by Eliminating Heuristic NMS**

- intro: Alibaba Group & Monash University & The University of Adelaide
- arxiv: [https://arxiv.org/abs/2101.11782](https://arxiv.org/abs/2101.11782)
- github: [https://github.com/txdet/FCOSPss](https://github.com/txdet/FCOSPss)

# Adversarial Examples

**Adversarial Examples that Fool Detectors**

- intro: University of Illinois
- arxiv: [https://arxiv.org/abs/1712.02494](https://arxiv.org/abs/1712.02494)

**Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods**

- project page: [http://nicholas.carlini.com/code/nn_breaking_detection/](http://nicholas.carlini.com/code/nn_breaking_detection/)
- arxiv: [https://arxiv.org/abs/1705.07263](https://arxiv.org/abs/1705.07263)
- github: [https://github.com/carlini/nn_breaking_detection](https://github.com/carlini/nn_breaking_detection)

# Knowledge Distillation

**Mimicking Very Efficient Network for Object Detection**

- intro: CVPR 2017. SenseTime & Beihang University
- paper: [http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Mimicking_Very_Efficient_CVPR_2017_paper.pdf)

**Quantization Mimic: Towards Very Tiny CNN for Object Detection**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1805.02152](https://arxiv.org/abs/1805.02152)

**Learning Efficient Detector with Semi-supervised Adaptive Distillation**

- intro: SenseTime Research
- arxiv: [https://arxiv.org/abs/1901.00366](https://arxiv.org/abs/1901.00366)
- github: [https://github.com/Tangshitao/Semi-supervised-Adaptive-Distillation](https://github.com/Tangshitao/Semi-supervised-Adaptive-Distillation)

**Distilling Object Detectors with Fine-grained Feature Imitation**

- intro: CVPR 2019
- intro: National University of Singapore & Huawei Noah’s Ark Lab
- keywords: mimic
- arxiv: [https://arxiv.org/abs/1906.03609](https://arxiv.org/abs/1906.03609)
- github: [https://github.com/twangnh/Distilling-Object-Detectors](https://github.com/twangnh/Distilling-Object-Detectors)

**GAN-Knowledge Distillation for one-stage Object Detection**

[https://arxiv.org/abs/1906.08467](https://arxiv.org/abs/1906.08467)

**Learning Lightweight Pedestrian Detector with Hierarchical Knowledge Distillation**

- intro: ICIP 2019 oral
- arxiv: [https://arxiv.org/abs/1909.09325](https://arxiv.org/abs/1909.09325)

**Improve Object Detection with Feature-based Knowledge Distillation: Towards Accurate and Efficient Detectors**

- intro: ICLR 2021 poster
- openreview: [https://openreview.net/forum?id=uKhGRvM8QNH](https://openreview.net/forum?id=uKhGRvM8QNH)
- paper: [https://openreview.net/pdf?id=uKhGRvM8QNH](https://openreview.net/pdf?id=uKhGRvM8QNH)
- github: [https://github.com/ArchipLab-LinfengZhang/Object-Detection-Knowledge-Distillation-ICLR2021](https://github.com/ArchipLab-LinfengZhang/Object-Detection-Knowledge-Distillation-ICLR2021)

**G-DetKD: Towards General Distillation Framework for Object Detectors via Contrastive and Semantic-guided Feature Imitation**

- intro: Hong Kong University of Science and Technology & Huawei Noah’s Ark Lab
- intro: ICCV 2021
- arxiv: [https://arxiv.org/abs/2108.07482](https://arxiv.org/abs/2108.07482)

**LGD: Label-guided Self-distillation for Object Detection**

- intro: MEGVII Technology & Xi’an Jiaotong University
- arxiv: [https://arxiv.org/abs/2109.11496](https://arxiv.org/abs/2109.11496)

**Deep Structured Instance Graph for Distilling Object Detectors**

- intro: ICCV 2021
- intro: The Chinese University of Hong Kong & SmartMore
- arxiv: [https://arxiv.org/abs/2109.12862](https://arxiv.org/abs/2109.12862)
- github: [https://github.com/dvlab-research/Dsig](https://github.com/dvlab-research/Dsig)

**Instance-Conditional Knowledge Distillation for Object Detection**

- intro: NeurIPS 2021 poster
- intro: Xi’an Jiaotong University & MEGVII Technology
- arxiv: [https://arxiv.org/abs/2110.12724](https://arxiv.org/abs/2110.12724)

**Distilling Object Detectors with Feature Richness**

- intro: University of Science and Technology of China & CAS & Cambricon Technologies & University of Chinese Academy of Sciences
- arxiv: [https://arxiv.org/abs/2111.00674](https://arxiv.org/abs/2111.00674)

**Focal and Global Knowledge Distillation for Detectors**

- intro: Tsinghua Shenzhen International Graduate School & ByteDance Inc & BeiHang University
- arxiv: [https://arxiv.org/abs/2111.11837](https://arxiv.org/abs/2111.11837)
- github: [https://github.com/yzd-v/FGD](https://github.com/yzd-v/FGD)

**Prediction-Guided Distillation for Dense Object Detection**

- intro: University of Edinburgh & Heriot-Watt University
- arxiv: [https://arxiv.org/abs/2203.05469](https://arxiv.org/abs/2203.05469)
- github: [https://github.com/ChenhongyiYang/PGD](https://github.com/ChenhongyiYang/PGD)

# Rotated Object Detection

**Rethinking Rotated Object Detection with Gaussian Wasserstein Distance Loss**

- intro: Shanghai Jiao Tong University & Huawei Inc. & Beijing Institute of Technology
- arxiv: [https://arxiv.org/abs/2101.11952](https://arxiv.org/abs/2101.11952)
- github: [https://github.com/yangxue0827/RotationDetection](https://github.com/yangxue0827/RotationDetection)

# Long-Tailed Object Detection

**Factors in Finetuning Deep Model for object detection**

**Factors in Finetuning Deep Model for Object Detection with Long-tail Distribution**

- intro: CVPR 2016.rank 3rd for provided data and 2nd for external data on ILSVRC 2015 object detection
- project page: [http://www.ee.cuhk.edu.hk/~wlouyang/projects/ImageNetFactors/CVPR16.html](http://www.ee.cuhk.edu.hk/~wlouyang/projects/ImageNetFactors/CVPR16.html)
- arxiv: [http://arxiv.org/abs/1601.05150](http://arxiv.org/abs/1601.05150)

**Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax**

- intro: CVPR 2020 oral
- arxiv: [https://arxiv.org/abs/2006.10408](https://arxiv.org/abs/2006.10408)
- github: [https://github.com/FishYuLi/BalancedGroupSoftmax](https://github.com/FishYuLi/BalancedGroupSoftmax)

**Equalization Loss v2: A New Gradient Balance Approach for Long-tailed Object Detection**

- intro: Tongji University & SenseTime Research & Tsinghua University
- arxiv: [https://arxiv.org/abs/2012.08548](https://arxiv.org/abs/2012.08548)

**A Simple and Effective Use of Object-Centric Images for Long-Tailed Object Detection**

- intro: The Ohio State University & University of Central Florida & University of Southern California & Google Research
- arxiv: [https://arxiv.org/abs/2102.08884](https://arxiv.org/abs/2102.08884)

**Adaptive Class Suppression Loss for Long-Tail Object Detection**

- intro: CVPR 2021
- arxiv: [https://arxiv.org/abs/2104.00885](https://arxiv.org/abs/2104.00885)
- github: [https://github.com/CASIA-IVA-Lab/ACSL](https://github.com/CASIA-IVA-Lab/ACSL)

# Weakly Supervised Object Detection

**Track and Transfer: Watching Videos to Simulate Strong Human Supervision for Weakly-Supervised Object Detection**

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1604.05766](http://arxiv.org/abs/1604.05766)

**Weakly supervised object detection using pseudo-strong labels**

- arxiv: [http://arxiv.org/abs/1607.04731](http://arxiv.org/abs/1607.04731)

**Saliency Guided End-to-End Learning for Weakly Supervised Object Detection**

- intro: IJCAI 2017
- arxiv: [https://arxiv.org/abs/1706.06768](https://arxiv.org/abs/1706.06768)

**Visual and Semantic Knowledge Transfer for Large Scale Semi-supervised Object Detection**

- intro: TPAMI 2017. National Institutes of Health (NIH) Clinical Center
- arxiv: [https://arxiv.org/abs/1801.03145](https://arxiv.org/abs/1801.03145)

# Video Object Detection

**Learning Object Class Detectors from Weakly Annotated Video**

- intro: CVPR 2012
- paper: [https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00905.pdf](https://www.vision.ee.ethz.ch/publications/papers/proceedings/eth_biwi_00905.pdf)

**Analysing domain shift factors between videos and images for object detection**

- arxiv: [https://arxiv.org/abs/1501.01186](https://arxiv.org/abs/1501.01186)

**Video Object Recognition**

- slides: [http://vision.princeton.edu/courses/COS598/2015sp/slides/VideoRecog/Video%20Object%20Recognition.pptx](http://vision.princeton.edu/courses/COS598/2015sp/slides/VideoRecog/Video%20Object%20Recognition.pptx)

**Deep Learning for Saliency Prediction in Natural Video**

- intro: Submitted on 12 Jan 2016
- keywords: Deep learning, saliency map, optical flow, convolution network, contrast features
- paper: [https://hal.archives-ouvertes.fr/hal-01251614/document](https://hal.archives-ouvertes.fr/hal-01251614/document)

**T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos**

- intro: Winning solution in ILSVRC2015 Object Detection from Video(VID) Task
- arxiv: [http://arxiv.org/abs/1604.02532](http://arxiv.org/abs/1604.02532)
- github: [https://github.com/myfavouritekk/T-CNN](https://github.com/myfavouritekk/T-CNN)

**Object Detection from Video Tubelets with Convolutional Neural Networks**

- intro: CVPR 2016 Spotlight paper
- arxiv: [https://arxiv.org/abs/1604.04053](https://arxiv.org/abs/1604.04053)
- paper: [http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf](http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf)
- gihtub: [https://github.com/myfavouritekk/vdetlib](https://github.com/myfavouritekk/vdetlib)

**Object Detection in Videos with Tubelets and Multi-context Cues**

- intro: SenseTime Group
- slides: [http://www.ee.cuhk.edu.hk/~xgwang/CUvideo.pdf](http://www.ee.cuhk.edu.hk/~xgwang/CUvideo.pdf)
- slides: [http://image-net.org/challenges/talks/Object%20Detection%20in%20Videos%20with%20Tubelets%20and%20Multi-context%20Cues%20-%20Final.pdf](http://image-net.org/challenges/talks/Object%20Detection%20in%20Videos%20with%20Tubelets%20and%20Multi-context%20Cues%20-%20Final.pdf)

**Context Matters: Refining Object Detection in Video with Recurrent Neural Networks**

- intro: BMVC 2016
- keywords: pseudo-labeler
- arxiv: [http://arxiv.org/abs/1607.04648](http://arxiv.org/abs/1607.04648)
- paper: [http://vision.cornell.edu/se3/wp-content/uploads/2016/07/video_object_detection_BMVC.pdf](http://vision.cornell.edu/se3/wp-content/uploads/2016/07/video_object_detection_BMVC.pdf)

**CNN Based Object Detection in Large Video Images**

- intro: WangTao @ 爱奇艺
- keywords: object retrieval, object detection, scene classification
- slides: [http://on-demand.gputechconf.com/gtc/2016/presentation/s6362-wang-tao-cnn-based-object-detection-large-video-images.pdf](http://on-demand.gputechconf.com/gtc/2016/presentation/s6362-wang-tao-cnn-based-object-detection-large-video-images.pdf)

**Object Detection in Videos with Tubelet Proposal Networks**

- arxiv: [https://arxiv.org/abs/1702.06355](https://arxiv.org/abs/1702.06355)

**Flow-Guided Feature Aggregation for Video Object Detection**

- intro: MSRA
- arxiv: [https://arxiv.org/abs/1703.10025](https://arxiv.org/abs/1703.10025)

**Video Object Detection using Faster R-CNN**

- blog: [http://andrewliao11.github.io/object_detection/faster_rcnn/](http://andrewliao11.github.io/object_detection/faster_rcnn/)
- github: [https://github.com/andrewliao11/py-faster-rcnn-imagenet](https://github.com/andrewliao11/py-faster-rcnn-imagenet)

**Improving Context Modeling for Video Object Detection and Tracking**

[http://image-net.org/challenges/talks_2017/ilsvrc2017_short(poster).pdf](http://image-net.org/challenges/talks_2017/ilsvrc2017_short(poster).pdf)

**Temporal Dynamic Graph LSTM for Action-driven Video Object Detection**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.00666](https://arxiv.org/abs/1708.00666)

**Mobile Video Object Detection with Temporally-Aware Feature Maps**

[https://arxiv.org/abs/1711.06368](https://arxiv.org/abs/1711.06368)

**Towards High Performance Video Object Detection**

[https://arxiv.org/abs/1711.11577](https://arxiv.org/abs/1711.11577)

**Impression Network for Video Object Detection**

[https://arxiv.org/abs/1712.05896](https://arxiv.org/abs/1712.05896)

**Spatial-Temporal Memory Networks for Video Object Detection**

[https://arxiv.org/abs/1712.06317](https://arxiv.org/abs/1712.06317)

**3D-DETNet: a Single Stage Video-Based Vehicle Detector**

[https://arxiv.org/abs/1801.01769](https://arxiv.org/abs/1801.01769)

**Object Detection in Videos by Short and Long Range Object Linking**

[https://arxiv.org/abs/1801.09823](https://arxiv.org/abs/1801.09823)

**Object Detection in Video with Spatiotemporal Sampling Networks**

- intro: University of Pennsylvania, 2Dartmouth College
- arxiv: [https://arxiv.org/abs/1803.05549](https://arxiv.org/abs/1803.05549)

**Towards High Performance Video Object Detection for Mobiles**

- intro: Microsoft Research Asia
- arxiv: [https://arxiv.org/abs/1804.05830](https://arxiv.org/abs/1804.05830)

**Optimizing Video Object Detection via a Scale-Time Lattice**

- intro: CVPR 2018
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/ST-Lattice/](http://mmlab.ie.cuhk.edu.hk/projects/ST-Lattice/)
- arxiv: [https://arxiv.org/abs/1804.05472](https://arxiv.org/abs/1804.05472)
- github: [https://github.com/hellock/scale-time-lattice](https://github.com/hellock/scale-time-lattice)

**Pack and Detect: Fast Object Detection in Videos Using Region-of-Interest Packing**

[https://arxiv.org/abs/1809.01701](https://arxiv.org/abs/1809.01701)

**Fast Object Detection in Compressed Video**

[https://arxiv.org/abs/1811.11057](https://arxiv.org/abs/1811.11057)

**Tube-CNN: Modeling temporal evolution of appearance for object detection in video**

- intro: INRIA/ENS
- arxiv: [https://arxiv.org/abs/1812.02619](https://arxiv.org/abs/1812.02619)

**AdaScale: Towards Real-time Video Object Detection Using Adaptive Scaling**

- intro: SysML 2019 oral
- arxiv: [https://arxiv.org/abs/1902.02910](https://arxiv.org/abs/1902.02910)

**SCNN: A General Distribution based Statistical Convolutional Neural Network with Application to Video Object Detection**

- intro: AAAI 2019
- arxiv: [https://arxiv.org/abs/1903.07663](https://arxiv.org/abs/1903.07663)

**Looking Fast and Slow: Memory-Guided Mobile Video Object Detection**

- intro: Cornell University & Google AI
- arxiv: [https://arxiv.org/abs/1903.10172](https://arxiv.org/abs/1903.10172)

**Progressive Sparse Local Attention for Video object detection**

- intro: NLPR,CASIA & Horizon Robotics
- arxiv: [https://arxiv.org/abs/1903.09126](https://arxiv.org/abs/1903.09126)

**Sequence Level Semantics Aggregation for Video Object Detection**

- intro: ICCV 2019 oral
- arxiv: [https://arxiv.org/abs/1907.06390](https://arxiv.org/abs/1907.06390)
- github(MXNet): [https://github.com/happywu/Sequence-Level-Semantics-Aggregation](https://github.com/happywu/Sequence-Level-Semantics-Aggregation)

**Object Detection in Video with Spatial-temporal Context Aggregation**

- intro: Huazhong University of Science and Technology & Horizon Robotics Inc.
- arxiv: [https://arxiv.org/abs/1907.04988](https://arxiv.org/abs/1907.04988)

**A Delay Metric for Video Object Detection: What Average Precision Fails to Tell**

- intro: ICCV 2019
- arxiv: [https://arxiv.org/abs/1908.06368](https://arxiv.org/abs/1908.06368)

**Minimum Delay Object Detection From Video**

- intro: ICCV 2019
- arxiv: [https://arxiv.org/abs/1908.11092](https://arxiv.org/abs/1908.11092)

**Learning Motion Priors for Efficient Video Object Detection**

[https://arxiv.org/abs/1911.05253](https://arxiv.org/abs/1911.05253)

**Object-aware Feature Aggregation for Video Object Detection**

- intro: Beihang University & Capital Normal University & The University of Hong Kong & Baidu, Inc.
- arxiv: [https://arxiv.org/abs/2010.12573](https://arxiv.org/abs/2010.12573)

**End-to-End Video Object Detection with Spatial-Temporal Transformers**

- arxiv: [https://arxiv.org/abs/2105.10920](https://arxiv.org/abs/2105.10920)
- github: [https://github.com/SJTU-LuHe/TransVOD](https://github.com/SJTU-LuHe/TransVOD)

# Object Detection on Mobile Devices

**Pelee: A Real-Time Object Detection System on Mobile Devices**

- intro: ICLR 2018 workshop track
- intro: based on the SSD
- arxiv: [https://arxiv.org/abs/1804.06882](https://arxiv.org/abs/1804.06882)
- github: [https://github.com/Robert-JunWang/Pelee](https://github.com/Robert-JunWang/Pelee)

# Object Detection on RGB-D

**Learning Rich Features from RGB-D Images for Object Detection and Segmentation**

- arxiv: [http://arxiv.org/abs/1407.5736](http://arxiv.org/abs/1407.5736)

**Differential Geometry Boosts Convolutional Neural Networks for Object Detection**

- intro: CVPR 2016
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html)

**A Self-supervised Learning System for Object Detection using Physics Simulation and Multi-view Pose Estimation**

[https://arxiv.org/abs/1703.03347](https://arxiv.org/abs/1703.03347)

**Cross-Modal Attentional Context Learning for RGB-D Object Detection**

- intro: IEEE Transactions on Image Processing
- arxiv: [https://arxiv.org/abs/1810.12829](https://arxiv.org/abs/1810.12829)

# Zero-Shot Object Detection

**Zero-Shot Detection**

- intro: Australian National University
- keywords: YOLO
- arxiv: [https://arxiv.org/abs/1803.07113](https://arxiv.org/abs/1803.07113)

**Zero-Shot Object Detection**

[https://arxiv.org/abs/1804.04340](https://arxiv.org/abs/1804.04340)

**Zero-Shot Object Detection: Learning to Simultaneously Recognize and Localize Novel Concepts**

- intro: Australian National University
- arxiv: [https://arxiv.org/abs/1803.06049](https://arxiv.org/abs/1803.06049)

**Zero-Shot Object Detection by Hybrid Region Embedding**

- intro: Middle East Technical University & Hacettepe University
- arxiv: [https://arxiv.org/abs/1805.06157](https://arxiv.org/abs/1805.06157)

# Visual Relationship Detection

**Visual Relationship Detection with Language Priors**

- intro: ECCV 2016 oral
- paper: [https://cs.stanford.edu/people/ranjaykrishna/vrd/vrd.pdf](https://cs.stanford.edu/people/ranjaykrishna/vrd/vrd.pdf)
- github: [https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection](https://github.com/Prof-Lu-Cewu/Visual-Relationship-Detection)

**ViP-CNN: A Visual Phrase Reasoning Convolutional Neural Network for Visual Relationship Detection**

- intro: Visual Phrase reasoning Convolutional Neural Network (ViP-CNN), Visual Phrase Reasoning Structure (VPRS)
- arxiv: [https://arxiv.org/abs/1702.07191](https://arxiv.org/abs/1702.07191)

**Visual Translation Embedding Network for Visual Relation Detection**

- arxiv: [https://www.arxiv.org/abs/1702.08319](https://www.arxiv.org/abs/1702.08319)

**Deep Variation-structured Reinforcement Learning for Visual Relationship and Attribute Detection**

- intro: CVPR 2017 spotlight paper
- arxiv: [https://arxiv.org/abs/1703.03054](https://arxiv.org/abs/1703.03054)

**Detecting Visual Relationships with Deep Relational Networks**

- intro: CVPR 2017 oral. The Chinese University of Hong Kong
- arxiv: [https://arxiv.org/abs/1704.03114](https://arxiv.org/abs/1704.03114)

**Identifying Spatial Relations in Images using Convolutional Neural Networks**

[https://arxiv.org/abs/1706.04215](https://arxiv.org/abs/1706.04215)

**PPR-FCN: Weakly Supervised Visual Relation Detection via Parallel Pairwise R-FCN**

- intro: ICCV
- arxiv: [https://arxiv.org/abs/1708.01956](https://arxiv.org/abs/1708.01956)

**Natural Language Guided Visual Relationship Detection**

[https://arxiv.org/abs/1711.06032](https://arxiv.org/abs/1711.06032)

**Detecting Visual Relationships Using Box Attention**

- intro: Google AI & IST Austria
- arxiv: [https://arxiv.org/abs/1807.02136](https://arxiv.org/abs/1807.02136)

**Google AI Open Images - Visual Relationship Track**

- intro: Detect pairs of objects in particular relationships
- kaggle: [https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track](https://www.kaggle.com/c/google-ai-open-images-visual-relationship-track)

**Context-Dependent Diffusion Network for Visual Relationship Detection**

- intro: 2018 ACM Multimedia Conference
- arxiv: [https://arxiv.org/abs/1809.06213](https://arxiv.org/abs/1809.06213)

**A Problem Reduction Approach for Visual Relationships Detection**

- intro: ECCV 2018 Workshop
- arxiv: [https://arxiv.org/abs/1809.09828](https://arxiv.org/abs/1809.09828)

**Exploring the Semantics for Visual Relationship Detection**

[https://arxiv.org/abs/1904.02104](https://arxiv.org/abs/1904.02104)

# Face Detection

**Multi-view Face Detection Using Deep Convolutional Neural Networks**

- intro: Yahoo
- arxiv: [http://arxiv.org/abs/1502.02766](http://arxiv.org/abs/1502.02766)
- github: [https://github.com/guoyilin/FaceDetection_CNN](https://github.com/guoyilin/FaceDetection_CNN)

**From Facial Parts Responses to Face Detection: A Deep Learning Approach**

![](http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/support/index.png)

- intro: ICCV 2015. CUHK
- project page: [http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/Faceness.html](http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/Faceness.html)
- arxiv: [https://arxiv.org/abs/1509.06451](https://arxiv.org/abs/1509.06451)
- paper: [http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yang_From_Facial_Parts_ICCV_2015_paper.pdf](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Yang_From_Facial_Parts_ICCV_2015_paper.pdf)

**Compact Convolutional Neural Network Cascade for Face Detection**

- arxiv: [http://arxiv.org/abs/1508.01292](http://arxiv.org/abs/1508.01292)
- github: [https://github.com/Bkmz21/FD-Evaluation](https://github.com/Bkmz21/FD-Evaluation)
- github: [https://github.com/Bkmz21/CompactCNNCascade](https://github.com/Bkmz21/CompactCNNCascade)

**Face Detection with End-to-End Integration of a ConvNet and a 3D Model**

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1606.00850](https://arxiv.org/abs/1606.00850)
- github(MXNet): [https://github.com/tfwu/FaceDetection-ConvNet-3D](https://github.com/tfwu/FaceDetection-ConvNet-3D)

**CMS-RCNN: Contextual Multi-Scale Region-based CNN for Unconstrained Face Detection**

- intro: CMU
- arxiv: [https://arxiv.org/abs/1606.05413](https://arxiv.org/abs/1606.05413)

**Towards a Deep Learning Framework for Unconstrained Face Detection**

- intro: overlap with CMS-RCNN
- arxiv: [https://arxiv.org/abs/1612.05322](https://arxiv.org/abs/1612.05322)

**Supervised Transformer Network for Efficient Face Detection**

- arxiv: [http://arxiv.org/abs/1607.05477](http://arxiv.org/abs/1607.05477)

**UnitBox: An Advanced Object Detection Network**

- intro: ACM MM 2016
- intro: University of Illinois at Urbana−Champaign & Megvii Inc
- keywords: IOULoss
- arxiv: [http://arxiv.org/abs/1608.01471](http://arxiv.org/abs/1608.01471)

**Bootstrapping Face Detection with Hard Negative Examples**

- author: 万韶华 @ 小米.
- intro: Faster R-CNN, hard negative mining. state-of-the-art on the FDDB dataset
- arxiv: [http://arxiv.org/abs/1608.02236](http://arxiv.org/abs/1608.02236)

**Grid Loss: Detecting Occluded Faces**

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1609.00129](https://arxiv.org/abs/1609.00129)
- paper: [http://lrs.icg.tugraz.at/pubs/opitz_eccv_16.pdf](http://lrs.icg.tugraz.at/pubs/opitz_eccv_16.pdf)
- poster: [http://www.eccv2016.org/files/posters/P-2A-34.pdf](http://www.eccv2016.org/files/posters/P-2A-34.pdf)

**A Multi-Scale Cascade Fully Convolutional Network Face Detector**

- intro: ICPR 2016
- arxiv: [http://arxiv.org/abs/1609.03536](http://arxiv.org/abs/1609.03536)

## MTCNN

**Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks**

**Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks**

![](https://kpzhang93.github.io/MTCNN_face_detection_alignment/support/index.png)

- project page: [https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
- arxiv: [https://arxiv.org/abs/1604.02878](https://arxiv.org/abs/1604.02878)
- github(official, Matlab): [https://github.com/kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- github: [https://github.com/pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
- github: [https://github.com/DaFuCoding/MTCNN_Caffe](https://github.com/DaFuCoding/MTCNN_Caffe)
- github(MXNet): [https://github.com/Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)
- github: [https://github.com/Pi-DeepLearning/RaspberryPi-FaceDetection-MTCNN-Caffe-With-Motion](https://github.com/Pi-DeepLearning/RaspberryPi-FaceDetection-MTCNN-Caffe-With-Motion)
- github(Caffe): [https://github.com/foreverYoungGitHub/MTCNN](https://github.com/foreverYoungGitHub/MTCNN)
- github: [https://github.com/CongWeilin/mtcnn-caffe](https://github.com/CongWeilin/mtcnn-caffe)
- github(OpenCV+OpenBlas): [https://github.com/AlphaQi/MTCNN-light](https://github.com/AlphaQi/MTCNN-light)
- github(Tensorflow+golang): [https://github.com/jdeng/goface](https://github.com/jdeng/goface)

**Face Detection using Deep Learning: An Improved Faster RCNN Approach**

- intro: DeepIR Inc
- arxiv: [https://arxiv.org/abs/1701.08289](https://arxiv.org/abs/1701.08289)

**Faceness-Net: Face Detection through Deep Facial Part Responses**

- intro: An extended version of ICCV 2015 paper
- arxiv: [https://arxiv.org/abs/1701.08393](https://arxiv.org/abs/1701.08393)

**Multi-Path Region-Based Convolutional Neural Network for Accurate Detection of Unconstrained "Hard Faces"**

- intro: CVPR 2017. MP-RCNN, MP-RPN
- arxiv: [https://arxiv.org/abs/1703.09145](https://arxiv.org/abs/1703.09145)

**End-To-End Face Detection and Recognition**

[https://arxiv.org/abs/1703.10818](https://arxiv.org/abs/1703.10818)

**Face R-CNN**

[https://arxiv.org/abs/1706.01061](https://arxiv.org/abs/1706.01061)

**Face Detection through Scale-Friendly Deep Convolutional Networks**

[https://arxiv.org/abs/1706.02863](https://arxiv.org/abs/1706.02863)

**Scale-Aware Face Detection**

- intro: CVPR 2017. SenseTime & Tsinghua University
- arxiv: [https://arxiv.org/abs/1706.09876](https://arxiv.org/abs/1706.09876)

**Detecting Faces Using Inside Cascaded Contextual CNN**

- intro: CVPR 2017. Tencent AI Lab & SenseTime
- paper: [http://ai.tencent.com/ailab/media/publications/Detecting_Faces_Using_Inside_Cascaded_Contextual_CNN.pdf](http://ai.tencent.com/ailab/media/publications/Detecting_Faces_Using_Inside_Cascaded_Contextual_CNN.pdf)

**Multi-Branch Fully Convolutional Network for Face Detection**

[https://arxiv.org/abs/1707.06330](https://arxiv.org/abs/1707.06330)

**SSH: Single Stage Headless Face Detector**

- intro: ICCV 2017. University of Maryland
- arxiv: [https://arxiv.org/abs/1708.03979](https://arxiv.org/abs/1708.03979)
- github(official, Caffe): [https://github.com/mahyarnajibi/SSH](https://github.com/mahyarnajibi/SSH)

**Dockerface: an easy to install and use Faster R-CNN face detector in a Docker container**

[https://arxiv.org/abs/1708.04370](https://arxiv.org/abs/1708.04370)

**FaceBoxes: A CPU Real-time Face Detector with High Accuracy**

- intro: IJCB 2017
- keywords: Rapidly Digested Convolutional Layers (RDCL), Multiple Scale Convolutional Layers (MSCL)
- intro: the proposed detector runs at 20 FPS on a single CPU core and 125 FPS using a GPU for VGA-resolution images
- arxiv: [https://arxiv.org/abs/1708.05234](https://arxiv.org/abs/1708.05234)
- github(official): [https://github.com/sfzhang15/FaceBoxes](https://github.com/sfzhang15/FaceBoxes)
- github(Caffe): [https://github.com/zeusees/FaceBoxes](https://github.com/zeusees/FaceBoxes)

**S3FD: Single Shot Scale-invariant Face Detector**

- intro: ICCV 2017. Chinese Academy of Sciences
- intro: can run at 36 FPS on a Nvidia Titan X (Pascal) for VGA-resolution images
- arxiv: [https://arxiv.org/abs/1708.05237](https://arxiv.org/abs/1708.05237)
- github(Caffe, official): [https://github.com/sfzhang15/SFD](https://github.com/sfzhang15/SFD)
- github: [https://github.com//clcarwin/SFD_pytorch](https://github.com//clcarwin/SFD_pytorch)

**Detecting Faces Using Region-based Fully Convolutional Networks**

[https://arxiv.org/abs/1709.05256](https://arxiv.org/abs/1709.05256)

**AffordanceNet: An End-to-End Deep Learning Approach for Object Affordance Detection**

[https://arxiv.org/abs/1709.07326](https://arxiv.org/abs/1709.07326)

**Face Attention Network: An effective Face Detector for the Occluded Faces**

[https://arxiv.org/abs/1711.07246](https://arxiv.org/abs/1711.07246)

**Feature Agglomeration Networks for Single Stage Face Detection**

[https://arxiv.org/abs/1712.00721](https://arxiv.org/abs/1712.00721)

**Face Detection Using Improved Faster RCNN**

- intro: Huawei Cloud BU
- arxiv: [https://arxiv.org/abs/1802.02142](https://arxiv.org/abs/1802.02142)

**PyramidBox: A Context-assisted Single Shot Face Detector**

- intro: Baidu, Inc
- arxiv: [https://arxiv.org/abs/1803.07737](https://arxiv.org/abs/1803.07737)

**PyramidBox++: High Performance Detector for Finding Tiny Face**

- intro: Chinese Academy of Sciences & Baidu, Inc.
- arxiv: [https://arxiv.org/abs/1904.00386](https://arxiv.org/abs/1904.00386)

**A Fast Face Detection Method via Convolutional Neural Network**

- intro: Neurocomputing
- arxiv: [https://arxiv.org/abs/1803.10103](https://arxiv.org/abs/1803.10103)

**Beyond Trade-off: Accelerate FCN-based Face Detector with Higher Accuracy**

- intro: CVPR 2018. Beihang University & CUHK & Sensetime
- arxiv: [https://arxiv.org/abs/1804.05197](https://arxiv.org/abs/1804.05197)

**Real-Time Rotation-Invariant Face Detection with Progressive Calibration Networks**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1804.06039](https://arxiv.org/abs/1804.06039)
- github(binary library): [https://github.com/Jack-CV/PCN](https://github.com/Jack-CV/PCN)

**SFace: An Efficient Network for Face Detection in Large Scale Variations**

- intro: Beihang University & Megvii Inc. (Face++)
- arxiv: [https://arxiv.org/abs/1804.06559](https://arxiv.org/abs/1804.06559)

**Survey of Face Detection on Low-quality Images**

[https://arxiv.org/abs/1804.07362](https://arxiv.org/abs/1804.07362)

**Anchor Cascade for Efficient Face Detection**

- intro: The University of Sydney
- arxiv: [https://arxiv.org/abs/1805.03363](https://arxiv.org/abs/1805.03363)

**Adversarial Attacks on Face Detectors using Neural Net based Constrained Optimization**

- intro: IEEE MMSP
- arxiv: [https://arxiv.org/abs/1805.12302](https://arxiv.org/abs/1805.12302)

**Selective Refinement Network for High Performance Face Detection**

[https://arxiv.org/abs/1809.02693](https://arxiv.org/abs/1809.02693)

**DSFD: Dual Shot Face Detector**

[https://arxiv.org/abs/1810.10220](https://arxiv.org/abs/1810.10220)

**Learning Better Features for Face Detection with Feature Fusion and Segmentation Supervision**

[https://arxiv.org/abs/1811.08557](https://arxiv.org/abs/1811.08557)

**FA-RPN: Floating Region Proposals for Face Detection**

[https://arxiv.org/abs/1812.05586](https://arxiv.org/abs/1812.05586)

**Robust and High Performance Face Detector**

[https://arxiv.org/abs/1901.02350](https://arxiv.org/abs/1901.02350)

**DAFE-FD: Density Aware Feature Enrichment for Face Detection**

[https://arxiv.org/abs/1901.05375](https://arxiv.org/abs/1901.05375)

**Improved Selective Refinement Network for Face Detection**

- intro: Chinese Academy of Sciences & JD AI Research
- arxiv: [https://arxiv.org/abs/1901.06651](https://arxiv.org/abs/1901.06651)

**Revisiting a single-stage method for face detection**

[https://arxiv.org/abs/1902.01559](https://arxiv.org/abs/1902.01559)

**MSFD:Multi-Scale Receptive Field Face Detector**

- intro: ICPR 2018
- arxiv: [https://arxiv.org/abs/1903.04147](https://arxiv.org/abs/1903.04147)

**LFFD: A Light and Fast Face Detector for Edge Devices**

- arxiv: [https://arxiv.org/abs/1904.10633](https://arxiv.org/abs/1904.10633)
- github: [https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices](https://github.com/YonghaoHe/A-Light-and-Fast-Face-Detector-for-Edge-Devices)

**RetinaFace: Single-stage Dense Face Localisation in the Wild**

- intro: CVPR 2020
- arxiv: [https://arxiv.org/abs/1905.00641](https://arxiv.org/abs/1905.00641)
- gihtub: [https://github.com/deepinsight/insightface/tree/master/RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

**BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs**

- intro: CVPR Workshop on Computer Vision for Augmented and Virtual Reality, 2019
- arxiv: [https://arxiv.org/abs/1907.05047](https://arxiv.org/abs/1907.05047)

**HAMBox: Delving into Online High-quality Anchors Mining for Detecting Outer Faces**

- intro: Baidu Inc. &  Chinese Academy of Sciences
- arxiv: [https://arxiv.org/abs/1912.09231](https://arxiv.org/abs/1912.09231)

**KPNet: Towards Minimal Face Detector**

- intro: AAAI 2020
- arxiv: [https://arxiv.org/abs/2003.07543](https://arxiv.org/abs/2003.07543)

**ASFD: Automatic and Scalable Face Detector**

- intro: Youtu Lab, Tencent & Southeast University & Xiamen University
- arxiv: [https://arxiv.org/abs/2003.11228](https://arxiv.org/abs/2003.11228)

**TinaFace: Strong but Simple Baseline for Face Detection**

- intro: Media Intelligence Technology Co.,Ltd
- arxiv: [https://arxiv.org/abs/2011.13183](https://arxiv.org/abs/2011.13183)
- github(PyTorch): [https://github.com/Media-Smart/vedadet](https://github.com/Media-Smart/vedadet)

**MogFace: Rethinking Scale Augmentation on the Face Detector**

- intro: Alibaba Group & Imperial College
- arxiv: [https://arxiv.org/abs/2103.11139](https://arxiv.org/abs/2103.11139)

**HLA-Face: Joint High-Low Adaptation for Low Light Face Detection**

- intro: CVPR 2021
- intro: Peking University
- project page: [https://daooshee.github.io/HLA-Face-Website/](https://daooshee.github.io/HLA-Face-Website/)
- arxiv: [https://arxiv.org/abs/2104.01984](https://arxiv.org/abs/2104.01984)
- github: [https://github.com/daooshee/HLA-Face-Code](https://github.com/daooshee/HLA-Face-Code)

**1st Place Solutions for UG2+ Challenge 2021 -- (Semi-)supervised Face detection in the low light condition**

- intro: Tomorrow Advancing Life (TAL) Education Group
- arxiv: [https://arxiv.org/abs/2107.00818](https://arxiv.org/abs/2107.00818)

**MOS: A Low Latency and Lightweight Framework for Face Detection, Landmark Localization, and Head Pose Estimation**

- intro: BMVC 2021
- arxiv: [https://arxiv.org/abs/2110.10953](https://arxiv.org/abs/2110.10953)
- github: [https://github.com/lyp-deeplearning/MOS-Multi-Task-Face-Detect](https://github.com/lyp-deeplearning/MOS-Multi-Task-Face-Detect)

## Detect Small Faces

**Finding Tiny Faces**

- intro: CVPR 2017. CMU
- project page: [http://www.cs.cmu.edu/~peiyunh/tiny/index.html](http://www.cs.cmu.edu/~peiyunh/tiny/index.html)
- arxiv: [https://arxiv.org/abs/1612.04402](https://arxiv.org/abs/1612.04402)
- github(official, Matlab): [https://github.com/peiyunh/tiny](https://github.com/peiyunh/tiny)
- github(inference-only): [https://github.com/chinakook/hr101_mxnet](https://github.com/chinakook/hr101_mxnet)
- github: [https://github.com/cydonia999/Tiny_Faces_in_Tensorflow](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)

**Detecting and counting tiny faces**

- intro: ENS Paris-Saclay. ExtendedTinyFaces
- intro: Detecting and counting small objects - Analysis, review and application to counting
- arxiv: [https://arxiv.org/abs/1801.06504](https://arxiv.org/abs/1801.06504)
- github: [https://github.com/alexattia/ExtendedTinyFaces](https://github.com/alexattia/ExtendedTinyFaces)

**Seeing Small Faces from Robust Anchor's Perspective**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1802.09058](https://arxiv.org/abs/1802.09058)

**Face-MagNet: Magnifying Feature Maps to Detect Small Faces**

- intro: WACV 2018
- keywords: Face Magnifier Network (Face-MageNet)
- arxiv: [https://arxiv.org/abs/1803.05258](https://arxiv.org/abs/1803.05258)
- github: [https://github.com/po0ya/face-magnet](https://github.com/po0ya/face-magnet)

**Robust Face Detection via Learning Small Faces on Hard Images**

- intro: Johns Hopkins University & Stanford University
- arxiv: [https://arxiv.org/abs/1811.11662](https://arxiv.org/abs/1811.11662)
- github: [https://github.com/bairdzhang/smallhardface](https://github.com/bairdzhang/smallhardface)

**SFA: Small Faces Attention Face Detector**

- intro: Jilin University
- arxiv: [https://arxiv.org/abs/1812.08402](https://arxiv.org/abs/1812.08402)

# Person Head Detection

**Context-aware CNNs for person head detection**

- intro: ICCV 2015
- project page: [http://www.di.ens.fr/willow/research/headdetection/](http://www.di.ens.fr/willow/research/headdetection/)
- arxiv: [http://arxiv.org/abs/1511.07917](http://arxiv.org/abs/1511.07917)
- github: [https://github.com/aosokin/cnn_head_detection](https://github.com/aosokin/cnn_head_detection)

**Detecting Heads using Feature Refine Net and Cascaded Multi-scale Architecture**

[https://arxiv.org/abs/1803.09256](https://arxiv.org/abs/1803.09256)

**A Comparison of CNN-based Face and Head Detectors for Real-Time Video Surveillance Applications**

[https://arxiv.org/abs/1809.03336](https://arxiv.org/abs/1809.03336)

**FCHD: A fast and accurate head detector**

- arxiv: [https://arxiv.org/abs/1809.08766](https://arxiv.org/abs/1809.08766)
- github(PyTorch, official): [https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector](https://github.com/aditya-vora/FCHD-Fully-Convolutional-Head-Detector)

**Relational Learning for Joint Head and Human Detection**

- keywords: JointDet, head-body Relationship Discriminating Module (RDM)
- arxiv: [https://arxiv.org/abs/1909.10674](https://arxiv.org/abs/1909.10674)

**Body-Face Joint Detection via Embedding and Head Hook**

- intro: ICCV 2021
- paper: [https://openaccess.thecvf.com/content/ICCV2021/papers/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Wan_Body-Face_Joint_Detection_via_Embedding_and_Head_Hook_ICCV_2021_paper.pdf)
- gihtub: [https://github.com/AibeeDetect/BFJDet](https://github.com/AibeeDetect/BFJDet)

# Pedestrian Detection / People Detection

**Pedestrian Detection aided by Deep Learning Semantic Tasks**

- intro: CVPR 2015
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/TA-CNN/](http://mmlab.ie.cuhk.edu.hk/projects/TA-CNN/)
- arxiv: [http://arxiv.org/abs/1412.0069](http://arxiv.org/abs/1412.0069)

**Deep Learning Strong Parts for Pedestrian Detection**

- intro: ICCV 2015. CUHK. DeepParts
- intro: Achieving 11.89% average miss rate on Caltech Pedestrian Dataset
- paper: [http://personal.ie.cuhk.edu.hk/~pluo/pdf/tianLWTiccv15.pdf](http://personal.ie.cuhk.edu.hk/~pluo/pdf/tianLWTiccv15.pdf)

**Taking a Deeper Look at Pedestrians**

- intro: CVPR 2015
- arxiv: [https://arxiv.org/abs/1501.05790](https://arxiv.org/abs/1501.05790)

**Convolutional Channel Features**

- intro: ICCV 2015
- arxiv: [https://arxiv.org/abs/1504.07339](https://arxiv.org/abs/1504.07339)
- github: [https://github.com/byangderek/CCF](https://github.com/byangderek/CCF)

**End-to-end people detection in crowded scenes**

- arxiv: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- github: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)
- youtube: [https://www.youtube.com/watch?v=QeWl0h3kQ24](https://www.youtube.com/watch?v=QeWl0h3kQ24)

**Learning Complexity-Aware Cascades for Deep Pedestrian Detection**

- intro: ICCV 2015
- arxiv: [https://arxiv.org/abs/1507.05348](https://arxiv.org/abs/1507.05348)

**Deep convolutional neural networks for pedestrian detection**

- arxiv: [http://arxiv.org/abs/1510.03608](http://arxiv.org/abs/1510.03608)
- github: [https://github.com/DenisTome/DeepPed](https://github.com/DenisTome/DeepPed)

**Scale-aware Fast R-CNN for Pedestrian Detection**

- arxiv: [https://arxiv.org/abs/1510.08160](https://arxiv.org/abs/1510.08160)

**New algorithm improves speed and accuracy of pedestrian detection**

- blog: [http://www.eurekalert.org/pub_releases/2016-02/uoc--nai020516.php](http://www.eurekalert.org/pub_releases/2016-02/uoc--nai020516.php)

**Pushing the Limits of Deep CNNs for Pedestrian Detection**

- intro: "set a new record on the Caltech pedestrian dataset, lowering the log-average miss rate from 11.7% to 8.9%"
- arxiv: [http://arxiv.org/abs/1603.04525](http://arxiv.org/abs/1603.04525)

**A Real-Time Deep Learning Pedestrian Detector for Robot Navigation**

- arxiv: [http://arxiv.org/abs/1607.04436](http://arxiv.org/abs/1607.04436)

**A Real-Time Pedestrian Detector using Deep Learning for Human-Aware Navigation**

- arxiv: [http://arxiv.org/abs/1607.04441](http://arxiv.org/abs/1607.04441)

**Is Faster R-CNN Doing Well for Pedestrian Detection?**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.07032](http://arxiv.org/abs/1607.07032)
- github: [https://github.com/zhangliliang/RPN_BF/tree/RPN-pedestrian](https://github.com/zhangliliang/RPN_BF/tree/RPN-pedestrian)

**Unsupervised Deep Domain Adaptation for Pedestrian Detection**

- intro: ECCV Workshop 2016
- arxiv: [https://arxiv.org/abs/1802.03269](https://arxiv.org/abs/1802.03269)

**Reduced Memory Region Based Deep Convolutional Neural Network Detection**

- intro: IEEE 2016 ICCE-Berlin
- arxiv: [http://arxiv.org/abs/1609.02500](http://arxiv.org/abs/1609.02500)

**Fused DNN: A deep neural network fusion approach to fast and robust pedestrian detection**

- arxiv: [https://arxiv.org/abs/1610.03466](https://arxiv.org/abs/1610.03466)

**Detecting People in Artwork with CNNs**

- intro: ECCV 2016 Workshops
- arxiv: [https://arxiv.org/abs/1610.08871](https://arxiv.org/abs/1610.08871)

**Deep Multi-camera People Detection**

- arxiv: [https://arxiv.org/abs/1702.04593](https://arxiv.org/abs/1702.04593)

**Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters**

- intro: CVPR 2017
- project page: [http://ml.cs.tsinghua.edu.cn:5000/publications/synunity/](http://ml.cs.tsinghua.edu.cn:5000/publications/synunity/)
- arxiv: [https://arxiv.org/abs/1703.06283](https://arxiv.org/abs/1703.06283)
- github(Tensorflow): [https://github.com/huangshiyu13/RPNplus](https://github.com/huangshiyu13/RPNplus)

**What Can Help Pedestrian Detection?**

- intro: CVPR 2017. Tsinghua University & Peking University & Megvii Inc.
- keywords: Faster R-CNN, HyperLearner
- arxiv: [https://arxiv.org/abs/1705.02757](https://arxiv.org/abs/1705.02757)
- paper: [http://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mao_What_Can_Help_CVPR_2017_paper.pdf)

**Illuminating Pedestrians via Simultaneous Detection & Segmentation**

[https://arxiv.org/abs/1706.08564](https://arxiv.org/abs/1706.08564)

**Rotational Rectification Network for Robust Pedestrian Detection**

- intro: CMU & Volvo Construction
- arxiv: [https://arxiv.org/abs/1706.08917](https://arxiv.org/abs/1706.08917)

**STD-PD: Generating Synthetic Training Data for Pedestrian Detection in Unannotated Videos**

- intro: The University of North Carolina at Chapel Hill
- arxiv: [https://arxiv.org/abs/1707.09100](https://arxiv.org/abs/1707.09100)

**Too Far to See? Not Really! --- Pedestrian Detection with Scale-aware Localization Policy**

[https://arxiv.org/abs/1709.00235](https://arxiv.org/abs/1709.00235)

**Aggregated Channels Network for Real-Time Pedestrian Detection**

[https://arxiv.org/abs/1801.00476](https://arxiv.org/abs/1801.00476)

**Exploring Multi-Branch and High-Level Semantic Networks for Improving Pedestrian Detection**

[https://arxiv.org/abs/1804.00872](https://arxiv.org/abs/1804.00872)

**Pedestrian-Synthesis-GAN: Generating Pedestrian Data in Real Scene and Beyond**

[https://arxiv.org/abs/1804.02047](https://arxiv.org/abs/1804.02047)

**PCN: Part and Context Information for Pedestrian Detection with CNNs**

- intro: British Machine Vision Conference(BMVC) 2017
- arxiv: [https://arxiv.org/abs/1804.04483](https://arxiv.org/abs/1804.04483)

**Improving Occlusion and Hard Negative Handling for Single-Stage Pedestrian Detectors**

- intro: CVPR 2018
- paper: [http://openaccess.thecvf.com/content_cvpr_2018/papers/Noh_Improving_Occlusion_and_CVPR_2018_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2018/papers/Noh_Improving_Occlusion_and_CVPR_2018_paper.pdf)

**Small-scale Pedestrian Detection Based on Somatic Topology Localization and Temporal Feature Aggregation**

- intro: ECCV 2018
- intro: Hikvision Research Institute
- arxiv: [https://arxiv.org/abs/1807.01438](https://arxiv.org/abs/1807.01438)

**Bi-box Regression for Pedestrian Detection and Occlusion Estimation**

- intro: ECCV 2018
- paper: [http://openaccess.thecvf.com/content_ECCV_2018/papers/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.pdf](http://openaccess.thecvf.com/content_ECCV_2018/papers/CHUNLUAN_ZHOU_Bi-box_Regression_for_ECCV_2018_paper.pdf)
- github(Pytorch): [https://github.com/rainofmine/Bi-box_Regression](https://github.com/rainofmine/Bi-box_Regression)

**Pedestrian Detection with Autoregressive Network Phases**

- intro: Michigan State University
- arxiv: [https://arxiv.org/abs/1812.00440](https://arxiv.org/abs/1812.00440)

**SSA-CNN: Semantic Self-Attention CNN for Pedestrian Detection**

[https://arxiv.org/abs/1902.09080](https://arxiv.org/abs/1902.09080)

**High-level Semantic Feature Detection:A New Perspective for Pedestrian Detection**

**Center and Scale Prediction: A Box-free Approach for Object Detection**

- intro: CVPR 2019
- intro: National University of Defense Technology & Chinese Academy of Sciences & Inception Institute of Artificial Intelligence (IIAI) & Horizon Robotics Inc.
- arxiv: [https://arxiv.org/abs/1904.02948](https://arxiv.org/abs/1904.02948)
- github(official, Keras): [https://github.com/liuwei16/CSP](https://github.com/liuwei16/CSP)

**Evading Real-Time Person Detectors by Adversarial T-shirt**

[https://arxiv.org/abs/1910.11099](https://arxiv.org/abs/1910.11099)

**Coupled Network for Robust Pedestrian Detection with Gated Multi-Layer Feature Extraction and Deformable Occlusion Handling**

[https://arxiv.org/abs/1912.08661](https://arxiv.org/abs/1912.08661)

**Scale Match for Tiny Person Detection**

- intro: WACV 2020
- arxiv: [https://arxiv.org/abs/1912.10664](https://arxiv.org/abs/1912.10664)
- github: [https://github.com/ucas-vg/TinyBenchmark](https://github.com/ucas-vg/TinyBenchmark)

**SM+: Refined Scale Match for Tiny Person Detection**

[https://arxiv.org/abs/2102.03558](https://arxiv.org/abs/2102.03558)

**Resisting the Distracting-factors in Pedestrian Detection**

- intro: Beihang University & Arizona State University
- arxiv: [https://arxiv.org/abs/2005.07344](https://arxiv.org/abs/2005.07344)

**SADet: Learning An Efficient and Accurate Pedestrian Detector**

[https://arxiv.org/abs/2007.13119](https://arxiv.org/abs/2007.13119)

**NOH-NMS: Improving Pedestrian Detection by Nearby Objects Hallucination**

- intro: ACM MM 2020
- intro: Tencent Youtu Lab
- arxiv: [https://arxiv.org/abs/2007.13376](https://arxiv.org/abs/2007.13376)

**Anchor-free Small-scale Multispectral Pedestrian Detection**

- intro: BMVC 2020
- arxiv: [https://arxiv.org/abs/2008.08418](https://arxiv.org/abs/2008.08418)
- github: [https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection](https://github.com/HensoldtOptronicsCV/MultispectralPedestrianDetection)

**LLA: Loss-aware Label Assignment for Dense Pedestrian Detection**

- arxiv: [https://arxiv.org/abs/2101.04307](https://arxiv.org/abs/2101.04307)
- github: [https://github.com/Megvii-BaseDetection/LLA](https://github.com/Megvii-BaseDetection/LLA)

**DETR for Pedestrian Detection**

[https://arxiv.org/abs/2012.06785](https://arxiv.org/abs/2012.06785)

**V2F-Net: Explicit Decomposition of Occluded Pedestrian Detection**

- intro: MEGVII Technology & Texas A&M University
- arxiv: [https://arxiv.org/abs/2104.03106](https://arxiv.org/abs/2104.03106)

## Pedestrian Detection in a Crowd

**Repulsion Loss: Detecting Pedestrians in a Crowd**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1711.07752](https://arxiv.org/abs/1711.07752)

**Occlusion-aware R-CNN: Detecting Pedestrians in a Crowd**

- intro: ECCV 2018
- arxiv: [https://arxiv.org/abs/1807.08407](https://arxiv.org/abs/1807.08407)

**Adaptive NMS: Refining Pedestrian Detection in a Crowd**

- intro: CVPR 2019 oral
- arxiv: [https://arxiv.org/abs/1904.03629](https://arxiv.org/abs/1904.03629)

**PedHunter: Occlusion Robust Pedestrian Detector in Crowded Scenes**

- keywords: SUR-PED
- arxiv: [https://arxiv.org/abs/1909.06826](https://arxiv.org/abs/1909.06826)

**Double Anchor R-CNN for Human Detection in a Crowd**

- intro: Megvii Inc. (Face++) & Tsinghua University & Xi’an Jiaotong University & Zhejiang University
- arxiv: [https://arxiv.org/abs/1909.09998](https://arxiv.org/abs/1909.09998)

**CSID: Center, Scale, Identity and Density-aware Pedestrian Detection in a Crowd**

[https://arxiv.org/abs/1910.09188](https://arxiv.org/abs/1910.09188)

**Semantic Head Enhanced Pedestrian Detection in a Crowd**

[https://arxiv.org/abs/1911.11985](https://arxiv.org/abs/1911.11985)

**Detection in Crowded Scenes: One Proposal, Multiple Predictions**

- intro: CVPR 2020 Oral
- arxiv: [https://arxiv.org/abs/2003.09163](https://arxiv.org/abs/2003.09163)
- github: [https://github.com/Purkialo/CrowdDet](https://github.com/Purkialo/CrowdDet)

**Visible Feature Guidance for Crowd Pedestrian Detection**

- intro: ECCV 2020 RLQ Workshop
- arxiv: [https://arxiv.org/abs/2008.09993](https://arxiv.org/abs/2008.09993)

# Occluded Pedestrian Detection

**Mask-Guided Attention Network for Occluded Pedestrian Detection**

- intro: ICCV 2019
- arxiv: [https://arxiv.org/abs/1910.06160](https://arxiv.org/abs/1910.06160)
- github: [https://github.com/Leotju/MGAN](https://github.com/Leotju/MGAN)

## Multispectral Pedestrian Detection

**Multispectral Deep Neural Networks for Pedestrian Detection**

- intro: BMVC 2016 oral
- arxiv: [https://arxiv.org/abs/1611.02644](https://arxiv.org/abs/1611.02644)

**Illumination-aware Faster R-CNN for Robust Multispectral Pedestrian Detection**

- intro: State Key Lab of CAD&CG, Zhejiang University
- arxiv: [https://arxiv.org/abs/1803.05347](https://arxiv.org/abs/1803.05347)

**Multispectral Pedestrian Detection via Simultaneous Detection and Segmentation**

- intro: BMVC 2018
- arxiv: [https://arxiv.org/abs/1808.04818](https://arxiv.org/abs/1808.04818)

**The Cross-Modality Disparity Problem in Multispectral Pedestrian Detection**

[https://arxiv.org/abs/1901.02645](https://arxiv.org/abs/1901.02645)

**Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pedestrian Detection**

[https://arxiv.org/abs/1902.05291](https://arxiv.org/abs/1902.05291)

**GFD-SSD: Gated Fusion Double SSD for Multispectral Pedestrian Detection**

[https://arxiv.org/abs/1903.06999](https://arxiv.org/abs/1903.06999)

**Unsupervised Domain Adaptation for Multispectral Pedestrian Detection**

[https://arxiv.org/abs/1904.03692](https://arxiv.org/abs/1904.03692)

# Vehicle Detection

**DAVE: A Unified Framework for Fast Vehicle Detection and Annotation**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.04564](http://arxiv.org/abs/1607.04564)

**Evolving Boxes for fast Vehicle Detection**

- arxiv: [https://arxiv.org/abs/1702.00254](https://arxiv.org/abs/1702.00254)

**Fine-Grained Car Detection for Visual Census Estimation**

- intro: AAAI 2016
- arxiv: [https://arxiv.org/abs/1709.02480](https://arxiv.org/abs/1709.02480)

**SINet: A Scale-insensitive Convolutional Neural Network for Fast Vehicle Detection**

- intro: IEEE Transactions on Intelligent Transportation Systems (T-ITS)
- arxiv: [https://arxiv.org/abs/1804.00433](https://arxiv.org/abs/1804.00433)

**Label and Sample: Efficient Training of Vehicle Object Detector from Sparsely Labeled Data**

- intro: UC Berkeley
- arxiv: [https://arxiv.org/abs/1808.08603](https://arxiv.org/abs/1808.08603)

**Domain Randomization for Scene-Specific Car Detection and Pose Estimation**

[https://arxiv.org/abs/1811.05939](https://arxiv.org/abs/1811.05939)

**ShuffleDet: Real-Time Vehicle Detection Network in On-board Embedded UAV Imagery**

- intro: ECCV 2018, UAVision 2018
- arxiv: [https://arxiv.org/abs/1811.06318](https://arxiv.org/abs/1811.06318)

# Traffic-Sign Detection

**Traffic-Sign Detection and Classification in the Wild**

- intro: CVPR 2016
- project page(code+dataset): [http://cg.cs.tsinghua.edu.cn/traffic-sign/](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)
- code & model: [http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip](http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip)

**Evaluating State-of-the-art Object Detector on Challenging Traffic Light Data**

- intro: CVPR 2017 workshop
- paper: [http://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/papers/Jensen_Evaluating_State-Of-The-Art_Object_CVPR_2017_paper.pdf](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w9/papers/Jensen_Evaluating_State-Of-The-Art_Object_CVPR_2017_paper.pdf)

**Detecting Small Signs from Large Images**

- intro: IEEE Conference on Information Reuse and Integration (IRI) 2017 oral
- arxiv: [https://arxiv.org/abs/1706.08574](https://arxiv.org/abs/1706.08574)

**Localized Traffic Sign Detection with Multi-scale Deconvolution Networks**

[https://arxiv.org/abs/1804.10428](https://arxiv.org/abs/1804.10428)

**Detecting Traffic Lights by Single Shot Detection**

- intro: ITSC 2018
- arxiv: [https://arxiv.org/abs/1805.02523](https://arxiv.org/abs/1805.02523)

**A Hierarchical Deep Architecture and Mini-Batch Selection Method For Joint Traffic Sign and Light Detection**

- intro: IEEE 15th Conference on Computer and Robot Vision
- arxiv: [https://arxiv.org/abs/1806.07987](https://arxiv.org/abs/1806.07987)
- demo: [https://www.youtube.com/watch?v=_YmogPzBXOw&feature=youtu.be](https://www.youtube.com/watch?v=_YmogPzBXOw&feature=youtu.be)

# Skeleton Detection

**Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs**

![](https://camo.githubusercontent.com/88a65f132aa4ae4b0477e3ad02c13cdc498377d9/687474703a2f2f37786e37777a2e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f44656570536b656c65746f6e2e706e673f696d61676556696577322f322f772f353030)

- arxiv: [http://arxiv.org/abs/1603.09446](http://arxiv.org/abs/1603.09446)
- github: [https://github.com/zeakey/DeepSkeleton](https://github.com/zeakey/DeepSkeleton)

**DeepSkeleton: Learning Multi-task Scale-associated Deep Side Outputs for Object Skeleton Extraction in Natural Images**

- arxiv: [http://arxiv.org/abs/1609.03659](http://arxiv.org/abs/1609.03659)

**SRN: Side-output Residual Network for Object Symmetry Detection in the Wild**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1703.02243](https://arxiv.org/abs/1703.02243)
- github: [https://github.com/KevinKecc/SRN](https://github.com/KevinKecc/SRN)

**Hi-Fi: Hierarchical Feature Integration for Skeleton Detection**

[https://arxiv.org/abs/1801.01849](https://arxiv.org/abs/1801.01849)

# Fruit Detection

**Deep Fruit Detection in Orchards**

- arxiv: [https://arxiv.org/abs/1610.03677](https://arxiv.org/abs/1610.03677)

**Image Segmentation for Fruit Detection and Yield Estimation in Apple Orchards**

- intro: The Journal of Field Robotics in May 2016
- project page: [http://confluence.acfr.usyd.edu.au/display/AGPub/](http://confluence.acfr.usyd.edu.au/display/AGPub/)
- arxiv: [https://arxiv.org/abs/1610.08120](https://arxiv.org/abs/1610.08120)

## Shadow Detection

**Fast Shadow Detection from a Single Image Using a Patched Convolutional Neural Network**

[https://arxiv.org/abs/1709.09283](https://arxiv.org/abs/1709.09283)

**A+D-Net: Shadow Detection with Adversarial Shadow Attenuation**

[https://arxiv.org/abs/1712.01361](https://arxiv.org/abs/1712.01361)

**Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal**

[https://arxiv.org/abs/1712.02478](https://arxiv.org/abs/1712.02478)

**Direction-aware Spatial Context Features for Shadow Detection**

- intro: CVPR 2018
- arxiv: [https://arxiv.org/abs/1712.04142](https://arxiv.org/abs/1712.04142)

**Direction-aware Spatial Context Features for Shadow Detection and Removal**

- intro: The Chinese University of Hong Kong & The Hong Kong Polytechnic University
- arxiv:  [https://arxiv.org/abs/1805.04635](https://arxiv.org/abs/1805.04635)

# Others Detection

**Deep Deformation Network for Object Landmark Localization**

- arxiv: [http://arxiv.org/abs/1605.01014](http://arxiv.org/abs/1605.01014)

**Fashion Landmark Detection in the Wild**

- intro: ECCV 2016
- project page: [http://personal.ie.cuhk.edu.hk/~lz013/projects/FashionLandmarks.html](http://personal.ie.cuhk.edu.hk/~lz013/projects/FashionLandmarks.html)
- arxiv: [http://arxiv.org/abs/1608.03049](http://arxiv.org/abs/1608.03049)
- github(Caffe): [https://github.com/liuziwei7/fashion-landmarks](https://github.com/liuziwei7/fashion-landmarks)

**Deep Learning for Fast and Accurate Fashion Item Detection**

- intro: Kuznech Inc.
- intro: MultiBox and Fast R-CNN
- paper: [https://kddfashion2016.mybluemix.net/kddfashion_finalSubmissions/Deep%20Learning%20for%20Fast%20and%20Accurate%20Fashion%20Item%20Detection.pdf](https://kddfashion2016.mybluemix.net/kddfashion_finalSubmissions/Deep%20Learning%20for%20Fast%20and%20Accurate%20Fashion%20Item%20Detection.pdf)

**OSMDeepOD - OSM and Deep Learning based Object Detection from Aerial Imagery (formerly known as "OSM-Crosswalk-Detection")**

![](https://raw.githubusercontent.com/geometalab/OSMDeepOD/master/imgs/process.png)

- github: [https://github.com/geometalab/OSMDeepOD](https://github.com/geometalab/OSMDeepOD)

**Selfie Detection by Synergy-Constraint Based Convolutional Neural Network**

- intro:  IEEE SITIS 2016
- arxiv: [https://arxiv.org/abs/1611.04357](https://arxiv.org/abs/1611.04357)

**Associative Embedding:End-to-End Learning for Joint Detection and Grouping**

- arxiv: [https://arxiv.org/abs/1611.05424](https://arxiv.org/abs/1611.05424)

**Deep Cuboid Detection: Beyond 2D Bounding Boxes**

- intro: CMU & Magic Leap
- arxiv: [https://arxiv.org/abs/1611.10010](https://arxiv.org/abs/1611.10010)

**Automatic Model Based Dataset Generation for Fast and Accurate Crop and Weeds Detection**

- arxiv: [https://arxiv.org/abs/1612.03019](https://arxiv.org/abs/1612.03019)

**Deep Learning Logo Detection with Data Expansion by Synthesising Context**

- arxiv: [https://arxiv.org/abs/1612.09322](https://arxiv.org/abs/1612.09322)

**Scalable Deep Learning Logo Detection**

[https://arxiv.org/abs/1803.11417](https://arxiv.org/abs/1803.11417)

**Pixel-wise Ear Detection with Convolutional Encoder-Decoder Networks**

- arxiv: [https://arxiv.org/abs/1702.00307](https://arxiv.org/abs/1702.00307)

**Automatic Handgun Detection Alarm in Videos Using Deep Learning**

- arxiv: [https://arxiv.org/abs/1702.05147](https://arxiv.org/abs/1702.05147)
- results: [https://github.com/SihamTabik/Pistol-Detection-in-Videos](https://github.com/SihamTabik/Pistol-Detection-in-Videos)

**Objects as context for part detection**

[https://arxiv.org/abs/1703.09529](https://arxiv.org/abs/1703.09529)

**Using Deep Networks for Drone Detection**

- intro: AVSS 2017
- arxiv: [https://arxiv.org/abs/1706.05726](https://arxiv.org/abs/1706.05726)

**Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.01642](https://arxiv.org/abs/1708.01642)

**Target Driven Instance Detection**

[https://arxiv.org/abs/1803.04610](https://arxiv.org/abs/1803.04610)

**DeepVoting: An Explainable Framework for Semantic Part Detection under Partial Occlusion**

[https://arxiv.org/abs/1709.04577](https://arxiv.org/abs/1709.04577)

**VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection and Recognition**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1710.06288](https://arxiv.org/abs/1710.06288)
- github: [https://github.com/SeokjuLee/VPGNet](https://github.com/SeokjuLee/VPGNet)

**Grab, Pay and Eat: Semantic Food Detection for Smart Restaurants**

[https://arxiv.org/abs/1711.05128](https://arxiv.org/abs/1711.05128)

**ReMotENet: Efficient Relevant Motion Event Detection for Large-scale Home Surveillance Videos**

- intro: WACV 2018
- arxiv: [https://arxiv.org/abs/1801.02031](https://arxiv.org/abs/1801.02031)

**Deep Learning Object Detection Methods for Ecological Camera Trap Data**

- intro: Conference of Computer and Robot Vision. University of Guelph
- arxiv: [https://arxiv.org/abs/1803.10842](https://arxiv.org/abs/1803.10842)

**EL-GAN: Embedding Loss Driven Generative Adversarial Networks for Lane Detection**

[https://arxiv.org/abs/1806.05525](https://arxiv.org/abs/1806.05525)

**Towards End-to-End Lane Detection: an Instance Segmentation Approach**

- arxiv: [https://arxiv.org/abs/1802.05591](https://arxiv.org/abs/1802.05591)
- github: [https://github.com/MaybeShewill-CV/lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

**Densely Supervised Grasp Detector (DSGD)**

[https://arxiv.org/abs/1810.03962](https://arxiv.org/abs/1810.03962)

# Object Proposal

**DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers**

- arxiv: [http://arxiv.org/abs/1510.04445](http://arxiv.org/abs/1510.04445)
- github: [https://github.com/aghodrati/deepproposal](https://github.com/aghodrati/deepproposal)

**Scale-aware Pixel-wise Object Proposal Networks**

- intro: IEEE Transactions on Image Processing
- arxiv: [http://arxiv.org/abs/1601.04798](http://arxiv.org/abs/1601.04798)

**Attend Refine Repeat: Active Box Proposal Generation via In-Out Localization**

- intro: BMVC 2016. AttractioNet
- arxiv: [https://arxiv.org/abs/1606.04446](https://arxiv.org/abs/1606.04446)
- github: [https://github.com/gidariss/AttractioNet](https://github.com/gidariss/AttractioNet)

**Learning to Segment Object Proposals via Recursive Neural Networks**

- arxiv: [https://arxiv.org/abs/1612.01057](https://arxiv.org/abs/1612.01057)

**Learning Detection with Diverse Proposals**

- intro: CVPR 2017
- keywords: differentiable Determinantal Point Process (DPP) layer, Learning Detection with Diverse Proposals (LDDP)
- arxiv: [https://arxiv.org/abs/1704.03533](https://arxiv.org/abs/1704.03533)

**ScaleNet: Guiding Object Proposal Generation in Supermarkets and Beyond**

- keywords: product detection
- arxiv: [https://arxiv.org/abs/1704.06752](https://arxiv.org/abs/1704.06752)

**Improving Small Object Proposals for Company Logo Detection**

- intro: ICMR 2017
- arxiv: [https://arxiv.org/abs/1704.08881](https://arxiv.org/abs/1704.08881)

**Open Logo Detection Challenge**

- intro: BMVC 2018
- keywords: QMUL-OpenLogo
- project page: [https://qmul-openlogo.github.io/](https://qmul-openlogo.github.io/)
- arxiv: [https://arxiv.org/abs/1807.01964](https://arxiv.org/abs/1807.01964)

**AttentionMask: Attentive, Efficient Object Proposal Generation Focusing on Small Objects**

- intro: ACCV 2018 oral
- arxiv: [https://arxiv.org/abs/1811.08728](https://arxiv.org/abs/1811.08728)
- github: [https://github.com/chwilms/AttentionMask](https://github.com/chwilms/AttentionMask)

# Localization

**Beyond Bounding Boxes: Precise Localization of Objects in Images**

- intro: PhD Thesis
- homepage: [http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html](http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html)
- phd-thesis: [http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.pdf](http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.pdf)
- github("SDS using hypercolumns"): [https://github.com/bharath272/sds](https://github.com/bharath272/sds)

**Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning**

- arxiv: [http://arxiv.org/abs/1503.00949](http://arxiv.org/abs/1503.00949)

**Weakly Supervised Object Localization Using Size Estimates**

- arxiv: [http://arxiv.org/abs/1608.04314](http://arxiv.org/abs/1608.04314)

**Active Object Localization with Deep Reinforcement Learning**

- intro: ICCV 2015
- keywords: Markov Decision Process
- arxiv: [https://arxiv.org/abs/1511.06015](https://arxiv.org/abs/1511.06015)

**Localizing objects using referring expressions**

- intro: ECCV 2016
- keywords: LSTM, multiple instance learning (MIL)
- paper: [http://www.umiacs.umd.edu/~varun/files/refexp-ECCV16.pdf](http://www.umiacs.umd.edu/~varun/files/refexp-ECCV16.pdf)
- github: [https://github.com/varun-nagaraja/referring-expressions](https://github.com/varun-nagaraja/referring-expressions)

**LocNet: Improving Localization Accuracy for Object Detection**

- intro: CVPR 2016 oral
- arxiv: [http://arxiv.org/abs/1511.07763](http://arxiv.org/abs/1511.07763)
- github: [https://github.com/gidariss/LocNet](https://github.com/gidariss/LocNet)

**Learning Deep Features for Discriminative Localization**

![](http://cnnlocalization.csail.mit.edu/framework.jpg)

- homepage: [http://cnnlocalization.csail.mit.edu/](http://cnnlocalization.csail.mit.edu/)
- arxiv: [http://arxiv.org/abs/1512.04150](http://arxiv.org/abs/1512.04150)
- github(Tensorflow): [https://github.com/jazzsaxmafia/Weakly_detector](https://github.com/jazzsaxmafia/Weakly_detector)
- github: [https://github.com/metalbubble/CAM](https://github.com/metalbubble/CAM)
- github: [https://github.com/tdeboissiere/VGG16CAM-keras](https://github.com/tdeboissiere/VGG16CAM-keras)

**ContextLocNet: Context-Aware Deep Network Models for Weakly Supervised Localization**

![](http://www.di.ens.fr/willow/research/contextlocnet/model.png)

- intro: ECCV 2016
- project page: [http://www.di.ens.fr/willow/research/contextlocnet/](http://www.di.ens.fr/willow/research/contextlocnet/)
- arxiv: [http://arxiv.org/abs/1609.04331](http://arxiv.org/abs/1609.04331)
- github: [https://github.com/vadimkantorov/contextlocnet](https://github.com/vadimkantorov/contextlocnet)

**Ensemble of Part Detectors for Simultaneous Classification and Localization**

[https://arxiv.org/abs/1705.10034](https://arxiv.org/abs/1705.10034)

**STNet: Selective Tuning of Convolutional Networks for Object Localization**

[https://arxiv.org/abs/1708.06418](https://arxiv.org/abs/1708.06418)

**Soft Proposal Networks for Weakly Supervised Object Localization**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1709.01829](https://arxiv.org/abs/1709.01829)

**Fine-grained Discriminative Localization via Saliency-guided Faster R-CNN**

- intro: ACM MM 2017
- arxiv: [https://arxiv.org/abs/1709.08295](https://arxiv.org/abs/1709.08295)

# Tutorials / Talks

**Convolutional Feature Maps: Elements of efficient (and accurate) CNN-based object detection**

- slides: [http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)

**Towards Good Practices for Recognition & Detection**

- intro: Hikvision Research Institute. Supervised Data Augmentation (SDA)
- slides: [http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf](http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf)

**Work in progress: Improving object detection and instance segmentation for small objects**

[https://docs.google.com/presentation/d/1OTfGn6mLe1VWE8D0q6Tu_WwFTSoLGd4OF8WCYnOWcVo/edit#slide=id.g37418adc7a_0_229](https://docs.google.com/presentation/d/1OTfGn6mLe1VWE8D0q6Tu_WwFTSoLGd4OF8WCYnOWcVo/edit#slide=id.g37418adc7a_0_229)

**Object Detection with Deep Learning: A Review**

[https://arxiv.org/abs/1807.05511](https://arxiv.org/abs/1807.05511)

# Projects

**Detectron**

- intro: FAIR's research platform for object detection research, implementing popular algorithms like Mask R-CNN and RetinaNet.
- github: [https://github.com/facebookresearch/Detectron](https://github.com/facebookresearch/Detectron)

**Detectron2**

- intro: Detectron2 is FAIR's next-generation platform for object detection and segmentation.
- github: [https://github.com/facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

**MMDetection**

- intro: MMDetection: Open MMLab Detection Toolbox and Benchmark
- arxiv: [https://arxiv.org/abs/1906.07155](https://arxiv.org/abs/1906.07155)
- github: [https://github.com/open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection)
- docs: [https://mmdetection.readthedocs.io/en/latest/](https://mmdetection.readthedocs.io/en/latest/)

**SimpleDet - A Simple and Versatile Framework for Object Detection and Instance Recognition**

- intro: A Simple and Versatile Framework for Object Detection and Instance Recognition
- github: [https://github.com/TuSimple/simpledet](https://github.com/TuSimple/simpledet)

**AdelaiDet**

- intro: AdelaiDet is an open source toolbox for multiple instance-level detection and recognition tasks.
- github: [https://github.com/aim-uofa/AdelaiDet/](https://github.com/aim-uofa/AdelaiDet/)

**TensorBox: a simple framework for training neural networks to detect objects in images**

- intro: "The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. 
We additionally provide an implementation of the [ReInspect](https://github.com/Russell91/ReInspect/) algorithm"
- github: [https://github.com/Russell91/TensorBox](https://github.com/Russell91/TensorBox)

**NanoDet**

- intro: Super fast and lightweight anchor-free object detection model. Real-time on mobile devices.
- arxiv: [https://github.com/RangiLyu/nanodet](https://github.com/RangiLyu/nanodet)

**Object detection in torch: Implementation of some object detection frameworks in torch**

- github: [https://github.com/fmassa/object-detection.torch](https://github.com/fmassa/object-detection.torch)

**Using DIGITS to train an Object Detection network**

- github: [https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/README.md](https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/README.md)

**FCN-MultiBox Detector**

- intro: Full convolution MultiBox Detector (like SSD) implemented in Torch.
- github: [https://github.com/teaonly/FMD.torch](https://github.com/teaonly/FMD.torch)

**KittiBox: A car detection model implemented in Tensorflow.**

- keywords: MultiNet
- intro: KittiBox is a collection of scripts to train out model FastBox on the Kitti Object Detection Dataset
- github: [https://github.com/MarvinTeichmann/KittiBox](https://github.com/MarvinTeichmann/KittiBox)

**Deformable Convolutional Networks + MST + Soft-NMS**

- github: [https://github.com/bharatsingh430/Deformable-ConvNets](https://github.com/bharatsingh430/Deformable-ConvNets)

**How to Build a Real-time Hand-Detector using Neural Networks (SSD) on Tensorflow**

- blog: [https://towardsdatascience.com/how-to-build-a-real-time-hand-detector-using-neural-networks-ssd-on-tensorflow-d6bac0e4b2ce](https://towardsdatascience.com/how-to-build-a-real-time-hand-detector-using-neural-networks-ssd-on-tensorflow-d6bac0e4b2ce)
- github: [https://github.com//victordibia/handtracking](https://github.com//victordibia/handtracking)

**Metrics for object detection**

- intro: Most popular metrics used to evaluate object detection algorithms
- github: [https://github.com/rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)

**MobileNetv2-SSDLite**

- intro: Caffe implementation of SSD and SSDLite detection on MobileNetv2, converted from tensorflow.
- github: [https://github.com/chuanqi305/MobileNetv2-SSDLite](https://github.com/chuanqi305/MobileNetv2-SSDLite)

# Leaderboard

**Detection Results: VOC2012**

- intro: Competition "comp4" (train on additional data)
- homepage: [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4)

# Tools

**BeaverDam: Video annotation tool for deep learning training labels**

[https://github.com/antingshen/BeaverDam](https://github.com/antingshen/BeaverDam)

# Blogs

**Convolutional Neural Networks for Object Detection**

[http://rnd.azoft.com/convolutional-neural-networks-object-detection/](http://rnd.azoft.com/convolutional-neural-networks-object-detection/)

**Introducing automatic object detection to visual search (Pinterest)**

- keywords: Faster R-CNN
- blog: [https://engineering.pinterest.com/blog/introducing-automatic-object-detection-visual-search](https://engineering.pinterest.com/blog/introducing-automatic-object-detection-visual-search)
- demo: [https://engineering.pinterest.com/sites/engineering/files/Visual%20Search%20V1%20-%20Video.mp4](https://engineering.pinterest.com/sites/engineering/files/Visual%20Search%20V1%20-%20Video.mp4)
- review: [https://news.developer.nvidia.com/pinterest-introduces-the-future-of-visual-search/?mkt_tok=eyJpIjoiTnpaa01UWXpPRE0xTURFMiIsInQiOiJJRjcybjkwTmtmallORUhLOFFFODBDclFqUlB3SWlRVXJXb1MrQ013TDRIMGxLQWlBczFIeWg0TFRUdnN2UHY2ZWFiXC9QQVwvQzBHM3B0UzBZblpOSmUyU1FcLzNPWXI4cml2VERwTTJsOFwvOEk9In0%3D](https://news.developer.nvidia.com/pinterest-introduces-the-future-of-visual-search/?mkt_tok=eyJpIjoiTnpaa01UWXpPRE0xTURFMiIsInQiOiJJRjcybjkwTmtmallORUhLOFFFODBDclFqUlB3SWlRVXJXb1MrQ013TDRIMGxLQWlBczFIeWg0TFRUdnN2UHY2ZWFiXC9QQVwvQzBHM3B0UzBZblpOSmUyU1FcLzNPWXI4cml2VERwTTJsOFwvOEk9In0%3D)

**Deep Learning for Object Detection with DIGITS**

- blog: [https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/](https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/)

**Analyzing The Papers Behind Facebook's Computer Vision Approach**

- keywords: DeepMask, SharpMask, MultiPathNet
- blog: [https://adeshpande3.github.io/adeshpande3.github.io/Analyzing-the-Papers-Behind-Facebook's-Computer-Vision-Approach/](https://adeshpande3.github.io/adeshpande3.github.io/Analyzing-the-Papers-Behind-Facebook's-Computer-Vision-Approach/)

**Easily Create High Quality Object Detectors with Deep Learning**

- intro: dlib v19.2
- blog: [http://blog.dlib.net/2016/10/easily-create-high-quality-object.html](http://blog.dlib.net/2016/10/easily-create-high-quality-object.html)

**How to Train a Deep-Learned Object Detection Model in the Microsoft Cognitive Toolkit**

- blog: [https://blogs.technet.microsoft.com/machinelearning/2016/10/25/how-to-train-a-deep-learned-object-detection-model-in-cntk/](https://blogs.technet.microsoft.com/machinelearning/2016/10/25/how-to-train-a-deep-learned-object-detection-model-in-cntk/)
- github: [https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FastRCNN](https://github.com/Microsoft/CNTK/tree/master/Examples/Image/Detection/FastRCNN)

**Object Detection in Satellite Imagery, a Low Overhead Approach**

- part 1: [https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7#.2csh4iwx9](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-i-cbd96154a1b7#.2csh4iwx9)
- part 2: [https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-ii-893f40122f92#.f9b7dgf64](https://medium.com/the-downlinq/object-detection-in-satellite-imagery-a-low-overhead-approach-part-ii-893f40122f92#.f9b7dgf64)

**You Only Look Twice — Multi-Scale Object Detection in Satellite Imagery With Convolutional Neural Networks**

- part 1: [https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571#.fmmi2o3of](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-38dad1cf7571#.fmmi2o3of)
- part 2: [https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-34f72f659588#.nwzarsz1t](https://medium.com/the-downlinq/you-only-look-twice-multi-scale-object-detection-in-satellite-imagery-with-convolutional-neural-34f72f659588#.nwzarsz1t)

**Faster R-CNN Pedestrian and Car Detection**

- blog: [https://bigsnarf.wordpress.com/2016/11/07/faster-r-cnn-pedestrian-and-car-detection/](https://bigsnarf.wordpress.com/2016/11/07/faster-r-cnn-pedestrian-and-car-detection/)
- ipn: [https://gist.github.com/bigsnarfdude/2f7b2144065f6056892a98495644d3e0#file-demo_faster_rcnn_notebook-ipynb](https://gist.github.com/bigsnarfdude/2f7b2144065f6056892a98495644d3e0#file-demo_faster_rcnn_notebook-ipynb)
- github: [https://github.com/bigsnarfdude/Faster-RCNN_TF](https://github.com/bigsnarfdude/Faster-RCNN_TF)

**Small U-Net for vehicle detection**

- blog: [https://medium.com/@vivek.yadav/small-u-net-for-vehicle-detection-9eec216f9fd6#.md4u80kad](https://medium.com/@vivek.yadav/small-u-net-for-vehicle-detection-9eec216f9fd6#.md4u80kad)

**Region of interest pooling explained**

- blog: [https://deepsense.io/region-of-interest-pooling-explained/](https://deepsense.io/region-of-interest-pooling-explained/)
- github: [https://github.com/deepsense-io/roi-pooling](https://github.com/deepsense-io/roi-pooling)

**Supercharge your Computer Vision models with the TensorFlow Object Detection API**

- blog: [https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html](https://research.googleblog.com/2017/06/supercharge-your-computer-vision-models.html)
- github: [https://github.com/tensorflow/models/tree/master/object_detection](https://github.com/tensorflow/models/tree/master/object_detection)

**Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning**

[https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)

**One-shot object detection**

[http://machinethink.net/blog/object-detection/](http://machinethink.net/blog/object-detection/)

**An overview of object detection: one-stage methods**

[https://www.jeremyjordan.me/object-detection-one-stage/](https://www.jeremyjordan.me/object-detection-one-stage/)

**deep learning object detection**

- intro: A paper list of object detection using deep learning.
- arxiv: [https://github.com/hoya012/deep_learning_object_detection](https://github.com/hoya012/deep_learning_object_detection)
