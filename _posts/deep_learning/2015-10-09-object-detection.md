---
layout: post
category: deep_learning
title: Object Detection
date: 2015-10-09
---

* TOC
{:toc}

# Papers

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

|  method  |ILSVRC 2013 mAP|
|:--------:|:-------------:|
|OverFeat  |24.3%          |

- intro: A deep version of the sliding window method, predicts bounding box directly from each location of the 
topmost feature map after knowing the confidences of the underlying object categories.
- arXiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- code: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

## R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation(R-CNN)**

|method                |VOC 2007 mAP|VOC 2010 mAP|VOC 2012 mAP|ILSVRC 2013 mAP|
|:--------------------:|:----------:|:----------:|:----------:|:-------------:|
|R-CNN,AlexNet         |54.2%       |50.2%       |49.6%       |               |
|R-CNN,bbox reg,AlexNet|58.5%       |53.7%       |53.3%       |31.4%          |
|R-CNN,bbox reg,ZFNet  |59.2%       |            |            |               |
|R-CNN,VGG-Net         |62.2%       |            |            |               |
|R-CNN,bbox reg,VGG-Net|66.0%       |            |            |               |

- arXiv: [http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- slides: [http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf](http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- code: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- notes: [http://zhangliliang.com/2014/07/23/paper-note-rcnn/](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)
- caffe-pr("Make R-CNN the Caffe detection example"): [https://github.com/BVLC/caffe/pull/482](https://github.com/BVLC/caffe/pull/482) 

## MultiBox

**Scalable Object Detection using Deep Neural Networks (MultiBox)**

- intro: Train a CNN to predict Region of Interest.
- arXiv: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

## SPP-Net

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

|method               |VOC 2007 mAP|ILSVRC 2013 mAP|
|:-------------------:|:----------:|:-------------:|
|SPP_net(ZF-5),1-model|54.2%       |31.84%         |
|SPP_net(ZF-5),2-model|60.9%       |               |
|SPP_net(ZF-5),6-model|            |35.11%         |

- arXiv: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- code: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

**Learning Rich Features from RGB-D Images for Object Detection and Segmentation**

- arxiv: [http://arxiv.org/abs/1407.5736](http://arxiv.org/abs/1407.5736)

**Scalable, High-Quality Object Detection**

- arXiv: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

## DeepID-Net

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

|method        |VOC 2007 mAP|ILSVRC 2013 mAP|
|:------------:|:----------:|:-------------:|
|DeepID-Net    |64.1%       |50.3%          |

- arXiv: [http://arxiv.org/abs/1412.5661](http://arxiv.org/abs/1412.5661)

**Object Detection Networks on Convolutional Feature Maps**

|method    |Trained on |mAP       |
|:--------:|:---------:|:--------:|
|NoC       |07+12      |68.8%     |
|NoC,bb    |07+12      |71.6%     |
|NoC,+EB   |07+12      |71.8%     |
|NoC,+EB,bb|07+12      |73.3%     |

- arXiv: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization
and Structured Prediction**

|Model            |BBoxReg? |VOC 2007 mAP(IoU>0.5)|
|:---------------:|:-------:|:-------------------:|
|R-CNN(AlexNet)   |No       |54.2%                |
|R-CNN(VGG)       |No       |60.6%                |
|+StructObj       |No       |61.2%                |
|+StructObj-FT    |No       |62.3%                |
|+FGS             |No       |64.8%                |
|+StructObj+FGS   |No       |65.9%                |
|+StructObj-FT+FGS|No       |66.5%                |

|Model            |BBoxReg? |VOC 2007 mAP(IoU>0.5)|
|:---------------:|:-------:|:-------------------:|
|R-CNN(AlexNet)   |Yes      |58.5%                |
|R-CNN(VGG)       |Yes      |65.4%                |
|+StructObj       |Yes      |66.6%                |
|+StructObj-FT    |Yes      |66.9%                |
|+FGS             |Yes      |67.2%                |
|+StructObj+FGS   |Yes      |68.5%                |
|+StructObj-FT+FGS|Yes      |68.4%                |

- arXiv: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- code: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

## Fast R-CNN

**Fast R-CNN**

|method    |data |VOC 2007 mAP|
|:--------:|:---:|:----------:|
|FRCN,VGG16|07   |66.9%       |
|FRCN,VGG16|07+12|70.0%       |

|method    |data  |VOC 2010 mAP|
|:--------:|:----:|:----------:|
|FRCN,VGG16|12    |66.1%       |
|FRCN,VGG16|07++12|68.8%       |

|method    |data  |VOC 2012 mAP|
|:--------:|:----:|:----------:|
|FRCN,VGG16|12    |65.7%       |
|FRCN,VGG16|07++12|68.4%       |

- arXiv: [http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- github: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- webcam demo: [https://github.com/rbgirshick/fast-rcnn/pull/29](https://github.com/rbgirshick/fast-rcnn/pull/29)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/](http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/)
- notes: [http://blog.csdn.net/linj_m/article/details/48930179](http://blog.csdn.net/linj_m/article/details/48930179)
- github("Train Fast-RCNN on Another Dataset"): [https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train](https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train)

## DeepBox

**DeepBox: Learning Objectness with Convolutional Networks**

- arXiv: [http://arxiv.org/abs/1505.02146](http://arxiv.org/abs/1505.02146)
- github: [https://github.com/weichengkuo/DeepBox](https://github.com/weichengkuo/DeepBox)

## MR-CNN

**Object detection via a multi-region & semantic segmentation-aware CNN model (MR-CNN)**

|Model     |Trained on|VOC 2007 mAP|
|:--------:|:--------:|:----------:|
|VGG-net   |07+12     |78.2%       |
|VGG-net   |07        |74.9%       |

|Model     |Trained on|VOC 2012 mAP|
|:--------:|:--------:|:----------:|
|VGG-net   |07+12     |73.9%       |
|VGG-net   |12        |70.7%       |

- arXiv: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- code: "Pdf and code will appear here shortly -- stay tuned"  <br />
 [http://imagine.enpc.fr/~komodakn/](http://imagine.enpc.fr/~komodakn/)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

## Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks(NIPS 2015)**

|                   |training data|test data    |mAP  |time/img|
|:-----------------:|:-----------:|:-----------:|:---:|:------:|
|Faster RCNN, VGG-16|07           |VOC 2007 test|69.9%|198ms   |
|Faster RCNN, VGG-16|07+12        |VOC 2007 test|73.2%|198ms   |
|Faster RCNN, VGG-16|12           |VOC 2007 test|67.0%|198ms   |
|Faster RCNN, VGG-16|07++12       |VOC 2007 test|70.4%|198ms   |

- arXiv: [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- github: [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- github: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)

## YOLO

**You Only Look Once: Unified, Real-Time Object Detection(YOLO)**

- intro: YOLO uses the whole topmost feature map to predict both confidences for multiple categories and 
bounding boxes (which are shared for these categories).
- arXiv: [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)
- github: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/](https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/)
- github(YOLO_tensorflow): [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)

**R-CNN minus R**

- arXiv: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)

## DenseBox

**DenseBox: Unifying Landmark Localization with End to End Object Detection**

- arXiv: [http://arxiv.org/abs/1509.04874](http://arxiv.org/abs/1509.04874)
- demo: [http://pan.baidu.com/s/1mgoWWsS](http://pan.baidu.com/s/1mgoWWsS)
- KITTI result: [http://www.cvlibs.net/datasets/kitti/eval_object.php](http://www.cvlibs.net/datasets/kitti/eval_object.php)

## SSD

**SSD: Single Shot MultiBox Detector**

![](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

- arXiv: [http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- github: [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
- video: [http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973](http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973)

## Inside-Outside Net

**Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks**

Detection results on VOC 2007 test:

| Method | R | S | W | D | Train   | mAP  |
|:------:|:-:|:-:|:-:|:-:|:-------:|:----:|
| FRCN   |   |   |   |   | 07+12   | 70.0 |
| RPN    |   |   |   |   | 07+12   | 73.2 |
| MR-CNN |   |   | √ |   | 07+12   | 78.2 |
|:------:|:-:|:-:|:-:|:-:|:-------:|:----:|
| ION    |   |   |   |   | 07+12   | 74.6 |
| ION    | √ |   |   |   | 07+12   | 75.6 |
| ION    | √ | √ |   |   | 07+12+S | 76.5 |
| ION    | √ | √ | √ |   | 07+12+S | 78.5 |
| ION    | √ | √ | √ | √ | 07+12+S | 79.2 |

Detection results on VOC 2012 test:

| Method    | R | S | W | D | Train   | mAP  |
|:---------:|:-:|:-:|:-:|:-:|:-------:|:----:|
| FRCN      |   |   |   |   | 07++12  | 68.4 |
| RPN       |   |   |   |   | 07++12  | 70.4 |
| FRCN+YOLO |   |   |   |   | 07++12  | 70.4 |
| HyperNet  |   |   |   |   | 07++12  | 71.4 |
| MR-CNN    |   |   | √ |   | 07+12   | 73.9 |
|:---------:|:-:|:-:|:-:|:-:|:-------:|:----:|
| ION       | √ | √ | √ | √ | 07+12+S | 76.4 |

- intro: "0.8s per image on a Titan X GPU (excluding proposal generation) without two-stage bounding-box regression
and 1.15s per image with it".
- arxiv: [http://arxiv.org/abs/1512.04143](http://arxiv.org/abs/1512.04143)
- slides: [http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf](http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf)
- coco-leaderboard: [http://mscoco.org/dataset/#detections-leaderboard](http://mscoco.org/dataset/#detections-leaderboard)

## G-CNN

**G-CNN: an Iterative Grid Based Object Detector**

- arxiv: [http://arxiv.org/abs/1512.07729](http://arxiv.org/abs/1512.07729)

**Learning Deep Features for Discriminative Localization**

- homepage: [http://cnnlocalization.csail.mit.edu/](http://cnnlocalization.csail.mit.edu/)
- arxiv: [http://arxiv.org/abs/1512.04150](http://arxiv.org/abs/1512.04150)

**Factors in Finetuning Deep Model for object detection**

- arxiv: [http://arxiv.org/abs/1601.05150](http://arxiv.org/abs/1601.05150)

**We don't need no bounding-boxes: Training object class detectors using only human verification**

- arxiv: [http://arxiv.org/abs/1602.08405](http://arxiv.org/abs/1602.08405)

# Specific Object Deteciton

**End-to-end people detection in crowded scenes**

![](/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg)

- arXiv: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- code: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)

# Tutorials

**Convolutional Feature Maps: Elements of efficient (and accurate) CNN-based object detection**

- slides: [http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)

# Codes

**TensorBox: a simple framework for training neural networks to detect objects in images**

- intro: "The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. 
We additionally provide an implementation of the [ReInspect](https://github.com/Russell91/ReInspect/) algorithm"
- github: [https://github.com/Russell91/TensorBox](https://github.com/Russell91/TensorBox)

# Blogs

**Convolutional Neural Networks for Object Detection**

[http://rnd.azoft.com/convolutional-neural-networks-object-detection/](http://rnd.azoft.com/convolutional-neural-networks-object-detection/)