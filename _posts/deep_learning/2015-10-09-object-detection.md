---
layout: post
category: deep_learning
title: Object Detection
date: 2015-10-09
---

# Papers

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

|  method  |ILSVRC 2013 mAP|
|:--------:|:-------------:|
|OverFeat  |24.3%          |

- intro: A deep version of the sliding window method, predicts bounding box directly from each location of the 
topmost feature map after knowing the confidences of the underlying object categories.
- arxiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- github: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
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

- arxiv: [http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- supp: [http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf](http://people.eecs.berkeley.edu/~rbg/papers/r-cnn-cvpr-supp.pdf)
- slides: [http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf](http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- github: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- notes: [http://zhangliliang.com/2014/07/23/paper-note-rcnn/](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)
- caffe-pr("Make R-CNN the Caffe detection example"): [https://github.com/BVLC/caffe/pull/482](https://github.com/BVLC/caffe/pull/482) 

## MultiBox

**Scalable Object Detection using Deep Neural Networks (MultiBox)**

- intro: Train a CNN to predict Region of Interest.
- arxiv: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)

**Scalable, High-Quality Object Detection**

- arxiv: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)

## SPP-Net

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

|method               |VOC 2007 mAP|ILSVRC 2013 mAP|
|:-------------------:|:----------:|:-------------:|
|SPP_net(ZF-5),1-model|54.2%       |31.84%         |
|SPP_net(ZF-5),2-model|60.9%       |               |
|SPP_net(ZF-5),6-model|            |35.11%         |

- arxiv: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

**Learning Rich Features from RGB-D Images for Object Detection and Segmentation**

- arxiv: [http://arxiv.org/abs/1407.5736](http://arxiv.org/abs/1407.5736)

## DeepID-Net

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

|method        |VOC 2007 mAP|ILSVRC 2013 mAP|
|:------------:|:----------:|:-------------:|
|DeepID-Net    |64.1%       |50.3%          |

- arxiv: [http://arxiv.org/abs/1412.5661](http://arxiv.org/abs/1412.5661)

**Object Detectors Emerge in Deep Scene CNNs**

- arxiv: [http://arxiv.org/abs/1412.6856](http://arxiv.org/abs/1412.6856)
- paper: [https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf](https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf)
- paper: [https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf](https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf)
- slides: [http://places.csail.mit.edu/slide_iclr2015.pdf](http://places.csail.mit.edu/slide_iclr2015.pdf)

**Object Detection Networks on Convolutional Feature Maps**

|method    |Trained on |mAP       |
|:--------:|:---------:|:--------:|
|NoC       |07+12      |68.8%     |
|NoC,bb    |07+12      |71.6%     |
|NoC,+EB   |07+12      |71.8%     |
|NoC,+EB,bb|07+12      |73.3%     |

- arxiv: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- github: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

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

- arxiv: [http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- github: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- webcam demo: [https://github.com/rbgirshick/fast-rcnn/pull/29](https://github.com/rbgirshick/fast-rcnn/pull/29)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/](http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/)
- notes: [http://blog.csdn.net/linj_m/article/details/48930179](http://blog.csdn.net/linj_m/article/details/48930179)
- github("Train Fast-RCNN on Another Dataset"): [https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train](https://github.com/zeyuanxy/fast-rcnn/tree/master/help/train)
- github("Fast R-CNN in MXNet"): [https://github.com/precedenceguo/mx-rcnn](https://github.com/precedenceguo/mx-rcnn)
- github: [https://github.com/mahyarnajibi/fast-rcnn-torch](https://github.com/mahyarnajibi/fast-rcnn-torch)
- github: [https://github.com/apple2373/chainer-simple-fast-rnn](https://github.com/apple2373/chainer-simple-fast-rnn)

## DeepBox

**DeepBox: Learning Objectness with Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1505.02146](http://arxiv.org/abs/1505.02146)
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

- arxiv: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- github: [https://github.com/gidariss/mrcnn-object-detection](https://github.com/gidariss/mrcnn-object-detection)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

## Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks(NIPS 2015)**

|                   |training data|test data    |mAP  |time/img|
|:-----------------:|:-----------:|:-----------:|:---:|:------:|
|Faster RCNN, VGG-16|07           |VOC 2007 test|69.9%|198ms   |
|Faster RCNN, VGG-16|07+12        |VOC 2007 test|73.2%|198ms   |
|Faster RCNN, VGG-16|COCO+07+12   |VOC 2007 test|78.8%|198ms   |
|Faster RCNN, VGG-16|12           |VOC 2012 test|67.0%|198ms   |
|Faster RCNN, VGG-16|07++12       |VOC 2012 test|70.4%|198ms   |
|Faster RCNN, VGG-16|COCO+07++12  |VOC 2012 test|75.9%|198ms   |

- arxiv: [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- github: [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- github: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- github: [https://github.com/mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn)

## YOLO

**You Only Look Once: Unified, Real-Time Object Detection(YOLO)**

- intro: YOLO uses the whole topmost feature map to predict both confidences for multiple categories and 
bounding boxes (which are shared for these categories).
- arxiv: [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)
- github: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/](https://www.reddit.com/r/MachineLearning/comments/3a3m0o/realtime_object_detection_with_yolo/)
- github: [https://github.com/gliese581gg/YOLO_tensorflow](https://github.com/gliese581gg/YOLO_tensorflow)
- github: [https://github.com/xingwangsfu/caffe-yolo](https://github.com/xingwangsfu/caffe-yolo)

**R-CNN minus R**

- arxiv: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)

## DenseBox

**DenseBox: Unifying Landmark Localization with End to End Object Detection**

- arxiv: [http://arxiv.org/abs/1509.04874](http://arxiv.org/abs/1509.04874)
- demo: [http://pan.baidu.com/s/1mgoWWsS](http://pan.baidu.com/s/1mgoWWsS)
- KITTI result: [http://www.cvlibs.net/datasets/kitti/eval_object.php](http://www.cvlibs.net/datasets/kitti/eval_object.php)

## SSD

**SSD: Single Shot MultiBox Detector**

![](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

|    System      | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes |
|:---------------|:------------------:|:-----------------:|:---------------:|
| SSD300 (VGG16) |      72.1          |       58          |    7308         |
| SSD500 (VGG16) |    **75.1**        |       23          |    20097        |

- arxiv: [http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- paper: [http://www.cs.unc.edu/~wliu/papers/ssd.pdf](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)
- github: [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
- video: [http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973](http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973)

## Inside-Outside Net

**Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks**

Detection results on VOC 2007 test:

| Method | R | S | W | D | Train   | mAP  |
|:------:|:-:|:-:|:-:|:-:|:-------:|:----:|
| ION    | √ | √ | √ | √ | 07+12+S | 79.2 |

Detection results on VOC 2012 test:

| Method    | R | S | W | D | Train   | mAP  |
|:---------:|:-:|:-:|:-:|:-:|:-------:|:----:|
| ION       | √ | √ | √ | √ | 07+12+S | 76.4 |

- intro: "0.8s per image on a Titan X GPU (excluding proposal generation) without two-stage bounding-box regression
and 1.15s per image with it".
- arxiv: [http://arxiv.org/abs/1512.04143](http://arxiv.org/abs/1512.04143)
- slides: [http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf](http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf)
- coco-leaderboard: [http://mscoco.org/dataset/#detections-leaderboard](http://mscoco.org/dataset/#detections-leaderboard)

**Adaptive Object Detection Using Adjacency and Zoom Prediction (CVPR 2016)**

VOC 2007 test:

| Method  | Train | mAP  |
|:-------:|:-----:|:----:|
| AZ-Net  | 79.0  | 70.2 |
| AZ-Net* | 78.4  | 70.4 |

MSCOCO 2015 test:

|     Method     |  AP  | AP IoU=0.50 |
|:--------------:|:----:|:-----------:|
| AZ-Net(VGG-16) | 22.3 |    41.0     |

- arxiv: [http://arxiv.org/abs/1512.07711](http://arxiv.org/abs/1512.07711)
- github: [https://github.com/luyongxi/az-net](https://github.com/luyongxi/az-net)
- youtube: [https://www.youtube.com/watch?v=YmFtuNwxaNM](https://www.youtube.com/watch?v=YmFtuNwxaNM)

## G-CNN

**G-CNN: an Iterative Grid Based Object Detector**

- arxiv: [http://arxiv.org/abs/1512.07729](http://arxiv.org/abs/1512.07729)

**Factors in Finetuning Deep Model for object detection**

- arxiv: [http://arxiv.org/abs/1601.05150](http://arxiv.org/abs/1601.05150)

**We don't need no bounding-boxes: Training object class detectors using only human verification**

- arxiv: [http://arxiv.org/abs/1602.08405](http://arxiv.org/abs/1602.08405)

## HyperNet

**HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection**

- arxiv: [http://arxiv.org/abs/1604.00600](http://arxiv.org/abs/1604.00600)

**A MultiPath Network for Object Detection**

- arxiv: [http://arxiv.org/abs/1604.02135](http://arxiv.org/abs/1604.02135)

**Training Region-based Object Detectors with Online Hard Example Mining**

- arxiv: [http://arxiv.org/abs/1604.03540](http://arxiv.org/abs/1604.03540)

**Track and Transfer: Watching Videos to Simulate Strong Human Supervision for Weakly-Supervised Object Detection (CVPR 2016)**

- arxiv: [http://arxiv.org/abs/1604.05766](http://arxiv.org/abs/1604.05766)

**Exploit All the Layers: Fast and Accurate CNN Object Detector with Scale Dependent Pooling and Cascaded Rejection Classifiers**

[http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf](http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf)

## R-FCN

**R-FCN: Object Detection via Region-based Fully Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)
- github: [https://github.com/daijifeng001/R-FCN](https://github.com/daijifeng001/R-FCN)

**Weakly supervised object detection using pseudo-strong labels**

- arxiv: [http://arxiv.org/abs/1607.04731](http://arxiv.org/abs/1607.04731)

**Recycle deep features for better object detection**

- arxiv: [http://arxiv.org/abs/1607.05066](http://arxiv.org/abs/1607.05066)

## MS-CNN

**A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection**

- intro: 640×480: 15 fps, 960×720: 8 fps
- arxiv: [http://arxiv.org/abs/1607.07155](http://arxiv.org/abs/1607.07155)

# Specific Object Deteciton

## People Detection

**End-to-end people detection in crowded scenes**

![](/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg)

- arxiv: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- github: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)

## Person Head Detection

**Context-aware CNNs for person head detection**

- arxiv: [http://arxiv.org/abs/1511.07917](http://arxiv.org/abs/1511.07917)
- github: [https://github.com/aosokin/cnn_head_detection](https://github.com/aosokin/cnn_head_detection)

## Pedestrian Detection

**Pedestrian Detection aided by Deep Learning Semantic Tasks**:

- paper: [http://arxiv.org/abs/1412.0069](http://arxiv.org/abs/1412.0069)
- project: [http://mmlab.ie.cuhk.edu.hk/projects/TA-CNN/](http://mmlab.ie.cuhk.edu.hk/projects/TA-CNN/)

**Deep convolutional neural networks for pedestrian detection**

- arxiv: [http://arxiv.org/abs/1510.03608](http://arxiv.org/abs/1510.03608)
- github: [https://github.com/DenisTome/DeepPed](https://github.com/DenisTome/DeepPed)

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

- arxiv: [http://arxiv.org/abs/1607.07032](http://arxiv.org/abs/1607.07032)

## Vehicle Detection

**DAVE: A Unified Framework for Fast Vehicle Detection and Annotation**

- arxiv: [http://arxiv.org/abs/1607.04564](http://arxiv.org/abs/1607.04564)

## Traffic-Sign Detection

**Traffic-Sign Detection and Classification in the Wild**

- project page(code+dataset): [http://cg.cs.tsinghua.edu.cn/traffic-sign/](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
- paper: [http://120.52.73.11/www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf](http://120.52.73.11/www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)
- code & model: [http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip](http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip)

## Boundary Detection

**Pushing the Boundaries of Boundary Detection using Deep Learning**

- arxiv: [http://arxiv.org/abs/1511.07386](http://arxiv.org/abs/1511.07386)

## Abnormality Detection

**Toward a Taxonomy and Computational Models of Abnormalities in Images**

- arxiv: [http://arxiv.org/abs/1512.01325](http://arxiv.org/abs/1512.01325)

## Edge Detection

**Holistically-Nested Edge Detection (ICCV 2015, Marr Prize)**

![](https://camo.githubusercontent.com/da32e7e3275c2a9693dd2a6925b03a1151e2b098/687474703a2f2f70616765732e756373642e6564752f7e7a74752f6865642e6a7067)

- paper: [http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf)
- arxiv: [http://arxiv.org/abs/1504.06375](http://arxiv.org/abs/1504.06375)
- github: [https://github.com/s9xie/hed](https://github.com/s9xie/hed)

**Unsupervised Learning of Edges (CVPR 2016. Facebook AI Research)**

- arxiv: [http://arxiv.org/abs/1511.04166](http://arxiv.org/abs/1511.04166)
- zn-blog: [http://www.leiphone.com/news/201607/b1trsg9j6GSMnjOP.html](http://www.leiphone.com/news/201607/b1trsg9j6GSMnjOP.html)

## Skeleton Detection

**Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs**

![](https://camo.githubusercontent.com/88a65f132aa4ae4b0477e3ad02c13cdc498377d9/687474703a2f2f37786e37777a2e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f44656570536b656c65746f6e2e706e673f696d61676556696577322f322f772f353030)

- arxiv: [http://arxiv.org/abs/1603.09446](http://arxiv.org/abs/1603.09446)
- github: [https://github.com/zeakey/DeepSkeleton](https://github.com/zeakey/DeepSkeleton)

## Others

**Deep Deformation Network for Object Landmark Localization**

- arxiv: [http://arxiv.org/abs/1605.01014](http://arxiv.org/abs/1605.01014)

# Detection From Video

## T-CNN

**T-CNN: Tubelets with Convolutional Neural Networks for Object Detection from Videos**

- arxiv: [http://arxiv.org/abs/1604.02532](http://arxiv.org/abs/1604.02532)
- github: [https://github.com/myfavouritekk/T-CNN](https://github.com/myfavouritekk/T-CNN)

**Object Detection from Video Tubelets with Convolutional Neural Networks (CVPR 2016 Spotlight paper)**

- arxiv: [https://arxiv.org/abs/1604.04053](https://arxiv.org/abs/1604.04053)
- paper: [http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf](http://www.ee.cuhk.edu.hk/~wlouyang/Papers/KangVideoDet_CVPR16.pdf)
- gihtub: [https://github.com/myfavouritekk/vdetlib](https://github.com/myfavouritekk/vdetlib)

**Context Matters: Refining Object Detection in Video with Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1607.04648](http://arxiv.org/abs/1607.04648)

# Object Proposal

## DeepProposal

**DeepProposal: Hunting Objects by Cascading Deep Convolutional Layers**

- arxiv: [http://arxiv.org/abs/1510.04445](http://arxiv.org/abs/1510.04445)
- github: [https://github.com/aghodrati/deepproposal](https://github.com/aghodrati/deepproposal)

## DeepMask

**Learning to Segment Object Candidates (DeepMask)**

- intro: learning segmentation proposals
- arxiv: [http://arxiv.org/abs/1506.06204](http://arxiv.org/abs/1506.06204)
- github: [https://github.com/abbypa/NNProject_DeepMask](https://github.com/abbypa/NNProject_DeepMask)

# Localization

**Beyond Bounding Boxes: Precise Localization of Objects in Images (PhD Thesis)**

- homepage: [http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html](http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.html)
- phd-thesis: [http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.pdf](http://www.eecs.berkeley.edu/Pubs/TechRpts/2015/EECS-2015-193.pdf)
- github("SDS using hypercolumns"): [https://github.com/bharath272/sds](https://github.com/bharath272/sds)

**Weakly Supervised Object Localization with Multi-fold Multiple Instance Learning**

- arxiv: [http://arxiv.org/abs/1503.00949](http://arxiv.org/abs/1503.00949)

## LocNet

**LocNet: Improving Localization Accuracy for Object Detection**

- arxiv: [http://arxiv.org/abs/1511.07763](http://arxiv.org/abs/1511.07763)
- github: [https://github.com/gidariss/LocNet](https://github.com/gidariss/LocNet)

**Learning Deep Features for Discriminative Localization**

![](http://cnnlocalization.csail.mit.edu/framework.jpg)

- homepage: [http://cnnlocalization.csail.mit.edu/](http://cnnlocalization.csail.mit.edu/)
- arxiv: [http://arxiv.org/abs/1512.04150](http://arxiv.org/abs/1512.04150)
- github(Tensorflow): [https://github.com/jazzsaxmafia/Weakly_detector](https://github.com/jazzsaxmafia/Weakly_detector)
- github: [https://github.com/metalbubble/CAM](https://github.com/metalbubble/CAM)
- github: [https://github.com/tdeboissiere/VGG16CAM-keras](https://github.com/tdeboissiere/VGG16CAM-keras)

# Tutorials

**Convolutional Feature Maps: Elements of efficient (and accurate) CNN-based object detection**

- slides: [http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)

# Projects

**TensorBox: a simple framework for training neural networks to detect objects in images**

- intro: "The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. 
We additionally provide an implementation of the [ReInspect](https://github.com/Russell91/ReInspect/) algorithm"
- github: [https://github.com/Russell91/TensorBox](https://github.com/Russell91/TensorBox)

**Object detection in torch: Implementation of some object detection frameworks in torch**

- github: [https://github.com/fmassa/object-detection.torch](https://github.com/fmassa/object-detection.torch)

**Using DIGITS to train an Object Detection network**

![](https://raw.githubusercontent.com/NVIDIA/DIGITS/master/examples/object-detection/select-object-detection-dataset.jpg)

- github: [https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/README.md](https://github.com/NVIDIA/DIGITS/blob/master/examples/object-detection/README.md)

# Blogs

**Convolutional Neural Networks for Object Detection**

[http://rnd.azoft.com/convolutional-neural-networks-object-detection/](http://rnd.azoft.com/convolutional-neural-networks-object-detection/)

**Introducing automatic object detection to visual search (Pinterest)**

![](https://engineering.pinterest.com/sites/engineering/files/Visual%20Search%20V1%20-%20Video.mp4)

- keywords: Faster R-CNN
- blog: [https://engineering.pinterest.com/blog/introducing-automatic-object-detection-visual-search](https://engineering.pinterest.com/blog/introducing-automatic-object-detection-visual-search)
- review: [https://news.developer.nvidia.com/pinterest-introduces-the-future-of-visual-search/?mkt_tok=eyJpIjoiTnpaa01UWXpPRE0xTURFMiIsInQiOiJJRjcybjkwTmtmallORUhLOFFFODBDclFqUlB3SWlRVXJXb1MrQ013TDRIMGxLQWlBczFIeWg0TFRUdnN2UHY2ZWFiXC9QQVwvQzBHM3B0UzBZblpOSmUyU1FcLzNPWXI4cml2VERwTTJsOFwvOEk9In0%3D](https://news.developer.nvidia.com/pinterest-introduces-the-future-of-visual-search/?mkt_tok=eyJpIjoiTnpaa01UWXpPRE0xTURFMiIsInQiOiJJRjcybjkwTmtmallORUhLOFFFODBDclFqUlB3SWlRVXJXb1MrQ013TDRIMGxLQWlBczFIeWg0TFRUdnN2UHY2ZWFiXC9QQVwvQzBHM3B0UzBZblpOSmUyU1FcLzNPWXI4cml2VERwTTJsOFwvOEk9In0%3D)

**CVPR 2016论文快讯：目标检测领域的新进展**

[http://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325043&idx=1&sn=bd016d98a40e8cf7d53ee674f201b4a7#rd](http://mp.weixin.qq.com/s?__biz=MzI1NTE4NTUwOQ==&mid=2650325043&idx=1&sn=bd016d98a40e8cf7d53ee674f201b4a7#rd)