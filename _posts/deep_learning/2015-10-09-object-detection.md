---
layout: post
category: deep_learning
title: Object Detection
date: 2015-10-09
---

| Method              | VOC2007     | VOC2010     | VOC2012     | ILSVRC 2013 | MSCOCO 2015 | Speed       |
|:-------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| OverFeat            |             |             |             | 24.3%       |             |             |
| R-CNN(AlexNet)      | 58.5%       | 53.7%       | 53.3%       | 31.4%       |             |             |
| R-CNN               | 59.2%(ZFNet), 66.0%(VGG16) | |          |             |             |             |
| SPP_net(ZF-5)       | 54.2%(1-model), 60.9%(2-model) | | |31.84%(1-model), 35.11%(6-model) |            |
| DeepID-Net          | 64.1%       |             |             | 50.3%       |             |             |
| NoC                 | 73.3%       |             | 68.8%       |             |             |             |
| Fast-RCNN (VGG16)   | 70.0%       | 68.8%       | 68.4%       |             |             |             |
| MR-CNN              | 78.2%       |             | 73.9%       |             |             |             |
| Faster-RCNN (VGG16) | 69.9%/(07), 73.2%/(07+12), 78.8%/(COCO+07+12) | | 67.0%/(12), 70.4%/(07++12), 75.9%/(COCO+07++12) | | | 198ms |
| SSD300 (VGG16)      | 72.1%       |             |             |             |             | 58 fps      |
| SSD500 (VGG16)      | 75.1%       |             |             |             |             | 23 fps      | 
| ION                 | 79.2%/(07+12+S) |       | 76.4%/(07+12+S)|            |             |             |
| AZ-Net              | 70.4%       |             |             |             | 22.3%, 41.0%(IoU=0.5) |   |
| CRAFT               | 75.7%       |             |  71.3%      |    48.5%    |             |             |
| R-FCN (ResNet-50)   | 77.4%(07+12)|             |             |             |             | 0.12sec(K40), 0.09sec(Titian X) |
| R-FCN (ResNet-101)  | 79.5%(07+12)|             |             |             |             | 0.17sec(K40), 0.12sec(Titian X) |

# Papers

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

- intro: A deep version of the sliding window method, predicts bounding box directly from each location of the 
topmost feature map after knowing the confidences of the underlying object categories.
- arxiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- github: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

## R-CNN

**Rich feature hierarchies for accurate object detection and semantic segmentation(R-CNN)**

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

- arxiv: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

**Learning Rich Features from RGB-D Images for Object Detection and Segmentation**

- arxiv: [http://arxiv.org/abs/1407.5736](http://arxiv.org/abs/1407.5736)

## DeepID-Net

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

- intro: an extension of R-CNN
- keywords: box pre-training, cascade on region proposals, deformation layers and context representations
- arxiv: [http://arxiv.org/abs/1412.5661](http://arxiv.org/abs/1412.5661)

**Object Detectors Emerge in Deep Scene CNNs**

- arxiv: [http://arxiv.org/abs/1412.6856](http://arxiv.org/abs/1412.6856)
- paper: [https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf](https://www.robots.ox.ac.uk/~vgg/rg/papers/zhou_iclr15.pdf)
- paper: [https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf](https://people.csail.mit.edu/khosla/papers/iclr2015_zhou.pdf)
- slides: [http://places.csail.mit.edu/slide_iclr2015.pdf](http://places.csail.mit.edu/slide_iclr2015.pdf)

**Object Detection Networks on Convolutional Feature Maps**

- intro: NoC
- arxiv: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- github: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

## Fast R-CNN

**Fast R-CNN**

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

- arxiv: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- github: [https://github.com/gidariss/mrcnn-object-detection](https://github.com/gidariss/mrcnn-object-detection)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

## Faster R-CNN

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks(NIPS 2015)**

- arxiv: [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- github: [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- github: [https://github.com/rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)
- github: [https://github.com/mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn)
- github(Torch7): [https://github.com/andreaskoepf/faster-rcnn.torch](https://github.com/andreaskoepf/faster-rcnn.torch)

**Faster R-CNN in MXNet with distributed implementation and data parallelization**

- github: [https://github.com/dmlc/mxnet/tree/master/example/rcnn](https://github.com/dmlc/mxnet/tree/master/example/rcnn)

## YOLO

**You Only Look Once: Unified, Real-Time Object Detection**

![](https://camo.githubusercontent.com/e69d4118b20a42de4e23b9549f9a6ec6dbbb0814/687474703a2f2f706a7265646469652e636f6d2f6d656469612f66696c65732f6461726b6e65742d626c61636b2d736d616c6c2e706e67)

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

**CVPR研讨会 余凯特邀报告：基于密集预测图的物体检测技术造就全球领先的ADAS系统**

- intro: DenseBox(V2)
- blog: [https://mp.weixin.qq.com/s?__biz=MzI4ODAyNjU3MQ==&mid=2649827746&idx=1&sn=aa66524f964ac87d7437fc7b162f95a6&scene=1&srcid=0704uqAhpgy2fZQecXXLu6VN&pass_ticket=V7q2djnsZpyMQSJrOri0pR%2Bd%2Fi063dE5bK3kRigh1vPo%2B9yRU0Xm7cRvRNbzVgqF#rd](https://mp.weixin.qq.com/s?__biz=MzI4ODAyNjU3MQ==&mid=2649827746&idx=1&sn=aa66524f964ac87d7437fc7b162f95a6&scene=1&srcid=0704uqAhpgy2fZQecXXLu6VN&pass_ticket=V7q2djnsZpyMQSJrOri0pR%2Bd%2Fi063dE5bK3kRigh1vPo%2B9yRU0Xm7cRvRNbzVgqF#rd)

## SSD

**SSD: Single Shot MultiBox Detector**

![](https://camo.githubusercontent.com/ad9b147ed3a5f48ffb7c3540711c15aa04ce49c6/687474703a2f2f7777772e63732e756e632e6564752f7e776c69752f7061706572732f7373642e706e67)

- arxiv: [http://arxiv.org/abs/1512.02325](http://arxiv.org/abs/1512.02325)
- paper: [http://www.cs.unc.edu/~wliu/papers/ssd.pdf](http://www.cs.unc.edu/~wliu/papers/ssd.pdf)
- github: [https://github.com/weiliu89/caffe/tree/ssd](https://github.com/weiliu89/caffe/tree/ssd)
- video: [http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973](http://weibo.com/p/2304447a2326da963254c963c97fb05dd3a973)

## Inside-Outside Net (ION)

**Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks**

- intro: "0.8s per image on a Titan X GPU (excluding proposal generation) without two-stage bounding-box regression
and 1.15s per image with it".
- arxiv: [http://arxiv.org/abs/1512.04143](http://arxiv.org/abs/1512.04143)
- slides: [http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf](http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf)
- coco-leaderboard: [http://mscoco.org/dataset/#detections-leaderboard](http://mscoco.org/dataset/#detections-leaderboard)

**Adaptive Object Detection Using Adjacency and Zoom Prediction (CVPR 2016)**

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

- intro: Facebook AI Research (FAIR)
- arxiv: [http://arxiv.org/abs/1604.02135](http://arxiv.org/abs/1604.02135)

## CRAFT

**CRAFT Objects from Images**

![](http://byangderek.github.io/projects/craft/approach.png)

- intro: CVPR 2016. Cascade Region-proposal-network And FasT-rcnn. an extension of Faster R-CNN
- project page: [http://byangderek.github.io/projects/craft.html](http://byangderek.github.io/projects/craft.html)
- arxiv: [https://arxiv.org/abs/1604.03239](https://arxiv.org/abs/1604.03239)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_CRAFT_Objects_From_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_CRAFT_Objects_From_CVPR_2016_paper.pdf)
- github: [https://github.com/byangderek/CRAFT](https://github.com/byangderek/CRAFT)

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

# Detection From Video

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

## T-CNN

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

## Datasets

**YouTube-Objects dataset v2.2**

- homepage: [http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/](http://calvin.inf.ed.ac.uk/datasets/youtube-objects-dataset/)

**ILSVRC2015: Object detection from video (VID)**

- homepage: [http://vision.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid](http://vision.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid)

# Salient Object Detection

This task involves predicting the salient regions of an image given by human eye fixations.

**Large-scale optimization of hierarchical features for saliency prediction in natural images**

- paper: [http://coxlab.org/pdfs/cvpr2014_vig_saliency.pdf](http://coxlab.org/pdfs/cvpr2014_vig_saliency.pdf)

**Predicting Eye Fixations using Convolutional Neural Networks**

- paper: [http://www.escience.cn/system/file?fileId=72648](http://www.escience.cn/system/file?fileId=72648)

## DeepFix

**DeepFix: A Fully Convolutional Neural Network for predicting Human Eye Fixations**

- arxiv: [http://arxiv.org/abs/1510.02927](http://arxiv.org/abs/1510.02927)

## DeepSaliency

**DeepSaliency: Multi-Task Deep Neural Network Model for Salient Object Detection**

- arxiv: [http://arxiv.org/abs/1510.05484](http://arxiv.org/abs/1510.05484)

## SuperCNN

**SuperCNN: A Superpixelwise Convolutional Neural Network for Salient Object Detection**

![](http://www.shengfenghe.com/uploads/1/5/1/3/15132160/445461979.png)

- paper: [www.shengfenghe.com/supercnn-a-superpixelwise-convolutional-neural-network-for-salient-object-detection.html](www.shengfenghe.com/supercnn-a-superpixelwise-convolutional-neural-network-for-salient-object-detection.html)

**Shallow and Deep Convolutional Networks for Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1603.00845](http://arxiv.org/abs/1603.00845)
- github: [https://github.com/imatge-upc/saliency-2016-cvpr](https://github.com/imatge-upc/saliency-2016-cvpr)

**Two-Stream Convolutional Networks for Dynamic Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1607.04730](http://arxiv.org/abs/1607.04730)

**Unconstrained Salient Object Detection (CVPR 2016)**

![](http://cs-people.bu.edu/jmzhang/images/pasted%20image%201465x373.jpg)

- project page: [http://cs-people.bu.edu/jmzhang/sod.html](http://cs-people.bu.edu/jmzhang/sod.html)
- paper: [http://cs-people.bu.edu/jmzhang/SOD/CVPR16SOD_camera_ready.pdf](http://cs-people.bu.edu/jmzhang/SOD/CVPR16SOD_camera_ready.pdf)
- github: [https://github.com/jimmie33/SOD](https://github.com/jimmie33/SOD)

**Salient Object Subitizing**

- intro: predicting the existence and the number of salient objects in an image using holistic cues
- arxiv: [http://arxiv.org/abs/1607.07525](http://arxiv.org/abs/1607.07525)

# Specific Object Deteciton

## Face Deteciton

**Multi-view Face Detection Using Deep Convolutional Neural Networks**

- intro: Yahoo
- arxiv: [http://arxiv.org/abs/1502.02766](http://arxiv.org/abs/1502.02766)

**From Facial Parts Responses to Face Detection: A Deep Learning Approach**

![](http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/support/index.png)

- project page: [http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/Faceness.html](http://personal.ie.cuhk.edu.hk/~ys014/projects/Faceness/Faceness.html)

**Compact Convolutional Neural Network Cascade for Face Detection**

- arxiv: [http://arxiv.org/abs/1508.01292](http://arxiv.org/abs/1508.01292)
- github: [https://github.com/Bkmz21/FD-Evaluation](https://github.com/Bkmz21/FD-Evaluation)

**Supervised Transformer Network for Efficient Face Detection**

- arxiv: [http://arxiv.org/abs/1607.05477](http://arxiv.org/abs/1607.05477)

### UnitBox

**UnitBox: An Advanced Object Detection Network**

- arxiv: [http://arxiv.org/abs/1608.01471](http://arxiv.org/abs/1608.01471)

**Bootstrapping Face Detection with Hard Negative Examples**

- arxiv: [http://arxiv.org/abs/1608.02236](http://arxiv.org/abs/1608.02236)

### Datasets / Benchmarks

**WIDER FACE: A Face Detection Benchmark**

![](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/intro.jpg)

- homepage: [http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)
- arxiv: [http://arxiv.org/abs/1511.06523](http://arxiv.org/abs/1511.06523)

## Facial Point / Landmark Detection

**Deep Convolutional Network Cascade for Facial Point Detection**

![](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/Picture1.png)

- homepage: [http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
- paper: [http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf)
- github: [https://github.com/luoyetx/deep-landmark](https://github.com/luoyetx/deep-landmark)

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

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.04564](http://arxiv.org/abs/1607.04564)

## Traffic-Sign Detection

**Traffic-Sign Detection and Classification in the Wild**

- project page(code+dataset): [http://cg.cs.tsinghua.edu.cn/traffic-sign/](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
- paper: [http://120.52.73.11/www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf](http://120.52.73.11/www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)
- code & model: [http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip](http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip)

## Boundary / Edge / Contour Detection

**Holistically-Nested Edge Detection**

![](https://camo.githubusercontent.com/da32e7e3275c2a9693dd2a6925b03a1151e2b098/687474703a2f2f70616765732e756373642e6564752f7e7a74752f6865642e6a7067)

- intro: ICCV 2015, Marr Prize
- paper: [http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf](http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Xie_Holistically-Nested_Edge_Detection_ICCV_2015_paper.pdf)
- arxiv: [http://arxiv.org/abs/1504.06375](http://arxiv.org/abs/1504.06375)
- github: [https://github.com/s9xie/hed](https://github.com/s9xie/hed)

**Unsupervised Learning of Edges**

- intro: CVPR 2016. Facebook AI Research
- arxiv: [http://arxiv.org/abs/1511.04166](http://arxiv.org/abs/1511.04166)
- zn-blog: [http://www.leiphone.com/news/201607/b1trsg9j6GSMnjOP.html](http://www.leiphone.com/news/201607/b1trsg9j6GSMnjOP.html)

**Pushing the Boundaries of Boundary Detection using Deep Learning**

- arxiv: [http://arxiv.org/abs/1511.07386](http://arxiv.org/abs/1511.07386)

**Convolutional Oriented Boundaries**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1608.02755](http://arxiv.org/abs/1608.02755)

## Abnormality Detection

**Toward a Taxonomy and Computational Models of Abnormalities in Images**

- arxiv: [http://arxiv.org/abs/1512.01325](http://arxiv.org/abs/1512.01325)

## Skeleton Detection

**Object Skeleton Extraction in Natural Images by Fusing Scale-associated Deep Side Outputs**

![](https://camo.githubusercontent.com/88a65f132aa4ae4b0477e3ad02c13cdc498377d9/687474703a2f2f37786e37777a2e636f6d312e7a302e676c622e636c6f7564646e2e636f6d2f44656570536b656c65746f6e2e706e673f696d61676556696577322f322f772f353030)

- arxiv: [http://arxiv.org/abs/1603.09446](http://arxiv.org/abs/1603.09446)
- github: [https://github.com/zeakey/DeepSkeleton](https://github.com/zeakey/DeepSkeleton)

## Others

**Deep Deformation Network for Object Landmark Localization**

- arxiv: [http://arxiv.org/abs/1605.01014](http://arxiv.org/abs/1605.01014)

**Fashion Landmark Detection in the Wild**

- arxiv: [http://arxiv.org/abs/1608.03049](http://arxiv.org/abs/1608.03049)

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

**Deep Learning for Object Detection with DIGITS**

- blog: [https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/](https://devblogs.nvidia.com/parallelforall/deep-learning-object-detection-digits/)