---
layout: post
category: deep_learning
title: Object Detection
date: 2015-10-09
---

| Method              | VOC2007     | VOC2010     | VOC2012     | ILSVRC 2013 | MSCOCO 2015 | Speed       |
|:-------------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| OverFeat            |             |             |             | 24.3%       |             |             |
| R-CNN (AlexNet)     | 58.5%       | 53.7%       | 53.3%       | 31.4%       |             |             |
| R-CNN (VGG16)       | 66.0%       |             |             |             |             |             |
| SPP_net(ZF-5)       | 54.2%(1-model), 60.9%(2-model) | | |31.84%(1-model), 35.11%(6-model) |            |
| DeepID-Net          | 64.1%       |             |             | 50.3%       |             |             |
| NoC                 | 73.3%       |             | 68.8%       |             |             |             |
| Fast-RCNN (VGG16)   | 70.0%       | 68.8%       | 68.4%       |             | 19.7%(@[0.5-0.95]), 35.9%(@0.5) | |
| MR-CNN              | 78.2%       |             | 73.9%       |             |             |             |
| Faster-RCNN (VGG16) | 78.8%       |             | 75.9%       |             | 21.9%(@[0.5-0.95]), 42.7%(@0.5) | 198ms |
| Faster-RCNN (ResNet-101) | 85.6%  |             | 83.8%       |             | 37.4%(@[0.5-0.95]), 59.0%(@0.5) | |
| YOLO                | 63.4%       |             | 57.9%       |             |             | 45 fps      |
| YOLO VGG-16         | 66.4%       |             |             |             |             | 21 fps      |
| YOLOv2 544 × 544    | 78.6%       |             | 73.4%       |             | 21.6%(@[0.5-0.95]), 44.0%(@0.5) | 40 fps |
| SSD300 (VGG16)      | 77.2%       |             | 75.8%       |             | 25.1%(@[0.5-0.95]), 43.1%(@0.5) | 46 fps |
| SSD512 (VGG16)      | 79.8%       |             | 78.5%       |             | 28.8%(@[0.5-0.95]), 48.5%(@0.5) | 19 fps |
| ION                 | 79.2%       |             | 76.4%       |             |             |             |
| CRAFT               | 75.7%       |             | 71.3%       | 48.5%       |             |             |
| OHEM                | 78.9%       |             | 76.3%       |             | 25.5%(@[0.5-0.95]), 45.9%(@0.5) | |
| R-FCN (ResNet-50)   | 77.4%       |             |             |             |             | 0.12sec(K40), 0.09sec(TitianX) |
| R-FCN (ResNet-101)  | 79.5%       |             |             |             |             | 0.17sec(K40), 0.12sec(TitianX) |
| R-FCN (ResNet-101),multi sc train | 83.6% |     | 82.0%       |             | 31.5%(@[0.5-0.95]), 53.2%(@0.5) | |
| PVANet 9.0          | 89.8%       |             | 84.2%       |             |             | 750ms(CPU), 46ms(TitianX) |

# Leaderboard

**Detection Results: VOC2012**

- intro: Competition "comp4" (train on additional data)
- homepage: [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4)

# Papers

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- github: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

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
- github: [https://github.com/mitmul/chainer-faster-rcnn](https://github.com/mitmul/chainer-faster-rcnn)
- github: [https://github.com/andreaskoepf/faster-rcnn.torch](https://github.com/andreaskoepf/faster-rcnn.torch)
- github: [https://github.com/ruotianluo/Faster-RCNN-Densecap-torch](https://github.com/ruotianluo/Faster-RCNN-Densecap-torch)
- github: [https://github.com/smallcorgi/Faster-RCNN_TF](https://github.com/smallcorgi/Faster-RCNN_TF)
- github: [https://github.com/CharlesShang/TFFRCNN](https://github.com/CharlesShang/TFFRCNN)
- github(C++ demo): [https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus](https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus)
- github: [https://github.com/yhenon/keras-frcnn](https://github.com/yhenon/keras-frcnn)

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

- - -

## MultiBox

**Scalable Object Detection using Deep Neural Networks**

- intro: first MultiBox. Train a CNN to predict Region of Interest.
- arxiv: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)
- blog: [https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html](https://research.googleblog.com/2014/12/high-quality-object-detection-at-scale.html)

**Scalable, High-Quality Object Detection**

- intro: second MultiBox
- arxiv: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- github: [https://github.com/google/multibox](https://github.com/google/multibox)

## SPP-Net

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

- intro: ECCV 2014 / TPAMI 2015
- arxiv: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- github: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

## DeepID-Net

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

## NoC

**Object Detection Networks on Convolutional Feature Maps**

- intro: TPAMI 2015
- arxiv: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization and Structured Prediction**

- arxiv: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- github: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

## DeepBox

**DeepBox: Learning Objectness with Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1505.02146](http://arxiv.org/abs/1505.02146)
- github: [https://github.com/weichengkuo/DeepBox](https://github.com/weichengkuo/DeepBox)

## MR-CNN

**Object detection via a multi-region & semantic segmentation-aware CNN model**

- intro: ICCV 2015. MR-CNN
- arxiv: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- github: [https://github.com/gidariss/mrcnn-object-detection](https://github.com/gidariss/mrcnn-object-detection)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

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

**Yolo_mark: GUI for marking bounded boxes of objects in images for training Yolo v2**

- github: [https://github.com/AlexeyAB/Yolo_mark](https://github.com/AlexeyAB/Yolo_mark)

- - -

**R-CNN minus R**

- arxiv: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)

## AttentionNet

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

**Enhancement of SSD by concatenating feature maps for object detection**

- intro: rainbow SSD (R-SSD)
- arxiv: [https://arxiv.org/abs/1705.09587](https://arxiv.org/abs/1705.09587)

## DSSD

**DSSD : Deconvolutional Single Shot Detector**

- intro: UNC Chapel Hill & Amazon Inc
- arxiv: [https://arxiv.org/abs/1701.06659](https://arxiv.org/abs/1701.06659)
- demo: [http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4](http://120.52.72.53/www.cs.unc.edu/c3pr90ntc0td/~cyfu/dssd_lalaland.mp4)

**Context-aware Single-Shot Detector**

- keywords: CSSD, DiCSSD, DeCSSD, effective receptive fields (ERFs),  theoretical receptive fields (TRFs)
- arxiv: [https://arxiv.org/abs/1707.08682](https://arxiv.org/abs/1707.08682)

- - -

## Inside-Outside Net (ION)

**Inside-Outside Net: Detecting Objects in Context with Skip Pooling and Recurrent Neural Networks**

- intro: "0.8s per image on a Titan X GPU (excluding proposal generation) without two-stage bounding-box regression
and 1.15s per image with it".
- arxiv: [http://arxiv.org/abs/1512.04143](http://arxiv.org/abs/1512.04143)
- slides: [http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf](http://www.seanbell.ca/tmp/ion-coco-talk-bell2015.pdf)
- coco-leaderboard: [http://mscoco.org/dataset/#detections-leaderboard](http://mscoco.org/dataset/#detections-leaderboard)

**Adaptive Object Detection Using Adjacency and Zoom Prediction**

- intro: CVPR 2016. AZ-Net
- arxiv: [http://arxiv.org/abs/1512.07711](http://arxiv.org/abs/1512.07711)
- github: [https://github.com/luyongxi/az-net](https://github.com/luyongxi/az-net)
- youtube: [https://www.youtube.com/watch?v=YmFtuNwxaNM](https://www.youtube.com/watch?v=YmFtuNwxaNM)

## G-CNN

**G-CNN: an Iterative Grid Based Object Detector**

- arxiv: [http://arxiv.org/abs/1512.07729](http://arxiv.org/abs/1512.07729)

**Factors in Finetuning Deep Model for object detection**

**Factors in Finetuning Deep Model for Object Detection with Long-tail Distribution**

- intro: CVPR 2016.rank 3rd for provided data and 2nd for external data on ILSVRC 2015 object detection
- project page: [http://www.ee.cuhk.edu.hk/~wlouyang/projects/ImageNetFactors/CVPR16.html](http://www.ee.cuhk.edu.hk/~wlouyang/projects/ImageNetFactors/CVPR16.html)
- arxiv: [http://arxiv.org/abs/1601.05150](http://arxiv.org/abs/1601.05150)

**We don't need no bounding-boxes: Training object class detectors using only human verification**

- arxiv: [http://arxiv.org/abs/1602.08405](http://arxiv.org/abs/1602.08405)

## HyperNet

**HyperNet: Towards Accurate Region Proposal Generation and Joint Object Detection**

- arxiv: [http://arxiv.org/abs/1604.00600](http://arxiv.org/abs/1604.00600)

## MultiPathNet

**A MultiPath Network for Object Detection**

- intro: BMVC 2016. Facebook AI Research (FAIR)
- arxiv: [http://arxiv.org/abs/1604.02135](http://arxiv.org/abs/1604.02135)
- github: [https://github.com/facebookresearch/multipathnet](https://github.com/facebookresearch/multipathnet)

## CRAFT

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

**Exploit All the Layers: Fast and Accurate CNN Object Detector with Scale Dependent Pooling and Cascaded Rejection Classifiers**

- intro: CVPR 2016
- keywords: scale-dependent pooling  (SDP), cascaded rejection classifiers (CRC)
- paper: [http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf](http://www-personal.umich.edu/~wgchoi/SDP-CRC_camready.pdf)

## R-FCN

**R-FCN: Object Detection via Region-based Fully Convolutional Networks**

- arxiv: [http://arxiv.org/abs/1605.06409](http://arxiv.org/abs/1605.06409)
- github: [https://github.com/daijifeng001/R-FCN](https://github.com/daijifeng001/R-FCN)
- github: [https://github.com/Orpine/py-R-FCN](https://github.com/Orpine/py-R-FCN)
- github: [https://github.com/PureDiors/pytorch_RFCN](https://github.com/PureDiors/pytorch_RFCN)
- github: [https://github.com/bharatsingh430/py-R-FCN-multiGPU](https://github.com/bharatsingh430/py-R-FCN-multiGPU)
- github: [https://github.com/xdever/RFCN-tensorflow](https://github.com/xdever/RFCN-tensorflow)

**Recycle deep features for better object detection**

- arxiv: [http://arxiv.org/abs/1607.05066](http://arxiv.org/abs/1607.05066)

## MS-CNN

**A Unified Multi-scale Deep Convolutional Neural Network for Fast Object Detection**

- intro: ECCV 2016
- intro: 640×480: 15 fps, 960×720: 8 fps
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

## PVANET

**PVANET: Deep but Lightweight Neural Networks for Real-time Object Detection**

- intro: "less channels with more layers", concatenated ReLU, Inception, and HyperNet, batch normalization, residual connections
- arxiv: [http://arxiv.org/abs/1608.08021](http://arxiv.org/abs/1608.08021)
- github: [https://github.com/sanghoon/pva-faster-rcnn](https://github.com/sanghoon/pva-faster-rcnn)
- leaderboard(PVANet 9.0): [http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=4)

**PVANet: Lightweight Deep Neural Networks for Real-time Object Detection**

- intro: Presented at NIPS 2016 Workshop on Efficient Methods for Deep Neural Networks (EMDNN). 
Continuation of [arXiv:1608.08021](https://arxiv.org/abs/1608.08021)
- arxiv: [https://arxiv.org/abs/1611.08588](https://arxiv.org/abs/1611.08588)

## GBD-Net

**Gated Bi-directional CNN for Object Detection**

- intro: The Chinese University of Hong Kong & Sensetime Group Limited
- paper: [http://link.springer.com/chapter/10.1007/978-3-319-46478-7_22](http://link.springer.com/chapter/10.1007/978-3-319-46478-7_22)
- mirror: [https://pan.baidu.com/s/1dFohO7v](https://pan.baidu.com/s/1dFohO7v)

**Crafting GBD-Net for Object Detection**

- intro: winner of the ImageNet object detection challenge of 2016. CUImage and CUVideo
- intro: gated bi-directional CNN (GBD-Net)
- arxiv: [https://arxiv.org/abs/1610.02579](https://arxiv.org/abs/1610.02579)
- github: [https://github.com/craftGBD/craftGBD](https://github.com/craftGBD/craftGBD)

## StuffNet

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

- intro: Google Research
- arxiv: [https://arxiv.org/abs/1611.10012](https://arxiv.org/abs/1611.10012)

**SqueezeDet: Unified, Small, Low Power Fully Convolutional Neural Networks for Real-Time Object Detection for Autonomous Driving**

- arxiv: [https://arxiv.org/abs/1612.01051](https://arxiv.org/abs/1612.01051)
- github: [https://github.com/BichenWuUCB/squeezeDet](https://github.com/BichenWuUCB/squeezeDet)

## Feature Pyramid Network (FPN)

**Feature Pyramid Networks for Object Detection**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/1612.03144](https://arxiv.org/abs/1612.03144)

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

## CC-Net

**Learning Chained Deep Features and Classifiers for Cascade in Object Detection**

- intro: chained cascade network (CC-Net). 81.1% mAP on PASCAL VOC 2007
- arxiv: [https://arxiv.org/abs/1702.07054](https://arxiv.org/abs/1702.07054)

**DeNet: Scalable Real-time Object Detection with Directed Sparse Sampling**

[https://arxiv.org/abs/1703.10295](https://arxiv.org/abs/1703.10295)

**Discriminative Bimodal Networks for Visual Localization and Detection with Natural Language Queries**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1704.03944](https://arxiv.org/abs/1704.03944)

**Spatial Memory for Context Reasoning in Object Detection**

- arxiv: [https://arxiv.org/abs/1704.04224](https://arxiv.org/abs/1704.04224)

**Accurate Single Stage Detector Using Recurrent Rolling Convolution**

- intro: CVPR 2017. SenseTime
- keywords: Recurrent Rolling Convolution (RRC)
- arxiv: [https://arxiv.org/abs/1704.05776](https://arxiv.org/abs/1704.05776)
- github: [https://github.com/xiaohaoChen/rrc_detection](https://github.com/xiaohaoChen/rrc_detection)

**Deep Occlusion Reasoning for Multi-Camera Multi-Target Detection**

[https://arxiv.org/abs/1704.05775](https://arxiv.org/abs/1704.05775)

**S-OHEM: Stratified Online Hard Example Mining for Object Detection**

[https://arxiv.org/abs/1705.02233](https://arxiv.org/abs/1705.02233)

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

**SMC Faster R-CNN: Toward a scene-specialized multi-object detector**

[https://arxiv.org/abs/1706.10217](https://arxiv.org/abs/1706.10217)

**Towards lightweight convolutional neural networks for object detection**

[https://arxiv.org/abs/1707.01395](https://arxiv.org/abs/1707.01395)

**RON: Reverse Connection with Objectness Prior Networks for Object Detection**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1707.01691](https://arxiv.org/abs/1707.01691)
- github: [https://github.com/taokong/RON](https://github.com/taokong/RON)

**Residual Features and Unified Prediction Network for Single Stage Detection**

[https://arxiv.org/abs/1707.05031](https://arxiv.org/abs/1707.05031)

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

## DSOD

**DSOD: Learning Deeply Supervised Object Detectors from Scratch**

![](https://user-images.githubusercontent.com/3794909/28934967-718c9302-78b5-11e7-89ee-8b514e53e23c.png)

- intro: ICCV 2017. Fudan University & Tsinghua University & Intel Labs China
- arxiv: [https://arxiv.org/abs/1708.01241](https://arxiv.org/abs/1708.01241)
- github: [https://github.com/szq0214/DSOD](https://github.com/szq0214/DSOD)

**Focal Loss for Dense Object Detection**

- intro: Facebook AI Research
- arxiv: [https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)

**CoupleNet: Coupling Global Structure with Local Parts for Object Detection**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.02863](https://arxiv.org/abs/1708.02863)

## NMS

**End-to-End Integration of a Convolutional Network, Deformable Parts Model and Non-Maximum Suppression**

- intro: CVPR 2015
- arxiv: [http://arxiv.org/abs/1411.5309](http://arxiv.org/abs/1411.5309)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wan_End-to-End_Integration_of_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Wan_End-to-End_Integration_of_2015_CVPR_paper.pdf)

**A convnet for non-maximum suppression**

- arxiv: [http://arxiv.org/abs/1511.06437](http://arxiv.org/abs/1511.06437)

**Improving Object Detection With One Line of Code**

- intro: University of Maryland
- keywords: Soft-NMS
- arxiv: [https://arxiv.org/abs/1704.04503](https://arxiv.org/abs/1704.04503)
- github: [https://github.com/bharatsingh430/soft-nms](https://github.com/bharatsingh430/soft-nms)

**Learning non-maximum suppression**

[https://arxiv.org/abs/1705.02950](https://arxiv.org/abs/1705.02950)

## Weakly Supervised Object Detection

**Track and Transfer: Watching Videos to Simulate Strong Human Supervision for Weakly-Supervised Object Detection**

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1604.05766](http://arxiv.org/abs/1604.05766)

**Weakly supervised object detection using pseudo-strong labels**

- arxiv: [http://arxiv.org/abs/1607.04731](http://arxiv.org/abs/1607.04731)

**Saliency Guided End-to-End Learning for Weakly Supervised Object Detection**

- intro: IJCAI 2017
- arxiv: [https://arxiv.org/abs/1706.06768](https://arxiv.org/abs/1706.06768)

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

# Object Detection in 3D

**Vote3Deep: Fast Object Detection in 3D Point Clouds Using Efficient Convolutional Neural Networks**

- arxiv: [https://arxiv.org/abs/1609.06666](https://arxiv.org/abs/1609.06666)

# Object Detection on RGB-D

**Learning Rich Features from RGB-D Images for Object Detection and Segmentation**

- arxiv: [http://arxiv.org/abs/1407.5736](http://arxiv.org/abs/1407.5736)

**Differential Geometry Boosts Convolutional Neural Networks for Object Detection**

- intro: CVPR 2016
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html](http://www.cv-foundation.org/openaccess/content_cvpr_2016_workshops/w23/html/Wang_Differential_Geometry_Boosts_CVPR_2016_paper.html)

**A Self-supervised Learning System for Object Detection using Physics Simulation and Multi-view Pose Estimation**

[https://arxiv.org/abs/1703.03347](https://arxiv.org/abs/1703.03347)

# Salient Object Detection

This task involves predicting the salient regions of an image given by human eye fixations.

**Best Deep Saliency Detection Models (CVPR 2016 & 2015)**

[http://i.cs.hku.hk/~yzyu/vision.html](http://i.cs.hku.hk/~yzyu/vision.html)

**Large-scale optimization of hierarchical features for saliency prediction in natural images**

- paper: [http://coxlab.org/pdfs/cvpr2014_vig_saliency.pdf](http://coxlab.org/pdfs/cvpr2014_vig_saliency.pdf)

**Predicting Eye Fixations using Convolutional Neural Networks**

- paper: [http://www.escience.cn/system/file?fileId=72648](http://www.escience.cn/system/file?fileId=72648)

**Saliency Detection by Multi-Context Deep Learning**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Saliency_Detection_by_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Zhao_Saliency_Detection_by_2015_CVPR_paper.pdf)

**DeepSaliency: Multi-Task Deep Neural Network Model for Salient Object Detection**

- arxiv: [http://arxiv.org/abs/1510.05484](http://arxiv.org/abs/1510.05484)

**SuperCNN: A Superpixelwise Convolutional Neural Network for Salient Object Detection**

- paper: [www.shengfenghe.com/supercnn-a-superpixelwise-convolutional-neural-network-for-salient-object-detection.html](www.shengfenghe.com/supercnn-a-superpixelwise-convolutional-neural-network-for-salient-object-detection.html)

**Shallow and Deep Convolutional Networks for Saliency Prediction**

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1603.00845](http://arxiv.org/abs/1603.00845)
- github: [https://github.com/imatge-upc/saliency-2016-cvpr](https://github.com/imatge-upc/saliency-2016-cvpr)

**Recurrent Attentional Networks for Saliency Detection**

- intro: CVPR 2016. recurrent attentional convolutional-deconvolution network (RACDNN)
- arxiv: [http://arxiv.org/abs/1604.03227](http://arxiv.org/abs/1604.03227)

**Two-Stream Convolutional Networks for Dynamic Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1607.04730](http://arxiv.org/abs/1607.04730)

**Unconstrained Salient Object Detection**

**Unconstrained Salient Object Detection via Proposal Subset Optimization**

![](http://cs-people.bu.edu/jmzhang/images/pasted%20image%201465x373.jpg)

- intro: CVPR 2016
- project page: [http://cs-people.bu.edu/jmzhang/sod.html](http://cs-people.bu.edu/jmzhang/sod.html)
- paper: [http://cs-people.bu.edu/jmzhang/SOD/CVPR16SOD_camera_ready.pdf](http://cs-people.bu.edu/jmzhang/SOD/CVPR16SOD_camera_ready.pdf)
- github: [https://github.com/jimmie33/SOD](https://github.com/jimmie33/SOD)
- caffe model zoo: [https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-object-proposal-models-for-salient-object-detection](https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-object-proposal-models-for-salient-object-detection)

**DHSNet: Deep Hierarchical Saliency Network for Salient Object Detection**

- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Liu_DHSNet_Deep_Hierarchical_CVPR_2016_paper.pdf)

**Salient Object Subitizing**

![](http://cs-people.bu.edu/jmzhang/images/frontpage.png?crc=123070793)

- intro: CVPR 2015
- intro: predicting the existence and the number of salient objects in an image using holistic cues
- project page: [http://cs-people.bu.edu/jmzhang/sos.html](http://cs-people.bu.edu/jmzhang/sos.html)
- arxiv: [http://arxiv.org/abs/1607.07525](http://arxiv.org/abs/1607.07525)
- paper: [http://cs-people.bu.edu/jmzhang/SOS/SOS_preprint.pdf](http://cs-people.bu.edu/jmzhang/SOS/SOS_preprint.pdf)
- caffe model zoo: [https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing](https://github.com/BVLC/caffe/wiki/Model-Zoo#cnn-models-for-salient-object-subitizing)

**Deeply-Supervised Recurrent Convolutional Neural Network for Saliency Detection**

- intro: ACMMM 2016. deeply-supervised recurrent convolutional neural network (DSRCNN)
- arxiv: [http://arxiv.org/abs/1608.05177](http://arxiv.org/abs/1608.05177)

**Saliency Detection via Combining Region-Level and Pixel-Level Predictions with CNNs**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1608.05186](http://arxiv.org/abs/1608.05186)

**Edge Preserving and Multi-Scale Contextual Neural Network for Salient Object Detection**

- arxiv: [http://arxiv.org/abs/1608.08029](http://arxiv.org/abs/1608.08029)

**A Deep Multi-Level Network for Saliency Prediction**

- arxiv: [http://arxiv.org/abs/1609.01064](http://arxiv.org/abs/1609.01064)

**Visual Saliency Detection Based on Multiscale Deep CNN Features**

- intro: IEEE Transactions on Image Processing
- arxiv: [http://arxiv.org/abs/1609.02077](http://arxiv.org/abs/1609.02077)

**A Deep Spatial Contextual Long-term Recurrent Convolutional Network for Saliency Detection**

- intro: DSCLRCN
- arxiv: [https://arxiv.org/abs/1610.01708](https://arxiv.org/abs/1610.01708)

**Deeply supervised salient object detection with short connections**

- arxiv: [https://arxiv.org/abs/1611.04849](https://arxiv.org/abs/1611.04849)

**Weakly Supervised Top-down Salient Object Detection**

- intro: Nanyang Technological University
- arxiv: [https://arxiv.org/abs/1611.05345](https://arxiv.org/abs/1611.05345)

**SalGAN: Visual Saliency Prediction with Generative Adversarial Networks**

- project page: [https://imatge-upc.github.io/saliency-salgan-2017/](https://imatge-upc.github.io/saliency-salgan-2017/)
- arxiv: [https://arxiv.org/abs/1701.01081](https://arxiv.org/abs/1701.01081)

**Visual Saliency Prediction Using a Mixture of Deep Neural Networks**

- arxiv: [https://arxiv.org/abs/1702.00372](https://arxiv.org/abs/1702.00372)

**A Fast and Compact Salient Score Regression Network Based on Fully Convolutional Network**

- arxiv: [https://arxiv.org/abs/1702.00615](https://arxiv.org/abs/1702.00615)

**Saliency Detection by Forward and Backward Cues in Deep-CNNs**

[https://arxiv.org/abs/1703.00152](https://arxiv.org/abs/1703.00152)

**Supervised Adversarial Networks for Image Saliency Detection**

[https://arxiv.org/abs/1704.07242](https://arxiv.org/abs/1704.07242)

**Group-wise Deep Co-saliency Detection**

[https://arxiv.org/abs/1707.07381](https://arxiv.org/abs/1707.07381)

**Towards the Success Rate of One: Real-time Unconstrained Salient Object Detection**

- intro: University of Maryland College Park & eBay Inc
- arxiv: [https://arxiv.org/abs/1708.00079](https://arxiv.org/abs/1708.00079)

**Amulet: Aggregating Multi-level Convolutional Features for Salient Object Detection**

- intro: ICCV 2017
- arixv: [https://arxiv.org/abs/1708.02001](https://arxiv.org/abs/1708.02001)

**Learning Uncertain Convolutional Features for Accurate Saliency Detection**

- intro: Accepted as a poster in ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.02031](https://arxiv.org/abs/1708.02031)

## Saliency Detection in Video

**Deep Learning For Video Saliency Detection**

- arxiv: [https://arxiv.org/abs/1702.00871](https://arxiv.org/abs/1702.00871)

**Video Salient Object Detection Using Spatiotemporal Deep Features**

[https://arxiv.org/abs/1708.01447](https://arxiv.org/abs/1708.01447)

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

# Specific Object Deteciton

## Face Deteciton

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

**Finding Tiny Faces**

- intro: CVPR 2017. CMU
- project page: [http://www.cs.cmu.edu/~peiyunh/tiny/index.html](http://www.cs.cmu.edu/~peiyunh/tiny/index.html)
- arxiv: [https://arxiv.org/abs/1612.04402](https://arxiv.org/abs/1612.04402)
- github: [https://github.com/peiyunh/tiny](https://github.com/peiyunh/tiny)
- github(inference-only): [https://github.com/chinakook/hr101_mxnet](https://github.com/chinakook/hr101_mxnet)

**Towards a Deep Learning Framework for Unconstrained Face Detection**

- intro: overlap with CMS-RCNN
- arxiv: [https://arxiv.org/abs/1612.05322](https://arxiv.org/abs/1612.05322)

**Supervised Transformer Network for Efficient Face Detection**

- arxiv: [http://arxiv.org/abs/1607.05477](http://arxiv.org/abs/1607.05477)

### UnitBox

**UnitBox: An Advanced Object Detection Network**

- intro: ACM MM 2016
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

### MTCNN

**Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks**

**Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks**

![](https://kpzhang93.github.io/MTCNN_face_detection_alignment/support/index.png)

- project page: [https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html](https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
- arxiv: [https://arxiv.org/abs/1604.02878](https://arxiv.org/abs/1604.02878)
- github(Matlab): [https://github.com/kpzhang93/MTCNN_face_detection_alignment](https://github.com/kpzhang93/MTCNN_face_detection_alignment)
- github: [https://github.com/pangyupo/mxnet_mtcnn_face_detection](https://github.com/pangyupo/mxnet_mtcnn_face_detection)
- github: [https://github.com/DaFuCoding/MTCNN_Caffe](https://github.com/DaFuCoding/MTCNN_Caffe)
- github(MXNet): [https://github.com/Seanlinx/mtcnn](https://github.com/Seanlinx/mtcnn)
- github: [https://github.com/Pi-DeepLearning/RaspberryPi-FaceDetection-MTCNN-Caffe-With-Motion](https://github.com/Pi-DeepLearning/RaspberryPi-FaceDetection-MTCNN-Caffe-With-Motion)
- github(Caffe): [https://github.com/foreverYoungGitHub/MTCNN](https://github.com/foreverYoungGitHub/MTCNN)
- github: [https://github.com/CongWeilin/mtcnn-caffe](https://github.com/CongWeilin/mtcnn-caffe)
- github: [https://github.com/AlphaQi/MTCNN-light](https://github.com/AlphaQi/MTCNN-light)

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

**Multi-Branch Fully Convolutional Network for Face Detection**

[https://arxiv.org/abs/1707.06330](https://arxiv.org/abs/1707.06330)

**SSH: Single Stage Headless Face Detector**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.03979](https://arxiv.org/abs/1708.03979)

## Facial Point / Landmark Detection

**Deep Convolutional Network Cascade for Facial Point Detection**

![](http://mmlab.ie.cuhk.edu.hk/archive/CNN/data/Picture1.png)

- homepage: [http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm)
- paper: [http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf](http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTcvpr13.pdf)
- github: [https://github.com/luoyetx/deep-landmark](https://github.com/luoyetx/deep-landmark)

**Facial Landmark Detection by Deep Multi-task Learning**

- intro: ECCV 2014
- project page: [http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)
- paper: [http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2014_deepfacealign.pdf)
- github(Matlab): [https://github.com/zhzhanp/TCDCN-face-alignment](https://github.com/zhzhanp/TCDCN-face-alignment)

**A Recurrent Encoder-Decoder Network for Sequential Face Alignment**

- intro: ECCV 2016
- arxiv: [https://arxiv.org/abs/1608.05477](https://arxiv.org/abs/1608.05477)

**Detecting facial landmarks in the video based on a hybrid framework**

- arxiv: [http://arxiv.org/abs/1609.06441](http://arxiv.org/abs/1609.06441)

**Deep Constrained Local Models for Facial Landmark Detection**

- arxiv: [https://arxiv.org/abs/1611.08657](https://arxiv.org/abs/1611.08657)

**Effective face landmark localization via single deep network**

- arxiv: [https://arxiv.org/abs/1702.02719](https://arxiv.org/abs/1702.02719)

**A Convolution Tree with Deconvolution Branches: Exploiting Geometric Relationships for Single Shot Keypoint Detection**

[https://arxiv.org/abs/1704.01880](https://arxiv.org/abs/1704.01880)

**Deep Alignment Network: A convolutional neural network for robust face alignment**

- intro: CVPRW 2017
- arxiv: [https://arxiv.org/abs/1706.01789](https://arxiv.org/abs/1706.01789)
- gihtub: [https://github.com/MarekKowalski/DeepAlignmentNetwork](https://github.com/MarekKowalski/DeepAlignmentNetwork)

# People Detection

**End-to-end people detection in crowded scenes**

![](/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg)

- arxiv: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- github: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)
- youtube: [https://www.youtube.com/watch?v=QeWl0h3kQ24](https://www.youtube.com/watch?v=QeWl0h3kQ24)

**Detecting People in Artwork with CNNs**

- intro: ECCV 2016 Workshops
- arxiv: [https://arxiv.org/abs/1610.08871](https://arxiv.org/abs/1610.08871)

**Deep Multi-camera People Detection**

- arxiv: [https://arxiv.org/abs/1702.04593](https://arxiv.org/abs/1702.04593)

## Person Head Detection

**Context-aware CNNs for person head detection**

- intro: ICCV 2015
- project page: [http://www.di.ens.fr/willow/research/headdetection/](http://www.di.ens.fr/willow/research/headdetection/)
- arxiv: [http://arxiv.org/abs/1511.07917](http://arxiv.org/abs/1511.07917)
- github: [https://github.com/aosokin/cnn_head_detection](https://github.com/aosokin/cnn_head_detection)

## Pedestrian Detection

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

**Reduced Memory Region Based Deep Convolutional Neural Network Detection**

- intro: IEEE 2016 ICCE-Berlin
- arxiv: [http://arxiv.org/abs/1609.02500](http://arxiv.org/abs/1609.02500)

**Fused DNN: A deep neural network fusion approach to fast and robust pedestrian detection**

- arxiv: [https://arxiv.org/abs/1610.03466](https://arxiv.org/abs/1610.03466)

**Multispectral Deep Neural Networks for Pedestrian Detection**

- intro: BMVC 2016 oral
- arxiv: [https://arxiv.org/abs/1611.02644](https://arxiv.org/abs/1611.02644)

**Expecting the Unexpected: Training Detectors for Unusual Pedestrians with Adversarial Imposters**

- intro: CVPR 2017
- project page: [http://ml.cs.tsinghua.edu.cn:5000/publications/synunity/](http://ml.cs.tsinghua.edu.cn:5000/publications/synunity/)
- arxiv: [https://arxiv.org/abs/1703.06283](https://arxiv.org/abs/1703.06283)
- github(Tensorflow): [https://github.com/huangshiyu13/RPNplus](https://github.com/huangshiyu13/RPNplus)

**Illuminating Pedestrians via Simultaneous Detection & Segmentation**

[https://arxiv.org/abs/1706.08564](https://arxiv.org/abs/1706.08564

**Rotational Rectification Network for Robust Pedestrian Detection**

- intro: CMU & Volvo Construction
- arxiv: [https://arxiv.org/abs/1706.08917](https://arxiv.org/abs/1706.08917)

**STD-PD: Generating Synthetic Training Data for Pedestrian Detection in Unannotated Videos**

- intro: The University of North Carolina at Chapel Hill
- arxiv: [https://arxiv.org/abs/1707.09100](https://arxiv.org/abs/1707.09100)

## Vehicle Detection

**DAVE: A Unified Framework for Fast Vehicle Detection and Annotation**

- intro: ECCV 2016
- arxiv: [http://arxiv.org/abs/1607.04564](http://arxiv.org/abs/1607.04564)

**Evolving Boxes for fast Vehicle Detection**

- arxiv: [https://arxiv.org/abs/1702.00254](https://arxiv.org/abs/1702.00254)

## Traffic-Sign Detection

**Traffic-Sign Detection and Classification in the Wild**

- project page(code+dataset): [http://cg.cs.tsinghua.edu.cn/traffic-sign/](http://cg.cs.tsinghua.edu.cn/traffic-sign/)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Traffic-Sign_Detection_and_CVPR_2016_paper.pdf)
- code & model: [http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip](http://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/newdata0411.zip)

**Detecting Small Signs from Large Images**

- intro: IEEE Conference on Information Reuse and Integration (IRI) 2017 oral
- arxiv: [https://arxiv.org/abs/1706.08574](https://arxiv.org/abs/1706.08574)

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

**Convolutional Oriented Boundaries: From Image Segmentation to High-Level Tasks**

- project page: [http://www.vision.ee.ethz.ch/~cvlsegmentation/](http://www.vision.ee.ethz.ch/~cvlsegmentation/)
- arxiv: [https://arxiv.org/abs/1701.04658](https://arxiv.org/abs/1701.04658)
- github: [https://github.com/kmaninis/COB](https://github.com/kmaninis/COB)

**Richer Convolutional Features for Edge Detection**

- intro: CVPR 2017
- keywords: richer convolutional features (RCF)
- arxiv: [https://arxiv.org/abs/1612.02103](https://arxiv.org/abs/1612.02103)
- github: [https://github.com/yun-liu/rcf](https://github.com/yun-liu/rcf)

**Contour Detection from Deep Patch-level Boundary Prediction**

[https://arxiv.org/abs/1705.03159](https://arxiv.org/abs/1705.03159)

**CASENet: Deep Category-Aware Semantic Edge Detection**

- intro: CVPR 2017
- arxiv: [https://arxiv.org/abs/1705.09759](https://arxiv.org/abs/1705.09759)

## Skeleton Detection

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

## Fruit Detection

**Deep Fruit Detection in Orchards**

- arxiv: [https://arxiv.org/abs/1610.03677](https://arxiv.org/abs/1610.03677)

**Image Segmentation for Fruit Detection and Yield Estimation in Apple Orchards**

- intro: The Journal of Field Robotics in May 2016
- project page: [http://confluence.acfr.usyd.edu.au/display/AGPub/](http://confluence.acfr.usyd.edu.au/display/AGPub/)
- arxiv: [https://arxiv.org/abs/1610.08120](https://arxiv.org/abs/1610.08120)

## Part Detection

**Objects as context for part detection**

[https://arxiv.org/abs/1703.09529](https://arxiv.org/abs/1703.09529)

## Others

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

**Pixel-wise Ear Detection with Convolutional Encoder-Decoder Networks**

- arxiv: [https://arxiv.org/abs/1702.00307](https://arxiv.org/abs/1702.00307)

**Automatic Handgun Detection Alarm in Videos Using Deep Learning**

- arxiv: [https://arxiv.org/abs/1702.05147](https://arxiv.org/abs/1702.05147)
- results: [https://github.com/SihamTabik/Pistol-Detection-in-Videos](https://github.com/SihamTabik/Pistol-Detection-in-Videos)

**Using Deep Networks for Drone Detection**

- intro: AVSS 2017
- arxiv: [https://arxiv.org/abs/1706.05726](https://arxiv.org/abs/1706.05726)

**Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection**

- intro: ICCV 2017
- arxiv: [https://arxiv.org/abs/1708.01642](https://arxiv.org/abs/1708.01642)

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

# Tutorials / Talks

**Convolutional Feature Maps: Elements of efficient (and accurate) CNN-based object detection**

- slides: [http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf](http://research.microsoft.com/en-us/um/people/kahe/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)

**Towards Good Practices for Recognition & Detection**

- intro: Hikvision Research Institute. Supervised Data Augmentation (SDA)
- slides: [http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf](http://image-net.org/challenges/talks/2016/Hikvision_at_ImageNet_2016.pdf)

# Projects

**TensorBox: a simple framework for training neural networks to detect objects in images**

- intro: "The basic model implements the simple and robust GoogLeNet-OverFeat algorithm. 
We additionally provide an implementation of the [ReInspect](https://github.com/Russell91/ReInspect/) algorithm"
- github: [https://github.com/Russell91/TensorBox](https://github.com/Russell91/TensorBox)

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
