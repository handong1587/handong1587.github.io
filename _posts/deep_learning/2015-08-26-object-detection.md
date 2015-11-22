---
layout: post
category: deep_learning
title: Object Detection
date: 2015-08-26
---

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

|  method  |ILSVRC 2013 mAP|
|:--------:|:-------------:|
|OverFeat  |24.3%          |

- paper: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- code: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

**Rich feature hierarchies for accurate object detection and semantic segmentation(R-CNN)**

|method                |VOC 2007 mAP|VOC 2010 mAP|VOC 2012 mAP|ILSVRC 2013 mAP|
|:--------------------:|:----------:|:----------:|:----------:|:-------------:|
|R-CNN,AlexNet         |54.2%       |50.2%       |49.6%       |               |
|R-CNN,bbox reg,AlexNet|58.5%       |53.7%       |53.3%       |31.4%          |
|R-CNN,bbox reg,ZFNet  |59.2%       |            |            |               |
|R-CNN,VGG-Net         |62.2%       |            |            |               |
|R-CNN,bbox reg,VGG-Net|66.0%       |            |            |               |

- paper: [http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- slides: [http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf](http://www.image-net.org/challenges/LSVRC/2013/slides/r-cnn-ilsvrc2013-workshop.pdf)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- code: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- notes: [http://zhangliliang.com/2014/07/23/paper-note-rcnn/](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)

**Scalable Object Detection using Deep Neural Networks**

- paper: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

|method               |VOC 2007 mAP|ILSVRC 2013 mAP|
|:-------------------:|:----------:|:-------------:|
|SPP_net(ZF-5),1-model|54.2%       |31.84%         |
|SPP_net(ZF-5),2-model|60.9%       |               |
|SPP_net(ZF-5),6-model|            |35.11%         |

- paper: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- code: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

**Scalable, High-Quality Object Detection**

- paper: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

**DeepID-Net: Deformable Deep Convolutional Neural Networks for Object Detection**

|method        |VOC 2007 mAP|ILSVRC 2013 mAP|
|:------------:|:----------:|:-------------:|
|DeepID-Net    |64.1%       |50.3%          |

- arXiv: [http://arxiv.org/abs/1412.5661](http://arxiv.org/abs/1412.5661)

**What makes for effective detection proposals?(PAMI 2015)**

- homepage: [https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/)
- arXiv: [http://arxiv.org/abs/1502.05082](http://arxiv.org/abs/1502.05082)
- github: [https://github.com/hosang/detection-proposals](https://github.com/hosang/detection-proposals)

**Object Detection Networks on Convolutional Feature Maps**

|method    |Trained on |mAP       |
|:--------:|:---------:|:--------:|
|NoC       |07+12      |68.8%     |
|NoC,bb    |07+12      |71.6%     |
|NoC,+EB   |07+12      |71.8%     |
|NoC,+EB,bb|07+12      |73.3%     |

- paper: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

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

- paper: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- code: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

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

- paper: [http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- code: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/](http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/)
- notes: [http://blog.csdn.net/linj_m/article/details/48930179](http://blog.csdn.net/linj_m/article/details/48930179)

**DeepBox: Learning Objectness with Convolutional Networks**

- arXiv: [http://arxiv.org/abs/1505.02146](http://arxiv.org/abs/1505.02146)
- github: [https://github.com/weichengkuo/DeepBox](https://github.com/weichengkuo/DeepBox)

**Object detection via a multi-region & semantic segmentation-aware CNN model**

|Model     |Trained on|VOC 2007 mAP|
|:--------:|:--------:|:----------:|
|VGG-net   |07+12     |78.2%       |
|VGG-net   |07        |74.9%       |

|Model     |Trained on|VOC 2012 mAP|
|:--------:|:--------:|:----------:|
|VGG-net   |07+12     |73.9%       |
|VGG-net   |12        |70.7%       |

- paper: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- code: "Pdf and code will appear here shortly -- stay tuned"  <br />
 [http://imagine.enpc.fr/~komodakn/](http://imagine.enpc.fr/~komodakn/)
- note: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

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

**You Only Look Once: Unified, Real-Time Object Detection(YOLO)**

- paper: [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)
- github: [https://github.com/pjreddie/darknet](https://github.com/pjreddie/darknet)

**End-to-end people detection in crowded scenes**

<img src="/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg"/>

- paper: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- code: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)

**R-CNN minus R**

- paper: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)

**DenseBox: Unifying Landmark Localization with End to End Object Detection**

- paper: [http://arxiv.org/abs/1509.04874](http://arxiv.org/abs/1509.04874)
- demo: [http://pan.baidu.com/s/1mgoWWsS](http://pan.baidu.com/s/1mgoWWsS)
- KITTI result: [http://www.cvlibs.net/datasets/kitti/eval_object.php](http://www.cvlibs.net/datasets/kitti/eval_object.php)
