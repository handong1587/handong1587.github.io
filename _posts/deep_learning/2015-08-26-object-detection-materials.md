---
layout: post
categories: deep_learning
title: Object Detection Materials
---

{{ page.title }}
================

<p class="meta">26 Aug 2015 - Beijing</p>

**Deep Neural Networks for Object Detection**

- paper: [http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf](http://papers.nips.cc/paper/5207-deep-neural-networks-for-object-detection.pdf)

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|}
\\hline
  \\text{method}   & \\text{ILSVRC 2013 mAP} \\T \\\\\\hline
  \\text{OverFeat} &  24.3\%                 \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1312.6229](http://arxiv.org/abs/1312.6229)
- code: [https://github.com/sermanet/OverFeat](https://github.com/sermanet/OverFeat)
- code: [http://cilvr.nyu.edu/doku.php?id=software:overfeat:start](http://cilvr.nyu.edu/doku.php?id=software:overfeat:start)

**Rich feature hierarchies for accurate object detection and semantic segmentation**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|c|c|}
\\hline
  \\text{method} & \\text{VOC 2007 mAP} & \\text{VOC 2010 mAP} & \\text{VOC 2012 mAP} & \\text{ILSVRC 2013 mAP} \\T \\\\\\hline
  \\text{R-CNN,AlexNet}           & 54.2\% & 50.2\% & 49.6\% &         \\\\\\hline
  \\text{R-CNN,bbox reg,AlexNet}  & 58.5\% & 53.7\% & 53.3\% & 31.4\%  \\\\\\hline
  \\text{R-CNN,bbox reg,ZFNet}    & 59.2\% &        &        &         \\\\\\hline
  \\text{R-CNN,VGG-Net}           & 62.2\% &        &        &         \\\\\\hline
  \\text{R-CNN,bbox reg,VGG-Net}  & 66.0\% &        &        &         \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1311.2524](http://arxiv.org/abs/1311.2524)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- code: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)
- notes: [http://zhangliliang.com/2014/07/23/paper-note-rcnn/](http://zhangliliang.com/2014/07/23/paper-note-rcnn/)

**Scalable Object Detection using Deep Neural Networks**

- paper: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}                 & \\text{VOC 2007 mAP} & \\text{ILSVRC 2013 mAP} \\T \\\\\\hline
  \\text{SPP_net(ZF-5),1-model}  & 59.2\%               & 31.84\%        \\\\\\hline
  \\text{SPP_net(ZF-5),2-model}  & 60.9\%               &                \\\\\\hline
  \\text{SPP_net(ZF-5),6-model}  &                      & 35.11\%        \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1406.4729](http://arxiv.org/abs/1406.4729)
- code: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)
- notes: [http://zhangliliang.com/2014/09/13/paper-note-sppnet/](http://zhangliliang.com/2014/09/13/paper-note-sppnet/)

**Scalable, High-Quality Object Detection**

- paper: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

**Object Detection Networks on Convolutional Feature Maps**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Method}      & \\text{Trained on} & \\text{mAP} \\T \\\\\\hline
  \\text{NoC}         & \\text{07+12}      & 68.8\% \\\\\\hline
  \\text{NoC,bb}      & \\text{07+12}      & 71.6\% \\\\\\hline
  \\text{NoC,+EB}     & \\text{07+12}      & 71.8\% \\\\\\hline
  \\text{NoC,+EB,bb}  & \\text{07+12}      & 73.3\% \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization
and Structured Prediction**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Model}             & \\text{BBoxReg} & \\text{VOC 2007 mAP(IoU > 0.5)} \\T \\\\\\hline
  \\text{R-CNN(AlexNet)}    & \\text{No}      & 54.2\%  \\\\\\hline
  \\text{R-CNN(VGG)}        & \\text{No}      & 60.6\%  \\\\\\hline
  \\text{+StructObj}        & \\text{No}      & 61.2\%  \\\\\\hline
  \\text{+StructObj-FT}     & \\text{No}      & 62.3\%  \\\\\\hline
  \\text{+FGS}              & \\text{No}      & 64.8\%  \\\\\\hline
  \\text{+StructObj+FGS}    & \\text{No}      & 65.9\%  \\\\\\hline
  \\text{+StructObj-FT+FGS} & \\text{No}      & 66.5\%  \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Model}             & \\text{BBoxReg} & \\text{VOC 2007 mAP(IoU > 0.5)} \\T \\\\\\hline
  \\text{R-CNN(AlexNet)}    & \\text{Yes}     & 58.5\%  \\\\\\hline
  \\text{R-CNN(VGG)}        & \\text{Yes}     & 65.4\%  \\\\\\hline
  \\text{+StructObj}        & \\text{Yes}     & 66.6\%  \\\\\\hline
  \\text{+StructObj-FT}     & \\text{Yes}     & 66.9\%  \\\\\\hline
  \\text{+FGS}              & \\text{Yes}     & 67.2\%  \\\\\\hline
  \\text{+StructObj+FGS}    & \\text{Yes}     & 68.5\%  \\\\\\hline
  \\text{+StructObj-FT+FGS} & \\text{Yes}     & 68.4\%  \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- code: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

**Fast R-CNN**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}     & \\text{data} & \\text{VOC 2007 mAP}  \\T \\\\\\hline
  \\text{FRCN,VGG16} & 07           & 66.9\%                \\\\\\hline
  \\text{FRCN,VGG16} & \\text{07+12} & 70.0\%                \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}     & \\text{data} & \\text{VOC 2010 mAP} \\T \\\\\\hline
  \\text{FRCN,VGG16} & 12           & 66.1\%               \\\\\\hline
  \\text{FRCN,VGG16} & \\text{07++12} & 68.8\%               \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}     & \\text{data} & \\text{VOC 2012 mAP}  \\T \\\\\\hline
  \\text{FRCN,VGG16} & 12           & 65.7\%                \\\\\\hline
  \\text{FRCN,VGG16} & \\text{07++12} & 68.4\%                \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- code: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)
- notes: [http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/](http://zhangliliang.com/2015/05/17/paper-note-fast-rcnn/)

**Object detection via a multi-region & semantic segmentation-aware CNN model**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Model}   & \\text{Trained on} & \\text{VOC 2007 mAP} \\T \\\\\\hline
  \\text{VGG-net} & \\text{07+12}      & 78.2\% \\\\\\hline
  \\text{VGG-net} & \\text{07}         & 74.9\% \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Model}   & \\text{Trained on} & \\text{VOC 2012 mAP} \\T \\\\\\hline
  \\text{VGG-net} & \\text{07+12}      & 73.9\% \\\\\\hline
  \\text{VGG-net} & \\text{12}         & 70.7\% \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1505.01749](http://arxiv.org/abs/1505.01749)
- code: "Pdf and code will appear here shortly -- stay tuned"  <br />
 [http://imagine.enpc.fr/~komodakn/](http://imagine.enpc.fr/~komodakn/)
- note: [http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/](http://zhangliliang.com/2015/05/17/paper-note-ms-cnn/)
- notes: [http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/](http://blog.cvmarcher.com/posts/2015/05/17/multi-region-semantic-segmentation-aware-cnn/)

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\  & \\text{training data} & \\text{test data} & \\text{VOC 2007 mAP} & \\text{time/img}  \\T \\\\\\hline
  \\text{Faster RCNN, VGG-16} & \\text{07}     & \\text{VOC 2007 test} & 69.9\% & 198ms \\\\\\hline
  \\text{Faster RCNN, VGG-16} & \\text{07+12}  & \\text{VOC 2007 test} & 73.2\% & 198ms \\\\\\hline
  \\text{Faster RCNN, VGG-16} & \\text{12}     & \\text{VOC 2012 test} & 67.0\% & 198ms \\\\\\hline
  \\text{Faster RCNN, VGG-16} & \\text{07++12} & \\text{VOC 2012 test} & 70.4\% & 198ms \\\\\\hline
\\end{array}
$$

- paper: [http://arxiv.org/abs/1506.01497](http://arxiv.org/abs/1506.01497)
- code: [https://github.com/ShaoqingRen/faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)

**You Only Look Once: Unified, Real-Time Object Detection**

- paper: [http://arxiv.org/abs/1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)

**End-to-end people detection in crowded scenes**

<img src="/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg"/>

- paper: [http://arxiv.org/abs/1506.04878](http://arxiv.org/abs/1506.04878)
- code: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)

**R-CNN minus R**

- paper: [http://arxiv.org/abs/1506.06981](http://arxiv.org/abs/1506.06981)
