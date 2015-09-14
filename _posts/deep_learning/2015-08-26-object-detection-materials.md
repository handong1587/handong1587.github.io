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

**OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks**:

*(Submitted on 21 Dec 2013 (v1), last revised 24 Feb 2014 (this version, v4))*

- paper: [arXiv:1312.6229](http://arxiv.org/abs/1312.6229)

**Rich feature hierarchies for accurate object detection and semantic segmentation**

*(Submitted on 11 Nov 2013 (v1), last revised 22 Oct 2014 (this version, v5))*

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|c|}
\\hline
  \\text{method} & \\text{VOC 2007 mAP} & \\text{VOC 2010 mAP} & \\text{VOC 2012 mAP} & \\text{ILSVRC2013 test mAP} \\T \\\\\\hline
  \\text{R-CNN,AlexNet}           & 54.2% & 50.2% & 49.6% &        \\\\\\hline
  \\text{R-CNN bbox reg,AlexNet}  & 58.5% & 53.7% & 53.3% & 31.4%  \\\\\\hline
  \\text{R-CNN,VGG-Net}           & 62.2% &       &       &        \\\\\\hline
  \\text{R-CNN bbox reg,VGG-Net}  & 66.0% &       &       &   \\\\\\hline
\\end{array}
$$

- paper: [arXiv:1311.2524](http://arxiv.org/abs/1311.2524)
- slides: [http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf](http://www.cs.berkeley.edu/~rbg/slides/rcnn-cvpr14-slides.pdf)
- code: [https://github.com/rbgirshick/rcnn](https://github.com/rbgirshick/rcnn)

**Scalable Object Detection using Deep Neural Networks**

*(Submitted on 8 Dec 2013)*

- paper: [http://arxiv.org/abs/1312.2249](http://arxiv.org/abs/1312.2249)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

**Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition**:

*(Submitted on 18 Jun 2014 (v1), last revised 23 Apr 2015 (this version, v4))*

- paper: [arXiv:1406.4729](http://arxiv.org/abs/1406.4729)
- code: [https://github.com/ShaoqingRen/SPP_net](https://github.com/ShaoqingRen/SPP_net)

**Scalable, High-Quality Object Detection**

*(Submitted on 3 Dec 2014 (v1), last revised 26 Feb 2015 (this version, v2))*

- paper: [http://arxiv.org/abs/1412.1441](http://arxiv.org/abs/1412.1441)
- code: [https://github.com/google/multibox](https://github.com/google/multibox)

**Object Detection Networks on Convolutional Feature Maps**

*(Submitted on 23 Apr 2015)*

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Method}      & \\text{Trained on} & \\text{mAP} \\T \\\\\\hline
  \\text{NoC}         & 07+12              & 68.8% \\\\\\hline
  \\text{NoC,bb}      & 07+12              & 71.6% \\\\\\hline
  \\text{NoC,+EB}     & 07+12              & 71.8% \\\\\\hline
  \\text{NoC,+EB,bb}  & 07+12              & 73.3% \\\\\\hline
\\end{array}
$$

- paper: [arXiv:1504.06066](http://arxiv.org/abs/1504.06066)

**Improving Object Detection with Deep Convolutional Networks via Bayesian Optimization
and Structured Prediction**

*(Submitted on 13 Apr 2015)*

Test set mAP of VOC 2007 with IoU > 0.5:
$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Model}             & \\text{BBoxReg} & \\text{mAP} \\T \\\\\\hline
  \\text{R-CNN(AlexNet)}    & No              & 54.2% \\\\\\hline
  \\text{R-CNN(VGG)}        & No              & 60.6% \\\\\\hline
  \\text{+StructObj}        & No              & 61.2% \\\\\\hline
  \\text{+StructObj-FT}     & No              & 62.3% \\\\\\hline
  \\text{+FGS}              & No              & 64.8% \\\\\\hline
  \\text{+StructObj+FGS}    & No              & 65.9% \\\\\\hline
  \\text{+StructObj-FT+FGS} & No              & 66.5% \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{Model}             & \\text{BBoxReg} & \\text{mAP} \\T \\\\\\hline
  \\text{R-CNN(AlexNet)}    & Yes             & 58.5% \\\\\\hline
  \\text{R-CNN(VGG)}        & Yes             & 65.4% \\\\\\hline
  \\text{+StructObj}        & Yes             & 66.6% \\\\\\hline
  \\text{+StructObj-FT}     & Yes             & 66.9% \\\\\\hline
  \\text{+FGS}              & Yes             & 67.2% \\\\\\hline
  \\text{+StructObj+FGS}    & Yes             & 68.5% \\\\\\hline
  \\text{+StructObj-FT+FGS} & Yes             & 68.4% \\\\\\hline
\\end{array}
$$

- paper: [arXiv:1504.03293](http://arxiv.org/abs/1504.03293)
- slides: [2015-cvpr-det-slides](http://www.ytzhang.net/files/publications/2015-cvpr-det-slides.pdf)
- code: [https://github.com/YutingZhang/fgs-obj](https://github.com/YutingZhang/fgs-obj)

**Fast R-CNN**

*(Submitted on 30 Apr 2015)*

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}     & \\text{data} & \\text{VOC 2007 mAP}  \\T \\\\\\hline
  \\text{FRCN,VGG16} & 07           & 66.9%                 \\\\\\hline
  \\text{FRCN,VGG16} & 07+12        & 70.0%                 \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}     & \\text{data} & \\text{VOC 2010 mAP} \\T \\\\\\hline
  \\text{FRCN,VGG16} & 12           & 66.1%                \\\\\\hline
  \\text{FRCN,VGG16} & 07++12       & 68.8%                \\\\\\hline
\\end{array}
$$

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}     & \\text{data} & \\text{VOC 2012 mAP}  \\T \\\\\\hline
  \\text{FRCN,VGG16} & 12           & 65.7%                 \\\\\\hline
  \\text{FRCN,VGG16} & 07++12       & 68.4%                 \\\\\\hline
\\end{array}
$$

- paper: [arXiv:1504.08083](http://arxiv.org/abs/1504.08083)
- slides: [caffe-cvpr15-detection](http://tutorial.caffe.berkeleyvision.org/caffe-cvpr15-detection.pdf)
- code: [https://github.com/rbgirshick/fast-rcnn](https://github.com/rbgirshick/fast-rcnn)

**Object detection via a multi-region & semantic segmentation-aware CNN model**

*(Submitted on 7 May 2015 (v1), last revised 9 Jun 2015 (this version, v2))*

- paper: [arXiv:1505.01749](http://arxiv.org/abs/1505.01749)

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**

*(Submitted on 4 Jun 2015)*

Detection results on PASCAL VOC 2007 test set:
$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}           & text{proposals} & \\text{data} & \\text{mAP} & \\text{time}  \\T \\\\\\hline
  \\text{RPN+VGG,unshared} & 300             & 07           & 68.5%       & 342ms             \\\\\\hline
  \\text{RPN+VGG,shared}   & 300             & 07           & 69.9%       & 196ms             \\\\\\hline
  \\text{RPN+VGG,shared}   & 300             & 07+12        & 73.2%       & 196ms             \\\\\\hline
\\end{array}
$$

Detection results on PASCAL VOC 2012 test set:
$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{method}           & text{proposals} & \\text{data} & \\text{mAP} & \\text{time}  \\T \\\\\\hline
  \\text{RPN+VGG,shared}   & 300             & 12           & 67.0%       & 196ms             \\\\\\hline
  \\text{RPN+VGG,shared}   & 300             & 07++12       & 70.4%       & 196ms             \\\\\\hline
\\end{array}
$$

- paper: [arXiv:1506.01497](http://arxiv.org/abs/1506.01497)
- code: [https://github.com/ShaoqingRen/caffe](https://github.com/ShaoqingRen/caffe)

**You Only Look Once: Unified, Real-Time Object Detection**

*(Submitted on 8 Jun 2015 (v1), last revised 11 Jun 2015 (this version, v3))*

- paper: [arXiv:1506.02640](http://arxiv.org/abs/1506.02640)
- code: [http://pjreddie.com/darknet/yolo/](http://pjreddie.com/darknet/yolo/)

**End-to-end people detection in crowded scenes**

*(Submitted on 16 Jun 2015 (v1), last revised 8 Jul 2015 (this version, v3))*

<img src="/assets/object-detection-materials/end_to_end_people_detection_in_crowded_scenes.jpg"/>

- paper: [arXiv:1506.04878](http://arxiv.org/abs/1506.04878)
- code: [https://github.com/Russell91/reinspect](https://github.com/Russell91/reinspect)
- ipn: [http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb](http://nbviewer.ipython.org/github/Russell91/ReInspect/blob/master/evaluation_reinspect.ipynb)

**R-CNN minus R**

*(Submitted on 23 Jun 2015)*

- paper: [arXiv:1506.06981](http://arxiv.org/abs/1506.06981)
