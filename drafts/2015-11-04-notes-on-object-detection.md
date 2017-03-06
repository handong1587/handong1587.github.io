---
layout: post
category: deep_learning
title: Notes On Object Detection
date: 2015-11-04
---

Key points on object detection: (1) Good region proposal generation method; (2) Early rejection.




**Make py-faster-rcnn support cudnn-v5**

```
git clone https://github.com/rbgirshick/py-faster-rcnn
```

Merge to latest Caffe:

```
cd caffe-fast-rcnn
git remote add caffe https://github.com/BVLC/caffe.git
git fetch caffe
git merge -X theirs caffe/master
```

Then uncomment code line in include/caffe/layers/python_layer.hpp:

```
self_.attr("phase") = static_cast<int>(this->phase_);
```
