---
layout: post
category: deep_learning
title: Notes On YOLO
date: 2015-12-14
---

# How YOLO organise training data?

**ground-truth:**

49 x (1 + 20 + 4) =>  <br />
  49 x (1 x obj_gt + 20 x classes_gt + 4 x box_gt)

**predict-data:**

49 x 20 + 49x(1x2) + 49x(4x2) =>  <br /> 
  49 x (20 x classes) + 49 x (2 x obj_confidence) + 49 x (2 x predict_boxes)
  
# Some questions..

(1) The multi-part loss function differ from the code implementation:

$$ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{S} \mathds{1}_{ij}^{obj} (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2
 + \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{S} \mathds{1}_{ij}^{obj} (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2
$$