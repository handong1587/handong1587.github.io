---
layout: post
category: deep_learning
title: Notes On YOLO
date: 2015-12-14
---

# How YOLO organise training data?

ground-truth:

49 x (1 + 20 + 4) =>  <br />
  49 x (1 obj+gt + 20 classes_gt + 4 box_gt)

predict data:

49 x 20 + 49x(1x2) + 49x(4x2) =>  <br /> 
  49x(20 classes) + 49x(2 obj_confidence) + 49x(2 predict_boxes)
  
# Some questions..

(1) The multi-part loss function differ from the code implementation:

$$ \textlambda_coord \sum_{i=0}^{S^2} \sum_{j=0}^{S} \textbari_{ij}^{obj} (x_i - x_i)^2 + (y_i - y_i)^2$$