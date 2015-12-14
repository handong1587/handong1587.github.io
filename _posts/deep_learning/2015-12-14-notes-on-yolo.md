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

