---
layout: post
category: deep_learning
title: Notes On Quantized Convolutional Neural Networks
date: 2016-01-07
---

Existing works:

(1) Speed-up convolutional layers: **Low-rank approximation** and **Tensor decomposition**

(2) Reduce storage consumption in fully-connected layers: **Parameter compression**

Drawback: hard to achieve significant acceleration and compression simultaneously for the whole network.