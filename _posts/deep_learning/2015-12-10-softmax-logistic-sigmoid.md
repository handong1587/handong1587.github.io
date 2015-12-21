---
layout: post
category: deep_learning
title: Softmax Vs Logistic Vs Sigmoid
date: 2015-12-10
---

# Softmax

$$
\sigma(x) = \frac{1} {1 + exp^{-x}}
$$

$$
\frac{d\sigma(x)} {dx} = (1 - \sigma(x)) * \sigma(x)
$$

# Logistic

# Sigmoid

$$
y_i = \frac{e^{\zeta_i}} {\sum_{j \in L} e^{\zeta_j}}
$$