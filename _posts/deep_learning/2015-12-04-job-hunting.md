---
layout: post
category: deep_learning
title: Deep Learning Job Hunting
date: 2015-12-04
---

**What are the toughest neural networks and deep learning interview questions?**

quora: [https://www.quora.com/What-are-the-toughest-neural-networks-and-deep-learning-interview-questions](https://www.quora.com/What-are-the-toughest-neural-networks-and-deep-learning-interview-questions)

1. What is an auto-encoder? Why do we "auto-encode"? Hint: it's really a misnomer.

2. What is a Boltzmann Machine? Why a Boltzmann Machine?

3. Why do we use sigmoid for an output function? Why tanh? Why not cosine? Why any function in particular?

4. Why are CNNs used primarily in imaging and not so much other tasks?

5. Explain backpropagation. Seriously. To the target audience described above.

6. Is it OK to connect from a Layer 4 output back to a Layer 2 input?

7. A data-scientist person recently put up a YouTube video explaining that the essential difference between 
a Neural Network and a Deep Learning network is that the former is trained from output back to input, 
while the latter is trained from input toward output. Do you agree? Explain.

8. Etc.

(by Christopher Cuong Nguyen, CEO & co-founder at Adatao (ah-'DAY-tao))

- - -

1. Can they derive the back-propagation and weights update?

2. Extend the above question to non-trivial layers such as convolutional layers, pooling layers, etc.

3. How to implement dropout?

4. Their intuition when and why some tricks such as max pooling, ReLU, maxout, etc. work. 
There are no right answers but it helps to understand their thoughts and research experience.

5. Can they abstract the forward, backward, update operations as matrix operations, to leverage BLAS and GPU?

(by Kenneth Tran, ML Scientist @ MSR)