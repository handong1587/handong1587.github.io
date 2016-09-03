
# Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognititon

4.4 Model Combination for detection

They pre-train two networks in ImageNet, using the same structure but different random initializations.
These two mAPs are similar: 59.1% vs. 59.2%.
Given the two models, first use either model to score all candidate windows on the test image.
Then  perform NMS on the union of the two sets of candidate windows (with their scores). 
A more confident window given by one method can suppress those less confident given by the other method. 
After combination, the mAP is boosted to 60.9%. In 17 out of all 20
categories the combination performs better than either individual model. 
This indicates that the two models are complementary.

They further find that the complementarity is mainly because of the convolutional layers. 
They have tried to combine two randomly initialized fine-tuned results of the same convolutional model, and found no gain.

# Deep Residual Learning for Image Recognition

B. Object Detection Improvements

Single-model achieve mAP@.5 of 55.7% and an mAP@[.5, .95] of 34.9%.

In Faster R-CNN, the system is designed to learn region proposals and also object classifiers, 
so an ensemble can be used to boost both tasks. 
They use an ensemble for proposing regions, and the union set of proposals are processed 
by an ensemble of per-region classifiers.
The mAP is 59.0% and 37.4% on the test-dev set.
