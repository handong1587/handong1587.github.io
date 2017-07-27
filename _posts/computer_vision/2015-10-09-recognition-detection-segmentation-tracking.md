---
layout: post
category: computer_vision
title: Recognition, Detection, Segmentation and Tracking
date: 2015-10-09
---

# Classification / Recognition

**Generalized Hierarchical Matching for Sub-category Aware Object Classification (VOC2012 classification task winner)**

slides: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/workshop/Towards_VOC2012_NUSPSL.pdf](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/workshop/Towards_VOC2012_NUSPSL.pdf)

## License Plate Recognition

- homepage: [http://www.openalpr.com/](http://www.openalpr.com/)
- github: [https://github.com/openalpr/openalpr](https://github.com/openalpr/openalpr)
- tech review: [http://arstechnica.com/business/2015/12/new-open-source-license-plate-reader-software-lets-you-make-your-own-hot-list/](http://arstechnica.com/business/2015/12/new-open-source-license-plate-reader-software-lets-you-make-your-own-hot-list/)

# Detection

**Contextualizing Object Detection and Classification**

- intro: CVPR 2010
- paper: [http://www.lv-nus.org/papers/2011/cvpr2010-context_final.pdf](http://www.lv-nus.org/papers/2011/cvpr2010-context_final.pdf)

**Diagnosing Error in Object Detectors**

- author: Derek Hoiem, Yodsawalai Chodpathumwan, and Qieyun Dai
- project page: [http://web.engr.illinois.edu/~dhoiem/projects/detectionAnalysis/](http://web.engr.illinois.edu/~dhoiem/projects/detectionAnalysis/)
- paper: [http://web.engr.illinois.edu/~dhoiem/publications/eccv2012_detanalysis_derek.pdf](http://web.engr.illinois.edu/~dhoiem/publications/eccv2012_detanalysis_derek.pdf)
- code+data: [http://web.engr.illinois.edu/~dhoiem/projects/counter.php?Down=detectionAnalysis/detectionAnalysis_eccv12_v2.zip&Save=detectionAnalysis_eccv12](http://web.engr.illinois.edu/~dhoiem/projects/counter.php?Down=detectionAnalysis/detectionAnalysis_eccv12_v2.zip&Save=detectionAnalysis_eccv12)
- slides: [http://web.engr.illinois.edu/~dhoiem/presentations/DetectionAnalysis_ECCV2012.pptx](http://web.engr.illinois.edu/~dhoiem/presentations/DetectionAnalysis_ECCV2012.pptx)

## DPM and DPM variants

**Object detection with discriminatively trained part based models (DPM)**

- paper: [https://www.cs.berkeley.edu/~rbg/papers/Object-Detection-with-Discriminatively-Trained-Part-Based-Models--Felzenszwalb-Girshick-McAllester-Ramanan.pdf](https://www.cs.berkeley.edu/~rbg/papers/Object-Detection-with-Discriminatively-Trained-Part-Based-Models--Felzenszwalb-Girshick-McAllester-Ramanan.pdf)
- project page: [http://www.cs.berkeley.edu/~rbg/latent/](http://www.cs.berkeley.edu/~rbg/latent/)
- FAQ: [http://people.cs.uchicago.edu/~rbg/latent/voc-release5-faq.html](http://people.cs.uchicago.edu/~rbg/latent/voc-release5-faq.html)
- github: [https://github.com/rbgirshick/voc-dpm](https://github.com/rbgirshick/voc-dpm)

**30hz object detection with dpm v5**

- paper: [http://web.engr.illinois.edu/~msadegh2/papers/DPM30Hz.pdf](http://web.engr.illinois.edu/~msadegh2/papers/DPM30Hz.pdf)
- project page: [http://vision.cs.illinois.edu/DPM30Hz/](http://vision.cs.illinois.edu/DPM30Hz/)
- slides: [http://web.engr.illinois.edu/~msadegh2/papers/DPM30Hz.pptx](http://web.engr.illinois.edu/~msadegh2/papers/DPM30Hz.pptx)

**The fastest deformable part model for object detection**

- paper: [http://www.cbsr.ia.ac.cn/users/jjyan/Fastest_DPM.pdf](http://www.cbsr.ia.ac.cn/users/jjyan/Fastest_DPM.pdf)
- Result on FDDB: [http://vis-www.cs.umass.edu/fddb/results.html](http://vis-www.cs.umass.edu/fddb/results.html)

**Fast, accurate detection of 100,000 object classes on a single machine**

- paper: [research.google.com/pubs/archive/41104.pdf](research.google.com/pubs/archive/41104.pdf)
- paper: [https://courses.cs.washington.edu/courses/cse590v/13au/40814.pdf](https://courses.cs.washington.edu/courses/cse590v/13au/40814.pdf)

**Deformable Part Models are Convolutional Neural Networks**

- arxiv: [http://arxiv.org/abs/1409.5403](http://arxiv.org/abs/1409.5403)
- github: [https://github.com/rbgirshick/DeepPyramid](https://github.com/rbgirshick/DeepPyramid)

**Tensor-based approach to accelerate deformable part models**

[https://arxiv.org/abs/1707.03268](https://arxiv.org/abs/1707.03268)

- - -

**Integrating Context and Occlusion for Car Detection by Hierarchical And-or Model**

![](http://www.stat.ucla.edu/~boli/projects/context_occlusion/img/demo.png)

- intro: ECCV 2014
- homepage: [http://www.stat.ucla.edu/~boli/projects/context_occlusion/context_occlusion.html](http://www.stat.ucla.edu/~boli/projects/context_occlusion/context_occlusion.html)
- paper: [http://www.stat.ucla.edu/~boli/projects/context_occlusion/publications/CarAOG_ECCV2014.pdf](http://www.stat.ucla.edu/~boli/projects/context_occlusion/publications/CarAOG_ECCV2014.pdf)
- code: [http://www.stat.ucla.edu/~boli/projects/context_occlusion/publications/RGM_Release1.zip](http://www.stat.ucla.edu/~boli/projects/context_occlusion/publications/RGM_Release1.zip)

**Learning And-Or Models to Represent Context and Occlusion for Car Detection and Viewpoint Estimation**

- homepage: [http://www.stat.ucla.edu/~boli/projects/context_occlusion/context_occlusion.html](http://www.stat.ucla.edu/~boli/projects/context_occlusion/context_occlusion.html)
- arxiv: [http://arxiv.org/abs/1501.07359](http://arxiv.org/abs/1501.07359)
- code: [http://www.stat.ucla.edu/~boli/projects/context_occlusion/publications/RGM_Release1.zip](http://www.stat.ucla.edu/~boli/projects/context_occlusion/publications/RGM_Release1.zip)

**Quickest Moving Object Detection**

- arxiv: [http://arxiv.org/abs/1605.07586](http://arxiv.org/abs/1605.07586)

**Automatic detection of moving objects in video surveillance**

- arxiv: [http://arxiv.org/abs/1608.03617](http://arxiv.org/abs/1608.03617)

**Real-time Webcam Barcode Detection with OpenCV and C++**

![](http://www.codepool.biz/wp-content/uploads/2016/05/dbr_opencv_cplusplus.png)

- blog: [http://www.codepool.biz/webcam-barcode-detection-opencv-cplusplus.html](http://www.codepool.biz/webcam-barcode-detection-opencv-cplusplus.html)
- github: [https://github.com/dynamsoftlabs/cplusplus-webcam-barcode-reader](https://github.com/dynamsoftlabs/cplusplus-webcam-barcode-reader)

**The Role of Context Selection in Object Detection**

- arxiv: [http://arxiv.org/abs/1609.02948](http://arxiv.org/abs/1609.02948)

## Detection in Video

**Expanding Object Detector’s HORIZON: Incremental Learning Framework for Object Detection in Videos**

![](https://www.disneyresearch.com/wp-content/uploads/Expanding-Object-Detector%E2%80%99s-HORIZON-Incremental-Learning-Framework-for-Object-Detection-in-Videos-Image.jpg)

- intro: CVPR 2015
- project page: [https://www.disneyresearch.com/publication/expanding-object-detectors-horizon/](https://www.disneyresearch.com/publication/expanding-object-detectors-horizon/)
- paper: [http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Kuznetsova_Expanding_Object_Detectors_2015_CVPR_paper.pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Kuznetsova_Expanding_Object_Detectors_2015_CVPR_paper.pdf)

## Face Detection

**Build a Face Detection App Using Node.js and OpenCV**

[http://www.sitepoint.com/face-detection-nodejs-opencv/](http://www.sitepoint.com/face-detection-nodejs-opencv/)

**FaceTracker: Real time deformable face tracking in C++ with OpenCV 2**

- github: [https://github.com/kylemcdonald/FaceTracker](https://github.com/kylemcdonald/FaceTracker)

**A Fast and Accurate Unconstrained Face Detector**

![](https://github.com/CitrusRokid/OpenNPD)

- homepage: [http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html](http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html)
- github: [https://github.com/CitrusRokid/OpenNPD](https://github.com/CitrusRokid/OpenNPD)

**libfacedetection: A binary library for face detection in images. You can use it free of charge with any purpose**

- github: [https://github.com/ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection)

**jQuery Face Detection Plugin: A jQuery plugin to detect faces on images, videos and canvases**

- website: [http://facedetection.jaysalvat.com/](http://facedetection.jaysalvat.com/)
- github: [https://github.com/jaysalvat/jquery.facedetection](https://github.com/jaysalvat/jquery.facedetection)

**Spoofing 2D Face Detection: Machines See People Who Aren't There**

- arxiv: [http://arxiv.org/abs/1608.02128](http://arxiv.org/abs/1608.02128)

**Fall-Detection: Human Fall Detection from CCTV camera feed**

![](https://camo.githubusercontent.com/472295e9f092c3a7224b90fe09bfe91c25a102c6/68747470733a2f2f73342e706f7374696d672e6f72672f3439743472656c6f642f67697068792e676966)

- github: [https://github.com/harishrithish7/Fall-Detection](https://github.com/harishrithish7/Fall-Detection)

## Edge detection

**Image-feature-detection-using-Phase-Stretch-Transform**

![](https://upload.wikimedia.org/wikipedia/commons/4/45/PST_edge_detection_on_barbara_image.tif)

- github: [https://github.com/JalaliLabUCLA/Image-feature-detection-using-Phase-Stretch-Transform](https://github.com/JalaliLabUCLA/Image-feature-detection-using-Phase-Stretch-Transform)
- wikipedia: [https://en.wikipedia.org/wiki/Phase_stretch_transform](https://en.wikipedia.org/wiki/Phase_stretch_transform)

## Object Proposals

**What makes for effective detection proposals?(PAMI 2015)**

- homepage: [https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/)
- arxiv: [http://arxiv.org/abs/1502.05082](http://arxiv.org/abs/1502.05082)
- github: [https://github.com/hosang/detection-proposals](https://github.com/hosang/detection-proposals)

**BING++: A Fast High Quality Object Proposal Generator at 100fps**

- arxiv: [http://arxiv.org/abs/1511.04511](http://arxiv.org/abs/1511.04511)

**Object Proposals**

- intro: This is a library/API which can be used to generate bounding box/region proposals 
using a large number of the existing object proposal approaches.
- github: [https://github.com/batra-mlp-lab/object-proposals](https://github.com/batra-mlp-lab/object-proposals)

**Segmentation Free Object Discovery in Video**

- arxiv: [http://arxiv.org/abs/1609.00221](http://arxiv.org/abs/1609.00221)

# Segmentation

**Graph Based Image Segmentation**

- project page: [http://cs.brown.edu/~pff/segment/](http://cs.brown.edu/~pff/segment/)

**Pixelwise Image Saliency by Aggregation Complementary Appearance Contrast Measures with Edge-Preserving Coherence**

![](http://vision.sysu.edu.cn/project/PISA/framework.png)

- project page: [http://vision.sysu.edu.cn/project/PISA/](http://vision.sysu.edu.cn/project/PISA/)
- paper: [http://vision.sysu.edu.cn/project/PISA/PISA_Final.pdf](http://vision.sysu.edu.cn/project/PISA/PISA_Final.pdf)
- github: [https://github.com/kezewang/pixelwiseImageSaliencyAggregation](https://github.com/kezewang/pixelwiseImageSaliencyAggregation)

**Joint Tracking and Segmentation of Multiple Targets**

- paper: [http://www.milanton.de/files/cvpr2015/cvpr2015-anton.pdf](http://www.milanton.de/files/cvpr2015/cvpr2015-anton.pdf)
- bitbuckt(Matlab): [https://bitbucket.org/amilan/segtracking](https://bitbucket.org/amilan/segtracking)

**Supervised Evaluation of Image Segmentation Methods**

- project page: [http://www.vision.ee.ethz.ch/~biwiproposals/seism/index.html](http://www.vision.ee.ethz.ch/~biwiproposals/seism/index.html)
- github: [https://github.com/jponttuset/seism](https://github.com/jponttuset/seism)

## Normalized Cut (N-cut)

## Graph Cut

## Grab Cut

**“GrabCut” — Interactive Foreground Extraction using Iterated Graph Cuts**

- paper: [http://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf](http://cvg.ethz.ch/teaching/cvl/2012/grabcut-siggraph04.pdf)

**OpenCV 3.1: Interactive Foreground Extraction using GrabCut Algorithm**

[http://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html#gsc.tab=0](http://docs.opencv.org/master/d8/d83/tutorial_py_grabcut.html#gsc.tab=0)

## Video Segmentation

**Bilateral Space Video Segmentation**

![](https://graphics.ethz.ch/~perazzif/bvs/files/bvs_teaser.jpg)

- intro: CVPR 2016
- project page: [https://graphics.ethz.ch/~perazzif/bvs/index.html](https://graphics.ethz.ch/~perazzif/bvs/index.html)
- github: [https://github.com/owang/BilateralVideoSegmentation](https://github.com/owang/BilateralVideoSegmentation)

# Tracking

**Online Object Tracking: A Benchmark**

- intro: CVPR 2013
- paper: [http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf](http://faculty.ucmerced.edu/mhyang/papers/cvpr13_benchmark.pdf)

**Object Tracking Benchmark**

- intro: PAMI 2015
- paper: [http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf](http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf)

**Visual Tracker Benchmark**

[http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html)

**MEEM: Robust Tracking via Multiple Experts using Entropy Minimization**

![](http://cs-people.bu.edu/jmzhang/MEEM/frontpage.png)

- intro: ECCV 2014
- project page(code+data): [http://cs-people.bu.edu/jmzhang/MEEM/MEEM.html](http://cs-people.bu.edu/jmzhang/MEEM/MEEM.html)
- paper: [http://cs-people.bu.edu/jmzhang/MEEM/MEEM-eccv-preprint.pdf](http://cs-people.bu.edu/jmzhang/MEEM/MEEM-eccv-preprint.pdf)
- code: [http://cs-people.bu.edu/jmzhang/MEEM/MEEM_v1_release.zip](http://cs-people.bu.edu/jmzhang/MEEM/MEEM_v1_release.zip)
- code: [http://cs-people.bu.edu/jmzhang/MEEM/MEEM_v1.1_release.zip](http://cs-people.bu.edu/jmzhang/MEEM/MEEM_v1.1_release.zip)

**Struck: Structured Output Tracking with Kernels**

- intro: ICCV 2011
- paper: [http://mftp.mmcheng.net/Papers/StruckPAMI.pdf](http://mftp.mmcheng.net/Papers/StruckPAMI.pdf)
- paper: [http://www.robots.ox.ac.uk/~tvg/publications/2015/struck-author.pdf](http://www.robots.ox.ac.uk/~tvg/publications/2015/struck-author.pdf)
- paper: [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7360205](http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=7360205)
- slides: [http://vision.stanford.edu/teaching/cs231b_spring1415/slides/struck_meng.pdf](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/struck_meng.pdf)
- github: [https://github.com/samhare/struck](https://github.com/samhare/struck)
- github: [https://github.com/gnebehay/STRUCK](https://github.com/gnebehay/STRUCK)

**High-Speed Tracking with Kernelized Correlation Filters**

- intro: TPAMI 2015
- project page: [http://www.robots.ox.ac.uk/~joao/circulant/index.html](http://www.robots.ox.ac.uk/~joao/circulant/index.html)
- paper: [http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)
- github: [https://github.com/foolwood/KCF](https://github.com/foolwood/KCF)
- github: [https://github.com/vojirt/kcf](https://github.com/vojirt/kcf)

**Learning to Track: Online Multi-Object Tracking by Decision Making**

![](http://cvgl.stanford.edu/projects/MDP_tracking/MDP.png)

- intro: ICCV 2015
- homepage: [http://cvgl.stanford.edu/projects/MDP_tracking/](http://cvgl.stanford.edu/projects/MDP_tracking/)
- paper: [http://cvgl.stanford.edu/papers/xiang_iccv15.pdf](http://cvgl.stanford.edu/papers/xiang_iccv15.pdf)
- slides: [https://yuxng.github.io/Xiang_ICCV15_12162015.pdf](https://yuxng.github.io/Xiang_ICCV15_12162015.pdf)
- github: [https://github.com/yuxng/MDP_Tracking](https://github.com/yuxng/MDP_Tracking)

**Joint Tracking and Segmentation of Multiple Targets**

- paper: [http://www.milanton.de/files/cvpr2015/cvpr2015-anton.pdf](http://www.milanton.de/files/cvpr2015/cvpr2015-anton.pdf)
- bitbuckt(Matlab): [https://bitbucket.org/amilan/segtracking](https://bitbucket.org/amilan/segtracking)

**Robust Visual Tracking Via Consistent Low-Rank Sparse Learning**

![](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_IJCV14/material/paper_framework.jpg)

- project page: [http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_IJCV14/Robust%20Visual%20Tracking%20Via%20Consistent%20Low-Rank%20Sparse.html](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_IJCV14/Robust%20Visual%20Tracking%20Via%20Consistent%20Low-Rank%20Sparse.html)
- paper: [http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/tianzhu%20zhang_files/Conference%20Papers/ECCV12_zhang_Low-Rank%20Sparse%20Learning.pdf](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/tianzhu%20zhang_files/Conference%20Papers/ECCV12_zhang_Low-Rank%20Sparse%20Learning.pdf)

**Staple: Complementary Learners for Real-Time Tracking**

![](http://www.robots.ox.ac.uk/~luca/stuff/pipeline_horizontal.png)

- intro: CVPR 2016
- arxiv: [http://arxiv.org/abs/1512.01355](http://arxiv.org/abs/1512.01355)
- homepage: [http://www.robots.ox.ac.uk/~luca/staple.html](http://www.robots.ox.ac.uk/~luca/staple.html)
- github: [https://github.com/bertinetto/staple](https://github.com/bertinetto/staple)

**Simple Online and Realtime Tracking**

- intro: ICIP 2016. SORT = Simple Online and Realtime Tracking
- intro: Simple, online, and realtime tracking of multiple objects in a video sequence
- keywords: Kalman Filter, Hungarian algorithm, 260 Hz
- arxiv: [http://arxiv.org/abs/1602.00763](http://arxiv.org/abs/1602.00763)
- github: [https://github.com/abewley/sort](https://github.com/abewley/sort)
- demo: [https://motchallenge.net/movies/ETH-Linthescher-SORT.mp4](https://motchallenge.net/movies/ETH-Linthescher-SORT.mp4)

**Visual Tracking via Reliable Memories**

- arxiv: [http://arxiv.org/abs/1602.01887](http://arxiv.org/abs/1602.01887)

**Tracking Completion**

- arxiv: [http://arxiv.org/abs/1608.08171](http://arxiv.org/abs/1608.08171)

**Real-Time Visual Tracking: Promoting the Robustness of Correlation Filter Learning**

- arxiv: [http://arxiv.org/abs/1608.08173](http://arxiv.org/abs/1608.08173)

## Projects

**Benchmark Results of Correlation Filters**

- intro: Collect and share results for correlation filter trackers.
- github: [https://github.com/HakaseH/CF_benchmark_results](https://github.com/HakaseH/CF_benchmark_results)

**JS-face-tracking-demo**

![](https://camo.githubusercontent.com/3cde9346fa47e7985235b22a9c032d149dcf7c6d/68747470733a2f2f692e696d6775722e636f6d2f6a36424f4b39662e676966)

- demo: [https://kdzwinel.github.io/JS-face-tracking-demo/](https://kdzwinel.github.io/JS-face-tracking-demo/)
- github: [https://github.com/kdzwinel/JS-face-tracking-demo](https://github.com/kdzwinel/JS-face-tracking-demo)

**VOTR: Visual Object Tracking Repository**

- intro: This repository aims at collecting state-of-the-art tracking algorithms that are freely available.
- github: [https://github.com/gnebehay/VOTR](https://github.com/gnebehay/VOTR)
