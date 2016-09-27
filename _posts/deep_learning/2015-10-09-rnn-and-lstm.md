---
layout: post
category: deep_learning
title: RNN and LSTM
date: 2015-10-09
---

# Types of RNN

1) Plain Tanh Recurrent Nerual Networks

2) Gated Recurrent Neural Networks (GRU)

3) Long Short-Term Memory (LSTM)

# Tutorials

**The Unreasonable Effectiveness of Recurrent Neural Networks**

- blog: [http://karpathy.github.io/2015/05/21/rnn-effectiveness/](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

**Understanding LSTM Networks**

- blog: [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- blog(zh): [http://www.jianshu.com/p/9dc9f41f0b29](http://www.jianshu.com/p/9dc9f41f0b29)

**A Beginner’s Guide to Recurrent Networks and LSTMs**

[http://deeplearning4j.org/lstm.html](http://deeplearning4j.org/lstm.html)

**A Deep Dive into Recurrent Neural Nets**

[http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/](http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/)

**A tutorial on training recurrent neural networks, covering BPPT, RTRL, EKF and the "echo state network" approach**

- paper: [http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf](http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf)
- slides: [http://deeplearning.cs.cmu.edu/notes/shaoweiwang.pdf](http://deeplearning.cs.cmu.edu/notes/shaoweiwang.pdf)

**Long Short-Term Memory: Tutorial on LSTM Recurrent Networks**

[http://people.idsia.ch/~juergen/lstm/index.htm](http://people.idsia.ch/~juergen/lstm/index.htm)

**LSTM implementation explained**

[http://apaszke.github.io/lstm-explained.html](http://apaszke.github.io/lstm-explained.html)

**Recurrent Neural Networks Tutorial**

- Part 1(Introduction to RNNs): [http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
- Part 2(Implementing a RNN using Python and Theano): [http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
- Part 3(Understanding the Backpropagation Through Time (BPTT) algorithm): [http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
- Part 4(Implementing a GRU/LSTM RNN): [http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

**Recurrent Neural Networks in DL4J**

[http://deeplearning4j.org/usingrnns.html](http://deeplearning4j.org/usingrnns.html)

**Learning RNN Hierarchies**

![](https://cloud.githubusercontent.com/assets/8753078/11612806/46e59834-9c2e-11e5-8309-7a93aa72383c.png)

- github: [https://github.com/pranv/lrh/blob/master/about.md](https://github.com/pranv/lrh/blob/master/about.md)

**Element-Research Torch RNN Tutorial for recurrent neural nets : let's predict time series with a laptop GPU**

- blog: [https://christopher5106.github.io/deep/learning/2016/07/14/element-research-torch-rnn-tutorial.html](https://christopher5106.github.io/deep/learning/2016/07/14/element-research-torch-rnn-tutorial.html)

**RNNs in Tensorflow, a Practical Guide and Undocumented Features**

- blog: [http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/](http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/)

**Learning about LSTMs using Torch**

- blog: [http://kbullaughey.github.io/lstm-play/](http://kbullaughey.github.io/lstm-play/)
- github: [https://github.com/kbullaughey/lstm-play](https://github.com/kbullaughey/lstm-play)

# Train RNN

**On the difficulty of training Recurrent Neural Networks**

- author: Razvan Pascanu, Tomas Mikolov, Yoshua Bengio
- arxiv: [http://arxiv.org/abs/1211.5063](http://arxiv.org/abs/1211.5063)
- video talks: [http://techtalks.tv/talks/on-the-difficulty-of-training-recurrent-neural-networks/58134/](http://techtalks.tv/talks/on-the-difficulty-of-training-recurrent-neural-networks/58134/)

**A Simple Way to Initialize Recurrent Networks of Rectified Linear Units**

- arxiv: [http://arxiv.org/abs/1504.00941](http://arxiv.org/abs/1504.00941)
- gitxiv: [http://gitxiv.com/posts/7j5JXvP3kn5Jf8Waj/irnn-experiment-with-pixel-by-pixel-sequential-mnist](http://gitxiv.com/posts/7j5JXvP3kn5Jf8Waj/irnn-experiment-with-pixel-by-pixel-sequential-mnist)
- github: [https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py)
- github: [https://gist.github.com/GabrielPereyra/353499f2e6e407883b32](https://gist.github.com/GabrielPereyra/353499f2e6e407883b32)
- blog("Implementing Recurrent Neural Net using chainer!"): [http://t-satoshi.blogspot.jp/2015/06/implementing-recurrent-neural-net-using.html](http://t-satoshi.blogspot.jp/2015/06/implementing-recurrent-neural-net-using.html)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/31rinf/150400941_a_simple_way_to_initialize_recurrent/](https://www.reddit.com/r/MachineLearning/comments/31rinf/150400941_a_simple_way_to_initialize_recurrent/)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/32tgvw/has_anyone_been_able_to_reproduce_the_results_in/](https://www.reddit.com/r/MachineLearning/comments/32tgvw/has_anyone_been_able_to_reproduce_the_results_in/)

**Sequence Level Training with Recurrent Neural Networks**

- intro: ICLR 2016
- arxiv: [http://arxiv.org/abs/1511.06732](http://arxiv.org/abs/1511.06732)
- github: [https://github.com/facebookresearch/MIXER](https://github.com/facebookresearch/MIXER)
- notes: [https://www.evernote.com/shard/s189/sh/ada01a82-70a9-48d4-985c-20492ab91e84/8da92be19e704996dc2b929473abed46](https://www.evernote.com/shard/s189/sh/ada01a82-70a9-48d4-985c-20492ab91e84/8da92be19e704996dc2b929473abed46)

**Training Recurrent Neural Networks (PhD thesis)**

- atuhor: Ilya Sutskever
- thesis: [https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf](https://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf)

**Deep learning for control using augmented Hessian-free optimization**

- blog: [https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/](https://studywolf.wordpress.com/2016/04/04/deep-learning-for-control-using-augmented-hessian-free-optimization/)
- github: [https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py](https://github.com/studywolf/blog/blob/master/train_AHF/train_hf.py)

- - -

**Hierarchical Conflict Propagation: Sequence Learning in a Recurrent Deep Neural Network**

- arxiv: [http://arxiv.org/abs/1602.08118](http://arxiv.org/abs/1602.08118)

**Recurrent Batch Normalization**

- arxiv: [http://arxiv.org/abs/1603.09025](http://arxiv.org/abs/1603.09025)
- github: [https://github.com/iassael/torch-bnlstm](https://github.com/iassael/torch-bnlstm)
- github: [https://github.com/cooijmanstim/recurrent-batch-normalization](https://github.com/cooijmanstim/recurrent-batch-normalization)
- github("LSTM with Batch Normalization"): [https://github.com/fchollet/keras/pull/2183](https://github.com/fchollet/keras/pull/2183)
- notes: [http://www.shortscience.org/paper?bibtexKey=journals/corr/CooijmansBLC16](http://www.shortscience.org/paper?bibtexKey=journals/corr/CooijmansBLC16)

**Batch normalized LSTM for Tensorflow**

- blog: [http://olavnymoen.com/2016/07/07/rnn-batch-normalization](http://olavnymoen.com/2016/07/07/rnn-batch-normalization)
- github: [https://github.com/OlavHN/bnlstm](https://github.com/OlavHN/bnlstm)

**Optimizing Performance of Recurrent Neural Networks on GPUs**

- arxiv: [http://arxiv.org/abs/1604.01946](http://arxiv.org/abs/1604.01946)
- github: [https://github.com/parallel-forall/code-samples/blob/master/posts/rnn/LSTM.cu](https://github.com/parallel-forall/code-samples/blob/master/posts/rnn/LSTM.cu)

**Path-Normalized Optimization of Recurrent Neural Networks with ReLU Activations**

- arxiv: [http://arxiv.org/abs/1605.07154](http://arxiv.org/abs/1605.07154)

**Explaining and illustrating orthogonal initialization for recurrent neural networks**

- blog: [http://smerity.com/articles/2016/orthogonal_init.html](http://smerity.com/articles/2016/orthogonal_init.html)

# Learn To Execute Programs

**Learning to Execute**

- arxiv: [http://arxiv.org/abs/1410.4615](http://arxiv.org/abs/1410.4615)
- github: [https://github.com/wojciechz/learning_to_execute](https://github.com/wojciechz/learning_to_execute)
- github(Tensorflow): [https://github.com/raindeer/seq2seq_experiments](https://github.com/raindeer/seq2seq_experiments)

**Neural Programmer-Interpreters**

<img src="/assets/dl-materials/rnn_lstm/NPI/add.gif" />

<img src="/assets/dl-materials/rnn_lstm/NPI/cars.gif" />

<img src="/assets/dl-materials/rnn_lstm/NPI/sort_full.gif" />

- intro:  Google DeepMind. ICLR 2016 Best Paper
- arxiv: [http://arxiv.org/abs/1511.06279](http://arxiv.org/abs/1511.06279)
- project page: [http://www-personal.umic (Google DeepMind. ICLR 2016 Best Paper)h.edu/~reedscot/iclr_project.html](http://www-personal.umich.edu/~reedscot/iclr_project.html)
- github: [https://github.com/mokemokechicken/keras_npi](https://github.com/mokemokechicken/keras_npi)

**A Programmer-Interpreter Neural Network Architecture for Prefrontal Cognitive Control**

- paper: [https://www.researchgate.net/publication/273912337_A_ProgrammerInterpreter_Neural_Network_Architecture_for_Prefrontal_Cognitive_Control](https://www.researchgate.net/publication/273912337_A_ProgrammerInterpreter_Neural_Network_Architecture_for_Prefrontal_Cognitive_Control)

**Convolutional RNN: an Enhanced Model for Extracting Features from Sequential Data**

- arxiv: [http://arxiv.org/abs/1602.05875](http://arxiv.org/abs/1602.05875)

**Neural Random-Access Machines**

- arxiv: [http://arxiv.org/abs/1511.06392](http://arxiv.org/abs/1511.06392)

# Attention Models

**Recurrent Models of Visual Attention**

- intro: Google DeepMind. NIPS 2014
- arxiv: [http://arxiv.org/abs/1406.6247](http://arxiv.org/abs/1406.6247)
- data: [https://github.com/deepmind/mnist-cluttered](https://github.com/deepmind/mnist-cluttered)
- github: [https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)

**Recurrent Model of Visual Attention**

- intro: Google DeepMind
- paper: [http://arxiv.org/abs/1406.6247](http://arxiv.org/abs/1406.6247)
- gitxiv: [http://gitxiv.com/posts/ZEobCXSh23DE8a8mo/recurrent-models-of-visual-attention](http://gitxiv.com/posts/ZEobCXSh23DE8a8mo/recurrent-models-of-visual-attention)
- blog: [http://torch.ch/blog/2015/09/21/rmva.html](http://torch.ch/blog/2015/09/21/rmva.html)
- github: [https://github.com/Element-Research/rnn/blob/master/scripts/evaluate-rva.lua](https://github.com/Element-Research/rnn/blob/master/scripts/evaluate-rva.lua)

**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**

- arxiv: [http://arxiv.org/abs/1502.03044](http://arxiv.org/abs/1502.03044)
- github: [https://github.com/kelvinxu/arctic-captions](https://github.com/kelvinxu/arctic-captions)

**A Neural Attention Model for Abstractive Sentence Summarization**

- intro: EMNLP 2015. Facebook AI Research
- arxiv: [http://arxiv.org/abs/1509.00685](http://arxiv.org/abs/1509.00685)
- github: [https://github.com/facebook/NAMAS](https://github.com/facebook/NAMAS)

**Effective Approaches to Attention-based Neural Machine Translation**

- intro: EMNLP 2015
- paper: [http://nlp.stanford.edu/pubs/emnlp15_attn.pdf](http://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
- project: [http://nlp.stanford.edu/projects/nmt/](http://nlp.stanford.edu/projects/nmt/)
- github: [https://github.com/lmthang/nmt.matlab](https://github.com/lmthang/nmt.matlab)

**Generating Images from Captions with Attention**

- arxiv: [http://arxiv.org/abs/1511.02793](http://arxiv.org/abs/1511.02793)
- github: [https://github.com/emansim/text2image](https://github.com/emansim/text2image)
- demo: [http://www.cs.toronto.edu/~emansim/cap2im.html](http://www.cs.toronto.edu/~emansim/cap2im.html)

**Attention and Memory in Deep Learning and NLP**

- blog: [http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

**Survey on the attention based RNN model and its applications in computer vision**

- arxiv: [http://arxiv.org/abs/1601.06823](http://arxiv.org/abs/1601.06823)

# Papers

**Generating Sequences With Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1308.0850](http://arxiv.org/abs/1308.0850)
- github: [https://github.com/hardmaru/write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow)
- github: [https://github.com/szcom/rnnlib](https://github.com/szcom/rnnlib)
- blog: [http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/)

**A Clockwork RNN**

- arxiv: [https://arxiv.org/abs/1402.3511](https://arxiv.org/abs/1402.3511)
- github: [https://github.com/makistsantekidis/clockworkrnn](https://github.com/makistsantekidis/clockworkrnn)
- github: [https://github.com/zergylord/ClockworkRNN](https://github.com/zergylord/ClockworkRNN)

**Unsupervised Learning of Video Representations using LSTMs**

- intro: ICML 2015
- project page: [http://www.cs.toronto.edu/~nitish/unsupervised_video/](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
- arxiv: [http://arxiv.org/abs/1502.04681](http://arxiv.org/abs/1502.04681)
- code: [http://www.cs.toronto.edu/~nitish/unsupervised_video/unsup_video_lstm.tar.gz](http://www.cs.toronto.edu/~nitish/unsupervised_video/unsup_video_lstm.tar.gz)
- github: [https://github.com/emansim/unsupervised-videos](https://github.com/emansim/unsupervised-videos)

**An Empirical Exploration of Recurrent Network Architectures**

- paper: [http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

**Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks**

- intro: ACL 2015. Tree RNNs aka Recursive Neural Networks
- arxiv: [https://arxiv.org/abs/1503.00075](https://arxiv.org/abs/1503.00075)
- slides: [http://lit.eecs.umich.edu/wp-content/uploads/2015/10/tree-lstms.pptx](http://lit.eecs.umich.edu/wp-content/uploads/2015/10/tree-lstms.pptx)
- gitxiv: [http://www.gitxiv.com/posts/esrArT2iLmSfNRrto/tree-structured-long-short-term-memory-networks](http://www.gitxiv.com/posts/esrArT2iLmSfNRrto/tree-structured-long-short-term-memory-networks)
- github: [https://github.com/stanfordnlp/treelstm](https://github.com/stanfordnlp/treelstm)
- github: [https://github.com/ofirnachum/tree_rnn](https://github.com/ofirnachum/tree_rnn)

**LSTM: A Search Space Odyssey**

- arxiv: [http://arxiv.org/abs/1503.04069](http://arxiv.org/abs/1503.04069)
- notes: [https://www.evernote.com/shard/s189/sh/48da42c5-8106-4f0d-b835-c203466bfac4/50d7a3c9a961aefd937fae3eebc6f540](https://www.evernote.com/shard/s189/sh/48da42c5-8106-4f0d-b835-c203466bfac4/50d7a3c9a961aefd937fae3eebc6f540)
- blog("Dissecting the LSTM"): [https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.crg8pztop](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.crg8pztop)
- github: [https://github.com/jimfleming/lstm_search](https://github.com/jimfleming/lstm_search)

**Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets**

- arxiv: [http://arxiv.org/abs/1503.01007](http://arxiv.org/abs/1503.01007)
- github: [https://github.com/facebook/Stack-RNN](https://github.com/facebook/Stack-RNN)

**A Critical Review of Recurrent Neural Networks for Sequence Learning**

- arxiv: [http://arxiv.org/abs/1506.00019](http://arxiv.org/abs/1506.00019)
- review: [http://blog.terminal.com/a-thorough-and-readable-review-on-rnns/](http://blog.terminal.com/a-thorough-and-readable-review-on-rnns/)

**Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks**

- intro: Winner of MSCOCO image captioning challenge, 2015
- arxiv: [http://arxiv.org/abs/1506.03099](http://arxiv.org/abs/1506.03099)

**Visualizing and Understanding Recurrent Networks**

- intro: ICLR 2016. Andrej Karpathy, Justin Johnson, Fei-Fei Li
- arxiv: [http://arxiv.org/abs/1506.02078](http://arxiv.org/abs/1506.02078)
- slides: [http://www.robots.ox.ac.uk/~seminars/seminars/Extra/2015_07_06_AndrejKarpathy.pdf](http://www.robots.ox.ac.uk/~seminars/seminars/Extra/2015_07_06_AndrejKarpathy.pdf)
- github: [https://github.com/karpathy/char-rnn](https://github.com/karpathy/char-rnn)

**Grid Long Short-Term Memory**

- arxiv: [http://arxiv.org/abs/1507.01526](http://arxiv.org/abs/1507.01526)
- github(Torch7): [https://github.com/coreylynch/grid-lstm/](https://github.com/coreylynch/grid-lstm/)

**Depth-Gated LSTM**

- arxiv: [http://arxiv.org/abs/1508.03790](http://arxiv.org/abs/1508.03790)
- github: [GitHub(dglstm.h+dglstm.cc)](https://github.com/kaishengyao/cnn/tree/master/cnn)

**Deep Knowledge Tracing**

- paper: [https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf](https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- github: [https://github.com/chrispiech/DeepKnowledgeTracing](https://github.com/chrispiech/DeepKnowledgeTracing)

**Top-down Tree Long Short-Term Memory Networks**

- arxiv: [http://arxiv.org/abs/1511.00060](http://arxiv.org/abs/1511.00060)
- github: [https://github.com/XingxingZhang/td-treelstm](https://github.com/XingxingZhang/td-treelstm)

**Alternative structures for character-level RNNs**

- intro: INRIA & Facebook AI Research. ICLR 2016
- arxiv: [http://arxiv.org/abs/1511.06303](http://arxiv.org/abs/1511.06303)
- github: [https://github.com/facebook/Conditional-character-based-RNN](https://github.com/facebook/Conditional-character-based-RNN)

**Long Short-Term Memory-Networks for Machine Reading**

- arxiv: [http://arxiv.org/abs/1601.06733](http://arxiv.org/abs/1601.06733)
- github: [https://github.com/cheng6076/SNLI-attention](https://github.com/cheng6076/SNLI-attention)

**Lipreading with Long Short-Term Memory**

- arxiv: [http://arxiv.org/abs/1601.08188](http://arxiv.org/abs/1601.08188)

**Associative Long Short-Term Memory**

- arxiv: [http://arxiv.org/abs/1602.03032](http://arxiv.org/abs/1602.03032)
- github: [https://github.com/mohammadpz/Associative_LSTM](https://github.com/mohammadpz/Associative_LSTM)

**Representation of linguistic form and function in recurrent neural networks**

- arxiv: [http://arxiv.org/abs/1602.08952](http://arxiv.org/abs/1602.08952)

**Architectural Complexity Measures of Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1602.08210](http://arxiv.org/abs/1602.08210)

**Easy-First Dependency Parsing with Hierarchical Tree LSTMs**

- arxiv: [http://arxiv.org/abs/1603.00375](http://arxiv.org/abs/1603.00375)

**Training Input-Output Recurrent Neural Networks through Spectral Methods**

- arxiv: [http://arxiv.org/abs/1603.00954](http://arxiv.org/abs/1603.00954)

**Neural networks with differentiable structure**

- arxiv: [http://arxiv.org/abs/1606.06216](http://arxiv.org/abs/1606.06216)
- github: [https://github.com/ThomasMiconi/DiffRNN](https://github.com/ThomasMiconi/DiffRNN)

**What You Get Is What You See: A Visual Markup Decompiler**

![](https://camo.githubusercontent.com/d5c6c528cdb25b504b1de298bc34d7109de06aea/687474703a2f2f6c73746d2e736561732e686172766172642e6564752f6c617465782f6e6574776f726b2e706e67)

- project page: [http://lstm.seas.harvard.edu/latex/](http://lstm.seas.harvard.edu/latex/)
- arxiv: [http://arxiv.org/abs/1609.04938](http://arxiv.org/abs/1609.04938)
- github: [https://github.com/harvardnlp/im2markup](https://github.com/harvardnlp/im2markup)

## LSTMVis

**Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks**

![](https://raw.githubusercontent.com/HendrikStrobelt/LSTMVis/master/docs/img/teaser_V2_small.png)

- homepage: [http://lstm.seas.harvard.edu/](http://lstm.seas.harvard.edu/)
- demo: [http://lstm.seas.harvard.edu/client/index.html](http://lstm.seas.harvard.edu/client/index.html)
- arxiv: [https://arxiv.org/abs/1606.07461](https://arxiv.org/abs/1606.07461)
- github: [https://github.com/HendrikStrobelt/LSTMVis](https://github.com/HendrikStrobelt/LSTMVis)

**Recurrent Memory Array Structures**

- arxiv: [https://arxiv.org/abs/1607.03085](https://arxiv.org/abs/1607.03085)
- github: [https://github.com/krocki/ArrayLSTM](https://github.com/krocki/ArrayLSTM)

**Recurrent Highway Networks**

- author: Julian Georg Zilly, Rupesh Kumar Srivastava, Jan Koutník, Jürgen Schmidhuber
- arxiv: [http://arxiv.org/abs/1607.03474](http://arxiv.org/abs/1607.03474)

## DeepSoft

**DeepSoft: A vision for a deep model of software**

- arxiv: [http://arxiv.org/abs/1608.00092](http://arxiv.org/abs/1608.00092)

**Recurrent Neural Networks With Limited Numerical Precision**

- arxiv: [http://arxiv.org/abs/1608.06902](http://arxiv.org/abs/1608.06902)

**Hierarchical Multiscale Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1609.01704](http://arxiv.org/abs/1609.01704)
- notes: [https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/hm-rnn.md](https://github.com/dennybritz/deeplearning-papernotes/blob/master/notes/hm-rnn.md)
- notes: [https://medium.com/@jimfleming/notes-on-hierarchical-multiscale-recurrent-neural-networks-7362532f3b64#.pag4kund0](https://medium.com/@jimfleming/notes-on-hierarchical-multiscale-recurrent-neural-networks-7362532f3b64#.pag4kund0)

# Projects

**NeuralTalk (Deprecated): a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences**

- github: [https://github.com/karpathy/neuraltalk](https://github.com/karpathy/neuraltalk)

**NeuralTalk2: Efficient Image Captioning code in Torch, runs on GPU**

- github: [https://github.com/karpathy/neuraltalk2](https://github.com/karpathy/neuraltalk2)

**char-rnn in Blocks**

- github: [https://github.com/johnarevalo/blocks-char-rnn](https://github.com/johnarevalo/blocks-char-rnn)

**Project: pycaffe-recurrent**

- code: [https://github.com/kuprel/pycaffe-recurrent/](https://github.com/kuprel/pycaffe-recurrent/)

**Using neural networks for password cracking**

- blog: [https://0day.work/using-neural-networks-for-password-cracking/](https://0day.work/using-neural-networks-for-password-cracking/)
- github: [https://github.com/gehaxelt/RNN-Passwords](https://github.com/gehaxelt/RNN-Passwords)

**torch-rnn: Efficient, reusable RNNs and LSTMs for torch**

- github: [https://github.com/jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn)

**Deploying a model trained with GPU in Torch into JavaScript, for everyone to use**

- blog: [http://testuggine.ninja/blog/torch-conversion](http://testuggine.ninja/blog/torch-conversion)
- demo: [http://testuggine.ninja/DRUMPF-9000/](http://testuggine.ninja/DRUMPF-9000/)
- github: [https://github.com/Darktex/char-rnn](https://github.com/Darktex/char-rnn)

**LSTM implementation on Caffe**

- github: [https://github.com/junhyukoh/caffe-lstm](https://github.com/junhyukoh/caffe-lstm)

**JNN: Java Neural Network Library**

- intro: C2W model, LSTM-based Language Model, LSTM-based Part-Of-Speech-Tagger Model
- github: [https://github.com/wlin12/JNN](https://github.com/wlin12/JNN)

**LSTM-Autoencoder: Seq2Seq LSTM Autoencoder**

- github: [https://github.com/cheng6076/LSTM-Autoencoder](https://github.com/cheng6076/LSTM-Autoencoder)

**RNN Language Model Variations**

- intro: Standard LSTM, Gated Feedback LSTM, 1D-Grid LSTM
- github: [https://github.com/cheng6076/mlm](https://github.com/cheng6076/mlm)

**keras-extra: Extra Layers for Keras to connect CNN with RNN**

- github: [https://github.com/anayebi/keras-extra](https://github.com/anayebi/keras-extra)

**Dynamic Vanilla RNN, GRU, LSTM,2layer Stacked LSTM with Tensorflow Higher Order Ops**

- github: [https://github.com/KnHuq/Dynamic_RNN_Tensorflow](https://github.com/KnHuq/Dynamic_RNN_Tensorflow)

**PRNN: A fast implementation of recurrent neural network layers in CUDA**

- intro: Baidu Research
- blog: [https://svail.github.io/persistent_rnns/](https://svail.github.io/persistent_rnns/)
- github: [https://github.com/baidu-research/persistent-rnn](https://github.com/baidu-research/persistent-rnn)

**min-char-rnn: Minimal character-level language model with a Vanilla Recurrent Neural Network, in Python/numpy**

- github: [https://github.com/weixsong/min-char-rnn](https://github.com/weixsong/min-char-rnn)

**rnn: Recurrent Neural Network library for Torch7's nn**

- github: [https://github.com/Element-Research/rnn](https://github.com/Element-Research/rnn)

**word-rnn-tensorflow: Multi-layer Recurrent Neural Networks (LSTM, RNN) for word-level language models in Python using TensorFlow**

- github: [https://github.com/hunkim/word-rnn-tensorflow](https://github.com/hunkim/word-rnn-tensorflow)

**tf-char-rnn: Tensorflow implementation of char-rnn**

- github: [https://github.com/shagunsodhani/tf-char-rnn](https://github.com/shagunsodhani/tf-char-rnn)

**translit-rnn: Automatic transliteration with LSTM**

- blog: [http://yerevann.github.io/2016/09/09/automatic-transliteration-with-lstm/](http://yerevann.github.io/2016/09/09/automatic-transliteration-with-lstm/)
- github: [https://github.com/YerevaNN/translit-rnn](https://github.com/YerevaNN/translit-rnn)

# Blogs

**Survey on Attention-based Models Applied in NLP**

[http://yanran.li/peppypapers/2015/10/07/survey-attention-model-1.html](http://yanran.li/peppypapers/2015/10/07/survey-attention-model-1.html)

**Survey on Advanced Attention-based Models**

[http://yanran.li/peppypapers/2015/10/07/survey-attention-model-2.html](http://yanran.li/peppypapers/2015/10/07/survey-attention-model-2.html)

**Online Representation Learning in Recurrent Neural Language Models**

[http://www.marekrei.com/blog/online-representation-learning-in-recurrent-neural-language-models/](http://www.marekrei.com/blog/online-representation-learning-in-recurrent-neural-language-models/)

**Fun with Recurrent Neural Nets: One More Dive into CNTK and TensorFlow**

[http://esciencegroup.com/2016/03/04/fun-with-recurrent-neural-nets-one-more-dive-into-cntk-and-tensorflow/](http://esciencegroup.com/2016/03/04/fun-with-recurrent-neural-nets-one-more-dive-into-cntk-and-tensorflow/)

**Materials to understand LSTM**

[https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1#.4mt3bzoau](https://medium.com/@shiyan/materials-to-understand-lstm-34387d6454c1#.4mt3bzoau)

**Understanding LSTM and its diagrams (★★★★★)**

- blog: [https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714](https://medium.com/@shiyan/understanding-lstm-and-its-diagrams-37e2f46f1714)
- slides: [https://github.com/shi-yan/FreeWill/blob/master/Docs/Diagrams/lstm_diagram.pptx](https://github.com/shi-yan/FreeWill/blob/master/Docs/Diagrams/lstm_diagram.pptx)

**Persistent RNNs: 30 times faster RNN layers at small mini-batch sizes (Greg Diamos, Baidu Silicon Valley AI Lab)**
**Persistent RNNs: Stashing Recurrent Weights On-Chip**

- paper: [http://jmlr.org/proceedings/papers/v48/diamos16.pdf](http://jmlr.org/proceedings/papers/v48/diamos16.pdf)
- blog: [http://svail.github.io/persistent_rnns/](http://svail.github.io/persistent_rnns/)
- slides: [http://on-demand.gputechconf.com/gtc/2016/presentation/s6673-greg-diamos-persisten-rnns.pdf](http://on-demand.gputechconf.com/gtc/2016/presentation/s6673-greg-diamos-persisten-rnns.pdf)

**All of Recurrent Neural Networks**

[https://medium.com/@jianqiangma/all-about-recurrent-neural-networks-9e5ae2936f6e#.q4s02elqg](https://medium.com/@jianqiangma/all-about-recurrent-neural-networks-9e5ae2936f6e#.q4s02elqg)

**Rolling and Unrolling RNNs**

[https://shapeofdata.wordpress.com/2016/04/27/rolling-and-unrolling-rnns/](https://shapeofdata.wordpress.com/2016/04/27/rolling-and-unrolling-rnns/)

**Sequence prediction using recurrent neural networks(LSTM) with TensorFlow: LSTM regression using TensorFlow**

- blog: [http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html](http://mourafiq.com/2016/05/15/predicting-sequences-using-rnn-in-tensorflow.html)
- github: [https://github.com/mouradmourafiq/tensorflow-lstm-regression](https://github.com/mouradmourafiq/tensorflow-lstm-regression)

**LSTMs**

![](https://shapeofdata.files.wordpress.com/2016/06/lstm.png?w=640)

- blog: [https://shapeofdata.wordpress.com/2016/06/04/lstms/](https://shapeofdata.wordpress.com/2016/06/04/lstms/)

**Machines and Magic: Teaching Computers to Write Harry Potter**

- blog: [https://medium.com/@joycex99/machines-and-magic-teaching-computers-to-write-harry-potter-37839954f252#.4fxemal9t](https://medium.com/@joycex99/machines-and-magic-teaching-computers-to-write-harry-potter-37839954f252#.4fxemal9t)
- github: [https://github.com/joycex99/hp-word-model](https://github.com/joycex99/hp-word-model)

**Crash Course in Recurrent Neural Networks for Deep Learning**

[http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/](http://machinelearningmastery.com/crash-course-recurrent-neural-networks-deep-learning/)

**Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras**

[http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/](http://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)

**Recurrent Neural Networks in Tensorflow**

- part I: [http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html](http://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)
- part II: [http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html](http://r2rt.com/recurrent-neural-networks-in-tensorflow-ii.html)

**Written Memories: Understanding, Deriving and Extending the LSTM**

[http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html](http://r2rt.com/written-memories-understanding-deriving-and-extending-the-lstm.html)

**Attention and Augmented Recurrent Neural Networks**

- blog: [http://distill.pub/2016/augmented-rnns/](http://distill.pub/2016/augmented-rnns/)
- github: [https://github.com/distillpub/post--augmented-rnns](https://github.com/distillpub/post--augmented-rnns)

**Interpreting and Visualizing Neural Networks for Text Processing**

[https://civisanalytics.com/blog/data-science/2016/09/22/neural-network-visualization/](https://civisanalytics.com/blog/data-science/2016/09/22/neural-network-visualization/)

## Optimizing RNN (Baidu Silicon Valley AI Lab)

**Optimizing RNN performance**

- blog: [http://svail.github.io/rnn_perf/](http://svail.github.io/rnn_perf/)

**Optimizing RNNs with Differentiable Graphs**

- blog: [http://svail.github.io/diff_graphs/](http://svail.github.io/diff_graphs/)
- notes: [http://research.baidu.com/svail-tech-notes-optimizing-rnns-differentiable-graphs/](http://research.baidu.com/svail-tech-notes-optimizing-rnns-differentiable-graphs/)

# Resources

**Awesome Recurrent Neural Networks - A curated list of resources dedicated to RNN**

- homepage: [http://jiwonkim.org/awesome-rnn/](http://jiwonkim.org/awesome-rnn/)
- github: [https://github.com/kjw0612/awesome-rnn](https://github.com/kjw0612/awesome-rnn)

**Jürgen Schmidhuber's page on Recurrent Neural Networks**

[http://people.idsia.ch/~juergen/rnn.html](http://people.idsia.ch/~juergen/rnn.html)

# Reading and Questions

**Are there any Recurrent convolutional neural network network implementations out there ?**

- reddit: [https://www.reddit.com/r/MachineLearning/comments/4chu3y/are_there_any_recurrent_convolutional_neural/](https://www.reddit.com/r/MachineLearning/comments/4chu3y/are_there_any_recurrent_convolutional_neural/)