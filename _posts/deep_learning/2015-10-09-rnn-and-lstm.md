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

**A Beginner’s Guide to Recurrent Networks and LSTMs**

[http://deeplearning4j.org/lstm.html](http://deeplearning4j.org/lstm.html)

**A Deep Dive into Recurrent Neural Nets**

[http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/](http://nikhilbuduma.com/2015/01/11/a-deep-dive-into-recurrent-neural-networks/)

**Long Short-Term Memory: Tutorial on LSTM Recurrent Networks**

[http://people.idsia.ch/~juergen/lstm/index.htm](http://people.idsia.ch/~juergen/lstm/index.htm)

**LSTM implementation explained**

[http://apaszke.github.io/lstm-explained.html](http://apaszke.github.io/lstm-explained.html)

**Recurrent Neural Networks Tutorial**

- Part 1(Introduction to RNNs): [http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/)
- Part 2(Implementing a RNN using Python and Theano): [http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/)
- Part 3(Understanding the Backpropagation Through Time (BPTT) algorithm): [http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/](http://www.wildml.com/2015/10/recurrent-neural-networks-tutorial-part-3-backpropagation-through-time-and-vanishing-gradients/)
- Part 4(Implementing a GRU/LSTM RNN): [http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/](http://www.wildml.com/2015/10/recurrent-neural-network-tutorial-part-4-implementing-a-grulstm-rnn-with-python-and-theano/)

**Understanding LSTM Networks**

- blog: [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- ZH: [http://www.jianshu.com/p/9dc9f41f0b29](http://www.jianshu.com/p/9dc9f41f0b29)

**Recurrent Neural Networks in DL4J**

[http://deeplearning4j.org/usingrnns.html](http://deeplearning4j.org/usingrnns.html)

# Train RNN

**A Simple Way to Initialize Recurrent Networks of Rectified Linear Units**

- arxiv: [http://arxiv.org/abs/1504.00941](http://arxiv.org/abs/1504.00941)
- gitxiv: [http://gitxiv.com/posts/7j5JXvP3kn5Jf8Waj/irnn-experiment-with-pixel-by-pixel-sequential-mnist](http://gitxiv.com/posts/7j5JXvP3kn5Jf8Waj/irnn-experiment-with-pixel-by-pixel-sequential-mnist)
- github: [https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py](https://github.com/fchollet/keras/blob/master/examples/mnist_irnn.py)
- github: [https://gist.github.com/GabrielPereyra/353499f2e6e407883b32](https://gist.github.com/GabrielPereyra/353499f2e6e407883b32)
- blog("Implementing Recurrent Neural Net using chainer!"): [http://t-satoshi.blogspot.jp/2015/06/implementing-recurrent-neural-net-using.html](http://t-satoshi.blogspot.jp/2015/06/implementing-recurrent-neural-net-using.html)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/31rinf/150400941_a_simple_way_to_initialize_recurrent/](https://www.reddit.com/r/MachineLearning/comments/31rinf/150400941_a_simple_way_to_initialize_recurrent/)
- reddit: [https://www.reddit.com/r/MachineLearning/comments/32tgvw/has_anyone_been_able_to_reproduce_the_results_in/](https://www.reddit.com/r/MachineLearning/comments/32tgvw/has_anyone_been_able_to_reproduce_the_results_in/)

**Sequence Level Training with Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1511.06732](http://arxiv.org/abs/1511.06732)
- notes: [https://www.evernote.com/shard/s189/sh/ada01a82-70a9-48d4-985c-20492ab91e84/8da92be19e704996dc2b929473abed46](https://www.evernote.com/shard/s189/sh/ada01a82-70a9-48d4-985c-20492ab91e84/8da92be19e704996dc2b929473abed46)

# Papers

**Generating Sequences With Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1308.0850](http://arxiv.org/abs/1308.0850)
- github: [https://github.com/hardmaru/write-rnn-tensorflow](https://github.com/hardmaru/write-rnn-tensorflow)
- blog: [http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/](http://blog.otoro.net/2015/12/12/handwriting-generation-demo-in-tensorflow/)

**DRAW: A Recurrent Neural Network For Image Generation**

- arXiv: [http://arxiv.org/abs/1502.04623](http://arxiv.org/abs/1502.04623)
- github: [https://github.com/vivanov879/draw](https://github.com/vivanov879/draw)
- github(Theano): [https://github.com/jbornschein/draw](https://github.com/jbornschein/draw)
- github(Lasagne): [https://github.com/skaae/lasagne-draw](https://github.com/skaae/lasagne-draw)

**Unsupervised Learning of Video Representations using LSTMs(ICML2015)**

- project: [http://www.cs.toronto.edu/~nitish/unsupervised_video/](http://www.cs.toronto.edu/~nitish/unsupervised_video/)
- paper: [http://arxiv.org/abs/1502.04681](http://arxiv.org/abs/1502.04681)
- code: [http://www.cs.toronto.edu/~nitish/unsupervised_video/unsup_video_lstm.tar.gz](http://www.cs.toronto.edu/~nitish/unsupervised_video/unsup_video_lstm.tar.gz)
- github: [https://github.com/emansim/unsupervised-videos](https://github.com/emansim/unsupervised-videos)

**LSTM: A Search Space Odyssey**

- paper: [http://arxiv.org/abs/1503.04069](http://arxiv.org/abs/1503.04069)
- notes: [https://www.evernote.com/shard/s189/sh/48da42c5-8106-4f0d-b835-c203466bfac4/50d7a3c9a961aefd937fae3eebc6f540](https://www.evernote.com/shard/s189/sh/48da42c5-8106-4f0d-b835-c203466bfac4/50d7a3c9a961aefd937fae3eebc6f540)
- blog("Dissecting the LSTM"): [https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.crg8pztop](https://medium.com/jim-fleming/implementing-lstm-a-search-space-odyssey-7d50c3bacf93#.crg8pztop)
- github: [https://github.com/jimfleming/lstm_search](https://github.com/jimfleming/lstm_search)

**Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets**

- paper: [http://arxiv.org/abs/1503.01007](http://arxiv.org/abs/1503.01007)
- code: [https://github.com/facebook/Stack-RNN](https://github.com/facebook/Stack-RNN)

**A Critical Review of Recurrent Neural Networks for Sequence Learning**

- arXiv: [http://arxiv.org/abs/1506.00019](http://arxiv.org/abs/1506.00019)
- intro: "A rigorous & readable review on RNNs"  <br /> [http://blog.terminal.com/a-thorough-and-readable-review-on-rnns/](http://blog.terminal.com/a-thorough-and-readable-review-on-rnns/)

**Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks(Winner of MSCOCO image captioning challenge, 2015)**

- arXiv: [http://arxiv.org/abs/1506.03099](http://arxiv.org/abs/1506.03099)

**Visualizing and Understanding Recurrent Networks(Andrej Karpathy, Justin Johnson, Fei-Fei Li)**

- paper: [http://arxiv.org/abs/1506.02078](http://arxiv.org/abs/1506.02078)
- slides: [http://www.robots.ox.ac.uk/~seminars/seminars/Extra/2015_07_06_AndrejKarpathy.pdf](http://www.robots.ox.ac.uk/~seminars/seminars/Extra/2015_07_06_AndrejKarpathy.pdf)

**Grid Long Short-Term Memory**

- arxiv: [http://arxiv.org/abs/1507.01526](http://arxiv.org/abs/1507.01526)
- github(Torch7): [https://github.com/coreylynch/grid-lstm/](https://github.com/coreylynch/grid-lstm/)

**Depth-Gated LSTM**

- paper: [http://arxiv.org/abs/1508.03790](http://arxiv.org/abs/1508.03790)
- code: [GitHub(dglstm.h+dglstm.cc)](https://github.com/kaishengyao/cnn/tree/master/cnn)

**Deep Knowledge Tracing**

- paper: [https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf](https://web.stanford.edu/~cpiech/bio/papers/deepKnowledgeTracing.pdf)
- github: [https://github.com/chrispiech/DeepKnowledgeTracing](https://github.com/chrispiech/DeepKnowledgeTracing)

**Alternative structures for character-level RNNs(INRIA & Facebook AI Research)**

- arXiv: [http://arxiv.org/abs/1511.06303](http://arxiv.org/abs/1511.06303)
- github: [https://github.com/facebook/Conditional-character-based-RNN](https://github.com/facebook/Conditional-character-based-RNN)

**Pixel Recurrent Neural Networks (Google DeepMind)**

- arxiv: [http://arxiv.org/abs/1601.06759](http://arxiv.org/abs/1601.06759)
- notes(by Hugo Larochelle): [https://www.evernote.com/shard/s189/sh/fdf61a28-f4b6-491b-bef1-f3e148185b18/aba21367d1b3730d9334ed91d3250848](https://www.evernote.com/shard/s189/sh/fdf61a28-f4b6-491b-bef1-f3e148185b18/aba21367d1b3730d9334ed91d3250848)

**Long Short-Term Memory-Networks for Machine Reading**

- arxiv: [http://arxiv.org/abs/1601.06733](http://arxiv.org/abs/1601.06733)

**Lipreading with Long Short-Term Memory**

- arxiv: [http://arxiv.org/abs/1601.08188](http://arxiv.org/abs/1601.08188)

**Associative Long Short-Term Memory**

- arxiv: [http://arxiv.org/abs/1602.03032](http://arxiv.org/abs/1602.03032)

**Representation of linguistic form and function in recurrent neural networks**

- arxiv: [http://arxiv.org/abs/1602.08952](http://arxiv.org/abs/1602.08952)

**Architectural Complexity Measures of Recurrent Neural Networks**

- arxiv: [http://arxiv.org/abs/1602.08210](http://arxiv.org/abs/1602.08210)

**Easy-First Dependency Parsing with Hierarchical Tree LSTMs**

- arxiv: [http://arxiv.org/abs/1603.00375](http://arxiv.org/abs/1603.00375)

**Training Input-Output Recurrent Neural Networks through Spectral Methods**

- arxiv: [http://arxiv.org/abs/1603.00954](http://arxiv.org/abs/1603.00954)

# Learn To Execute Programs

**Learning to Execute**

- arXiv: [http://arxiv.org/abs/1410.4615](http://arxiv.org/abs/1410.4615)
- github: [https://github.com/wojciechz/learning_to_execute](https://github.com/wojciechz/learning_to_execute)

**Neural Programmer-Interpreters (Google DeepMind)**

<img src="/assets/dl-materials/rnn_lstm/NPI/add.gif" />

<img src="/assets/dl-materials/rnn_lstm/NPI/cars.gif" />

<img src="/assets/dl-materials/rnn_lstm/NPI/sort_full.gif" />

- arXiv: [http://arxiv.org/abs/1511.06279](http://arxiv.org/abs/1511.06279)
- project page: [http://www-personal.umich.edu/~reedscot/iclr_project.html](http://www-personal.umich.edu/~reedscot/iclr_project.html)

**A Programmer-Interpreter Neural Network Architecture for Prefrontal Cognitive Control**

- paper: [https://www.researchgate.net/publication/273912337_A_ProgrammerInterpreter_Neural_Network_Architecture_for_Prefrontal_Cognitive_Control](https://www.researchgate.net/publication/273912337_A_ProgrammerInterpreter_Neural_Network_Architecture_for_Prefrontal_Cognitive_Control)

**Convolutional RNN: an Enhanced Model for Extracting Features from Sequential Data**

- arxiv: [http://arxiv.org/abs/1602.05875](http://arxiv.org/abs/1602.05875)

# Attention Models

**Recurrent Models of Visual Attention** (Google DeepMind. NIPS2014)

- paper: [http://arxiv.org/abs/1406.6247](http://arxiv.org/abs/1406.6247)
- data: [https://github.com/deepmind/mnist-cluttered](https://github.com/deepmind/mnist-cluttered)
- code: [https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua](https://github.com/Element-Research/rnn/blob/master/examples/recurrent-visual-attention.lua)

**Recurrent Model of Visual Attention(Google DeepMind)**

- paper: [http://arxiv.org/abs/1406.6247](http://arxiv.org/abs/1406.6247)
- GitXiv: [http://gitxiv.com/posts/ZEobCXSh23DE8a8mo/recurrent-models-of-visual-attention](http://gitxiv.com/posts/ZEobCXSh23DE8a8mo/recurrent-models-of-visual-attention)
- blog: [http://torch.ch/blog/2015/09/21/rmva.html](http://torch.ch/blog/2015/09/21/rmva.html)
- code: [https://github.com/Element-Research/rnn/blob/master/scripts/evaluate-rva.lua](https://github.com/Element-Research/rnn/blob/master/scripts/evaluate-rva.lua)

**Show, Attend and Tell: Neural Image Caption Generation with Visual Attention**

- paper: [http://arxiv.org/abs/1502.03044](http://arxiv.org/abs/1502.03044)
- code: [https://github.com/kelvinxu/arctic-captions](https://github.com/kelvinxu/arctic-captions)

**A Neural Attention Model for Abstractive Sentence Summarization(EMNLP 2015. Facebook AI Research)**

- arXiv: [http://arxiv.org/abs/1509.00685](http://arxiv.org/abs/1509.00685)
- github: [https://github.com/facebook/NAMAS](https://github.com/facebook/NAMAS)

**Effective Approaches to Attention-based Neural Machine Translation(EMNLP2015)**

- paper: [http://nlp.stanford.edu/pubs/emnlp15_attn.pdf](http://nlp.stanford.edu/pubs/emnlp15_attn.pdf)
- project: [http://nlp.stanford.edu/projects/nmt/](http://nlp.stanford.edu/projects/nmt/)

**Generating Images from Captions with Attention**

- arxiv: [http://arxiv.org/abs/1511.02793](http://arxiv.org/abs/1511.02793)
- github: [https://github.com/emansim/text2image](https://github.com/emansim/text2image)
- demo: [http://www.cs.toronto.edu/~emansim/cap2im.html](http://www.cs.toronto.edu/~emansim/cap2im.html)

**Attention and Memory in Deep Learning and NLP**

- blog: [http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

**Survey on the attention based RNN model and its applications in computer vision**

- arxiv: [http://arxiv.org/abs/1601.06823](http://arxiv.org/abs/1601.06823)

# Train RNN

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

# Codes

**NeuralTalk (Deprecated): a Python+numpy project for learning Multimodal Recurrent Neural Networks that describe images with sentences**

- github: [https://github.com/karpathy/neuraltalk](https://github.com/karpathy/neuraltalk)

**NeuralTalk2: Efficient Image Captioning code in Torch, runs on GPU**

- github: 	[https://github.com/karpathy/neuraltalk2](https://github.com/karpathy/neuraltalk2)

**char-rnn in Blocks**

- github: [https://github.com/johnarevalo/blocks-char-rnn](https://github.com/johnarevalo/blocks-char-rnn)

**Project: pycaffe-recurrent**

- code: [https://github.com/kuprel/pycaffe-recurrent/](https://github.com/kuprel/pycaffe-recurrent/)

**Using neural networks for password cracking**

- blog: [https://0day.work/using-neural-networks-for-password-cracking/](https://0day.work/using-neural-networks-for-password-cracking/)
- github: [https://github.com/gehaxelt/RNN-Passwords](https://github.com/gehaxelt/RNN-Passwords)

**Recurrent neural networks for decoding CAPTCHAS**

- blog: [https://deepmlblog.wordpress.com/2016/01/12/recurrent-neural-networks-for-decoding-captchas/](https://deepmlblog.wordpress.com/2016/01/12/recurrent-neural-networks-for-decoding-captchas/)
- demo: [http://simplecaptcha.sourceforge.net/](http://simplecaptcha.sourceforge.net/)
- code: [http://sourceforge.net/projects/simplecaptcha/](http://sourceforge.net/projects/simplecaptcha/)

**torch-rnn: Efficient, reusable RNNs and LSTMs for torch**

- github: [https://github.com/jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn)

**Deploying a model trained with GPU in Torch into JavaScript, for everyone to use**

- blog: [http://testuggine.ninja/blog/torch-conversion](http://testuggine.ninja/blog/torch-conversion)
- demo: [http://testuggine.ninja/DRUMPF-9000/](http://testuggine.ninja/DRUMPF-9000/)
- github: [https://github.com/Darktex/char-rnn](https://github.com/Darktex/char-rnn)

# Blog

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

[http://svail.github.io/persistent_rnns/](http://svail.github.io/persistent_rnns/)

**All of Recurrent Neural Networks**

[https://medium.com/@jianqiangma/all-about-recurrent-neural-networks-9e5ae2936f6e#.q4s02elqg](https://medium.com/@jianqiangma/all-about-recurrent-neural-networks-9e5ae2936f6e#.q4s02elqg)

# Resources

**Awesome-rnn - A curated list of resources dedicated to RNN**

[http://jiwonkim.org/awesome-rnn/](http://jiwonkim.org/awesome-rnn/)

**Jürgen Schmidhuber's page on Recurrent Neural Networks**

[http://people.idsia.ch/~juergen/rnn.html](http://people.idsia.ch/~juergen/rnn.html)

# Reading and Questions

**Are there any Recurrent convolutional neural network network implementations out there ?**

- reddit: [https://www.reddit.com/r/MachineLearning/comments/4chu3y/are_there_any_recurrent_convolutional_neural/](https://www.reddit.com/r/MachineLearning/comments/4chu3y/are_there_any_recurrent_convolutional_neural/)