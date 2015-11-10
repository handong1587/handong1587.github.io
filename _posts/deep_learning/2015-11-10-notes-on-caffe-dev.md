---
layout: post
categories: deep_learning
title: Notes On Caffe Development
---

{{ page.title }}
================

<p class="meta">10 Nov 2015 - Beijing</p>

# HDF5 vs. LMDB

HDF5:

Simple format to read/write.

The HDF5 files are always read entirely into memory, so can't have any HDF5 files exceed memory capacity. But can easily split data into several HDF5 files though (just put several paths to h5 files in text file). Caffe dev group suggests that divide original large h5 data into small ones so that each h5 data fits < 2 GB.

I/O performance won't be nearly as good as LMDB.

Other DataLayers do prefetching in a separate thread, HDF5Layer does not.

Can only store float32 and float64 data - no uint8 means image data will be huge.

LMDB:

Uses memory-mapped files, giving much better I/O performance. Works well with large dataset.

Example:

```shell
layer {
  type: "HDF5Data"
  top: "X" # same name as given in create_dataset!
  top: "y"
  hdf5_data_param {
    source: "train_h5_list.txt" # do not give the h5 files directly, but the list.
    batch_size: 32
  }
  include { phase:TRAIN }
}
```

# Create LMDB from float data

[http://stackoverflow.com/questions/31774953/test-labels-for-regression-caffe-float-not-allowed](http://stackoverflow.com/questions/31774953/test-labels-for-regression-caffe-float-not-allowed)

[https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045](https://github.com/BVLC/caffe/issues/1698#issuecomment-70211045)

# Reference

[http://vision.stanford.edu/teaching/cs231n/slides/caffe_tutorial.pdf](http://vision.stanford.edu/teaching/cs231n/slides/caffe_tutorial.pdf)

[http://deepdish.io/](http://deepdish.io/)

[https://github.com/BVLC/caffe/issues/1470](https://github.com/BVLC/caffe/issues/1470)
