---
layout: post
category: programming_study
title: Python2 with UTF-8
date: 2016-11-11
---

How to make directories (mkdir) with ascii-unicode mixed name in Python2 on Windows?

Here is a solution:

```
# -*- coding:utf-8 -*-

import os
import codecs

sys.stdin = codecs.getwriter('utf-8')(sys.stdin)
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

path = u'F:\\Ali\\test中文'

os.mkdir(path)
```
