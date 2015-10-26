---
layout: post
categories: programming_study
title: Glog Build Problems on Windows X86 and Visual Studio 2015
---

{{ page.title }}
================

<p class="meta">23 Sep 2015 - Beijing</p>

Error:

gflags.lib(gflags.obj) : error LNK2001: unresolved external symbol __imp__PathMatchSpecA@8

Resolve:

Add "shlwapi.lib" to "Project - Property - Linker - Input - Additional Dependencies".

Reference:

[https://groups.google.com/forum/#!topic/google-gflags/cM4DuGOS_GI](https://groups.google.com/forum/#!topic/google-gflags/cM4DuGOS_GI)
