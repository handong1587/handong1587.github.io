---
layout: post
category: programming_study
title: Gflags Build Problems on Windows X86 and Visual Studio 2015
date: 2015-10-23
---

# Gflags Build Problems on Windows X86 and Visual Studio 2015

Error:

gflags.lib(gflags.obj) : error LNK2001: unresolved external symbol __imp__PathMatchSpecA@8

Resolve:

Add "shlwapi.lib" to "Project - Property - Linker - Input - Additional Dependencies".

Reference:

[https://groups.google.com/forum/#!topic/google-gflags/cM4DuGOS_GI](https://groups.google.com/forum/#!topic/google-gflags/cM4DuGOS_GI)
