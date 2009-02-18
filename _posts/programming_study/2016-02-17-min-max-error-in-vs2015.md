---
layout: post
category: programming_study
title: Fix min/max Error In VS2015
date: 2016-02-17
---

VS2010 projects using min/max functions can be built successfully, 
while in VS2015 it will keep printing build error:

error C3861: 'min': identifier not found

I finally find out that std::max() requires the `<algorithm>` header.