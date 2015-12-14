---
layout: post
category: programming_study
title: Enable Large Addresses On VS2015
date: 2015-12-14
---

A Win32 process running under Win64 can use up to 4 GB if IMAGE_FILE_LARGE_ADDRESS_AWARE is set, 
make sure Visual Studio Project Properties have this enabled:

**Configuration Properties->Linker->System:**

"Enable Large Addresses"

set to:

**Support Addresses Larger Than 2 Gigabytes (/LARGEADDRESSAWARE)**

# Reference

[https://social.msdn.microsoft.com/Forums/en-US/bd656b83-b715-4e08-af4d-c6372979fa9a/increase-memory-for-visual-studio-2010-c?forum=Vsexpressvc](https://social.msdn.microsoft.com/Forums/en-US/bd656b83-b715-4e08-af4d-c6372979fa9a/increase-memory-for-visual-studio-2010-c?forum=Vsexpressvc)