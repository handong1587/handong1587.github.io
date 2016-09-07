---
layout: post
category: programming_study
title: Glog Build Problems on Windows X86 and Visual Studio 2015
date: 2015-10-23
---

I git clone glog from [https://github.com/google/glog](https://github.com/google/glog).

System info: Windows X86, Visual Studio 2015.

Two errors:

(1)

```
glog-0.3.4\src\windows\port.h(116): error C2375: 'snprintf': redefinition; different linkage
c:\program files\windows kits\10\include\10.0.10150.0\ucrt\stdio.h(1932): note: see declaration of 'snprintf'
```

Resolve:

Add "HAVE_SNPRINTF" to "C/C++ - Preprocessor - Preprocessor definitions".

(2)

```
glog-0.3.4\src\windows\glog\logging.h(1268): error C2280: 'std::basic_ios<char,std::char_traits<char>>::basic_ios(const std::basic_ios<char,std::char_traits<char>> &)': attempting to reference a deleted function  <br />
c:\program files\microsoft visual studio 14.0\vc\include\ios(189): note: see declaration of 'std::basic_ios<char,std::char_traits<char>>::basic_ios'
```

Resolve:

Follow this modification:

[https://github.com/google/glog/commit/856ff81a8268a5c22d026a65d4c12a2e1136f73f](https://github.com/google/glog/commit/856ff81a8268a5c22d026a65d4c12a2e1136f73f)

(3)

```
common.obj : error LNK2001: unresolved external symbol "__declspec(dllimport) void __cdecl google::InstallFailureSignalHandler(void)" (__imp_?InstallFailureSignalHandler@google@@YAXXZ)
```

Resolve:

This function appears in "glog-0.3.4\src\signalhandler.cc". But can just comment out target line:

```
::google::InstallFailureSignalHandler();
```
