---
layout: post
category: android_dev
title: Android Development Resources
date: 2016-05-23
---

# Eclipse

## DDMS

**How to enable native heap tracking in DDMS**

![](http://bricolsoftconsulting.com/wp-content/uploads/2012/05/ddms_native.png)

- blog: [http://bricolsoftconsulting.com/how-to-enable-native-heap-tracking-in-ddms/](http://bricolsoftconsulting.com/how-to-enable-native-heap-tracking-in-ddms/)

**Tips for Optimizing Android* Application Memory Usage**

![](https://software.intel.com/sites/default/files/managed/15/01/tips-for-optimizing-android-app-memory-fig2-ddms-heap-updates-track-allocation.png)

- blog: [https://software.intel.com/en-us/android/articles/tips-for-optimizing-android-application-memory-usage](https://software.intel.com/en-us/android/articles/tips-for-optimizing-android-application-memory-usage)

**使用DDMS中的内存监测工具Heap来优化内存**

![](http://images.cnitblog.com/blog/651487/201502/021532244064511.gif)

- blog: [http://www.cnblogs.com/tianzhijiexian/p/4267919.html](http://www.cnblogs.com/tianzhijiexian/p/4267919.html)

## Memory Analyzer Tool (MAT)

**Memory Analyzer 1.5.0 Release**

![](http://www.eclipse.org/mat/home/mat_thumb.png)

- homepage: [http://www.eclipse.org/mat/](http://www.eclipse.org/mat/)
- download page: [http://www.eclipse.org/mat/downloads.php](http://www.eclipse.org/mat/downloads.php)

**Eclipse Memory Analyzer (MAT) - Tutorial**

- blog: [http://www.vogella.com/tutorials/EclipseMemoryAnalyzer/article.html](http://www.vogella.com/tutorials/EclipseMemoryAnalyzer/article.html)

**10 Tips for using the Eclipse Memory Analyzer**

[http://eclipsesource.com/blogs/2013/01/21/10-tips-for-using-the-eclipse-memory-analyzer/](http://eclipsesource.com/blogs/2013/01/21/10-tips-for-using-the-eclipse-memory-analyzer/)

**[Android] 内存泄漏调试经验分享 (二)**

- intro: 内存监测工具 DDMS --> Heap, 内存分析工具 MAT(Memory Analyzer Tool)
- blog: [http://rayleeya.iteye.com/blog/755657](http://rayleeya.iteye.com/blog/755657)

**Hunting Your Leaks: Memory Management in Android (Part 2 of 2)**

- blog: [http://www.raizlabs.com/dev/2014/04/hunting-your-leaks-memory-management-in-android-part-2-of-2/](http://www.raizlabs.com/dev/2014/04/hunting-your-leaks-memory-management-in-android-part-2-of-2/)

# Valgrind

**Valgrind**

- intro: Valgrind is an instrumentation framework for building dynamic analysis tools. 
There are Valgrind tools that can automatically detect many memory management and threading bugs, 
and profile your programs in detail.
- homepage: [http://valgrind.org/](http://valgrind.org/)

**The compiler under Windows Valgrind for Android**

- intro: Windows 7, Cygwin, Valgrind 3.9.0
- blog: [http://www.programering.com/a/MjM3UzMwATE.html](http://www.programering.com/a/MjM3UzMwATE.html)
- csdn: [http://blog.csdn.net/foruok/article/details/20701991](http://blog.csdn.net/foruok/article/details/20701991)

**Building and running valgrind on Android**

- blog: [https://blog.frals.se/2014/07/02/building-and-running-valgrind-on-android/](https://blog.frals.se/2014/07/02/building-and-running-valgrind-on-android/)
- gist: [https://gist.github.com/frals/7775c413a52763d80de3](https://gist.github.com/frals/7775c413a52763d80de3)

**android valgrind build**

- blog: [http://none53.hatenablog.com/entry/20150325/1427242876](http://none53.hatenablog.com/entry/20150325/1427242876)

**valgrind: failed to start tool 'memcheck' for platform 'arm-linux': Permission denied**

- blog: [http://none53.hatenablog.com/entry/20150325/1427249228](http://none53.hatenablog.com/entry/20150325/1427249228)
- my notes: this blog really helped me..

# LeakCanary

**LeakCanary: A memory leak detection library for Android and Java**

![](https://raw.githubusercontent.com/square/leakcanary/master/assets/screenshot.png)

- github: [https://github.com/square/leakcanary](https://github.com/square/leakcanary)