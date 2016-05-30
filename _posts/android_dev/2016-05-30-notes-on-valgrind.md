---
layout: post
category: android_dev
title: Notes On Valgrind and Others
date: 2016-05-30
---

# Valgrind

Valgrind is a suite of tools for Profiling and debugging, I intend to use Memcheck to check my APP's memory leak issue.
Check out Vlagrind homepage at: [http://valgrind.org/](http://valgrind.org/), the version I use is: valgrind-3.11.0.

I follow the blog to configure Valgrind:

[https://blog.frals.se/2014/07/02/building-and-running-valgrind-on-android/](https://blog.frals.se/2014/07/02/building-and-running-valgrind-on-android/)

The author also provides some shell files on his gist: [ttps://gist.github.com/frals/7775c413a52763d80de3](ttps://gist.github.com/frals/7775c413a52763d80de3). 
They are useful, but since I work on Windows 7 x86, I add some modifications. [https://github.com/handong1587/run_valgrind](https://github.com/handong1587/run_valgrind)

(How to access gist in China? Add `192.30.252.141 gist.github.com` to C:/Windows/System32/drivers/etc/hosts)

One thing to note is that on Windows sometimes there will be some '\r', '\n\r' problems, 
so we'd better use a dos2unix tool to convert the text format after every time we edit shell files on Windows.
I use a tool from: [http://dos2unix.sourceforge.net/](http://dos2unix.sourceforge.net/) 

# Refs