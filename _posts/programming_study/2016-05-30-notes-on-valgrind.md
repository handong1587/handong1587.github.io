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

The author also provides some shell files on his gist: [https://gist.github.com/frals/7775c413a52763d80de3](https://gist.github.com/frals/7775c413a52763d80de3). 
They are useful, but since I work on Windows 7 x86, I add some modifications. [https://github.com/handong1587/run_valgrind](https://github.com/handong1587/run_valgrind)

(How to access gist in China? Add 

`192.30.252.141 gist.github.com` 

to 

`C:/Windows/System32/drivers/etc/hosts`

)

One thing to note is that on Windows sometimes there will be some '\r', '\r\n' problems, 
so we'd better use a dos2unix tool to convert the text format after every time we edit shell files on Windows.
I use a tool from: [http://dos2unix.sourceforge.net/](http://dos2unix.sourceforge.net/)

To successfully build, we need to modify the kernel version in configure 
(my VM's kernel verison is 2.0.1, which is not valid by default):

Change:

```
case "${kernel}" in
0.*|1.*|2.0.*|2.1.*|2.2.*|2.3.*|2.4.*|2.5.*)
```

to:

```
case "${kernel}" in
0.*|1.*|2.1.*|2.2.*|2.3.*|2.4.*|2.5.*)
```

Normally we should wait for about half an hour for bulding. All looks okay by far. But when I try to test it, things become nasty.

When I build a 32-bit Valgrind, every time I try to start memcheck, it will output an error log:

```
$ $SDKROOT/adb.exe shell "/data/local/Inst/bin/valgrind am start -a android.intent.action.MAIN -n com.example.MyAPP/.MyAPPMain"
valgrind: failed to start tool 'memcheck' for platform 'arm64-linux': No such file or directory
```

Wired thing is that there is a memcheck-arm-linux in /data/local/Inst/lib/valgrind/, 
and obviously Valgrind should call it since they are of 32-bit. But Valgrind always try to call a 64-bit memcheck. 
Use `strace` command can show the calling procedure easily:

```
$ $SDKROOT/adb.exe shell "strace /data/local/Inst/bin/valgrind am start -a android.intent.action.MAIN -n com.example.MyAPP/.MyAPPMain"
```

Outputs:

```
execve("/data/local/Inst/lib/valgrind/memcheck-arm64-linux", ["/data/local/Inst/bin/valgrind", "am", "start", "-a", "android.intent.action.MAIN", "-n", "com.example.SGallery/.SGalleryMa"...], [/* 17 vars */]) = -1 ENOENT (No such file or directory)
```

So I was stucked on this.
And it is more frustrating that if I change to 64-bit configure, many Valgrind tools are failed to build. Still not memcheck-arm64-linux.

# DDMS

DDMS (Dalvik Debug Monitor Server) is a debugging tool used in the Android platform, often downloaded as part of the Android SDK.
You can launch it from: "Eclipse > Window > Open Perspective > Other... > DDMS" or 
directly from: "PathToAndroidSDK\android-sdk-windows\tools\ddms.bat".

Need to modify the ddms.cfg (`c:/Users/username/.android/ddms.cfg`) file to enable native heap tracking in DDMS:

Add

```
native=true
```

in ddms.cfg.

Basicly I use DDMS to check if my Android app has memory leaks issues.
After DDMS detect your app is running, you can see that there will be updatings in "Heap > data object > Total Size"
everytime you interact with your app. That value indicates how much memory your app occupies while running.
If app does not free all allocted Java data objects, then this value will keep increasing after every GC.
So app will explode if excess some memory limitations and the process will be killed.

![](https://software.intel.com/sites/default/files/managed/15/01/tips-for-optimizing-android-app-memory-fig2-ddms-heap-updates-track-allocation.png)

# MAT

MAT (Memory Analyzer Tool) is a Java heap analyzer that can help to find memory leaks. 
You can download it from: [http://www.eclipse.org/mat/](http://www.eclipse.org/mat/).

Usually we use DDMS to dump a HPROF file, then use MAT to get reports.

# Refs

1. [http://www.programering.com/a/MjM3UzMwATE.html](http://www.programering.com/a/MjM3UzMwATE.html)
2. [https://blog.frals.se/2014/07/02/building-and-running-valgrind-on-android/](https://blog.frals.se/2014/07/02/building-and-running-valgrind-on-android/)