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

(How to access gist in China? Add <pre class="terminal"><code>192.30.252.141 gist.github.com</code></pre> 
to C:/Windows/System32/drivers/etc/hosts)

One thing to note is that on Windows sometimes there will be some '\r', '\r\n' problems, 
so we'd better use a dos2unix tool to convert the text format after every time we edit shell files on Windows.
I use a tool from: [http://dos2unix.sourceforge.net/](http://dos2unix.sourceforge.net/)

To successfully build, we need to modify the kernel version in configure 
(my VM's kernel verison is 2.0.1, which is not valid by default):

{% highlight bash %}
case "${kernel}" in
0.*|1.*|2.0.*|2.1.*|2.2.*|2.3.*|2.4.*|2.5.*)
{% endhighlight %}

{% highlight bash %}
case "${kernel}" in
0.*|1.*|2.1.*|2.2.*|2.3.*|2.4.*|2.5.*)
{% endhighlight %}

Normally we should wait for about half an hour for bulding. All looks okay by far. But when I try to test it, things become nasty.

When I build a 32-bit Valgrind, every time I try to start memcheck, it will output an error log:

<pre class="terminal">
<code>$ $SDKROOT/adb.exe shell "/data/local/Inst/bin/valgrind am start -a android.intent.action.MAIN -n com.example.MyAPP/.MyAPPMain"
valgrind: failed to start tool 'memcheck' for platform 'arm64-linux': No such file or directory</code>
</pre>

Wired thing is that there is a memcheck-arm-linux in /data/local/Inst/lib/valgrind/, 
and obviously Valgrind should call it since they are of 32-bit. But Valgrind always try to call a 64-bit memcheck. 

<pre class="terminal">
<code>$ $SDKROOT/adb.exe shell "strace /data/local/Inst/bin/valgrind am start -a android.intent.action.MAIN -n com.example.MyAPP/.MyAPPMain"</code>
</pre>

<pre class="terminal"><code>
execve("/data/local/Inst/lib/valgrind/memcheck-arm64-linux", ["/data/local/Inst/bin/valgrind", "am", "start", "-a", "android.intent.action.MAIN", "-n", "com.example.SGallery/.SGalleryMa"...], [/* 17 vars */]) = -1 ENOENT (No such file or directory)
</code></pre>

So I was stucked on this.
And it is more frustrating that if I change to 64-bit configure, many Valgrind tools are failed to build. Still not memcheck-arm64-linux.

# DDMS and MAT

# Refs