---
layout: post
category: linux_study
title: Linux Unpack and Decompression Commands
date: 2015-08-01
---

**.tar**

{% highlight bash %}
unpack: tar xvf FileName.tar
pack: tar cvf FileName.tar DirName
{% endhighlight %}

**.gz**

{% highlight bash %}
uncompress: gunzip FileName.gz
uncompress: gzip -d FileName.gz
compress: gzip FileName
{% endhighlight %}

**.tar.gz, .tgz**

{% highlight bash %}
uncompress: tar zxvf FileName.tar.gz
compress: tar zcvf FileName.tar.gz DirName
{% endhighlight %}

**.bz2**

{% highlight bash %}
uncompress: bzip2 -d FileName.bz2
uncompress: bunzip2 FileName.bz2
compress: bzip2 -z FileName
{% endhighlight %}

**.tar.bz2**

{% highlight bash %}
uncompress: tar jxvf FileName.tar.bz2
compress: tar jcvf FileName.tar.bz2 DirName
{% endhighlight %}

**.bz**

{% highlight bash %}
uncompress: bzip2 -d FileName.bz
uncompress: bunzip2 FileName.bz
compress: (unknown)
{% endhighlight %}

**.tar.bz**

{% highlight bash %}
uncompress: tar jxvf FileName.tar.bz
compress: (unknown)
{% endhighlight %}

**.Z**

{% highlight bash %}
uncompress FileName.Z
compress FileName
{% endhighlight %}

**.tar.Z**

{% highlight bash %}
uncompress: tar Zxvf FileName.tar.Z
compress: tar Zcvf FileName.tar.Z DirName
{% endhighlight %}

**.zip**

{% highlight bash %}
uncompress: unzip FileName.zip
compress: zip FileName.zip DirName
{% endhighlight %}

**.rar**

{% highlight bash %}
uncompress: rar x FileName.rar
compress: rar a FileName.rar DirName
{% endhighlight %}

**.lha**

{% highlight bash %}
uncompress: lha -e FileName.lha
compress: lha -a FileName.lha FileName
{% endhighlight %}

**.rpm**

{% highlight bash %}
unpack: rpm2cpio FileName.rpm | cpio -div
{% endhighlight %}

**.deb**

{% highlight bash %}
unpack: ar p FileName.deb data.tar.gz | tar zxf -
{% endhighlight %}

**.tar .tgz .tar.gz .tar.Z .tar.bz .tar.bz2 .zip .cpio .rpm .deb .slp .arj .rar .ace .lha .lzh .lzx .lzs .arc .sda .sfx .lnx .zoo .cab .kar .cpt .pit .sit .sea**

{% highlight bash %}
uncompress: sEx x FileName.*
compress: sEx a FileName.* FileName
{% endhighlight %}

**uncompress .xz files**

{% highlight bash %}
xz -d linux-3.12.tar.xz
tar -xf linux-3.12.tar
tar -Jxf linux-3.12.tar.xz
{% endhighlight %}
