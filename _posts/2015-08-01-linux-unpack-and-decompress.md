---
layout: post
title: Linux Unpack and Decompression Commands
---

{{ page.title }}
================

<p class="meta">01 Aug 2015 - Beijing</p>

**.tar**

```bash
unpack: tar xvf FileName.tar
pack: tar cvf FileName.tar DirName
```

**.gz**

```bash
uncompress: gunzip FileName.gz
uncompress: gzip -d FileName.gz
compress: gzip FileName
```

**.tar.gz, .tgz**

```bash
uncompress: tar zxvf FileName.tar.gz
compress: tar zcvf FileName.tar.gz DirName
```

**.bz2**

```bash
uncompress: bzip2 -d FileName.bz2
uncompress: bunzip2 FileName.bz2
compress: bzip2 -z FileName
```

**.tar.bz2**

```bash
uncompress: tar jxvf FileName.tar.bz2
compress: tar jcvf FileName.tar.bz2 DirName
```

**.bz**

```bash
uncompress: bzip2 -d FileName.bz
uncompress: bunzip2 FileName.bz
compress: (unknown)
```

**.tar.bz**

```bash
uncompress: tar jxvf FileName.tar.bz
compress: (unknown)
```

**.Z**

```bash
uncompress FileName.Z
compress FileName
```

**.tar.Z**

```bash
uncompress: tar Zxvf FileName.tar.Z
compress: tar Zcvf FileName.tar.Z DirName
```

**.zip**

```bash
uncompress: unzip FileName.zip
compress: zip FileName.zip DirName
```

**.rar**

```bash
uncompress: rar x FileName.rar
compress: rar a FileName.rar DirName
```

**.lha**

```bash
uncompress: lha -e FileName.lha
compress: lha -a FileName.lha FileName
```

**.rpm**

```bash
unpack: rpm2cpio FileName.rpm | cpio -div
```

**.deb**

```bash
unpack: ar p FileName.deb data.tar.gz | tar zxf -
```

**.tar .tgz .tar.gz .tar.Z .tar.bz .tar.bz2 .zip .cpio .rpm .deb .slp .arj .rar .ace .lha .lzh .lzx .lzs .arc .sda .sfx .lnx .zoo .cab .kar .cpt .pit .sit .sea**

```bash
uncompress: sEx x FileName.*
compress: sEx a FileName.* FileName
```

**uncompress .xz files**

```bash
xz -d linux-3.12.tar.xz
tar -xf linux-3.12.tar
tar -Jxf linux-3.12.tar.xz
```
