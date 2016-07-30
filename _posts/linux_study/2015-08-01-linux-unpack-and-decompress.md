---
layout: post
category: linux_study
title: Linux Pack/Unpack/Compress/Decompress Commands
date: 2015-08-01
---

|file type    |  Pack/Compress                      | Unpack/Decompress           |
|:-----------:|:-----------------------------------:|:---------------------------:|
|.tar         |  tar cvf FileName.tar DirName       |  tar xvf FileName.tar       |
|.gz          |  gzip FileName                      |  gunzip FileName.gz         |
|.gz          |  gzip FileName                      |  gzip -d FileName.gz        |
|.tar.gz, .tgz|  tar zcvf FileName.tar.gz DirName   |  tar zxvf FileName.tar.gz   |
|.bz2         |  bzip2 -z FileName                  |  bzip2 -d FileName.bz2      |
|.bz2         |                                     |  bunzip2 FileName.bz2       |
|.tar.bz2     |  tar jcvf FileName.tar.bz2 DirName  |  tar jxvf FileName.tar.bz2  |
|.bz          |                                     |  bzip2 -d FileName.bz       |
|.bz          |                                     |  bunzip2 FileName.bz        |
|.tar.bz      |                                     |  tar jxvf FileName.tar.bz   |
|.Z           |  compress FileName                  |  uncompress FileName.Z      |
|.tar.Z       |  tar Zcvf FileName.tar.Z DirName    |  tar Zxvf FileName.tar.Z    |
|.zip         |  zip FileName.zip DirName           |  unzip FileName.zip         |
|.rar         |  rar a FileName.rar DirName         |  rar x FileName.rar         |
|.lha         |  lha -a FileName.lha FileName       |  lha -e FileName.lha        |
|.rpm         |                                     |  rpm2cpio FileName.rpm \| cpio -div |
|.deb         |                                     |  ar x example.deb           |
|.deb         |                                     |  ar p FileName.deb data.tar.gz \| tar zx |
|.deb         |                                     |  dpkg -x somepackage.deb ~/temp/ |
|.xz          |                                     |  xz -d linux-3.12.tar.xz    |
|.xz          |                                     |  tar -xf linux-3.12.tar     |
|.xz          |                                     |  tar -Jxf linux-3.12.tar.xz |

For compress/uncompress those files:

.tar .tgz .tar.gz .tar.Z .tar.bz .tar.bz2 .zip .cpio .rpm .deb .slp .arj .rar .ace .lha .lzh .lzx .lzs .arc .sda .sfx .lnx .zoo .cab .kar .cpt .pit .sit .sea

```
compress:   sEx a FileName.* FileName
uncompress: sEx x FileName.*
```