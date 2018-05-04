---
layout: post
title: Install Therubyracer Failure
date: 2016-07-03
category: "web_dev"
---

I try to install therubyracer via gem on Windows 10, but keep getting an error associated with -rdynamic flag, 
which results in failure to build the native extensions:

![](/assets/web_dev/gem_install_therubyracer.png)

You can find -rdynamic flag in extconf.rb and Makefile:

`C:\Ruby23-x64\lib\ruby\gems\2.3.0\gems\therubyracer-0.12.2\ext\v8\extconf.rb`

```
$CPPFLAGS += " -rdynamic" unless $CPPFLAGS.split.include? "-rdynamic"
$CPPFLAGS += " -fPIC" unless $CPPFLAGS.split.include? "-rdynamic" or RUBY_PLATFORM =~ /darwin/
```

`C:\Ruby23-x64\lib\ruby\gems\2.3.0\gems\therubyracer-0.12.2\ext\v8\Makefile`

```
CPPFLAGS =  -DFD_SETSIZE=2048 -D_WIN32_WINNT=0x0501 -D__MINGW_USE_VC2005_COMPAT $(DEFS) $(cppflags) -Wall -g -rdynamic
```

Somebody figures it out by changing gcc compiler to version 4.2 (mine is 4.9.3).
For some reason the newer gcc version don't just ignore the -rdynamic flag, 
which is only present for compiling on Linux and is not actually compatible with Windows and OS X. 

[https://stackoverflow.com/questions/35741536/trouble-installing-therubyracer-gem-due-to-compiler-issue-on-mac](https://stackoverflow.com/questions/35741536/trouble-installing-therubyracer-gem-due-to-compiler-issue-on-mac)

More detailed explanation is that:
[-rdynamic passes the flag -export-dynamic to ELF linker, on targets that support it](https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html)

Executable formats in OS X and Windows are not ELF, 
thus the option -rdynamic is not supported building for these operating systems.

[http://stackoverflow.com/questions/29534519/why-gcc-doesnt-recognize-rdynamic-option](http://stackoverflow.com/questions/29534519/why-gcc-doesnt-recognize-rdynamic-option)

One solution is using an older gcc compiler, like gcc-4.2.

[http://preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/](http://preshing.com/20141108/how-to-install-the-latest-gcc-on-windows/)

```
wget http://ftpmirror.gnu.org/gcc/gcc-4.2.4/gcc-4.2.4.tar.gz
```

```
wget http://ftpmirror.gnu.org/gcc/gcc-4.2.4/gcc-g++-4.2.4.tar.bz2
```