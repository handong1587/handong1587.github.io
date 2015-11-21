---
layout: post
category: programming_study
title: Horrible Wired Errors Come From Simple Stupid Mistake
date: 2015-10-16
---

Several days ago I was transplanting some codes from Linux to Windows x86 platform.

The VS 2010 project worked fine on Windows server; But when I tried to re-run the program on my PC, many wired errors occurred: run-time errors, memory leaks, and many unexplainable exceptions.

After many aimless, exhausted searching on Google, I started to think about giving up. I opened up the module/call stack windows of Visual Studio, compared those with Windows server's version. And I found something strange: My program was unexpectedly calling msvcp120.dll! I suddenly realized that I should check the entire calling stack of the program.

I launched one utility software: Dependency Walker (depends.exe), loaded my program, and finally, everything was clear: the program depended on one 3rd-party .dll file which has been put into target directory. But in fact it was calling the other same name .dll file on my PC's system directory: "C:/Windows". All this shit-staff - I spent several days to debug and google, wasted - resulted from my one stupid operation, dated back almost ten months ago. I was self-studying some tutorials about Deep Learning, CNN staff, meanwhile I built some open-source programs to generate some 3rd-party .dll files. I put one .dll to "C:/Windows" - this careless mistake finally led to all this nasty, unintended consequences.
