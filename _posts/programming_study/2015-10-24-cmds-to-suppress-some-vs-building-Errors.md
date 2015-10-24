---
layout: post
categories: programming_study
title: Commands To Suppress Some Building Errors With Visual Studio
---

{{ page.title }}
================

<p class="meta">24 Sep 2015 - Beijing</p>

Here are some commands you would probably frequently use when you're building Linux codes with VS2013~VS2015. Go to "C/C++ - Project - Properties - Additional Options", add following commands(each command separated by one blank):

(1) **/D _CRT_SECURE_NO_WARNINGS**:  to suppress warnings:

error C4996: 'strcpy': This function or variable may be unsafe. Consider using strcpy_s instead. To disable deprecation, use _CRT_SECURE_NO_WARNINGS. See online help for details.

(2) **/D _SCL_SECURE_NO_WARNINGS**:

c:\program files\microsoft visual studio 14.0\vc\include\xutility(2230): error C4996: 'std::_Copy_impl': Function call with parameters that may be unsafe - this call relies on the caller to check that the passed values are correct. To disable this warning, use -D_SCL_SECURE_NO_WARNINGS. See documentation on how to use Visual C++ 'Checked Iterators'

(3) **/D _CRT_NONSTDC_NO_DEPRECATE**: to suppress warnings:

error C4996: 'close': The POSIX name for this item is deprecated. Instead, use the ISO C and C++ conformant name: _close. See online help for details.
