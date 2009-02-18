---
layout: post
category: programming_study
title: Embedding Python In C/C++
date: 2015-10-10
---

# Preparatory Work

Copy all necessary Python files to your project directory. It would be convenient that your program could still work even if other people's computer doesn't install a Python toolkit.

What We need:

(1) "C:\\Python27\\include". This directory will be added to VS-C++

(2) "C:\\Python27\\lib". This directory will be added to

(3) Copy all files in "C:\\Python27\\DLLS" and "C:\\Python27\\Lib" to one directory, e.g: "Python27". This is gonna be a huge directory..but you can remove many packages that you don't actually need.

So finally my project structure just as below:

# Coding Work

Now that we created a huge Python27 directory containing all packages, we can directly start our coding work by:

{% highlight python %}
from time import time
time.print()
{% endhighlight %}

However if you try to import those 3rd-party packages, such as numpy/cv2, your program will crash without any warning. Well, you probably already installed those packages into "C:\\Python27\\Lib\\site-packages"(now what we got is a Python27 directory, so it is "Python27\\site-packages" in your project directory). Before import any package, you should add one more line ahead:

{% highlight python %}
import cv2
{% endhighlight %}

Now you can import cv2 and manipulate images successfully!

# Hard Wording Remainded

It would be a frustrated if you've written a lot of codes but the program can not run - it just crash, without any warning. Debugging would be a nightmare.

# Reference

(1) "在 C++ 程序中嵌入 Python 脚本": [http://www.yangyuan.info/post.php?id=1071](http://www.yangyuan.info/post.php?id=1071)

(2) "Embedding Python in C/C++: Part I": [http://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I](http://www.codeproject.com/Articles/11805/Embedding-Python-in-C-C-Part-I)

(3) "Embedding Python in C/C++: Part II": [http://www.codeproject.com/Articles/11843/Embedding-Python-in-C-C-Part-II](http://www.codeproject.com/Articles/11843/Embedding-Python-in-C-C-Part-II)
