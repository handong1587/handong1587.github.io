---
layout: post
title: Install Jekyll To Fix Some Local Github-pages Defects
date: 2015-11-21 00:02:26
category: "web_dev"
---

# Install jekyll

I follow the blog: [http://blog.csdn.net/itmyhome1990/article/details/41982625](http://blog.csdn.net/itmyhome1990/article/details/41982625) 
to install Ruby, Devkit, and Jekyll.

# Intall github-pages

When I try to install github-pages by "gem install github-pages", an error(FetchError) is encountered: 

Seems like it is because the Ruby website is blocked. So I follow the instructions by @fighterleslie in 
[http://segmentfault.com/q/1010000003891086](http://segmentfault.com/q/1010000003891086), create a .gemrc 
file into "C:\Users\MyName", and problem solved:

<pre class="terminal"><code>
:sources:
- https://ruby.taobao.org
:update_sources: true
</code></pre>

# Try jekyll build!

Follow [http://rockhong.github.io/github-pages-fails-to-update.html](http://rockhong.github.io/github-pages-fails-to-update.html)
to detect my github-pages defects.

OK, try the instruction below:

<pre class="terminal"><code>
$ jekyll build --safe
</code></pre>

Then I get:

<img src="/assets/web_dev/jekyll_build_reuslts.png"/>

Follow the error information, do some minor changes, and finally my github-pages can successfully be shown.

# Something else to note

The above-mentioned instructions just work fine for me on my laptop(Windows 8.1, X64). 
But some other errors may happen, like Cygwin and Windows git can play nicely 
together(on my working PC, Windows 7, X32. Cygwin installed).
One particulr error message is like:

<img src="/assets/web_dev/gem_install_github-pages_cygwin_error.jpg"/>

When I try to execute a "gem install github-pages".

I found two posts helpful:

[http://stackoverflow.com/questions/19259272/error-installing-gem-couldnt-reserve-space-for-cygwins-heap-win32-error-487](http://stackoverflow.com/questions/19259272/error-installing-gem-couldnt-reserve-space-for-cygwins-heap-win32-error-487)

[http://blog.arithm.com/2014/02/14/couldnt-reserve-space-for-cygwins-heap-win32-error-0/](http://blog.arithm.com/2014/02/14/couldnt-reserve-space-for-cygwins-heap-win32-error-0/)

In a nutshell, I download rebase-1.18-1.exe from [http://www.tishler.net/jason/software/rebase/](http://www.tishler.net/jason/software/rebase/),
run it. 
Go to RubyDevkit directory, run devkitvars.bat. 
Fire up a git bash (or Windows Prompt, must run as administrator), execute instruction below:

<pre class="terminal"><code>
$ rebase.exe -b 0x50000000 msys-1.0.dll
</code></pre>

Retry "gem install github-pages". Now *voila*, everything work nicely!