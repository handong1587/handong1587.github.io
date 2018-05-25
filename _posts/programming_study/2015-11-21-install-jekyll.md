---
layout: post
title: Install Jekyll To Fix Some Local Github-pages Defects
date: 2015-11-21 00:02:26
category: "web_dev"
---

# Install jekyll

I follow the blog: [http://blog.csdn.net/itmyhome1990/article/details/41982625](http://blog.csdn.net/itmyhome1990/article/details/41982625) 
to install Ruby, Devkit, and Jekyll.

1. Download Ruby and DevKit: [http://rubyinstaller.org/downloads/](http://rubyinstaller.org/downloads/)

2. Check "Add Ruby executables to your PATH" when installing Ruby. You can execute:

<pre class="terminal"><code>ruby -v</code></pre>

to detect if Ruby successfully installed.

3. Install DevKit. After that, cd to RubyDevKit directory:

```
C:\> cd RubyDevKit
C:\RubyDevKit> ruby dk.rb init
C:\RubyDevKit> ruby dk.rb install
```

# Intall github-pages

When I try to install github-pages by "gem install github-pages", an error(FetchError) is encountered: 

![](/assets/web_dev/fetch_error.PNG)

Seems like it is because the Ruby website is blocked. So I follow the instructions by @fighterleslie in 
[http://segmentfault.com/q/1010000003891086](http://segmentfault.com/q/1010000003891086), create a .gemrc 
file into "C:\Users\MyName", and problem solved:

```
:sources:
- https://ruby.taobao.org
:update_sources: true
```

You may still encounter this FetchError:

![](/assets/web_dev/fetch_error_2.PNG)

![](/assets/web_dev/fetch_error_3.png)

I found a workaround here (Really a "workaround". It opens a possible security vulnerability):
add the line to your .gemrc file:

```
:ssl_verify_mode: 0
```

So now my .gemrc file is like:

```
:ssl_verify_mode: 0
:sources:
- https://ruby.taobao.org
:update_sources: true
```

Anyway, now jekyll is successfully installed on my system.

# Try jekyll build!

Follow [http://rockhong.github.io/github-pages-fails-to-update.html](http://rockhong.github.io/github-pages-fails-to-update.html)
to detect my github-pages defects.

OK, try the instruction below:

<pre class="terminal">
<code>$ jekyll build --safe</code>
</pre>

Then I get:

![](/assets/web_dev/jekyll_build_reuslts.png)

Follow the error information, do some minor changes, and finally my github-pages can successfully be shown.

Don't miss the chance to use jekyll to preview your site! Running the line:

```
jekyll serve –watch
```

and then open website `http://localhost:4000/` in browser.
If want to open a specific page, use `http://localhost:4000/web_dev/2015/11/21/install-jekyll.html`.

# Something else to note..

The above-mentioned instructions just work fine for me on my laptop(Windows 8.1, X64). 
But some other errors may happen, like Cygwin and Windows git can't play nicely 
together(on my work PC, Windows 7, X32, with Cygwin installed).
One particulr error message is like:

![](/assets/web_dev/gem_install_github-pages_cygwin_error.jpg)

When I try to execute a "gem install github-pages".

I found two posts helpful:

[http://stackoverflow.com/questions/19259272/error-installing-gem-couldnt-reserve-space-for-cygwins-heap-win32-error-487](http://stackoverflow.com/questions/19259272/error-installing-gem-couldnt-reserve-space-for-cygwins-heap-win32-error-487)

[http://blog.arithm.com/2014/02/14/couldnt-reserve-space-for-cygwins-heap-win32-error-0/](http://blog.arithm.com/2014/02/14/couldnt-reserve-space-for-cygwins-heap-win32-error-0/)

In a nutshell, I download rebase-1.18-1.exe from [http://www.tishler.net/jason/software/rebase/](http://www.tishler.net/jason/software/rebase/),
run it. 
Go to RubyDevkit directory, run devkitvars.bat. 
Fire up a git bash (or Windows Prompt, must run as administrator), execute instruction below:

<pre class="terminal">
<code>$ rebase.exe -b 0x50000000 msys-1.0.dll</code>
</pre>

Retry `gem install github-pages`. Now *voila*, everything works nicely!

# Refs

[1] [http://blog.csdn.net/itmyhome1990/article/details/41982625](http://blog.csdn.net/itmyhome1990/article/details/41982625)

[2] [http://segmentfault.com/q/1010000003891086](http://segmentfault.com/q/1010000003891086)

[3] [http://railsapps.github.io/openssl-certificate-verify-failed.html](http://railsapps.github.io/openssl-certificate-verify-failed.html)

[4] [http://rockhong.github.io/github-pages-fails-to-update.html](http://rockhong.github.io/github-pages-fails-to-update.html)