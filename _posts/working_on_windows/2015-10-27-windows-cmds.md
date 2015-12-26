---
layout: post
category: working_on_windows
title: Windows Commands and Utilities
date: 2015-10-27
---

* TOC
{:toc}

# Commands

**Create symbolic link in Windows 7**

Use admin privileges to run Command Prompt.

Example:

<pre class="terminal"><code>
D:\coding\Python_VS2015\densebox\fast-rcnn\data>mklink /D VOCdevkit2007 D:\data\public_dataset\VOCdevkit
</code></pre>

Command Parameters:

{% highlight bash %}
MKLINK [[/D] | [/H] | [/J]] Link Target

/D      Creates a directory symbolic link.  Default is a file
symbolic link.
/H      Creates a hard link instead of a symbolic link.
/J      Creates a Directory Junction.
Link    specifies the new symbolic link name.
Target  specifies the path (relative or absolute) that the new link refers to.
{% endhighlight %}

# Utilities

**A Utility to Unassociate File Types in Windows 7 and Vista**

[http://www.winhelponline.com/blog/unassociate-file-types-windows-7-vista/](http://www.winhelponline.com/blog/unassociate-file-types-windows-7-vista/)

## Video DownloadHelper

- filefox-plugin: [https://addons.mozilla.org/zh-cn/firefox/addon/video-downloadhelper/](https://addons.mozilla.org/zh-cn/firefox/addon/video-downloadhelper/)
- homepage: [http://www.downloadhelper.net/welcome.php?version=5.4.2](http://www.downloadhelper.net/welcome.php?version=5.4.2)