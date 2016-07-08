---
layout: post
category: working_on_windows
title: Windows Commands and Utilities
date: 2015-10-27
---

# Commands

**Create symbolic link in Windows 7**

Use admin privileges to run Command Prompt.

Example:

```
D:\coding\Python_VS2015\densebox\fast-rcnn\data>mklink /D VOCdevkit2007 D:\data\public_dataset\VOCdevkit
```

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

Ref:

**DIR: Display a list of files and subfolders**

[http://ss64.com/nt/dir.html](http://ss64.com/nt/dir.html)

**List all files/folders into a txt**

```
dir /s /b /a *.jpg > list.txt
``` 

**List only folders into a txt**

```
dir /s /b /ad *.jpg > list.txt
```

**List only files into a txt**

```
dir /s /b /a:-d *.mp4 *.mkv > list.txt
```

# Utilities

**A Utility to Unassociate File Types in Windows 7 and Vista**

[http://www.winhelponline.com/blog/unassociate-file-types-windows-7-vista/](http://www.winhelponline.com/blog/unassociate-file-types-windows-7-vista/)

**Video DownloadHelper**

- chrome-plugin: [https://chrome.google.com/webstore/detail/video-downloadhelper/lmjnegcaeklhafolokijcfjliaokphfk](https://chrome.google.com/webstore/detail/video-downloadhelper/lmjnegcaeklhafolokijcfjliaokphfk)
- filefox-plugin: [https://addons.mozilla.org/zh-cn/firefox/addon/video-downloadhelper/](https://addons.mozilla.org/zh-cn/firefox/addon/video-downloadhelper/)
- homepage: [http://www.downloadhelper.net/welcome.php?version=5.4.2](http://www.downloadhelper.net/welcome.php?version=5.4.2)

**BulkFileChanger**

- intro: "BulkFileChanger is a small utility that allows you to create files list from multiple folders, 
and then make some action on them - Modify their created/modified/accessed time, 
change their file attribute (Read Only, Hidden, System), 
run an executable with these files as parameter, and copy/cut paste into Explorer."
- website: [http://www.nirsoft.net/utils/bulk_file_changer.html](http://www.nirsoft.net/utils/bulk_file_changer.html)

**AVS Video ReMaker**

- intro: Edit AVI, VOB, MP4, DVD, Blu-ray, TS, MKV, 
HD-videos fast and without reconversion.
Burn Blu-ray or DVD discs with menus.
- website: [http://www.avs4you.com/AVS-Video-ReMaker.aspx](http://www.avs4you.com/AVS-Video-ReMaker.aspx)

**AVS Audio Editor**

- intro: Cut, join, trim, mix, delete parts, split audio files.
Apply various effects and filters. Record audio from various inputs. Save files to all key audio formats.
- website: [http://www.avs4you.com/AVS-Audio-Editor.aspx](http://www.avs4you.com/AVS-Audio-Editor.aspx)

**Dos2Unix / Unix2Dos - Text file format converters**

![](http://dos2unix.sourceforge.net/dos2unix2mac.png)

- homepage: [http://waterlan.home.xs4all.nl/dos2unix.html](http://waterlan.home.xs4all.nl/dos2unix.html)
- homepage: [http://dos2unix.sourceforge.net/](http://dos2unix.sourceforge.net/)
- download(dos2unix): [http://waterlan.home.xs4all.nl/dos2unix.html#DOS2UNIX](http://waterlan.home.xs4all.nl/dos2unix.html#DOS2UNIX)

**Git for Windows 国内下载站**

- github: [https://github.com/waylau/git-for-win](https://github.com/waylau/git-for-win)