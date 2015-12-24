---
layout: post
category: linux_study
title: Useful Linux Commands
date: 2015-07-24
---

* TOC
{:toc}

## Count

Count lines in a document

<pre class="terminal"><code>$ wc -l /dir/file.txt</code></pre>

<pre class="terminal"><code>$ cat /dir/file.txt | wc -l</code></pre>

Filter and count only lines with pattern, or with -v to invert match

{% highlight bash %}
$ grep -w "pattern" -c file
{% endhighlight %}

{% highlight bash %}
$ grep -w "pattern" -c -v file
{% endhighlight %} 

Count files in the current directory:

<pre class="terminal"><code>$ ls -l | grep “^-” | wc -l</code></pre>

example: counting all js files in directory "/home/account/" (recursively):

<pre class="terminal"><code>$ ls -lR /home/account | grep js | wc -l</code></pre>
<pre class="terminal"><code>$ ls -l "/home/account" | grep "js" | wc -l</code></pre>

Count files in the current directory, recursively:

<pre class="terminal"><code>$ ls -lR | grep “^-” | wc -l</code></pre>

Count folders in the current directory, recursively:

<pre class="terminal"><code>$ ls -lR | grep “^d” | wc -l</code></pre>

## Transfer Files

Remote transfer files, remote -> local:

<pre class="terminal"><code>$ scp account@111.111.111.111:/path/to/remote/file /path/to/local/</code></pre>

Remote transfer files, local -> remote:

<pre class="terminal"><code>$ scp /path/to/local/file account@111.111.111.111:/path/to/remote/</code></pre>

Print directory structure:

<pre class="terminal"><code>$ ls -l -R</code></pre>

Print folder in current directory:

<pre class="terminal"><code>$ ls -lF |grep /</code></pre>
<pre class="terminal"><code>$ ls -l |grep '^d'</code></pre>


Print history command:

<pre class="terminal"><code>$ history</code></pre>
<pre class="terminal"><code>$ history | less</code></pre>

## Print info

Print system info:

<pre class="terminal"><code>$ cat /proc/version</code></pre>

Print software info:

<pre class="terminal">
<code>$ whereis SOFEWARE
$ which SOFEWARE
$ locate SOFEWARE
</code></pre>

Print CPU info:

<pre class="terminal"><code>$ cat /proc/cpuinfo</code></pre>
dmesg |grep -i xeon</code></pre>

Print memory info:

<pre class="terminal"><code>$ cat /proc/meminfo</code></pre>
<pre class="terminal"><code>$ free -m</code></pre>

Print graphics card version:

<pre class="terminal"><code>$ nvcc --version</code></pre>

Print graphics card GPU info:

<pre class="terminal"><code>$ nvidia-smi</code></pre>

Print disk free space:

<pre class="terminal"><code>$ df -h
$ df -hl</code></pre>

Print current folder size

<pre class="terminal"><code>$ du -sh DIRNAME</code></pre>

Print target folder volume (in MB)

<pre class="terminal"><code>$ du -sm</code></pre>

## Download

Download file:

<pre class="terminal"><code>$ wget "http://domain.com/directory/4?action=AttachFile&do=view&target=file.tgz"</code></pre>

Download file to specific directory:

<pre class="terminal"><code>$ wget -O /home/omio/Desktop/ "http://thecanadiantestbox.x10.mx/CC.zip"</code></pre>

Download all files from a folder on a website or FTP:

<pre class="terminal"><code>$ wget -r --no-parent --reject "index.html*" http://vision.cs.utexas.edu/voc/</code></pre>

Perl-based rename commands(-n: test commands; -v: print renamed files):

<pre class="terminal"><code>$ rename -n 's/\.htm$/\.html/' \*.htm</code></pre>
<pre class="terminal"><code>$ rename -v 's/\.htm$/\.html/' \*.htm</code></pre>

## Screen

Run program in screen mode:

<pre class="terminal"><code>$ screen python demo.py --gpu 1</code></pre>

Detach screen: Ctrl + a, c

Re-connect screen:

<pre class="terminal"><code>$ screen -r pid</code></pre>

Display all screens:

<pre class="terminal"><code>$ screen -ls</code></pre>

Delete screen:

<pre class="terminal"><code>$ kill pid</code></pre>

Naming a screen:

<pre class="terminal"><code>$ screen -S sessionName</code></pre>

## Ctags

<code>ctags –R * </code>: Generate tags files in source code root directory

<code>vim -t func/var</code>: find func/var definition

<code>:ts</code>: give a list if func/var has multiple definitions

<code>Ctrl+]</code>: jump to definition

<code>Ctrl+T</code>: jump back

## nohup

<pre class="terminal"><code>$ nohup command-with-options &</code></pre>

<pre class="terminal"><code>$ nohup xxx.sh 1 > log.txt 2>&1 &</code></pre>

**nohup - get the process ID to kill a nohup process**

<pre class="terminal"><code>$ nohup command-with-options &</code></pre>

<pre class="terminal"><code>$ echo $! > save_pid.txt</code></pre>

<pre class="terminal"><code>$ kill -9 `cat save_pid.txt`</code></pre>

## Others

Launch terminal in Ubuntu: Ctrl+Alt+T

Create symbol link:

<pre class="terminal"><code>$ ln -s EXISTING_FILE SYMLINK_FILE
$ ln -s /path/to/file /path/to/symlink</code></pre>

Open image:

<pre class="terminal"><code>$ eog /path/to/image/im.jpg</code></pre>
<pre class="terminal"><code>$ display /path/to/image/im.jpg</code></pre>

Convert text files with DOS or MAC line breaks to Unix line breaks

{% highlight bash %}
$ sudo dos2unix /path/to/file
$ sudo sed -i -e 's/\r$//' /path/to/file
{% endhighlight %}

Create new file list:

{% highlight bash %}
$ sed 's?^?'`pwd`'/detection_images/?; s?$?.jpg?' trainval.txt > voc.2007trainval.list
{% endhighlight %}