---
layout: post
title: Some Useful Linux Commands
---

{{ page.title }}
================

<p class="meta">24 Jul 2015 - Beijing</p>

Counting files in the current directory:

<pre class="terminal"><code>$ ls -l | grep “^-” | wc -l</code></pre>

example: counting all js files in directory "/home/account/" (recursively):

<pre class="terminal"><code>$ ls -lR /home/account | grep js | wc -l</code></pre>
<pre class="terminal"><code>$ ls -l "/home/account" | grep "js" | wc -l</code></pre>

Counting files in the current directory, recursively:

<pre class="terminal"><code>$ ls -lR | grep “^-” | wc -l</code></pre>

Counting folders in the current directory, recursively:

<pre class="terminal"><code>$ ls -lR | grep “^d” | wc -l</code></pre>

Remote transfer files, remote -> local:

<pre class="terminal"><code>$ scp account@111.111.111.111:/path/to/remote/file /path/to/local/</code></pre>

Remote transfer files, local -> remote:

<pre class="terminal"><code>$ scp /path/to/local/file account@111.111.111.111:/path/to/remote/</code></pre>

Launch terminal in Ubuntu: Ctrl+Alt+T

Print system info:

<pre class="terminal"><code>$ cat /proc/version</code></pre>

Open image:

<pre class="terminal"><code>$ eog /path/to/image/im.jpg</code></pre>
<pre class="terminal"><code>$ display /path/to/image/im.jpg</code></pre>

Print directory structure:

<pre class="terminal"><code>$ ls -l -R</code></pre>

Print folder in current directory:

<pre class="terminal"><code>$ ls -lF |grep /</code></pre>
<pre class="terminal"><code>$ ls -l |grep '^d'</code></pre>

Print history command:

<pre class="terminal"><code>$ history</code></pre>
<pre class="terminal"><code>$ history | less</code></pre>

Print software info:

<pre class="terminal"><code>$ whereis SOFEWARE</code></pre>
which SOFEWARE
locate SOFEWARE</code></pre>

Print CPU info:

<pre class="terminal"><code>$ cat /proc/cpuinfo</code></pre>
dmesg |grep -i xeon</code></pre>

Print memory info:

<pre class="terminal"><code>$ cat /proc/meminfo</code></pre>
<pre class="terminal"><code>$ free -m</code></pre>

Print disk free space:

<pre class="terminal"><code>$ df -h
df -hl</code></pre>

Print current folder size

<pre class="terminal"><code>$ du -sh DIRNAME</code></pre>

Print target folder volume (in MB)

<pre class="terminal"><code>$ du -sm</code></pre>

Create symbol link:

<pre class="terminal"><code>$ ln -s EXISTING_FILE SYMLINK_FILE
ln -s /path/to/file /path/to/symlink</code></pre>

Download file:

<pre class="terminal"><code>$ wget "http://domain.com/directory/4?action=AttachFile&do=view&target=file.tgz"</code></pre>

Download file to specific directory:

<pre class="terminal"><code>$ wget -O /home/omio/Desktop/ "http://thecanadiantestbox.x10.mx/CC.zip"</code></pre>

Download all files from a folder on a website or FTP:

<pre class="terminal"><code>$ wget -r --no-parent --reject "index.html*" http://vision.cs.utexas.edu/voc/</code></pre>

Perl-based rename commands(-n: test commands; -v: print renamed files):

<pre class="terminal"><code>$ rename -n 's/\.htm$/\.html/' \*.htm</code></pre>
<pre class="terminal"><code>$ rename -v 's/\.htm$/\.html/' \*.htm</code></pre>

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

Generate Cscope database:

<pre class="terminal"><code>$ find . -name "\*.c" -o -name "\*.cc" -o -name "\*.cpp" -o -name "\*.cu" 
-o -name "\*.h" -o -name "\*.hpp" -o -name "\*.py" -o -name "\*.proto" > cscope.files</code></pre>

Build a Cscope reference database:

<pre class="terminal"><code>$ cscope -q -R -b -i cscope.files</code></pre>

Start the Cscope browser:

<pre class="terminal"><code>$ cscope -d</code></pre>

Exit a Cscope browser: Ctrl + d

Some Cscope parameters:

```bash
-b  Build the cross-reference only.
-C  Ignore letter case when searching.
-c  Use only ASCII characters in the cross-ref file (don’t compress).
-d  Do not update the cross-reference.
-e  Suppress the -e command prompt between files.
-F  symfile Read symbol reference lines from symfile.
-f  reffile Use reffile as cross-ref file name instead of cscope.out.
-h  This help screen.
-I  incdir Look in incdir for any #include files.
-i  namefile Browse through files listed in namefile, instead of cscope.files
-k  Kernel Mode – don’t use /usr/include for #include files.
-L  Do a single search with line-oriented output.
-l  Line-oriented interface.
-num  pattern Go to input field num (counting from 0) and find pattern.
-P  path Prepend path to relative file names in pre-built cross-ref file.
-p  n Display the last n file path components.
-q  Build an inverted index for quick symbol searching.
-R  Recurse directories for files.
-s  dir Look in dir for additional source files.
-T  Use only the first eight characters to match against C symbols.
-U  Check file time stamps.
-u  Unconditionally build the cross-reference file.
-v  Be more verbose in line mode.
-V  Print the version number.
```

Convert text files with DOS or MAC line breaks to Unix line breaks

<pre class="terminal"><code>$ sudo dos2unix /path/to/file
sudo sed -i -e 's/\r$//' /path/to/file</code></pre>

Print graphics card version:

<pre class="terminal"><code>$ nvcc --version</code></pre>

Print graphics card GPU info: 

<pre class="terminal"><code>$ nvidia-smi</code></pre>

Comment multi-lines in Matlab: Ctrl+R, Ctrl+T

Launch Matlab:

<pre class="terminal">
<code>$ cd /usr/local/bin/
$ sudo ln -s /usr/local/MATLAB/R2012a/bin/matlab Matlab
$ gedit ~/.bashrc
$ alias matlab="/usr/local/MATLAB/R2012a/bin/matlab"
</code></pre>

Create new file list:

<pre class="terminal"><code>$ sed 's?^?'`pwd`'/detection_images/?; s?$?.jpg?' trainval.txt > voc.2007trainval.list</code></pre>

**Ctags**:

<code>ctags –R * </code>: Generate tags files in source code root directory

<code>vim -t func/var</code>: find func/var definition

<code>:ts</code>: give a list if func/var has multiple definitions

<code>Ctrl+]</code>: jump to definition

<code>Ctrl+T</code>: jump back
