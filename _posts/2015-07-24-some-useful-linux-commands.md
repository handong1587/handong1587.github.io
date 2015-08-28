---
layout: post
title: Some Useful Linux Commands
---

{{ page.title }}
================

<p class="meta">24 Jul 2015 - Beijing</p>

Counting files in the current directory:

```bash
ls -l | grep “^-” | wc -l
```
example: counting all js files in directory "/home/account/" (recursively:)

```bash
ls -lR /home/account | grep js | wc -l
```
```bash
ls -l "/home/account" | grep "js" | wc -l
```

Counting files in the current directory, recursively:

```bash
ls -lR | grep “^-” | wc -l
```

Counting folders in the current directory, recursively:

```bash
ls -lR | grep “^d” | wc -l
```

Remote transfer files, remote -> local:

```bash
scp account@111.111.111.111:/path/to/remote/file /path/to/local/
```

Remote transfer files, local -> remote:

```bash
scp /path/to/local/file account@111.111.111.111:/path/to/remote/
```

Launch terminal in Ubuntu: Ctrl+Alt+T

Print system info:

```bash
cat /proc/version
```

Open image:

```bash
eog /path/to/image/im.jpg
```
```bash
display /path/to/image/im.jpg
```

Print directory structure:

```bash
ls -l -R
```

Print folder in current directory:

```bash
ls -lF |grep /
```
```bash
ls -l |grep '^d'
```

Print history command:

```bash
history
```
```bash
history | less
```

Print software info:

```bash
whereis SOFEWARE
```
```bash
which SOFEWARE
```
```bash
locate SOFEWARE
```

Print CPU info:

```bash
cat /proc/cpuinfo
```

```bash
dmesg |grep -i xeon
```

Print memory info:

```bash
cat /proc/meminfo
```

```bash
free -m
```

Print disk free space:

```bash
df -h
```
```bash
df -hl
```
Print current folder size

```bash
du -sh DIRNAME
```

Print target folder volume (in MB)

```bash
du -sm
```

Create symbol link:

```bash
ln -s EXISTING_FILE SYMLINK_FILE
```

```bash
ln -s /path/to/file /path/to/symlink
```

Download file:

```bash
wget "http://domain.com/directory/4?action=AttachFile&do=view&target=file.tgz"
```

Download file to specific directory:

```bash
wget -O /home/omio/Desktop/ "http://thecanadiantestbox.x10.mx/CC.zip"
```

Download all files from a folder on a website or FTP:

```bash
wget -r --no-parent --reject "index.html*" http://vision.cs.utexas.edu/voc/
```

Perl-based rename commands(-n: test commands; -v: print renamed files):

```bash
rename -n 's/\.htm$/\.html/' *.htm
```
```bash
rename -v 's/\.htm$/\.html/' *.htm
```

Run program in screen mode:

```bash
screen python demo.py --gpu 1
```

Detach screen: Ctrl + a, c

Re-connect screen:

```bash
screen -r pid
```

Display all screens:

```bash
screen -ls
```

Delete screen:

```bash
kill pid
```

Naming a screen:

```bash
screen -S sessionName
```

Generate Cscope database:

```bash
find . -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cu" 
-o -name "*.h" -o -name "*.hpp" -o -name "*.py" -o -name "*.proto" > cscope.files
```

Build a Cscope reference database:

```bash
cscope -q -R -b -i cscope.files
```

Start the Cscope browser:

```bash
cscope -d
```

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

```bash
sudo dos2unix /path/to/file
```
```bash
sudo sed -i -e 's/\r$//' /path/to/file
```

Print graphics card version:

```bash
nvcc --version
```

Print graphics card GPU info: nvidia-smi

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
