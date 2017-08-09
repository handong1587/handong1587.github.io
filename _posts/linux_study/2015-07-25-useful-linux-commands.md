---
layout: post
category: linux_study
title: Useful Linux Commands
date: 2015-07-25
---

# Pack/Unpack/Compress/Uncompress

|file type    |  Pack/Compress                      | Unpack/Uncompress           |
|:-----------:|:-----------------------------------:|:---------------------------:|
|.tar         |  tar cvf FileName.tar DirName       |  tar xvf FileName.tar       |
|.gz          |  gzip FileName                      |  gunzip FileName.gz         |
|.gz          |  gzip FileName                      |  gzip -d FileName.gz        |
|.tar.gz, .tgz|  tar zcvf FileName.tar.gz DirName   |  tar zxvf FileName.tar.gz   |
|.bz2         |  bzip2 -z FileName                  |  bzip2 -d FileName.bz2      |
|.bz2         |                                     |  bunzip2 FileName.bz2       |
|.tar.bz2     |  tar jcvf FileName.tar.bz2 DirName  |  tar jxvf FileName.tar.bz2  |
|.bz          |                                     |  bzip2 -d FileName.bz       |
|.bz          |                                     |  bunzip2 FileName.bz        |
|.tar.bz      |                                     |  tar jxvf FileName.tar.bz   |
|.Z           |  compress FileName                  |  uncompress FileName.Z      |
|.tar.Z       |  tar Zcvf FileName.tar.Z DirName    |  tar Zxvf FileName.tar.Z    |
|.zip         |  zip FileName.zip DirName           |  unzip FileName.zip         |
|.zip         |  zip -r target.zip dir1 dir2 dir3   |  unzip FileName.zip -d targetFolder |
|.rar         |  rar a FileName.rar DirName         |  rar x FileName.rar         |
|.lha         |  lha -a FileName.lha FileName       |  lha -e FileName.lha        |
|.rpm         |                                     |  rpm2cpio FileName.rpm \| cpio -div |
|.deb         |                                     |  ar x example.deb           |
|.deb         |                                     |  ar p FileName.deb data.tar.gz \| tar zx |
|.deb         |                                     |  dpkg -x somepackage.deb ~/temp/ |
|.xz          |                                     |  xz -d myfiles.tar.xz       |
|.xz          |                                     |  tar -xf myfiles.tar        |
|.xz          |                                     |  tar -Jxf myfiles.tar.xz    |
|.7z          |  7za a myfiles.7z myfiles/          |  7za x myfiles.7z           |

For compress/uncompress those files:

.tar .tgz .tar.gz .tar.Z .tar.bz .tar.bz2 .zip .cpio .rpm .deb .slp .arj .rar .ace .lha .lzh .lzx .lzs .arc .sda .sfx .lnx .zoo .cab .kar .cpt .pit .sit .sea

```
compress:   sEx a FileName.* FileName
uncompress: sEx x FileName.*
```

## Wiew a detailed table of contents for an archive

|file type    |  view contents cmd                  |
|:-----------:|:-----------------------------------:|
|.tar.gz      |  tar -tvf my-data.tar.gz            |

# Print info

| task                                              | command                 |
| :-----------------------------                    | :---------------------: |
| Print system info                                 | cat /proc/version       |
| Print software info                               | whereis SOFEWARE        |
|                                                   | which SOFEWARE          |
|                                                   | locate SOFEWARE         |
| Print CPU info                                    | cat /proc/cpuinfo       |
|                                                   | dmesg \| grep -i xeon   |
| Print memory info                                 | cat /proc/meminfo       |
|                                                   | free -m                 |
| Print pid info                                    | ps aux \| grep 'target_pid' |
| Print graphics card version                       | nvcc --version          |
| Print graphics card GPU info                      | nvidia-smi              |
| Print disk free space                             | df -h                   |
|                                                   | df -hl                  |
| Print current folder size                         | du -sh DIRNAME          |
| Print target folder volume                        | du -sh                  |
| Print target folder volume (in MB)                | du -sm                  |
| Prints one entry per line of output (bare format) | ls -1a                  |

CuDNN Version Check:

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

Print lines 20 to 40:

```
sed -n '20,40p' file_name
```

or:

```
sed -n '20,40p;41q' file_name
```

or:

```
awk 'FNR>=20 && FNR<=40' file_name
```

To print range with other specific line (5 - 8 & 10):

```
$ sed -n -e 5,8p -e 10p file
Line 5
Line 6
Line 7
Line 8
Line 10
```

# Download

Download file:

```
wget "http://domain.com/directory/4?action=AttachFile&do=view&target=file.tgz"
```

Download file to specific directory:

```
wget -O /home/omio/Desktop/ "http://thecanadiantestbox.x10.mx/CC.zip"
```

Download all files from a folder on a website or FTP:

```
wget -r --no-parent --reject "index.html*" http://vision.cs.utexas.edu/voc/
```

# Transfer Files

Remote transfer files, remote -> local:

```
scp account@111.111.111.111:/path/to/remote/file /path/to/local/
```

Remote transfer files, local -> remote:

```
scp /path/to/local/file account@111.111.111.111:/path/to/remote/
```

Print directory structure:

```
ls -l -R
```

Print folder in current directory:

```
ls -lF |grep /
```

```
ls -l |grep '^d'
```


Print history command:

```
history
```

```
history | less
```

# Rename

Perl-based rename commands (-n: test commands; -v: print renamed files):

```
rename -n 's/\.htm$/\.html/' \*.htm
```

```
rename -v 's/\.htm$/\.html/' \*.htm
```

**1. Replace first letter of all files' name with 'q':**

```
for i in `ls`; do mv -f $i `echo $i | sed 's/^./q/'`; done
```

**same with a bash script:**

```
for file in `ls`
do
  newfile =`echo $i | sed 's/^./q/'`
　mv $file $newfile
done
```

**2. Replace first 5 letters with 'abcde'**

```
for i in `ls`; do mv -f $i `echo $i | sed 's/^...../abcde/'`;
```

**3. Replace last 5 letters with 'abcde'**

```
for i in `ls`; do mv -f $i `echo $i | sed 's/.....$/abcde/'`;
```

**4. Add 'abcde' to the front**

```
for i in `ls`; do mv -f $i `echo "abcde"$i`; done
```

**5. Convert all lower case to upper case**

```
for i in `ls`; do mv -f $i `echo $i | tr a-z A-Z`; done
```

# Count

Count lines in a document

```
wc -l /dir/file.txt
```

```
cat /dir/file.txt | wc -l
```

Filter and count only lines with pattern, or with -v to invert match

```
grep -w "pattern" -c file
```

```
grep -w "pattern" -c -v file
```

Count files in the current directory:

```
ls -l | grep “^-” | wc -l
```

example: counting all js files in directory "/home/account/" (recursively):

```
ls -lR /home/account | grep js | wc -l
```

```
ls -l "/home/account" | grep "js" | wc -l
```

Count files in the current directory, recursively:

```
ls -lR | grep “^-” | wc -l
```

Count folders in the current directory, recursively:

```
ls -lR | grep “^d” | wc -l
```

# Search

**Search for a file by its file name**

The command below will search for the query in the current directory and any subdirectories.
Using -iname instead of -name ignores the case of your query. The -name command is case-sensitive.

```
find -iname "filename"
```

**Finding all files containing a text string**

```
grep -rnw '/path/to/somewhere/' -e "pattern"

grep --include=\*.{c,h} -rnw '/path/to/somewhere/' -e "pattern"

grep --exclude=*.o -rnw '/path/to/somewhere/' -e "pattern"

grep --exclude-dir={dir1,dir2,*.dst} -rnw '/path/to/somewhere/' -e "pattern"
```

1. -r or -R is recursive,
2. -n is line number, and
3. -w stands match the whole word.
4. -l (lower-case L) can be added to just give the file name of matching files.
5. Along with these, --exclude or --include parameter could be used for efficient searching.

**Finding all files containing a text string on Linux**

- stackoverflow: [http://stackoverflow.com/questions/16956810/finding-all-files-containing-a-text-string-on-linux](http://stackoverflow.com/questions/16956810/finding-all-files-containing-a-text-string-on-linux)

Count occurrences of a char(e.g, 'aaa') in plain text file

```
fgrep -o 'aaa' <file> | wc -l
```

**references**

**How to Find a File in Linux**

[http://www.wikihow.com/Find-a-File-in-Linux](http://www.wikihow.com/Find-a-File-in-Linux)

**How to find files in Linux using 'find'**

[http://www.codecoffee.com/tipsforlinux/articles/21.html](http://www.codecoffee.com/tipsforlinux/articles/21.html)

**35 Practical Examples of Linux Find Command**

[http://www.tecmint.com/35-practical-examples-of-linux-find-command/](http://www.tecmint.com/35-practical-examples-of-linux-find-command/)

**How to use grep to search for strings in files on the shell**

[https://www.howtoforge.com/tutorial/linux-grep-command/](https://www.howtoforge.com/tutorial/linux-grep-command/)

**Find All Files of a Particular Size**

```
find /home/ -type f -size +512k -exec ls -lh {} \;
```

As units you can use:
```
    b – for 512-byte blocks (this is the default if no suffix is used)
    c – for bytes
    w – for two-byte words
    k – for Kilobytes (units of 1024 bytes)
    M – for Megabytes (units of 1048576 bytes)
    G – for Gigabytes (units of 1073741824 bytes)
```

ref: [http://www.ducea.com/2008/02/12/linux-tips-find-all-files-of-a-particular-size/](http://www.ducea.com/2008/02/12/linux-tips-find-all-files-of-a-particular-size/)

# GDB

**GDB reference card**

[https://web.stanford.edu/class/cs107/gdb_refcard.pdf](https://web.stanford.edu/class/cs107/gdb_refcard.pdf)

**GDB Tutorial: A Walkthrough with Examples**

- slides: [https://www.cs.umd.edu/~srhuang/teaching/cmsc212/gdb-tutorial-handout.pdf](https://www.cs.umd.edu/~srhuang/teaching/cmsc212/gdb-tutorial-handout.pdf)

**GNU GDB Debugger Command Cheat Sheet**

[http://www.yolinux.com/TUTORIALS/GDB-Commands.html](http://www.yolinux.com/TUTORIALS/GDB-Commands.html)

**Cheat Sheet**:

| shortcut   | command       | explanation      |
|:----------:|:-------------:|:----------------:|
| q          | quit          |                  |
| h          | help          |                  |
| h command  | help command  | print command meaning|
| b          | break         | example: b 5     |
| disable codenum|           |                  |
| enable codenum |           ||
| condition codenum xxx|     | set break condition|
| c          | continue      |                  |
| n          | next          |                  |
| s          | step          |                  |
| w          |               | print code in current execution point|
| j codenum  |               | jump to line j   |
| l          |               | list nearby code |
| p          |               | print var value  |
| a          |               | print current func/var value|
| Enter      |               | repeat last command|

```
gdb --arg python demo.py 2>&1 | tee demo.log
```

# pdb

```
import pdb
pdb.set_trace()
```

**exit pdb and allow program to continue**

1. To remove the breakpoint (if inserted it manually):

    ```
    (Pdb) break
    Num Type         Disp Enb   Where
    1   breakpoint   keep yes   at /path/to/test.py:5
    (Pdb) clear 1
    Deleted breakpoint 1
    (Pdb) continue
    ```

2. if you're using pdb.set_trace(), you can try this (although if you're using pdb in more fancy ways, this may break things...)

    ```
    (Pdb) pdb.set_trace = lambda: None  # This replaces the set_trace() function!
    (Pdb) continue
    # No more breaks!
    ```

- ref: [http://stackoverflow.com/questions/17820618/how-to-exit-pdb-and-allow-program-to-continue](http://stackoverflow.com/questions/17820618/how-to-exit-pdb-and-allow-program-to-continue)

# Ctags

| command         | explanation                                       |
|:---------------:|:-------------------------------------------------:|
| ctags –R *      | Generate tags files in source code root directory |
| vim -t func/var | find func/var definition                          |
| :ts             | give a list if func/var has multiple definitions  |
| Ctrl+]          | jump to definition                                |
| Ctrl+T          | jump back                                         |

# screen

| task                        | command                       |
|:---------------------------:|:-----------------------------:|
| Run program in screen mode  | screen python demo.py --gpu 1 |
| Detach screen               | Ctrl + a, c                   |
| Detach screen               | Ctrl + a, d                   |
| Re-connect screen           | screen -r pid                 |
| Display all screens         | screen -ls                    |
| Delete screen               | kill pid                      |
| Naming a screen             | screen -S sessionName         |

# nohup

```
nohup command-with-options &
```

```
nohup xxx.sh 1 > log.txt 2>&1 &
```

**nohup - get the process ID to kill a nohup process**

```
nohup command-with-options & 
echo $! > save_pid.txt
kill -9 `cat save_pid.txt`
```

# cscope

| task                              | command                         |
|:---------------------------------:|:-------------------------------:|
| Generate Cscope database          | find . -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.py" -o -name "*.proto" > cscope.files |
| Build a Cscope reference database | cscope -q -R -b -i cscope.files |
| Start the Cscope browser          | cscope -d                       |
| Exit a Cscope browser             | Ctrl + d                        |

**Cheat Sheet**

```
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

# Vim

**Shifting blocks visually**

[http://vim.wikia.com/wiki/Shifting_blocks_visually](http://vim.wikia.com/wiki/Shifting_blocks_visually)

| mode        | task                      | command   |
|:-----------:|:-------------------------:|:---------:|
| normal mode | indent the current line   | type \>\> |
| normal mode | unindent the current line | type \<\< |
| insert mode | indent the current line   | Ctrl-T    |
| insert mode | unindent the current line | Ctrl-D    |

For all commands, pressing **.** repeats the operation.

For example, typing **5>>..** shifts five lines to the right, and then repeats
the operation twice so that the five lines are shifted three times.

Insert current file name:

```
:r! echo %
```

**Insert characters at specific lines head**

```
:80,90s/^/#/
```

**Switch windows**

```
gt
```

**Fold code block under spf-13 Vim**

```
za
```

# Matlab

Comment multi-lines in Matlab: Ctrl+R, Ctrl+T

Launch Matlab:

```
cd /usr/local/bin/
sudo ln -s /usr/local/MATLAB/R2012a/bin/matlab Matlab
gedit ~/.bashrc
alias matlab="/usr/local/MATLAB/R2012a/bin/matlab"
```

Start MATLAB Without Desktop:

```
matlab -nojvm -nodisplay -nosplash
```

Matlab + nohup:

runGenerareSSProposals.sh:

```
#!/bin/sh
cd /path/to/detection-proposals
matlab -nojvm -nodisplay -nosplash -r "startup; callRunCOCO; exit"
```

runNohup.sh:

```
time=`date +%Y%m%d_%H%M%S`
cd /path/to/detection-proposals
nohup ./runGenerareSSProposals.sh > runGenerareSSProposals_${time}.log 2>&1 &
echo $! > save_runGenerareSSProposals_val_pid.txt
```

# Others

**Hotkeys to speed up Linux CLI navigation:**

| hotkey    |                                                                                                              |
| :------:  | :----------------------------------------------------------------------------------------------------------: |
| Ctrl + a  | go to the start of the command line                                                                          |
| Ctrl + e  | go to the end of the command line                                                                            |
| Ctrl + k  | delete from cursor to the end of the command line                                                            |
| Ctrl + u  | delete from cursor to the start of the command line                                                          |
| Ctrl + w  | delete from cursor to start of word (i.e. delete backwards one word)                                         |
| Ctrl + y  | paste word or text that was cut using one of the deletion shortcuts (such as the one above) after the cursor |
| Ctrl + xx | move between start of command line and current cursor position (and back again)                              |
| Alt + b   | move backward one word (or go to start of word the cursor is currently on)                                   |
| Alt + f   | move forward one word (or go to end of word the cursor is currently on)                                      |
| Alt + d   | delete to end of word starting at cursor (whole word if cursor is at the beginning of word)                  |
| Alt + c   | capitalize to end of word starting at cursor (whole word if cursor is at the beginning of word)              |
| Alt + u   | make uppercase from cursor to end of word                                                                    |
| Alt + l   | make lowercase from cursor to end of word                                                                    |
| Alt + t   | swap current word with previous                                                                              |
| Ctrl + f  | move forward one character                                                                                   |
| Ctrl + b  | move backward one character                                                                                  |
| Ctrl + d  | delete character under the cursor                                                                            |
| Ctrl + h  | delete character before the cursor                                                                           |
| Ctrl + t  | swap character under cursor with the previous one                                                            |

Launch terminal in Ubuntu: Ctrl+Alt+T

Create symbol link:

```
ln -s EXISTING_FILE SYMLINK_FILE
ln -s /path/to/file /path/to/symlink
```

Open image:

```
eog /path/to/image/im.jpg
```

```
display /path/to/image/im.jpg
```

Convert text files with DOS or MAC line breaks to Unix line breaks:

```
sudo dos2unix /path/to/file
```

or:

```
sed -i 's/\r//' /path/to/file
```

```
sudo sed -i -e 's/\r$//' /path/to/file
```

Create new file list:

```
sed 's?^?'`pwd`'/detection_images/?; s?$?.jpg?' trainval.txt > voc.2007trainval.list
```

**Merge two files consistently line by line**

```
paste -d " " file1.txt file2.txt
```

[http://stackoverflow.com/questions/16394176/how-to-merge-two-files-consistently-line-by-line](http://stackoverflow.com/questions/16394176/how-to-merge-two-files-consistently-line-by-line)

**shuffle file lines**

```
shuf file.list > file_shuffled.list
```

**Combine multiple files into one file**

```
cat file1 file2 file3 .... >> merged_file
```

**Show all hidden characters**:

```
cat -A filename
```

**Get recursive full-path listing**

```
find /path/to/folder
```

want files only (omit directories, devices, etc):

```
find /path/to/folder -type f
```

**Remove specific file types**

```
rm `find -type f /path/to/dir/ | grep "filetype"`
```

```
rm `find -type f /path/to/dir/ | grep -E "filetype1 | filetype2"`
```

**Zip multiple files/folers to one named zip file**

```
zip -r target.zip file1 file2 folder1 folder2
```

**Remove Windows format line breaks**

```
sed -i 's/^M$//g'
```

Note:

^M = Ctrl+v,Ctrl+m

**Replace tab characters with spaces**

```
sed -i 's/^I//g'
```

Note:

^I = Ctrl+v,Ctrl+I

**Remove duplicate text lines**

```
sort {file-name} | uniq -u
```

**Remove duplicate text lines and only keep one line**
```
perl -lne '$seen{$_}++ and next or print;' data.txt > output.txt
```

**Exit a shell if some commands do not execute correctly**

```
./do_something.sh || exit 1
```
