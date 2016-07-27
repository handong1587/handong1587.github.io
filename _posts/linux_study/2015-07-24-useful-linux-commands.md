---
layout: post
category: linux_study
title: Useful Linux Commands
date: 2015-07-24
---

## Count

Count lines in a document

```
$ wc -l /dir/file.txt
```

```
$ cat /dir/file.txt | wc -l
```

Filter and count only lines with pattern, or with -v to invert match

```
$ grep -w "pattern" -c file
```

```
$ grep -w "pattern" -c -v file
``` 

Count files in the current directory:

```
$ ls -l | grep “^-” | wc -l
```

example: counting all js files in directory "/home/account/" (recursively):

```
$ ls -lR /home/account | grep js | wc -l
```

```
$ ls -l "/home/account" | grep "js" | wc -l
```

Count files in the current directory, recursively:

```
$ ls -lR | grep “^-” | wc -l
```

Count folders in the current directory, recursively:

```
$ ls -lR | grep “^d” | wc -l
```

## Transfer Files

Remote transfer files, remote -> local:

```
$ scp account@111.111.111.111:/path/to/remote/file /path/to/local/
```

Remote transfer files, local -> remote:

```
$ scp /path/to/local/file account@111.111.111.111:/path/to/remote/
```

Print directory structure:

```
$ ls -l -R
```

Print folder in current directory:

```
$ ls -lF |grep /
```

```
$ ls -l |grep '^d'
```


Print history command:

```
$ history
```

```
$ history | less
```

## Print info

Print system info:

```
$ cat /proc/version
```

Print software info:

```
$ whereis SOFEWARE
$ which SOFEWARE
$ locate SOFEWARE
```

Print CPU info:

```
$ cat /proc/cpuinfo
```

```
dmesg |grep -i xeon
```

Print memory info:

```
$ cat /proc/meminfo
```

```
$ free -m
```

Print graphics card version:

```
$ nvcc --version
```

Print graphics card GPU info:

```
$ nvidia-smi
```

Print disk free space:

```
$ df -h
$ df -hl
```

Print current folder size

```
$ du -sh DIRNAME
```

Print target folder volume (in MB)

```
$ du -sm
```

## Download

Download file:

```
$ wget "http://domain.com/directory/4?action=AttachFile&do=view&target=file.tgz"
```

Download file to specific directory:

```
$ wget -O /home/omio/Desktop/ "http://thecanadiantestbox.x10.mx/CC.zip"
```

Download all files from a folder on a website or FTP:

```
$ wget -r --no-parent --reject "index.html*" http://vision.cs.utexas.edu/voc/
```

Perl-based rename commands(-n: test commands; -v: print renamed files):

```
$ rename -n 's/\.htm$/\.html/' \*.htm
```

```
$ rename -v 's/\.htm$/\.html/' \*.htm
```

## Screen

Run program in screen mode:

```
$ screen python demo.py --gpu 1
```

Detach screen: Ctrl + a, c

Re-connect screen:

```
$ screen -r pid
```

Display all screens:

```
$ screen -ls
```

Delete screen:

```
$ kill pid
```

Naming a screen:

```
$ screen -S sessionName
```

## Ctags

`ctags –R * `: Generate tags files in source code root directory

`vim -t func/var`: find func/var definition

`:ts`: give a list if func/var has multiple definitions

`Ctrl+]`: jump to definition

`Ctrl+T`: jump back

## nohup

```$ nohup command-with-options &```

```$ nohup xxx.sh 1 > log.txt 2>&1 &```

**nohup - get the process ID to kill a nohup process**

```$ nohup command-with-options &```

```$ echo $! > save_pid.txt```

```$ kill -9 `cat save_pid.txt````

## Others

Launch terminal in Ubuntu: Ctrl+Alt+T

Create symbol link:

```$ ln -s EXISTING_FILE SYMLINK_FILE
$ ln -s /path/to/file /path/to/symlink```

Open image:

```$ eog /path/to/image/im.jpg```
```$ display /path/to/image/im.jpg```

Convert text files with DOS or MAC line breaks to Unix line breaks

```
$ sudo dos2unix /path/to/file
$ sudo sed -i -e 's/\r$//' /path/to/file
```

Create new file list:

```
$ sed 's?^?'`pwd`'/detection_images/?; s?$?.jpg?' trainval.txt > voc.2007trainval.list
```