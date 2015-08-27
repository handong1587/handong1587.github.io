---
layout: post
title: Some Useful Linux Commands
---

{{ page.title }}
================

<p class="meta">24 Jul 2015 - Beijing</p>

1. Counting files in the Current Directory:

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

2. Counting files in the current directory, recursively:

```bash
ls -lR | grep “^-” | wc -l
```

3. Counting folders in the current directory, recursively:

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

Launch terminal in Ubuntu:

Ctrl+Alt+T

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
