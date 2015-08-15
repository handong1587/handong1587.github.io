---
layout: page
mathjax: true
permalink: /linux_svn/
---

#### 1. Create a repository:
```shell
svnadmin create /svn/foo/mydirname
```
#### 2. Want to version control /home/user/mydirname:
```shell
cd /home/user/mydirname
```
#### 3. This only creates the ".svn" folder for version control:
```shell
svn co file:///svn/foo/mydirname .
```
#### 4. Tell svn you want to version control all files in this directory:
```shell
svn add ./*
```
#### 5. Check the files in:
```shell
svn ci
```
#### 6. Check file in with comment:
```shell
svn ci -m "your_comment"
```
#### 7. Checkout project
```shell
cd /home/user/projectx
svn checkout file:///svnrepo/projectx .
```
#### 8. Show only the last 4 log entries(need to svn update first in working copy directory)
```shell
svn log --limit 4
svn log -l 4
```
#### 9. Add all files
```shell
svn add --force path/to/dir
svn add --force .
```
#### 10. Checkout specified revision
```shell
svn checkout svn://somepath@1234 working-directory
svn checkout -r 1234 url://repository/path
```
