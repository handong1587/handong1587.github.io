---
layout: page
permalink: /linux_svn/
---

**1. Create a repository:**

```bash
svnadmin create /svn/foo/mydirname
```

**2. Want to version control /home/user/mydirname:**

```bash
cd /home/user/mydirname
```

**3. This only creates the ".svn" folder for version control:**

```bash
svn co file:///svn/foo/mydirname .
```

**4. Tell svn you want to version control all files in this directory:**

```bash
svn add ./*
```

**5. Check the files in:**

```bash
svn ci
```

**6. Check file in with comment:**

```bash
svn ci -m "your_comment"
```

**7. Checkout project**

```bash
cd /home/user/projectx
svn checkout file:///svnrepo/projectx .
```

**8. Show only the last 4 log entries(need to svn update first in working copy directory)**

```bash
svn log --limit 4
svn log -l 4
```

**9. Add all files**

```bash
svn add --force path/to/dir
svn add --force .
```

**10. Checkout specified revision**

```bash
svn checkout svn://somepath@1234 working-directory
svn checkout -r 1234 url://repository/path
```
