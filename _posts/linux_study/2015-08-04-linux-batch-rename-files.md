---
layout: post
categories: linux_study
title: Linux Batch Rename Commands
---

{{ page.title }}
================

<p class="meta">04 Aug 2015 - Beijing</p>

**1. Replace first letter of all files' name with 'q':**

```bash
for i in `ls`; do mv -f $i `echo $i | sed 's/^./q/'`; done
```

**same with a bash script:**

```bash
for file in `ls`
do
  newfile =`echo $i | sed 's/^./q/'`
　mv $file $newfile
done
```

**2. Replace first 5 letters with 'abcde'**

```bash
for i in `ls`; do mv -f $i `echo $i | sed 's/^...../abcde/'`;
```

**3. Replace last 5 letters with 'abcde'**

```bash
for i in `ls`; do mv -f $i `echo $i | sed 's/.....$/abcde/'`;
```

**4. Add 'abcde' to the front**

```bash
for i in `ls`; do mv -f $i `echo "abcde"$i`; done
```

**5. Convert all lower case to upper case**

```bash
for i in `ls`; do mv -f $i `echo $i | tr a-z A-Z`; done
```
