---
layout: post
category: linux_study
title: Useful GDB and PDB Commands
date: 2015-07-24
---

<code>gdb --arg python demo.py 2>&1 | tee demo.log</code>

{% highlight python %}
import pdb
pdb.set_trace()
{% endhighlight %}

**Shortcuts**:

<code>q</code>  quit

<code>h</code>  help

<code>b</code>  break, example: b 5

<code>h command</code>  print command meaning

<code>disable codenum</code>

<code>enable codenum</code>

<code>condition codenum xxx</code>  set break condition

<code>c</code>  continue

<code>n</code>  next

<code>s</code>  step

<code>w</code>  print code in current execution point

<code>j codenum</code>  jump to line j

<code>l</code>  list nearby code

<code>p</code>  print var value

<code>a</code>  print current func/var value

**Enter**  repeat last command
