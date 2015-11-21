---
layout: post
category: linux_study
title: Linux Batch Rename Commands
date: 2015-08-04
---

**1. Replace first letter of all files' name with 'q':**

{% highlight bash %}
for i in `ls`; do mv -f $i `echo $i | sed 's/^./q/'`; done
{% endhighlight %}

**same with a bash script:**

{% highlight bash %}
for file in `ls`
do
  newfile =`echo $i | sed 's/^./q/'`
　mv $file $newfile
done
{% endhighlight %}

**2. Replace first 5 letters with 'abcde'**

{% highlight bash %}
for i in `ls`; do mv -f $i `echo $i | sed 's/^...../abcde/'`;
{% endhighlight %}

**3. Replace last 5 letters with 'abcde'**

{% highlight bash %}
for i in `ls`; do mv -f $i `echo $i | sed 's/.....$/abcde/'`;
{% endhighlight %}

**4. Add 'abcde' to the front**

{% highlight bash %}
for i in `ls`; do mv -f $i `echo "abcde"$i`; done
{% endhighlight %}

**5. Convert all lower case to upper case**

{% highlight bash %}
for i in `ls`; do mv -f $i `echo $i | tr a-z A-Z`; done
{% endhighlight %}
