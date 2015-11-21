---
layout: post
category: linux_study
title: Linux Matlab Commands
date: 2015-08-04
---

Comment multi-lines in Matlab: Ctrl+R, Ctrl+T

Launch Matlab:

<pre class="terminal">
<code>$ cd /usr/local/bin/
$ sudo ln -s /usr/local/MATLAB/R2012a/bin/matlab Matlab
$ gedit ~/.bashrc
$ alias matlab="/usr/local/MATLAB/R2012a/bin/matlab"
</code></pre>

Start MATLAB Without Desktop:

<pre class="terminal">
<code>$ matlab -nojvm -nodisplay -nosplash</code>
</pre>

Matlab + nohup:

runGenerareSSProposals.sh:

{% highlight bash %}
#!/bin/sh
cd /path/to/detection-proposals
matlab -nojvm -nodisplay -nosplash -r "startup; callRunCOCO; exit"
{% endhighlight %}

runNohup.sh:

{% highlight bash %}
time=`date +%Y%m%d_%H%M%S`
cd /path/to/detection-proposals
nohup ./runGenerareSSProposals.sh > runGenerareSSProposals_${time}.log 2>&1 &
echo $! > save_runGenerareSSProposals_val_pid.txt
{% endhighlight %}