---
layout: post
title: Add Lunr Search Plugin For Blog
date: 2016-07-31
category: "web_dev"
---

I decided to add a full-text search plugin to my blog:

[https://github.com/slashdotdash/jekyll-lunr-js-search](https://github.com/slashdotdash/jekyll-lunr-js-search) .

Although it should be an easy work, there are still some rules I think are somewhat crucial to follow (for me..).

First rule: DO NOT try to do this on Windows.

On windows (and OS X), you can not even manage to gem install therubyracer, which is essential component required by jekyll-lunr-js-search. 
See my previous post: 

[http://handong1587.github.io/web_dev/2016/07/03/install-therubyracer.html](http://handong1587.github.io/web_dev/2016/07/03/install-therubyracer.html)

Keep yourself aware that you don't include jQuery twice. It can really cause all sorts of issues.

This post explains in a more detail: 

**Double referencing jQuery deletes all assigned plugins.**

[https://bugs.jquery.com/ticket/10066](https://bugs.jquery.com/ticket/10066)

It kept me receiving one wired error like:

```
TypeError: $(...).lunrSearch is not a function
```

and took me a long time to find out why this happened.

For a newbie like me who *know nothing at all* about front-end web development, 
all the work become trial and error, and google plus stackoverflow. So great now it can work.

Thanks to *My Chemical Romance* for helping me through those tough debugging nights!