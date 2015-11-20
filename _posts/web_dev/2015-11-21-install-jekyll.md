---
layout: post
title: Install Jekyll To Fix Some Local Github-pages Defects
date: 2015-11-21 00:02:26
category: "web_dev"
---

# install jekyll

I follow the blog: [http://blog.csdn.net/itmyhome1990/article/details/41982625](http://blog.csdn.net/itmyhome1990/article/details/41982625) 
to install Ruby, Devkit, and Jekyll.

# intall github-pages

When I try to install github-pages by "gem install github-pages", an error(FetchError) is encountered: 

Seems like it is because the Ruby website is blocked. So I follow the instructions in 
[http://segmentfault.com/q/1010000003891086](http://segmentfault.com/q/1010000003891086), create a .gemrc file into
"C:\Users\MyName", and problem solved.

# try jekyll build

Follow [http://rockhong.github.io/github-pages-fails-to-update.html](http://rockhong.github.io/github-pages-fails-to-update.html)
to detect my github-pages defects.

OK, try the instruction below:

jekyll build --safe

Then I get:

<img src="/assets/web_dev/jekyll_build_reuslts.png"/>

Follow the error informations, do some minor changes, and finally my github-pages can successfully be shown.