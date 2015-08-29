---
layout: post
categories: leisure
title: Image Showing Test
---

{{ page.title }}
================

<p class="meta">24 Jul 2015 - Beijing</p>

One way to show a image in markdown file:

<div class="fig figcenter fighighlight">
  <img src="/assets/leisure/girl.jpg">
  <div class="figcaption">Girl</div>
</div>

Way two:

<div style="float:center">
    <img src="/assets/leisure/girl_2.jpg">
</div>

Way three:

<div style="float:left;margin:0 10px 10px 0" markdown="1">
    <img src="/assets/leisure/girl_3.jpg">
</div>

Way four:

![alt text](/assets/leisure/Sugimoto_Yumi_15_1.jpg "Sugimoto Yumi")

Way five:

<img src="/assets/leisure/Ygritte.jpg"
alt="ygritte" title="She is not Ygritte" />

Way five with constrained width:

<img src="/assets/leisure/Ygritte.jpg"
alt="ygritte" title="She is not Ygritte" width="640" />
