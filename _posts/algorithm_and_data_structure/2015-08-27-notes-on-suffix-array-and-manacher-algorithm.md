---
layout: post
categories: algorithm_and_data_structure
title: Notes on Suffix Array and Manacher Algorithm
---

{{ page.title }}
================

<p class="meta">27 Aug 2015 - Beijing</p>

Recently I am working on some data structure questions, and I kind of
dive into the "Longest Palindromic Substring" problem, master(do I?) the Suffix
Array and Manacher algorithm. I found that they are impressively amazing and have something in common.

**What is Suffix Array?**

A suffix array is a sorted array of all suffixes of a string. For example, "abracadabra":

i          |sa[i]           |S[sa[i]...]
:----------|:----------|:---------------
0          |11         |(empty string)
1          |10         |a
2          |7          |abra
3          |0          |abracadabra
4          |3          |acadabra
5          |5          |adabra
6          |8          |bra
7          |1          |bracadabra
8          |4          |cadabra
9          |6          |dabra
10         |9          |ra
11         |2          |racadabra

**How to compute suffix array efficiently?**

Manber and Myers invented a algorithm with runtime complexity \\(O(n log^{2} n)\\).
