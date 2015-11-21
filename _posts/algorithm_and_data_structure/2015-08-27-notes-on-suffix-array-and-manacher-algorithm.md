---
layout: post
category: algorithm_and_data_structure
title: Notes on Suffix Array and Manacher Algorithm
date: 2015-08-27
---

Recently I am working on some data structure questions, and I kind of
dive into the "Longest Palindromic Substring" problem, master(do I?) the Suffix
Array and Manacher algorithm. I found that they are impressively amazing and have something in common.

**What is Suffix Array?**

A suffix array is a sorted array of all suffixes of a string. For example, "abracadabra":

$$
\\newcommand\\T{\\Rule{0pt}{1em}{.3em}}
\\begin{array}{|c|c|c|}
\\hline
  \\text{i}  & \\text{sa[i]} & \\text{S[sa[i]...]}    \\T \\\\\\hline
  \\text{0}  & \\text{11}    & \\text{(empty string)} \\\\\\hline
  \\text{1}  & \\text{10}    & \\text{a}              \\\\\\hline
  \\text{2}  & \\text{7}     & \\text{abra}           \\\\\\hline
  \\text{3}  & \\text{0}     & \\text{abracadabra}    \\\\\\hline
  \\text{4}  & \\text{3}     & \\text{acadabra}       \\\\\\hline
  \\text{5}  & \\text{5}     & \\text{adabra}         \\\\\\hline
  \\text{6}  & \\text{8}     & \\text{bra}            \\\\\\hline
  \\text{7}  & \\text{1}     & \\text{bracadabra}     \\\\\\hline
  \\text{8}  & \\text{4}     & \\text{cadabra}        \\\\\\hline
  \\text{9}  & \\text{6}     & \\text{dabra}          \\\\\\hline
  \\text{10} & \\text{9}     & \\text{ra}             \\\\\\hline
  \\text{11} & \\text{2}     & \\text{racadabra}      \\\\\\hline
\\end{array}
$$

**How to compute suffix array efficiently?**

Manber and Myers invented a algorithm with runtime complexity \\(O(n log^{2} n)\\). The basic idea is *doubling*. Start with the character at each location, we first compute the orders of substrings with length 2 at each location, then use the results to compute the orders of substrings with length 4.  The assessed prefix length doubles in each iteration of the algorithm until a prefix is unique (or length >= n) and provides the rank of the associated suffix.

**What is Manacher Algorithm?**
