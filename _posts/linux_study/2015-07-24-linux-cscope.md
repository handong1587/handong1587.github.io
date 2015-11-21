---
layout: post
category: linux_study
title: Linux Cscope Commands
date: 2015-07-24
---

Generate Cscope database:

<pre class="terminal"><code>$ find . -name "*.c" -o -name "*.cc" -o -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" -o -name "*.py" -o -name "*.proto" > cscope.files</code></pre>

Build a Cscope reference database:

<pre class="terminal"><code>$ cscope -q -R -b -i cscope.files</code></pre>

Start the Cscope browser:

<pre class="terminal"><code>$ cscope -d</code></pre>

Exit a Cscope browser: Ctrl + d

Some Cscope parameters:

<pre class="terminal"><code>
-b  Build the cross-reference only.
-C  Ignore letter case when searching.
-c  Use only ASCII characters in the cross-ref file (don’t compress).
-d  Do not update the cross-reference.
-e  Suppress the -e command prompt between files.
-F  symfile Read symbol reference lines from symfile.
-f  reffile Use reffile as cross-ref file name instead of cscope.out.
-h  This help screen.
-I  incdir Look in incdir for any #include files.
-i  namefile Browse through files listed in namefile, instead of cscope.files
-k  Kernel Mode – don’t use /usr/include for #include files.
-L  Do a single search with line-oriented output.
-l  Line-oriented interface.
-num  pattern Go to input field num (counting from 0) and find pattern.
-P  path Prepend path to relative file names in pre-built cross-ref file.
-p  n Display the last n file path components.
-q  Build an inverted index for quick symbol searching.
-R  Recurse directories for files.
-s  dir Look in dir for additional source files.
-T  Use only the first eight characters to match against C symbols.
-U  Check file time stamps.
-u  Unconditionally build the cross-reference file.
-v  Be more verbose in line mode.
-V  Print the version number.
</code></pre>