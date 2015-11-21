---
layout: post
category: linux_study
title: Useful Vim Shortcuts
date: 2015-07-24
---

**Some powerful VIM configuration projects on Github**:

- spf13-vim

[https://github.com/spf13/spf13-vim](https://github.com/spf13/spf13-vim)

- dot-vimrc

[https://github.com/humiaozuzu/dot-vimrc](https://github.com/humiaozuzu/dot-vimrc)

- vimrc

[https://github.com/amix/vimrc](https://github.com/amix/vimrc)

**Shortcuts**:

[Shifting blocks visually](http://vim.wikia.com/wiki/Shifting_blocks_visually)

In normal mode: type **>>** to indent the current line, or **<<** to unindent.

In insert mode, **Ctrl-T** indents the current line, and **Ctrl-D** unindents.

For all commands, pressing **.** repeats the operation.

For example, typing **5>>..** shifts five lines to the right, and then repeats
the operation twice so that the five lines are shifted three times.

Insert current file name: <code>:r! echo %</code>
