---
layout: post
category: linux_study
title: Useful Vim Shortcuts
date: 2015-07-24
---

# Pwerful VIM config on Github

**spf13-vim: The Ultimate Vim Distribution**

![(http://i.imgur.com/kZWj1.png)]

- homepage: [http://vim.spf13.com/](http://vim.spf13.com/)
- github: [https://github.com/spf13/spf13-vim](https://github.com/spf13/spf13-vim)

**dot-vimrc: Maple's vim config files**

- github: [https://github.com/humiaozuzu/dot-vimrc](https://github.com/humiaozuzu/dot-vimrc)

**vimrc: The Ultimate vimrc**

- github: [https://github.com/amix/vimrc](https://github.com/amix/vimrc)

# Vim Shortcuts

**Shifting blocks visually**

[http://vim.wikia.com/wiki/Shifting_blocks_visually](http://vim.wikia.com/wiki/Shifting_blocks_visually)

In normal mode: type **>>** to indent the current line, or **<<** to unindent.

In insert mode, **Ctrl-T** indents the current line, and **Ctrl-D** unindents.

For all commands, pressing **.** repeats the operation.

For example, typing **5>>..** shifts five lines to the right, and then repeats
the operation twice so that the five lines are shifted three times.

Insert current file name:
```
:r! echo %
```