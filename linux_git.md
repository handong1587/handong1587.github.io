---
layout: page
mathjax: true
permalink: /linux_svn/
---

**1. Push local modification to server**

```shell
git add deep learning/paper/reinforcement
git commit -m "xxxxxx"
git push -u origin master
```

**2. Solution to ERROR: "fatal: The remote end hung up unexpectedly"**

```shell
git config http.postBuffer 524288000
git config --global http.postBuffer 157286400
```

**3. Undo last two commits which not pushed yet**
**DANGEROUS: this will also delete relevant local files**

```shell
git reset --hard HEAD~2
```

**4. Undo commit, also roll-back codes to previous commit**

```shell
git reset --hard commit_id
```

**5. Undo commit, but won't undo local codes modification**
**can re-commit local changes by "git commit"**

```shell
git reset commit_id
```

**6. Only view how many non-pushed commits**

```shell
git status
```

**7. Only view comments/descriptions of non-pushed commits**

```shell
git cherry -v
```

**8. View detailed informations of non-pushed commits**

```shell
git log master ^origin/master
```

**9. Find id of last commit**
```shell
git log
```
