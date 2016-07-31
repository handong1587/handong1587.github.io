---
layout: post
category: linux_study
title: Create Multiple Forks of a GitHub Repo
date: 2015-12-18
---

# Step-1: Clone the original repo to your local machine

```
git clone https://github.com/handong1587/caffe.git caffe-yolo
```

![](/assets/linux_study/create_multi_fork_1.jpg)

```
cd caffe-yolo
```

# Step-2: Create a new empty repo in your GitHub account

![](/assets/linux_study/create_multi_fork_2.jpg)

# Step-3: Manually create the necessary remote links

```
git remote -v
```

![](/assets/linux_study/create_multi_fork_3.jpg)

# Step-4: Rename origin to upstream and add our new empty repo as the origin

```
git remote rename origin upstream
git remote add origin https://github.com/handong1587/caffe-yolo.git
git remote -v
```

![](/assets/linux_study/create_multi_fork_4.jpg)

# Step-5: Push from your local repo to your new remote one

```
git push -u origin master
```

![](/assets/linux_study/create_multi_fork_5.jpg)

Done.

# Reference

(In step-4 the author use a SSH method to "git remote add" while I can only use HTTPS method to finally succeed)

**Create multiple forks of a GitHub repo**

[https://adrianshort.org/create-multiple-forks-of-a-github-repo/](https://adrianshort.org/create-multiple-forks-of-a-github-repo/)