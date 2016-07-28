---
layout: post
category: linux_study
title: Setup vsftpd on Ubuntu 14.10
date: 2016-07-27
---

# Setup vsftpd

Install vsftpd:

```
sudo apt-get install vsftpd
```

Check if vsftpd installed successfully:

```
sudo service vsftpd status
```

Add `/home/uftp` as user home directory:

```
sudo mkdir /data/jinbin.lin/uftp
```

Add user `uftp` and set password:

```
sudo useradd -d /data/jinbin.lin/uftp -s /bin/bash uftp
```

Set user password (need to enter password twice):

```
sudo passwd uftp
```

Edit vsftpd configuration file:

```
/etc/vsftpd.conf
```

Add following commands at the end of `vsftpd.conf`:

```
userlist_deny=NO
userlist_enable=YES
userlist_file=/etc/allowed_users
```

Modify following configurations:

```
local_enable=YES
write_enable=YES
```

Edit `/etc/allowed_users`，add username: uftp

Check file `/etc/ftpusers`, delete `uftp` (if file contains this username). 
This file recording usernames which are forbidden to access FTP server.

Restart vsftpd:

```
sudo service vsftpd restart
```

# Close FTP server

```
sudo service vsftpd stop
```

# Visit FTP server 

(By default, the anonymous user is disabled)

```
ftp://user:password@hostname/
```

# Forbid user access top level directory

Create file `vsftpd.chroot_list` but don't add anything:

```
sudo touch /etc/vsftpd.chroot_list
```

Modify configurations as following:

```
chroot_local_user=YES
chroot_list_enable=NO
chroot_list_file=/etc/vsftpd.chroot_list
```

If want to have write permission to user home directory (otherwise you would meet this error when login: 
"500 OOPS: vsftpd: refusing to run with writable root inside chroot ()"):

```
allow_writeable_chroot=YES
```

Restart vsftpd:

```
sudo service vsftpd restart
```

# Does not allow the user to change the specified chroot_list_file root

```
chroot_local_user=NO
chroot_list_enable=YES
chroot_list_file=/etc/vsftpd.chroot_list
```

# Allows only specified users to change chroot_list_file root

```
chroot_local_user=YES
chroot_list_enable=YES
chroot_list_file=/etc/vsftpd.chroot_list
```

# Frequently used command

`mkdir`

`dir` or `ls`

`put`

`get`

# Refs

**How to Install and Configure vsftpd on Ubuntu 14.04 LTS**

[http://www.liquidweb.com/kb/how-to-install-and-configure-vsftpd-on-ubuntu-14-04-lts/](http://www.liquidweb.com/kb/how-to-install-and-configure-vsftpd-on-ubuntu-14-04-lts/)

**vsftpd 配置:chroot_local_user与chroot_list_enable详解**

[http://blog.csdn.net/bluishglc/article/details/42398811](http://blog.csdn.net/bluishglc/article/details/42398811)