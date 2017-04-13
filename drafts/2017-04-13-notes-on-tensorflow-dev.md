---
layout: post
category: deep_learning
title: Notes On Tensorflow Development
date: 2017-04-13
---

# Install Bazel on Ubuntu 14.04

**Install with installer:**

[https://bazel.build/versions/master/docs/install-ubuntu.html#install-with-installer](https://bazel.build/versions/master/docs/install-ubuntu.html#install-with-installer)

Official instruction is simple:

```
sudo apt-get install openjdk-8-jdk
```

But due to network/proxy problem I cannot make it work. 
After some goolings I found a work-around if someone else also fails on installing openjdk-8-jdk following above command: intalling Oracle's jdk.

Step 1: Download jdk8

I choose to use jdk-8u121-linux-x64.tar.gz:

[http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

Step 2: Uncompression and Copy to one target directory:

```
sudo mkdir /usr/lib/jvm
sudo tar -zxvf jdk-8u121-linux-x64.tar.gz -C /usr/lib/jvm
```

Step 3: Modify some system PATHs:

Add following lines to your ~/.bashrc :

```
#set oracle jdk environment @20170413
export JAVA_HOME=/usr/lib/jvm/jdk1.8.0_121
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH
```

Don't forget to enable these settings:

```
source ~/.bashrc
```

Step 4: Set system default jdk version:

```
jdk_ver=jdk1.8.0_121

sudo update-alternatives --install /usr/bin/java java /usr/lib/jvm/${jdk_ver}/bin/java 300
sudo update-alternatives --install /usr/bin/javac javac /usr/lib/jvm/${jdk_ver}/bin/javac 300
sudo update-alternatives --install /usr/bin/jar jar /usr/lib/jvm/${jdk_ver}/bin/jar 300
sudo update-alternatives --install /usr/bin/javah javah /usr/lib/jvm/${jdk_ver}/bin/javah 300
sudo update-alternatives --install /usr/bin/javap javap /usr/lib/jvm/${jdk_ver}/bin/javap 300

sudo update-alternatives --config java
```

You might get similar outputs like below:

```
my_account@node7:~/my_account/sw$ sudo ./set_system_default_jdk_version.sh
update-alternatives: using /usr/lib/jvm/jdk1.8.0_121/bin/javac to provide /usr/bin/javac (javac) in auto mode
update-alternatives: using /usr/lib/jvm/jdk1.8.0_121/bin/jar to provide /usr/bin/jar (jar) in auto mode
update-alternatives: using /usr/lib/jvm/jdk1.8.0_121/bin/javah to provide /usr/bin/javah (javah) in auto mode
update-alternatives: using /usr/lib/jvm/jdk1.8.0_121/bin/javap to provide /usr/bin/javap (javap) in auto mode
There are 2 choices for the alternative java (providing /usr/bin/java).

  Selection    Path                                Priority   Status
------------------------------------------------------------
* 0            /usr/bin/gij-4.8                     1048      auto mode
  1            /usr/bin/gij-4.8                     1048      manual mode
  2            /usr/lib/jvm/jdk1.8.0_121/bin/java   300       manual mode

Press enter to keep the current choice[*], or type selection number: 2
update-alternatives: using /usr/lib/jvm/jdk1.8.0_121/bin/java to provide /usr/bin/java (java) in manual mode
```

Run `java -version`, you will get following output:

```
my_account@node7:~/my_account/sw$ java -version
java version "1.8.0_121"
Java(TM) SE Runtime Environment (build 1.8.0_121-b13)
Java HotSpot(TM) 64-Bit Server VM (build 25.121-b13, mixed mode)
```

Step 5:

And finally you can successfully execute below command:

```
bazel-0.4.5-installer-linux-x86_64.sh --user
```

It would be useful to add following line to your ~/.bashrc :
```
# for Bazel
export PATH="$PATH:$HOME/bin"
```

Done.
