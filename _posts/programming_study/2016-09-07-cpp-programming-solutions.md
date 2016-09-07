---
layout: post
category: programming_study
title: C++ Programming Solutions
date: 2016-09-07
---

# Reference a nonstatic MFC class member in a static thread function

Declare a thread function:

```
static DWORD WINAPI ThreadFunc(LPVOID lpParameter);
```

Pass a `this` pointer to thread function:

```
HANDLE hThread = CreateThread(NULL, 0, ThreadFunc, this, 0, NULL);
```

In the thread function definition:

```
DWORD WINAPI CMFCDemoDlg::ThreadFunc(LPVOID lpParameter)
{
    //convert lpParameter to class pointer type
    CMFCDemoDlg* pMfcDemo = (CMFCDemoDlg*)lpParameter;

    // Now you can reference the CMFCDemoDlg class members
    ......
}
```
