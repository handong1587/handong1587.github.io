---
layout: post
category: programming_study
title: PyInstsaller and Others
date: 2016-12-24
---

# Quick introduction

I recently need to convert one Python program into binary mode program. 
That is, you don't want to expose any of your source code, data files, 
only one binary executable file will be provided.

[PyInstaller](http://www.pyinstaller.org/) is a fairly good choice to use, 
and can work on many platforms like Linux, Windows, etc.

You can check out its official git repository at 
[https://github.com/pyinstaller/pyinstaller](https://github.com/pyinstaller/pyinstaller).

It is recommended that first try out its officially, stable release -- 
but when something weird come just around, you can turn to the github dev branch for help -- actually that is what I did.

# hidden-import

There are 2 basic ways to process Python scripts. I chose to use pyinstaller.py directly,
although you can use *spec* file if you want.

When building Python scripts, you probably will get some build errors telling you that some Python packages cannot be imported.
Like:

```
ImportError: The 'packaging' package is required
```

```
ImportError: No module named core_cy
```

I might explain it in the future, but to put it simply, some Python packages need to be "hidden-import" to get around this issue.
So now we can setup a fundamental build script to help our work:

```
/path/to/git/pyinstaller/pyinstaller.py \
    --onefile \
    --hidden-import=skimage.io \
    --hidden-import=skimage.transform \
    --hidden-import=skimage.filter.rank.core_cy \
    --hidden-import=packaging \
    --hidden-import=packaging.version \
    --hidden-import=packaging.specifiers \
    --hidden-import=packaging.requirements \
    --hidden-import=scipy.linalg \
    --hidden-import=scipy.linalg.cython_blas \
    --hidden-import=scipy.linalg.cython_lapack \
    --hidden-import=scipy.ndimage \
    --hidden-import=skimage._shared.interpolation \
    --hidden-import=google.protobuf.internal \
    --hidden-import=google.protobuf.internal.enum_type_wrapper \
    --hidden-import=google.protobuf.descriptor \
    target_program.py
```

# What is wrong with MKL

One weird error I met was the Intel MKL FATAL ERROR:

```
Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.
```

Since I use anaconda, I find MKL has already been installed on the anaconda install location 
and can find these 2 files easily, but this error still pop out.
If I remember correctly, the solution is even more weird:
simply update numpy to a latest version:

```
conda update numpy
```

or:

```
conda install linux-64_numpy-1.11.2-py27_0.tar.bz2
```

I don't know what happened exactly but looks like it been fixed. Hmm...

# --add-data and _MEIPASS

PyInstaller can also bundle data files to your programs. When bundled app runs, 
it will load these data files, in a different location.
Here is a helper function to locate your data files:

```
def resource_path(relative):
    bundle_dir = os.environ.get("_MEIPASS2", os.path.abspath("."))
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        bundle_dir = sys._MEIPASS
    else:
        # we are running in a normal Python environment
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(bundle_dir, relative)
```

You can put your data file in your local directory, 
but need to specify the data file name in Python script in a right way:

```
target_file = resource_path('target_data_file1')
```

In build script, you need to configure the data files or folders:

   ```
    --add-data="target_data_file1:." \
    --add-data="target_data_file2:." \
    --add-data="folder1/sub_folder1/target_data_file3:folder1/sub_folder1/target_data_file3" \
   ```

# Missing libs

   ```
    --add-binary="libgfortran.so.1:lib" \
   ```

The build error told me one \*so file is required. So just add it.

# Config PYTHONPATH

Some of your Python scripts might depends on some relative path, 
so you will need to put this dependencies into the build script:

```
--paths="../dependency_folder" \
```

# Continue tackling weird stuffs

Util now it sounds like an easy task.
But what happened next consumed me about 2 days -- I wish I could have known how to avoid it :-(

My Python project includes A Caffe module which run a simple image classification process.
One basic function is [Caffe](https://github.com/BVLC/caffe) calling skimage.io to load image:

[https://github.com/BVLC/caffe/blob/master/python/caffe/io.py](https://github.com/BVLC/caffe/blob/master/python/caffe/io.py)

```
def load_image(filename, color=True):
    img = skimage.img_as_float(skimage.io.imread(filename, as_grey=not color)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img
```

I wonder if PyInstaller currently has a good support for Python package skimage.
But from what I know by now, it doesn't.

Run from Python source code files, it works fine. But when I packed all things into one single binary file,
it can not load image at all. And after debugging and googleing for a long time -- 
I always thought maybe I did something wrong -- I get rid of this. PyInstaller hates skimage! 
So at last I use cv2 instead. And it works smoothly.

```
def cv2_load_image(filename, color=True):
    img = cv2.imread(filename).astype(np.float32) / 255
    if img.ndim == 3:
        img[:,:,:] = img[:,:,2::-1]

    if img.ndim == 2:
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img
```

For all above details, please do check out PyInstaller Documentation: 
[https://media.readthedocs.org/pdf/pyinstaller/latest/pyinstaller.pdf](https://media.readthedocs.org/pdf/pyinstaller/latest/pyinstaller.pdf)

# Looks like we make it!

```
/path/to/git/pyinstaller/pyinstaller.py \
    --onefile \
    --hidden-import=skimage.io \
    --hidden-import=skimage.transform \
    --hidden-import=skimage.filter.rank.core_cy \
    --hidden-import=packaging \
    --hidden-import=packaging.version \
    --hidden-import=packaging.specifiers \
    --hidden-import=packaging.requirements \
    --hidden-import=scipy.linalg \
    --hidden-import=scipy.linalg.cython_blas \
    --hidden-import=scipy.linalg.cython_lapack \
    --hidden-import=scipy.ndimage \
    --hidden-import=skimage._shared.interpolation \
    --hidden-import=google.protobuf.internal \
    --hidden-import=google.protobuf.internal.enum_type_wrapper \
    --hidden-import=google.protobuf.descriptor \
    --add-binary="libgfortran.so.1:lib" \
    --add-data="target_data_file1:." \
    --add-data="target_data_file2:." \
    --add-data="folder1/sub_folder1/target_data_file3:folder1/sub_folder1/target_data_file3" \
    --paths="../dependency_folder" \
    target_program.py
```

# Misc

I just find a simple method to read/write binary file via Python: 
using cPickle to dump data to file in binary format.

```
import cPickle

a = ('img_path1', 1111, 222.222, 333, 444, 555, 6666)
b = ('img_path2', 777, 88.8888, 9999, 1010, 1111, 1212)
c = []
c.append(a)
c.append(b)

with open('wb_txt', 'wb') as f:
    cPickle.dump(c, f, cPickle.HIGHEST_PROTOCOL)

with open('wb_txt', 'rb') as f:
    data = cPickle.load(f)
    print data
```

Hopefully this note can guide someone new to PyInstaller like me to walk out of sloughy.
