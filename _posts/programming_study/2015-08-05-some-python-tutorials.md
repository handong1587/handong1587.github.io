---
layout: post
categories: programming_study
title: Some Python Tutorials
---

{{ page.title }}
================

<p class="meta">05 Aug 2015 - Beijing</p>

**1. Draw rectangle:**

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(img, (20, 20), (411, 411), (55, 255, 155), 2)
plt.imshow(img, 'brg')
cv2.imwrite("/path/to/save/img.jpg", img)
```

**2. Use PIL to show ndarray:**

```python
import Image
import numpy as np
w,h = 512,512
data = np.zeros( (w,h,3), dtype=np.uint8)
data[256,256] = [255,0,0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')
```

**3. Convert numpy multi-dim ndarray to 1-dim array:**

```python
np.asarray(a).reshape(-1)
```

**4. Simple operations**

```python
os.getcwd()
os.chdir(path)
os.path.dirname(os.path.abspath(__file__))
```

```python
from operator import itemgetter  # sort
import numpy as np
np.amax
np.sum
```
