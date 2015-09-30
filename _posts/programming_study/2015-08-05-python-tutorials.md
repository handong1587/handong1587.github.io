---
layout: post
categories: programming_study
title: Python Tutorials
---

{{ page.title }}
================

<p class="meta">05 Aug 2015 - Beijing</p>

**Draw rectangle:**

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(img, (20, 20), (411, 411), (55, 255, 155), 2)
plt.imshow(img, 'brg')
cv2.imwrite("/path/to/save/img.jpg", img)
```

**Use PIL to show ndarray:**

```python
import Image
import numpy as np
w,h = 512,512
data = np.zeros( (w,h,3), dtype=np.uint8)
data[256,256] = [255,0,0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')
```

**Convert numpy multi-dim ndarray to 1-dim array:**

```python
np.asarray(a).reshape(-1)
```

**Remove duplicate elements**

```python
# Our input list.
values = [5, 5, 1, 1, 2, 3, 4, 4, 5]

# Convert to a set and back into a list.
set = set(values)
result = list(set)
print(result)
```

Output:

<pre class="terminal"><code>>>> [1, 2, 3, 4, 5] </code></pre>

**Simple operations**

```python
os.getcwd()
os.chdir(path)
os.path.dirname(os.path.abspath(__file__))
```

```python
# sort array by column
from operator import itemgetter
a = ([2, 2, 2, 40], [5, 5, 5, 10], [1, 1, 1, 50], [3, 3, 3, 30], [4, 4, 4, 20])
sorted(a, key=itemgetter(3))
```
<pre class="terminal"><code>>>> [[5, 5, 5, 10], [4, 4, 4, 20], [3, 3, 3, 30], [2, 2, 2, 40], [1, 1, 1, 50]]</code></pre>

```python
import numpy as np
np.amax
np.sum
```
