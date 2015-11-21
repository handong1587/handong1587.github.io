---
layout: post
category: programming_study
title: Python Tutorials
date: 2015-08-05
---

**Draw rectangle:**

{% highlight python %}
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = np.zeros((512, 512, 3), np.uint8)
cv2.rectangle(img, (20, 20), (411, 411), (55, 255, 155), 2)
plt.imshow(img, 'brg')
cv2.imwrite("/path/to/save/img.jpg", img)
{% endhighlight %}

**Use PIL to show ndarray:**

{% highlight python %}
import Image
import numpy as np
w,h = 512,512
data = np.zeros( (w,h,3), dtype=np.uint8)
data[256,256] = [255,0,0]
img = Image.fromarray(data, 'RGB')
img.save('my.png')
{% endhighlight %}

**Convert numpy multi-dim ndarray to 1-dim array:**

{% highlight python %}
np.asarray(a).reshape(-1)
{% endhighlight %}

**Remove duplicate elements**

{% highlight python %}
# Our input list.
values = [5, 5, 1, 1, 2, 3, 4, 4, 5]

# Convert to a set and back into a list.
set = set(values)
result = list(set)
print(result)
{% endhighlight %}

Output:

<pre class="terminal"><code>>>> [1, 2, 3, 4, 5] </code></pre>

**Simple operations**

{% highlight python %}
os.getcwd()
os.chdir(path)
os.path.dirname(os.path.abspath(__file__))
{% endhighlight %}

{% highlight python %}
# sort array by column
from operator import itemgetter
a = ([2, 2, 2, 40], [5, 5, 5, 10], [1, 1, 1, 50], [3, 3, 3, 30], [4, 4, 4, 20])
sorted(a, key=itemgetter(3))
{% endhighlight %}
<pre class="terminal"><code>>>> [[5, 5, 5, 10], [4, 4, 4, 20], [3, 3, 3, 30], [2, 2, 2, 40], [1, 1, 1, 50]]</code></pre>

{% highlight python %}
import numpy as np
np.amax
np.sum
{% endhighlight %}
