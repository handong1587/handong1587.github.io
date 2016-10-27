---
layout: post
category: computer_vision
title: Use FFmpeg to Capture I Frames of Video
date: 2016-03-03
---

First we need to download ffmpeg.exe from: [http://ffmpeg.zeranoe.com/builds/](http://ffmpeg.zeranoe.com/builds/)

Use the commands below to extract all key frames (I, P, B) and their corresponding timecodes:

On Linux, it can be an one-off accomplishment:

```
ffmpeg -i yourvideo.mp4 -vf select="eq(pict_type\,PICT_TYPE_I)" -vsync 2 -s 160x90 -f image2 thumbnails-%02d.jpeg -loglevel debug 2>&1 | grep "pict_type:I -> select:1" | cut -d " " -f 6 - > ffmpeg_decode-info.txt
```

On Windows:

```
ffmpeg.exe -i yourvideo.mp4 -vf select='eq(pict_type\,I)' -vsync 2 -s 160x90 -f image2 thumbnails-%02d.jpeg -loglevel debug 2>&1 | findstr "pict_type:I" > ffmpeg_decode_info.txt
```

Now we will get a text file filled with formated text lines like:

```
\[Parsed_select_0 @ 05253d60\] n:134.000000 pts:135000.000000 t:5.400000 key:1 interlace_type:P pict_type:I scene:nan -> select:1.000000 select_out:0
```

Since we need only I frames' index, we can split each text line and get the target columns:

```
(for /f "tokens=5 delims= " %i in (ffmpeg_decode_info.txt) DO echo %i) > ffmpeg_iframe_index.txt
```

(Note: the file name "ffmpeg_decode_info.txt" should not contain any space. 
Otherwise we should add another option `usebackq`:

```
(for /f "usebackq tokens=5 delims= " %i in ("ffmpeg decode info.txt") DO echo %i) > ffmpeg_iframe_index.txt
```

)

Sometimes we would find some strange lines insert to ffmpeg_decode_info.txt 
(I don't know what it means, its insert location even whether it is inserted or not is unpredictable)

```
frame=   57 fps=2.7 q=2.1 size=N/A time=00:00:54.01 bitrate=N/A speed=2.56x   
```

Conflict will appear if we don't add some special handling. 
Usually this line is appended with the normal text line that we need, just like below:

```
frame=   57 fps=2.7 q=2.1 size=N/A time=00:00:54.01 bitrate=N/A speed=2.56x   **\r**\[Parsed_select_0 @ 05253d60\] n:134.000000 pts:135000.000000 t:5.400000 key:1 interlace_type:P pict_type:I scene:nan -> select:1.000000 select_out:0
```

Split it by **\r** will be Okay.

# Reference

[http://forum.doom9.org/archive/index.php/t-163553.html](http://forum.doom9.org/archive/index.php/t-163553.html)
