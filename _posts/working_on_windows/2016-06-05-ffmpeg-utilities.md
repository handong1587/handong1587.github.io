---
layout: post
category: working_on_windows
title: FFmpeg Collection of Utility Methods
date: 2016-06-05
---

- homepage: [http://ffmpeg.org/](http://ffmpeg.org/)
- github: [https://github.com/FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg)

# Split audio from video

ffmpeg -i video.mp4 output_audio.wav

[http://superuser.com/questions/609740/extracting-wav-from-mp4-while-preserving-the-highest-possible-quality](http://superuser.com/questions/609740/extracting-wav-from-mp4-while-preserving-the-highest-possible-quality)

# Merge audio to a video

ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -strict experimental output.mp4

[http://superuser.com/questions/277642/how-to-merge-audio-and-video-file-in-ffmpeg](http://superuser.com/questions/277642/how-to-merge-audio-and-video-file-in-ffmpeg)

# Compress MP4

ffmpeg -i video.mp4 -vcodec h264 -b:v 1000k -acodec mp2 output.mp4
