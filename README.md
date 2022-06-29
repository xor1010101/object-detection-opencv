# object-detection-opencv

Small object detection example in Python3 based on the HAAR cascade classifier from OpenCV

![Sample record of face detection](https://raw.githubusercontent.com/xor1010101/object-detection-opencv/main/record.gif)

# Usage

You may use the builtin USB webcam of your laptop to run a face detection:

```bash
$ python3 object-recognition.py
```

It's also possible to use a video as input:

```bash
$ python3 object-recognition.py --input myvideo.mp4
```

For more arguments & usage check the help:

```bash
$ python3 object-recognition.py -h
usage: object-recognition.py [-h] [--face_cascade FACE_CASCADE]
                             [--input [INPUT]] [--record]

A HAAR cascade object detector based on a given camera stream or video
playback

optional arguments:
  -h, --help            show this help message and exit
  --face_cascade FACE_CASCADE
                        HAAR cascade model file (default:
                        haarcascade_frontalface_default.xml)
  --input [INPUT]       Specify a camera device number / path or a recorded
                        video to play (default: 0)
  --record              Saves the image sequence with detected objects to a
                        GIF (default: False)
```