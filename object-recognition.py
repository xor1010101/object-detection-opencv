""" Object detection (frontal faces, car plates) w/ OpenCV HAAR cascade classifier
Reference: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
"""

import argparse # command line arguments
import cv2 as cv # OpenCV Python binding
import imageio # for saving a recorded object detection to a GIF
import os
import urllib.request # used for downloading a haarcascade

def grayscaleAndNormalize(frame):
    """ Turn the frame into grayscale & normalize
    This is mandatory before performing any object detection
    """

    frame_grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_grayscale = cv.equalizeHist(frame_grayscale)
    return frame_grayscale

def detectObjects(frame, classifier: cv.CascadeClassifier):
    """ Detect objects in a given frame (grayscale) with the OpenCV classifier
    """

    frame_grayscale = grayscaleAndNormalize(frame)
    face_coords = classifier.detectMultiScale(frame_grayscale)
    return face_coords

def addRectangleToFrame(frame, face_coords):
    """ Adds rectangles of detected objects to a given image frame
    """

    for (x, y, w, h) in face_coords:
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return frame

parser = argparse.ArgumentParser(description='A HAAR cascade object detector based on a given camera stream or video playback')
parser.add_argument('--face_cascade', help='HAAR cascade model file (default: haarcascade_frontalface_default.xml)', 
                    type=str, default='haarcascade_frontalface_default.xml')
parser.add_argument('--input', help='Specify a camera device number / path or a recorded video to play (default: 0)', 
                    default=0, nargs='?')
parser.add_argument('--record', help='Saves the image sequence with detected objects to a GIF (default: False)', action='store_true')

args = parser.parse_args()
cascade_file = args.face_cascade
input = args.input
record = args.record

cascade_filepath = os.path.join("data/haarcascades/", cascade_file)
url = "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/" + cascade_file

if not os.path.isfile(cascade_filepath):
    print("HAAR cascade file not available locally. Downloading from " + url)
    os.makedirs(os.path.dirname("data/haarcascades/"), exist_ok=True)
    urllib.request.urlretrieve(url, cascade_filepath)

capture = cv.VideoCapture(input)

if capture.isOpened() == False:
    print('-- (!) Error opening video capture')
    exit(0)

# load once during init phase
classifier = cv.CascadeClassifier(cascade_filepath)

if record:
    frames = []

while True:
    ret, frame = capture.read()
    if frame is None:
        print('-- (!) No captured frame -- exiting!')
        break

    detected_objs_coords = detectObjects(frame, classifier)
    frame = addRectangleToFrame(frame, detected_objs_coords)
    cv.imshow('HAAR Classifier Object Detection', frame)

    if (record):
        # imageio takes RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    key = cv.waitKey(10)

    if key == 27: # escape to break the loop and exit the program
        break

if (record):
    imageio.mimsave('record.gif', frames, fps=25)

capture.release()
cv.destroyAllWindows()
