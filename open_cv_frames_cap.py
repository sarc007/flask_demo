import cv2
import numpy as np
import os
import imutils

# Playing video from file:
input = "/Videos/debut/shayan.avi"
FPS = 25
# Playing video from file:
cap = cv2.VideoCapture(input)
cap.set(cv2.CAP_PROP_FPS, FPS)
frame_dir = 'framesdata/shayan'
try:
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
except OSError:
    print('Error: Creating directory of data')

currentFrame = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = imutils.resize(frame, width=209, height=209)

    # Saves image of the current frame in jpg file
    name = './'+frame_dir+'/frame' + str(currentFrame) + '.jpg'
    print('Creating...' + name)
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()