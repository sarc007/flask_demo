# USAGE
# python recognize_faces_video.py --encodings encodings.pickle
# python recognize_faces_video.py --encodings encodings.pickle --output output/jurassic_park_trailer_output.avi --display 0

# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import sys
import numpy

# class VideoCamera(object):
#     def __init__(self):
#         # Using OpenCV to capture from device 0. If you have trouble capturing
#         # from a webcam, comment the line below out and use a video file
#         # instead.
#         self.video = cv2.VideoCapture(0)
#         # If you decide to use video.mp4, you must have this file in the folder
#         # as the main.py.
#         # self.video = cv2.VideoCapture('video.mp4')
#
#     def __del__(self):
#         self.video.release()

    def get_frame():
        success, frame = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        # load the known faces and embeddings
        print("[INFO] loading encodings...")

        with open('encodings.pickle', 'rb') as f:
            contents = f.read()
        data = pickle.loads(contents)
        # convert the input frame from BGR to RGB then resize it to have
        # a width of 750px (to speedup processing)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
                                                model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # rescale the face coordinates
            top = int(top * r)
            right = int(right * r)
            bottom = int(bottom * r)
            left = int(left * r)

            # draw the predicted face name on the image
            cv2.rectangle(frame, (left, top), (right, bottom),
                          (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def identify_face_from_db(frame):


        return frame
        # if the video writer is None *AND* we are supposed to write
        # the output video to disk initialize the writer
        # if writer is None and args["output"] is not None:
        #     fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        #     writer = cv2.VideoWriter(args["output"], fourcc, 20,
        #                              (frame.shape[1], frame.shape[0]), True)

        # if the writer is not None, write the frame with recognized
        # faces to disk
        #if writer is not None:
         #   writer.write(frame)

        # check to see if we are supposed to display the output frame to
        # the screen
        # if args["display"] > 0:
        #     cv2.imshow("Frame", frame)
        #     key = cv2.waitKey(1) & 0xFF
        #
        #     # if the `q` key was pressed, break from the loop
        #     if key == ord("q"):
        #         break

    # do a bit of cleanup
    #cv2.destroyAllWindows()
    #vs.stop()

    # check to see if the video writer point needs to be released
    #if writer is not None:
     #   writer.release()