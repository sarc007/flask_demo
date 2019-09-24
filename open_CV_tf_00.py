import cv2
import imutils
import face_recognition
from imutils.video import VideoStream
import pickle
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pathlib

data = pickle.loads(open("encodings.pickle", "rb").read())
data1 = pickle.loads(open("encoding/suhail_chougule/encodings.pickle", "rb").read())

for encodings in data['names']:
	print(encodings)

for encodings in data1['names']:
	print(encodings)

print(len(data1['names']))
print(len(data1['encodings']))

print(len(data['names']))
print(len(data['encodings']))

#stream = cv2.VideoCapture(0)

# while True:
# 	(grabbed, frame) = stream.read()
# 	# if the frame was not grabbed, then we have reached the
# 	# end of the stream
# 	if not grabbed:
# 		break
#
# 	# convert the input frame from BGR to RGB then resize it to have
# 	# a width of 750px (to speedup processing)
# 	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	rgb = imutils.resize(frame, width=750, height=750)
# 	r = frame.shape[1] / float(rgb.shape[1])
# 	#rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
# 	tf_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 	tf_frame = imutils.resize(tf_frame, width=750, height=750)
#
#
#
# 	# detect the (x, y)-coordinates of the bounding boxes
# 	# corresponding to each face in the input frame, then compute
# 	# the facial embeddings for each face
# 	boxes = face_recognition.face_locations(rgb, model="cnn")
# 	encodings = face_recognition.face_encodings(rgb, boxes)
# 	names = []
#
# 	# loop over the facial embeddings
# 	for encoding in encodings:
# 		# attempt to match each face in the input image to our known
# 		# encodings
# 		matches = face_recognition.compare_faces(data["encodings"], encoding)
# 		name = "Unknown"
# 		# check to see if we have found a match
# 		if True in matches:
# 			# find the indexes of all matched faces then initialize a
# 			# dictionary to count the total number of times each face
# 			# was matched
# 			matchedIdxs = [i for (i, b) in enumerate(matches) if b]
# 			counts = {}
#
# 			# loop over the matched indexes and maintain a count for
# 			# each recognized face face
# 			for i in matchedIdxs:
# 				name = data["names"][i]
# 				counts[name] = counts.get(name, 0) + 1
#
# 			# determine the recognized face with the largest number
# 			# of votes (note: in the event of an unlikely tie Python
# 			# will select first entry in the dictionary)
# 			name = max(counts, key=counts.get)
#
# 		# update the list of names
# 		names.append(name)
#
# 	# loop over the recognized faces
# 	for ((top, right, bottom, left), name) in zip(boxes, names):
# 		# rescale the face coordinates
# 		top = int(top * r)
# 		right = int(right * r)
# 		bottom = int(bottom * r)
# 		left = int(left * r)
#
# 			# draw the predicted face name on the image
# 		cv2.rectangle(rgb, (left, top), (right, bottom),
# 		              (0, 255, 0), 2)
# 		y = top - 15 if top - 15 > 15 else top + 15
# 		cv2.putText(rgb, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
#
# 	cv2.imshow("Frame", rgb)
# 	key = cv2.waitKey(100) & 0xFF
# # if the `q` key was pressed, break from the loop
# 	if key == ord("q"):
# 		break
