# USAGE
# python encode_faces.py --dataset dataset/suhail_chougule --encodings encodings.pickle

# import the necessary packages
from flask import flash
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
                help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,
                help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",
                help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []
knownNames1 = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print("[INFO] processing image {}/{}".format(i + 1,
                                                 len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]

    # load the input image and convert it from RGB (OpenCV ordering)
    # to dlib ordering (RGB)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes
    # corresponding to each face in the input image
    boxes = face_recognition.face_locations(rgb,
                                            model=args["detection_method"])

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings
    for encoding in encodings:
        # add each encoding + name to our set of known names and
        # encodings

        knownEncodings.append(encoding)
        knownNames.append(os.path.basename(name))
        knownNames1.append(os.path.basename(name))

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
data1 = {"encodings": knownEncodings, "names": knownNames1}

abs_path_encoding = os.path.dirname(os.path.abspath(__file__))

if os.path.isfile(args['encodings']):
    f3 = open(args["encodings"], 'rb')
    mydict1 = pickle.load(f3)         # load file content as mydict
    f3.close()
    del f3
    for encode in knownEncodings:
        mydict1['encodings'].append(encode)
    for name2 in knownNames1:
        mydict1['names'].append(name2)
    with open(args["encodings"], "wb") as f:
        print('opened the file successfully')
        pickle.dump(mydict1, f)
        print('file dumpped successfully')
else:
    with open(args["encodings"], "wb") as f:
        print('opened the file successfully')
        pickle.dump(data, f)
        print('file dumpped successfully')


with open(args["encodings"], "wb") as f:
    print('opened the file successfully')
    pickle.dump(data, f)
    print('file dumpped successfully')

print('Encoding Image For Common DB Please Wait ....', 'information')

f2 = open("encodings.pickle", 'rb')
mydict = pickle.load(f2)         # load file content as mydict
f2.close()
del f2

for encode in knownEncodings:
    mydict['encodings'].append(encode)
for name1 in knownNames1:
    mydict['names'].append(name1)



with open("encodings.pickle", "wb") as f1:
    print(' encodings opened the file successfully')
    pickle.dump(mydict, f1)
    print('encoding file dumpped successfully')
