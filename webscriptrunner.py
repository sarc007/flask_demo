from flask import Flask, render_template, send_from_directory,session, Response,   redirect, url_for, send_file, flash
from forms import CustomDemoForm_Train, CustomDemoForm_Test
import datetime
import pycuda
from imutils.video import VideoStream
import requests
import cv2
import time
import imutils
import face_recognition
import pickle
import sys
import numpy
#from flask import static_file
#from flask import get, post, request # or route
#from flask import template,view
from invoke import run as in_run
#from bottle import cheetah_view as view, cheetah_template as template
import os
import platform


app = Flask(__name__)
app.config['SECRET_KEY'] = '98af3fc211680ff2895d30f7bf10f640'
app.config["CACHE_TYPE"] = "null"
#flask_cache.Cache.clear(app.app_context)

from markdown.serializers import to_html_string
filename ='example_02.png'
video_file_name = ''
curr_mod = 'demo'
video_file = 1
@app.route('/demo')
#@app.view('image_display.tpl')
def demo():
    global curr_mod
    curr_mod = 'demo'

    session['url_is_video'] = '0'
    example_pic = filename
    session['test_img'] = filename
    return render_template('image_display.html', picture=example_pic)

@app.route('/demo/<example_pic>')
def serve_pictures_example(example_pic):
    return send_from_directory("examples", example_pic)

#@app.route('/output_pic')
#def serve_pictures_output():
#    return send_from_directory(session['output_path_dir'], session['output_filename'])
#
# @app.route('/custom_output_pic')
# def serve_pictures_custom_output():
#     return send_from_directory(session['output_path_dir'], session['output_filename'])

@app.route('/out_img')
#@app.view('img_output.tpl')
def do_out_img():
    testimage = session['test_img']
    fname = 'Lawrance'.lower()
    lname = 'McDonalds'.lower()
    session['fname'] = fname
    session['lname'] = lname
    output_path = ''
    output_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    session['output_filename'] = output_filename

    if platform.system() == 'Windows':
        output_path_dir = get_save_path(fname, lname, 'output', True)
        output_path = output_path_dir+'\\'+output_filename
        session['output_path_dir'] = output_path_dir
        session['output_path'] =output_path
        cmd = "python recognize_faces_image.py -e encodings.pickle -i examples/" + testimage + " -o " + output_path
    else:
        output_path_dir = get_save_path(fname, lname, 'output', True)
        output_path = output_path_dir+'/'+output_filename
        session['output_path_dir'] = output_path_dir
        session['output_path'] = output_path
        cmd = "python3 recognize_faces_image.py -e encodings.pickle -i examples/" + testimage + " -o " + output_path

    result = in_run(cmd, hide=False)

    #flask_cache.Cache.clear(app.app_context())
    return render_template('img_output.html', face_identified_img=output_filename)

@app.route('/out_img/<output_filename>')
def send_out_image(output_filename):
    out_dir="output/"+session['fname']+"_"+session['lname']
    print(out_dir)
    return send_from_directory(out_dir, output_filename)

@app.route('/custom_out_img')
#@app.view('img_output.tpl')
def do_custom_out_img():
    testimage = session['test_img']
    fname =session['fname']
    lname = session['lname']
    output_path = ''
    output_filename = datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.png'
    session['output_filename'] = output_filename
    if platform.system() == 'Windows':
        output_path_dir = get_save_path(fname, lname, 'output', True)
        output_path = output_path_dir + '\\' + output_filename
        session['output_path_dir'] = output_path_dir
        session['output_path'] = output_path
        cmd = "python recognize_faces_image.py -e encoding/"+fname+"_"+lname +"/encodings.pickle -i examples/" + testimage + " -o " + output_path
    else:
        output_path_dir = get_save_path(fname, lname, 'output', True)
        output_path = output_path_dir + '/' + output_filename
        session['output_path_dir'] = output_path_dir
        session['output_path'] = output_path
        cmd = "python3 recognize_faces_image.py -e encoding/"+fname+"_"+lname +"/encodings.pickle -i examples/" + testimage + " -o " + output_path
    result = in_run(cmd, hide=False)
    return render_template('custom_img_output.html', face_identified_img=output_filename)

@app.route('/custom_out_img/<output_filename>')
def send_custom_out_image(output_filename):
    out_dir = "output/" + session['fname'] + "_" + session['lname']
    print(out_dir)
    return send_from_directory(out_dir, output_filename)


# @app.route('/default_out_img')
# #@app.view('img_output.tpl')
# def do_default_out_img():
#     return do_out_img(filename)

# @app.route('/custom_out_img')
# #@app.view('img_output.tpl')
# def do_custom_out_img():
#     return do_custom_out_img(session['test_img'])


@app.route('/customdemo_upload_train', methods=['POST','GET'])
def do_customdemo_upload_train():
    global video_file
    form = CustomDemoForm_Train()
    if form.validate_on_submit():
        fname = form.fname.data.lower()
        lname = form.lname.data.lower()
        session['fname'] = fname
        session['lname'] = lname
        _ = get_save_path(fname, lname, 'encoding', True)
        save_path = get_save_path(fname, lname, 'dataset', True)
        uploads = form.trainfileuploads.data
        for upload in uploads:
            fullpath = os.path.join(save_path, upload.filename)
            upload.save(fullpath)  # appends upload.filename automatically

        #            name, ext = os.path.splitext(upload.filename)
        #            if ext not in ('.png', '.jpg', '.jpeg'):
        #                return 'File extension not allowed.'

        flash('Upload Successful', 'success')
        flash('Encoding Image Please Wait ....', 'information')
        # cmd = "python encode_faces.py --dataset dataset/" + fname + "_" + lname + " --encodings encodings.pickle"
        if platform.system() == 'Windows':

            cmd = "python encode_faces.py --dataset dataset/"+fname+"_"+lname+" --encodings encoding/"+fname+"_"+lname+"/encodings.pickle"
        else:
            cmd = "python3 encode_faces.py --dataset dataset/" + fname + "_" + lname + " --encodings encoding/" + fname + "_" + lname + "/encodings.pickle"
        result = in_run(cmd, hide=False)
        # flash('Encoding Image For Common DB Please Wait ....', 'information')
        # if platform.system() == 'Windows':
        #     cmd = "python encode_faces.py --dataset dataset/"+fname+"_"+lname+" --encodings encodings.pickle"
        # else:
        #     cmd = "python3 encode_faces.py --dataset dataset/" + fname + "_" + lname + " --encodings encodings.pickle"
        # result = in_run(cmd, hide=False)

        if result.ok:
            flash('Encoding successful proceed to test ', 'success')
        else:
            flash('Encoding failed contact the tech support ', 'failure')
    return render_template('customdemoform_upload_train.html', form=form, video_file=0)


@app.route('/customdemo_upload_test', methods=['POST', 'GET'])
def do_customdemo_upload_test():
    global video_file
    global video_file_name
    form = CustomDemoForm_Test()
    if session.get('fname') is not None:
        fname = session['fname']
    else:
        fname = 'lawrance'
    if session.get('lname') is not None:
        lname = session['lname']
    else:
        lname = 'mcdonald'
    if form.validate_on_submit():
        save_path = get_save_path(fname, lname,'examples',False)
        upload = form.testfileupload.data
        fullpath = os.path.join(save_path, upload.filename)
        upload.save(fullpath)  # appends upload.filename automatically
        video_file_name = fullpath
        session['test_img'] = upload.filename
        flash('Upload Successful', 'success')
        #flask_cache.Cache.clear(app.app_context())
        #            name, ext = os.path.splitext(upload.filename)
            #            if ext not in ('.png', '.jpg', '.jpeg'):
            #                return 'File extension not allowed.'
    return render_template('customdemoform_upload_test.html', form=form, video_file=1)
    # fname = request.forms.get('fname').lower()
    # lname = request.forms.get('lname').lower()
    # uploads = request.files.getall('upload')
    # save_path = get_save_path_for_name(fname, lname)
    # for upload in uploads:
    #     name, ext = os.path.splitext(upload.filename)
    #     if ext not in ('.png', '.jpg', '.jpeg'):
    #         return 'File extension not allowed.'
    #     upload.save(save_path) # appends upload.filename automatically
@app.route('/video')
def video():
    return render_template('video_out.html')
def gen():
    i = 1
    while i<10:
        yield (b'--frame\r\n'
            b'Content-Type: text/plain\r\n\r\n'+str(i)+b'\r\n')
        i+=1
def get_frame():
    global video_file
    global video_file_name
    file_name = video_file_name
    camera_port = 1
    ramp_frames = 100

    if video_file == 0:
        camera = cv2.VideoCapture(camera_port)  # this makes a web cam object
    else:
        print(file_name)
        camera = cv2.VideoCapture(file_name)

    output_filename = "output/web_cam_"+datetime.datetime.now().strftime("%Y%m%d%H%M%S") + '.avi'
    print("[INFO] loading encodings...")
    with open('encodings.pickle', 'rb') as f:
        contents = f.read()
    data = pickle.loads(contents)
    time.sleep(2.0)
    i = 1
    writer = None
    while True:
        if video_file == 0:
            if camera.isOpened():
                retval, frame = camera.read()
                if not retval:
                    break
                # convert the input frame from BGR to RGB then resize it to have
                # a width of 750px (to speedup processing)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = imutils.resize(frame, width=750)
                r = frame.shape[1] / float(rgb.shape[1])

                # detect the (x, y)-coordinates of the bounding boxes
                # corresponding to each face in the input frame, then compute
                # the facial embeddings for each face
                boxes = face_recognition.face_locations(rgb, model="cnn")
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
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    y = top - 15 if top - 15 > 15 else top + 15
                    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                imgencode = cv2.imencode('.jpg', frame)[1]
                stringdata = imgencode.tostring()
                yield (b'--frame\r\n '
                       b'Content-Type: text/plain\r\n\r\n' + stringdata + b'\r\n')

                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter(output_filename, fourcc, 24, (frame.shape[1], frame.shape[0]), True)

            # if the writer is not None, write the frame with recognized
            # faces to disk
            if writer is not None:
                writer.write(frame)
        else:
            retval, frame = camera.read()
            if not retval:
                break
            # convert the input frame from BGR to RGB then resize it to have
            # a width of 750px (to speedup processing)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = imutils.resize(frame, width=750)
            r = frame.shape[1] / float(rgb.shape[1])

            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input frame, then compute
            # the facial embeddings for each face
            boxes = face_recognition.face_locations(rgb, model="cnn")
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
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                y = top - 15 if top - 15 > 15 else top + 15
                cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            imgencode = cv2.imencode('.jpg', frame)[1]
            stringdata = imgencode.tostring()
            yield (b'--frame\r\n '
                   b'Content-Type: text/plain\r\n\r\n' + stringdata + b'\r\n')

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output_filename, fourcc, 24, (frame.shape[1], frame.shape[0]), True)

            # if the writer is not None, write the frame with recognized
            # faces to disk
            if writer is not None:
                writer.write(frame)
        i += 1
    if writer is not None:
        writer.release()



@app.route('/calc')
def calc():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_save_path(fname,lname,location,createfolder=True):
    path =''
    if platform.system() == 'Windows':
        if createfolder:
            path = app.root_path + '\\' + location + '\\' + fname + '_' + lname
            path = path.lower()
            if os.path.exists(path):
                os.system("erase "+path+"\\*.* /Q ")
                os.system("rd "+path)
            os.mkdir(path)
        else:
            path = app.root_path + '\\' + location
            path = path.lower()

    else:
        if createfolder:
            path = app.root_path + '/' + location + '/' + fname + '_' + lname
            path = path.lower()
            if os.path.exists(path):
                os.system("rm -rf " + path)
            print("creating folder " + path)
            os.mkdir(path)
            print("creating folder '"+path+"' successful!!! ")
        else:
            path = app.root_path + '/' + location
            path = path.lower()

    return path

# @bottle.route('/dl')
# def dl():
#     #os.access(filename, R_OK)
#     print(filename)
#     return static_file(filename, root='./examples', download=True)
# @route('/static/C:/examples/example_01.png')
# def server_static(filepath):
#     return static_file(filepath, root='./examples')
if __name__=='__main__':
    app.run(debug = True, host='127.0.0.1',  port=5000)
