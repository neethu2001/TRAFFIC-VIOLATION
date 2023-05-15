import cv2
import numpy as np
import yolo_image_tri 
import yolo_image_hel
from flask import Flask, render_template, Response,request
from yolo_video import detect_video
from accident import detect_accident
from live import detect_realtime
from video_analysis import ana_video
from anpr_video import anpr_detect
# from ocr import ocr 
import csv




# Define a flask app
app = Flask(__name__)





@app.route('/helmet')
def index():
    return render_template('helmet.html')

@app.route('/report')
def report():
    with open('violation_report.csv', 'r') as file:
        reader = csv.reader(file)
        data = list(reader) # skip header row
    return render_template('report.html',data=data)

@app.route('/anpr', methods=['GET'])
def anpr():
    return render_template('anpr.html')


@app.route('/analysis')
def analysis():
    return render_template('video_analysis.html')


@app.route('/accident')
def accident():
    return render_template('accident.html')



@app.route('/video')
def yolo_video():
    return render_template('video.html')

import os

@app.route('/helvideo', methods=['POST'])
def helvideo():
    if 'video' not in request.files:
        return 'No file part in the request', 400
    file = request.files['video']
    if file.filename == '':
        return 'No file selected', 400
   

    file.save("test.mp4")
    
    detect_video()
    return "success"
@app.route('/video_analysis', methods=['POST'])
def anvideo():
    location = request.form['place']
    if 'video' not in request.files:
        return 'No file part in the request', 400
    file = request.files['video']
    if file.filename == '':
        return 'No file selected', 400
   

    file.save("test.mp4")
    
    ana_video(location)
    return "success"

@app.route('/anpr_video', methods=['POST'])
def anprvideo():
    if 'video' not in request.files:
        return 'No file part in the request', 400
    file = request.files['video']
    if file.filename == '':
        return 'No file selected', 400
   

    file.save("test.mp4")
    
    anpr_detect()
    return "success"

@app.route('/accvideo', methods=['POST'])
def accvideo():
    if 'video' not in request.files:
        return 'No file part in the request', 400
    file = request.files['video']
    if file.filename == '':
        return 'No file selected', 400
   

    file.save("test.mp4")
    
    detect_accident()
    return "success"



@app.route('/')
def detec():
    return render_template("main.html")

@app.route('/triple')
def triple():
    return render_template("triple.html")


@app.route('/live')
def live():
    detect_realtime()
    return render_template("result.html")


@app.route('/helpredict', methods=['GET', 'POST']) 
def helemt():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save("img.jpg")
        yolo_image_hel.detect_hel()
        return render_template('result.html')



@app.route('/tripredict', methods=['GET', 'POST']) 
def mobile():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f.save("img.jpg")
        yolo_image_tri.detect_tri()
        return render_template('result.html')
        


app.run(debug=False)