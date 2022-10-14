import base64
import numpy as np
import os
import cv2
from flask import render_template, request, flash, redirect, session, Blueprint
from werkzeug.utils import secure_filename
from vehicle_detector import VehicleDetector


# Load Veichle Detector
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

mod = Blueprint("home", __name__)


@mod.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@mod.route('/', methods=['POST'])
def upload_file():
    vd = VehicleDetector()

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        vehicle_boxes, class_ids_list = vd.detect_vehicles(img)
        vehicle_count = len(vehicle_boxes)

        i = 0
        for box in vehicle_boxes:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (25, 0, 180), 3)
            cv2.putText(img, get_class_name(class_ids_list[i]), (x, y + 25), 0, 1, (250, 206, 135), 2)
            i += 1

        cv2.putText(img, str(vehicle_count), (5, 25), 0, 1, (100, 200, 0), 2)

        string = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

    return render_template("index.html", image=string, vehicle_count=vehicle_count)


def get_class_name(class_id):
    if class_id == 2:
        return "mobil"
    elif class_id == 3:
        return "motor"
    elif class_id == 7:
        return "truk"
    else:
        return "unknown"


@mod.route('/video', methods=['POST'])
def video():
    if 'file' not in request.files:
        return redirect("/")
    file = request.files['file']
    if file.filename == '':
        return redirect("/")
    else:
        filepath = os.path.join("./static/uploads/", "video.mp4")
        file.save(filepath)
        # print('upload_video filename: ' + filename)
        os.system("python3 %s" % "vehicle_video_count.py")
        return redirect("/")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
