import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask
from flask import request
from flask_cors import CORS
from werkzeug.utils import secure_filename

import CountObject
from reader import read_files

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return {'success': 'loaded'}


@app.route('/count', methods=['POST'])
def count():
    if 'file_data' not in request.files:
        return {'error': 'no file found'}
    file = request.files['file_data']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        num_obj_found = CountObject.count_object(file_path=file_path, obj_to_count=request.form['obj_to_count'],
                                                 file_name=filename, upload_folder=UPLOAD_FOLDER)
        return {'success': num_obj_found}
    else:
        return 'file not allowed'


@app.route('/get_labels', methods=['GET'])
def get_all_labels():
    return {'labels': CountObject.class_labels,
            'total': len(CountObject.class_labels)
            }


@app.route('/run_start', methods=['GET'])
def run_start():
    yolo_dir = os.path.join(os.getcwd(), 'YoloConfig')
    os.mkdir(yolo_dir)
    executor = ThreadPoolExecutor(2)
    yolo_config_dir = os.path.join(yolo_dir, 'yolov3.cfg')
    yolo_weight_dir = os.path.join(yolo_dir, 'yolov3.weights')
    executor.submit(read_files('https://weightobj.s3.amazonaws.com/yolov3.weights', yolo_weight_dir))
    executor.submit(read_files('https://weightobj.s3.amazonaws.com/yolov3.cfg', yolo_config_dir))
    return {'success': 'Background task started to create yolo configs'}


@app.route('/config_complete', methods=['GET'])
def config_completed():
    yolo_dir = os.path.join(os.getcwd(), 'YoloConfig')
    weight_created = os.path.isfile(os.path.join(yolo_dir, 'yolov3.weights'))
    config_created = os.path.isfile(os.path.join(yolo_dir, 'yolov3.cfg'))
    if weight_created and config_created:
        return {'success': 'Background task completed'}
    return {'failed': 'Background task not completed'}


if __name__ == '__main__':
    app.run(debug=False)
