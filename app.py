# -*- coding: utf-8 -*-
import os
import numpy as np
import urllib.request
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import io

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

#my_project_id = "/subscriptions/7c20a5c8-956e-4eb4-b5e4-946870fff696/resourceGroups/ClassificatinTest/providers/Microsoft.CognitiveServices/accounts/ClassificatinTest" 
my_project_id = "78c68f9f-7790-40da-ae8d-29b352b8c20c"

#prediction_key = "cff6e899bb3f4404af5abdc15eb23784"
prediction_key = "e4fbf500de3642269089be472230e6b1"

ENDPOINT = "https://cv-morii-1.cognitiveservices.azure.com/"

#https://docs.microsoft.com/en-us/azure/cognitive-services/Custom-Vision-Service/quickstarts/image-classification?tabs=visual-studio&pivots=programming-language-python
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
#https://teratail.com/questions/262076
#predictor = CustomVisionPredictionClient(ENDPOINT, prediction_key)
my_iteration_name = "Iteration3"
#my_iteration_name = "376b0226-1d6e-45b3-80ef-2900a409aae6"
#https://dev.to/stratiteq/puffins-detection-with-azure-custom-vision-and-python-2ca5

# https://roytuts.com/upload-and-display-multiple-images-using-python-and-flask/
# https://qiita.com/kagami-r0927/items/3d426997467f0a975143
# https://ensekitt.hatenablog.com/entry/2018/06/27/200000
# pip install Pillow

# https://kinacon.hatenablog.com/entry/2019/03/13/Opencv4%E3%82%92Opencv3%E3%81%AB%E3%83%80%E3%82%A6%E3%83%B3%E3%82%B0%E3%83%AC%E3%83%BC%E3%83%89%E3%81%99%E3%82%8B%E3%80%82
# pip3 uninstall opencv-python
# pip3 install opencv-python

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # とりあえずサイズは小さくする
        img = cv2.resize(img, (640, 480))
        raw_img = cv2.resize(img, (320, 240))

        # サイズだけ変えたものも保存する
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_' + filename)
        cv2.imwrite(raw_img_url, raw_img)

        # なにがしかの加工
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        # --- prediction ---
        ret, jpeg = cv2.imencode('.jpg', img)
        results = predictor.detect_image(my_project_id, my_iteration_name, jpeg)

        fontType = cv2.FONT_HERSHEY_COMPLEX

        result_image = img
        img_h, img_w = result_image.shape[:2]
        for prediction in results.predictions:
            if prediction.probability > 0.6:
                bbox = prediction.bounding_box
                result_image = cv2.rectangle(result_image, (int(bbox.left * img_w), int(bbox.top * img_h)), (int((bbox.left + bbox.width) * img_w), int((bbox.top + bbox.height) * img_h)), (0, 255, 0), 3)
                label = prediction.tag_name
                x_text = int(bbox.left * img_w)
                y_text = int(bbox.top * img_h - 12)
                if y_text < 6:
                    y_text = 6
                if label == "makino":
                    cv2.putText(result_image, label,(x_text,  y_text), fontType, 1, (0, 0, 255),4)
                
                #print(prediction.tag_name)
                #cv2.imwrite('result.png', result_image)

        result_image = cv2.resize(result_image.copy(), (320, 240))

        # 加工したものを保存する
        gray_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'gray_'+filename)
        cv2.imwrite(gray_img_url, result_image)

        return render_template('index.html', raw_img_url=raw_img_url, gray_img_url=gray_img_url)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
