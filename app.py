import tensorflow as tf
import keras
from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads,IMAGES
from scipy.misc import imsave, imread, imresize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

import numpy as np
from werkzeug import secure_filename
import keras.models
import re
import sys
import os

app = Flask(__name__)

model = load_model('model2.h5')
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

global graph
graph = tf.compat.v1.get_default_graph()

photos = UploadSet('photos', IMAGES)

app.config['UPLOADED_PHOTOS_DEST'] = '.'
configure_uploads(app, photos)

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['photo']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        x = preprocess_image(file_path)

        out = model.predict(x)
        u = decode_predictions(out, top=3)[0]
        s1 = u[0][1]
        s2 = u[0][2]*100
        s3 = u[1][1]
        s4 = u[1][2]*100
        s5 = u[2][1]
        s6 = u[2][2]*100
        print(s1,s2,s3)
        return render_template("index2.html",s1=s1,s2=s2,s3=s3,s4=s4,s5=s5,s6=s6)

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	
	#app.run(debug=True)
