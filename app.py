from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os

RELATIVE_URL = 'image/upload'
UPLOAD_FOLDER = os.path.join(os.path.abspath(os.getcwd()), 'image/upload')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345<>?'


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/image")
def Image():

    model = tf.keras.models.load_model('ml_models/model_plant.h5')

    img_contoh = tf.keras.preprocessing.image.load_img(
        "image/ini septoria.jpg", target_size=(150, 150))
    img_contoh_array = tf.keras.preprocessing.image.img_to_array(img_contoh)
    img_contoh_batch = np.expand_dims(img_contoh_array, axis=0)
    prediction = model.predict(img_contoh_batch)
    result = np.argmax(prediction[0])

    label = {'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 0,
             'Corn_(maize)___Common_rust_': 1,
             'Corn_(maize)___Northern_Leaf_Blight': 2,
             'Corn_(maize)___healthy': 3,
             'Tomato___Bacterial_spot': 4,
             'Tomato___Early_blight': 5,
             'Tomato___Late_blight': 6,
             'Tomato___Septoria_leaf_spot': 7,
             'Tomato___healthy': 8}
    for key, val in label.items():
        if val == result:
            predicted_label = key
            break
    return predicted_label


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/detectImage", methods=['GET', 'POST'])
def detect():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img_contoh = tf.keras.preprocessing.image.load_img(os.path.join(RELATIVE_URL, filename), target_size=(150, 150))
            img_contoh_array = tf.keras.preprocessing.image.img_to_array(img_contoh)
            img_contoh_batch = np.expand_dims(img_contoh_array, axis=0)

            model = tf.keras.models.load_model('ml_models/model_plant.h5')
            prediction = model.predict(img_contoh_batch)
            result = np.argmax(prediction[0])

            label = {'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 0,
                    'Corn_(maize)___Common_rust_': 1,
                    'Corn_(maize)___Northern_Leaf_Blight': 2,
                    'Corn_(maize)___healthy': 3,
                    'Tomato___Bacterial_spot': 4,
                    'Tomato___Early_blight': 5,
                    'Tomato___Late_blight': 6,
                    'Tomato___Septoria_leaf_spot': 7,
                    'Tomato___healthy': 8}
            for key, val in label.items():
                if val == result:
                    predicted_label = key
                    break
            return predicted_label
    else:
        return render_template('detect.html')


if __name__ == "__main__":
    app.run(debug=True)
