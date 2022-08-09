from unittest import result
from flask import Flask
import tensorflow as tf
import numpy as np

app = Flask(__name__)


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


if __name__ == "__main__":
    app.run(debug=True)
