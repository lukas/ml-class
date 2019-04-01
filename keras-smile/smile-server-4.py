import flask
import keras
import numpy as np
import os
from keras.models import load_model
from PIL import Image, ExifTags
from flask import Flask, request
from jinja2 import Template
import base64
import cv2

app = Flask(__name__)

model = load_model('smile.h5')
model._make_predict_function()

detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def maybe_rotate(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        return image
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        return image


def detect_face(image, offsets=(0, 0)):
    faces = detection_model.detectMultiScale(image, 1.3, 5)
    if len(faces) > 0:
        x, y, width, height = faces[0]
        x_off, y_off = offsets
        x1, x2, y1, y2 = (x - x_off, x + width + x_off,
                          y - y_off, y + height + y_off)
        return cv2.resize(image[y1:y2, x1:x2], (64, 64))
    else:
        print("No faces found... using entire image")
        return cv2.resize(image, (64, 64))


def predict_image(image):
    image = maybe_rotate(image)
    image = image.convert(mode="L")
    im = np.asarray(image)
    face = detect_face(im)
    im_reshape = face.reshape(1, 64, 64, 1)
    im_rescale = im_reshape / 255.0
    pred = model.predict(im_rescale)
    return pred[0], face


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files['file']
    image = Image.open(f.stream)
    pred, image = predict_image(image)
    cv2.imwrite("/tmp/thumb_file.jpg", image)
    with open("/tmp/thumb_file.jpg", "rb") as img:
        thumb_string = base64.b64encode(img.read())
        base64out = "data:image/jpeg;base64," + str(thumb_string)[2:-1]
    template = Template("""
        <html>
            <body>
                <img src="{{face}}" style="width:200px" />
                <p>Probability of Smiling: {{smile_prob}}</p>
                <p>Probability of Not Smiling: {{no_smile_prob}}</p>
            </body>
        </html>
    """)

    return template.render(smile_prob="%.4f" % pred[1], no_smile_prob="%.4f" % pred[0], face=base64out)


@app.route("/")
def index():
    html = """
    <html>
        <body>
            <form action="predict" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*;capture=camera">
                <input type="submit"/>
            </form>
        </body>
    </html>
    """
    return(html)


if __name__ == '__main__' and not os.getenv("FLASK_DEBUG"):
    app.run(port=8080)
