# from app import app
import base64
import io
from os.path import abspath

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)

        return resized

    images = []
    for file in request.files.getlist("file[]"):  # multiple or single file.

        # print("image: ", file)
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filestr = file.read()  # Reads the contents of the file
            # Interpret a buffer as a 1-dimensional array.
            npimg = np.frombuffer(filestr, np.uint8)
            # reads an image from the specified buffer in the memory
            image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
            # ratio = image.shape[0] / 500.0
            # orig = image.copy()
            image = resize(image, height=500)

# The function converts an input image from one color space to another. In case of a transformation . to-from RGB color space, the order of the channels should be specified explicitly(RGB or BGR). Note . that the default color format in OpenCV is often referred to as RGB but it is actually BGR(the . bytes are reversed). So the first byte in a standard(24-bit) color image will be an 8-bit Blue
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # calculates laplacian variance,blur score
            fm = cv2.Laplacian(gray, cv2.CV_64F).var()
            # print(type(fm))
            # print(fm)
            result = "Not Blurry"

            if fm < 100:
                result = "Blurry"

            # :.0f used for rounding up upto nearest integer
            sharpness_value = "{:.0f}".format(fm)
            message = [result, sharpness_value, file]

            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            file_object = io.BytesIO()  # converts image to bytes
            # print(file_object)
            img = Image.fromarray(resize(img, width=500)
                                  )  # op image dimensions

            img.save(file_object, 'PNG')
            base64img = "data:image/png;base64," + \
                base64.b64encode(file_object.getvalue()).decode(
                    'ascii')  # ascii data decoding of image

            # print("base-->", type(base64img))

            images.append([message, base64img])

    # print(message)  # [result, sharpness_value, file]
    # print(images)  # [message, base64img]

    # sharpnesss value=i[0][1]
    # image=i[1]

    # storing image to the elasticsearch

    from elasticsearch import Elasticsearch
    es = Elasticsearch()

    es.indices.create(index='data', ignore=400)  # creating first index
    ls1 = []

    for i in images:
        dct = {"blurness": int(i[0][1]), "image": i[1]}
        ls1.append(dct)

    ls = sorted(ls1, key=lambda item: item['blurness'])

    for i in range(len(ls)):
        es.index(index="data", doc_type='photos', body=ls[i])

    # es.indices.delete(index='data')  # to delete all the data from elasticsearch

    return render_template('upload.html', images=images)


if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)
