from flask import Flask, render_template, request, send_from_directory
from flask_bootstrap import Bootstrap
from datetime import datetime
from uuid import uuid4
import os

from main import infer_by_webapp

app = Flask(__name__)
Bootstrap(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def index():
    
    return render_template('upload.html')

@app.route('/about')
def about():

    return render_template('about.html')

@app.route('/contact')
def contact():

    return render_template('contact.html')

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    else:
        print("Couldn't create upload directory:{}".format(target))
    print(request.files.getlist("file"))
    option = request.form.get('optionsPrediction')
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)
        result = predict_image(destination, option)
        print(result)

    return render_template("complete.html", image_name=filename, result=result)

@app.route('/upload/<filename>')
def send_image(filename):

    return send_from_directory("images", filename)

def predict_image(path, type):
    print(path)
    return infer_by_webapp(path, type)


if __name__ == '__main__':
    app.run(debug=True, port=5000)