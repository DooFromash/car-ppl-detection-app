from flask import Flask,render_template,redirect,request, url_for, jsonify
from ultralytics import YOLO
import json
import pandas as pd
import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom


app = Flask(__name__)


@app.route('/')
def hello():
    return render_template("model.html")

@app.route("/home")
def home():
    return redirect('/')

@app.route('/submit',methods=['POST'])
def submit_data():
    
    f=request.files['userfile']
    f.save(f.filename)
    results = model(f.filename)
   

    
    
    # response = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    response = results
    print(response)

   

    return render_template('model.html', response = response)


if __name__ =="__main__":
    
    app.run(debug=True, host='0.0.0.0')