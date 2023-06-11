from flask import Flask,render_template,redirect,request
from ultralytics import YOLO
import os
import atexit
import shutil
import cv2
import torch


# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")  # or yolov5n - yolov5x6, custom

app = Flask(__name__)

runs_folder = os.path.join("static", "runs")

@atexit.register
def delete_runs_folder_at_exit():
    if os.path.exists(runs_folder):
        shutil.rmtree(runs_folder)


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
     # Generate a unique filename for the saved results
    output_filename = "runs/detect/exp"
    output_filepath = os.path.join("static", output_filename)
    
    # Save the results to the static folder with the specified filename
    results.save(save_dir=output_filepath)

    folder_path = 'static/runs/detect'
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    latest_subfolder = max(subfolders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    image_path = folder_path+'/'+latest_subfolder+'/'+f.filename
    
    
    
    response_details = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    response = results
       
    
    print(response)

   

    return render_template('model.html', response = response, image_path = image_path, response_details = response_details)


if __name__ =="__main__":
    
    app.run(debug=True, host='0.0.0.0')