'''
import cv2
import time

img = cv2.imread('vc.jpg')
width = img.shape[1]
height = img.shape[0]
bicubic = cv2.resize(img,(width*4,height*4))
cv2.imshow('Image',img)
cv2.imshow('BICUBIC',bicubic)

super_res = cv2.dnn_superres.DnnSuperResImpl_create()

start = time.time()
super_res.readModel('LapSRN_x4.pb')
super_res.setModel('lapsrn',4)
lapsrn_image = super_res.upsample(img)
end = time.time()
print('Time taken in seconds by lapsrn', end-start)
cv2.imshow('LAPSRN',lapsrn_image)

start = time.time()
super_res.readModel('EDSR_x4.pb')
super_res.setModel('edsr',4)
edsr_image = super_res.upsample(img)
end = time.time()
print('Time taken in seconds by edsr', end-start)
cv2.imshow('EDSR',edsr_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
'''

import json
from flask import Flask, request, render_template, send_file, send_from_directory
import os

app = Flask(__name__)

# Specify the directory where uploaded images will be stored
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Render the index.html template
@app.route('/')
def index():
    return render_template('index.html')

# Handle image uploads
@app.route('/upload', methods=['POST'])
def upload():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return 'No file selected', 400
    
    # Save the file to the server
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    return 'File uploaded successfully', 200

# Serve uploaded images
'''
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    print(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
'''
@app.route('/get_images')
def get_images():
    
    filenames = os.listdir(app.config['UPLOAD_FOLDER'])
    print(filenames)
    return json.dumps(filenames)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True)