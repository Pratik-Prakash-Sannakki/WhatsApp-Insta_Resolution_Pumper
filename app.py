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
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import json
from flask import Flask, request, render_template, send_file, send_from_directory
import os

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))


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
    idx = 0
    for path in glob.glob(test_img_folder):
        idx += 1
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)
        # read images
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        cv2.imwrite('results/{:s}_rlt.png'.format(base), output)
    
    return 'File uploaded and resolution pumped up successfully', 200

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