import cv2
from flask_cors import CORS
from ViT.ViTmodel import load_video_ViT
from GRU.GRUTensorFlow import load_video
from LRCN.LRCNTensorFlow import gen_frames
from flask import Flask, render_template, Response

'''
for ip camera use - rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' 
for local webcam use cv2.VideoCapture(0)
'''

app = Flask(__name__)
CORS(app)
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed/LRCN')
def video_feed_LRCN():
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/GRU')
def video_feed_GRU():
    return Response(load_video(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/ViT')
def video_feed_ViT():
    return Response(load_video_ViT(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":    
    app.run(debug=True)