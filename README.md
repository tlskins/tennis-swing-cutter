# video-cutter

Python lambda for cutting and formatting videos. Built with moviepy which uses ffmpeg. The video-cutter is designed to listen to new file events in an S3 then cut and format those videos to be processed by the tennis-swing-cutter lambda.

## Lambda Architecture

### Basic Settings:

Runtime: Python 3.8  
Memory: 2048 MB  
Timeout: 15min

### Environment vars:

ACCESS_KEY_ID: "XXX"  
CLIP_LENGTH_SECS: 45  
MAX_CLIPS: 5  
SECRET_ACCESS_KEY: "XXX"  
TARGET_BUCKET: "bucket-name"  
TARGET_ROOT_FOLDER: "clips"

### Trigger:

S3 ObjectCreated  
Prefix: originals/

## Dependencies

### Layers:

AWSLambda-Python38-SciPy1x (v29)  
custom-moviepy-layer

Build custom moviepy layer using the below commands. Will use requirements.txt in the repo to build these dependencies on the aws python3.8 env.
reference: (https://stackoverflow.com/questions/64016819/cant-use-opencv-python-in-aws-lambda)

```
terminal1
$ mkdir /tmp/moviepy-aubio-layer && cp requirements.txt /tmp/moviepy-aubio-layer/requirements.txt && cd /tmp/moviepy-aubio-layer

terminal2
$ docker run -it -v /tmp/moviepy-aubio-layer:/moviepy-aubio-layer lambci/lambda:build-python3.8 bash
$ cd /moviepy-aubio-layer
$ pip install -t python/lib/python3.8/site-packages/ -r requirements.txt

terminal1
zip -r -9 moviepy-layer.zip python
```

# tennis-swing-cutter

Python lambda for detecting and cutting swings in tennis videos. Built using opencv, yolov3, and coco. Intended to run on videos cut by video-cutter repo due to 15 min lambda timeout limitations. Videos uploaded by the video-cutter trigger S3 events that will invoke the tennis-swing-cutter lambda. Finally, after videos are cut we need to use a transcoder to format the cut swing videos to HTML5 compatible mp4.

Directly invoking the tennis-swing-cutter lambda through an AWS apigateway is unadviseable due to its 30 second timeout. Asynchronous S3 events triggered by uploading to an S3 seems to be the more appropriate solution.

Yolov2 and Yolov3-tiny model weights ended up not performing well enough for swing detection on an iphone video, the problem with Yolov3 is that its weights are 248 mb unzipped which is impossible to fit on the lambda's 250 mb layer limit so the lambda must download the Yolov3 weights every invocation. But, downloading this file from an S3 does not seem to effect performance very much.

Videos cut by opencv to mp4 format do not automatically play in the browser since it is not HTML5 compatible and needs to be written using H.264 codec. To handle this we use AWS elastic transcoder to run as an asynchronous job after the swings are cut and uploaded to a temp folder in S3.

It appears, from initial tests, that videos of 80 seconds timeout in just under 15 minutes in this lambda architecture but due to the possible concurrency of this application it is advisable to tune the clip length shorter to achieve greater throughput.

## swing_cutter.py

Runs the swing cutter on a local video and uploads to S3. Input from /videos outputs to /out and model weights in /models. Requires settings.py

```
ACCESS_KEY_ID = "XXX"
POST_STRIKE_FRAMES = "22"
PRE_STRIKE_FRAMES = "23"
SECRET_ACCESS_KEY = "XXX"
TARGET_BUCKET = "bucket=name"
TRANSCODE_PRESET_ID = "1351620000001-000020"
TRANSCODE_PIPELINE_ID = "123432143215321-dafwef"

```

## Lambda Architecture

### Basic Settings:

```
Runtime: Python 3.8
Memory: 10240 MB
Timeout: 15min
```

### Environment vars:

```
ACCESS_KEY_ID: "XXX"
POST_STRIKE_FRAMES: 15
PRE_STRIKE_FRAMES: 15
SECRET_ACCESS_KEY: "XXX"
S3_BUCKET_KEY_YOLO: "yolov3.weights"
S3_BUCKET_YOLO: "configs-bucket-name"
TARGET_BUCKET: "bucket-name"
TRANSCODE_PRESET_ID: "1351620000001-000020"
TRANSCODE_PIPELINE_ID: "1608416327955-eky1hs"
```

### Trigger:

```
S3 ObjectCreated
Prefix: clips/
```

### Elastic Encoder:

Run as an asynchronous job that is queued after clips have been cut.

```
IAM Role: Elastic_Transcoder_Default_Role
H.264 (HTML5 Compatible MP4) Preset ID: 1351620000001-000020
```

## Dependencies

### Layers:

AWSLambda-Python38-SciPy1x (v29)  
custom-opencv_python3_8_numpy-layer

Build custom layer that includes opencv and numpy using the below commands. Will use requirements.txt in the repo to build these dependencies on the aws python3.8 env.
reference: (https://stackoverflow.com/questions/64016819/cant-use-opencv-python-in-aws-lambda)

```
terminal1
mkdir /tmp/opencv-numpy-py3-8-layer && cp requirements.txt /tmp/opencv-numpy-py3-8-layer/requirements.txt && cd /tmp/opencv-numpy-py3-8-layer

terminal2
docker run -it -v /tmp/opencv_imgio_aubio_layer:/opencv_imgio_aubio_layer  lambci/lambda:build-python3.8 bash

cd /opencv_imgio_aubio_layer

pip install -t python/lib/python3.8/site-packages/ -r opencv_requirements.txt

yum install -y mesa-libGL

cp -v /usr/lib64/libGL.so.1 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libGL.so.1.7.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libgthread-2.0.so.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libgthread-2.0.so.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libglib-2.0.so.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libGLX.so.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libX11.so.6 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libXext.so.6 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libGLdispatch.so.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libGLESv1_CM.so.1.2.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libGLX_mesa.so.0.0.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libGLESv2.so.2.1.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libxcb.so.1 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libXau.so.6 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /usr/lib64/libXau.so.6 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/
cp -v /lib64/libGLdispatch.so.0.0.0 /opencv_imgio_aubio_layer/python/lib/python3.8/site-packages/opencv_python.libs/


pip install -t python/lib/python3.8/site-packages/ -r requirements.txt
* aubio fails to install (https://github.com/aubio/aubio/issues/320)
pip install numpy
python -c 'import numpy as np; print(np.get_include())'
/var/lang/lib/python3.8/site-packages/numpy/core/include
export CFLAGS=-I/var/lang/lib/python3.8/site-packages/numpy/core/include
pip install -t python/lib/python3.8/site-packages/ -r requirements.txt --upgrade


terminal1
zip -r -9 opencv_imgio_aubio_layer.zip python

cp opencv_imgio_aubio_layer.zip ~/my_projects/layers/opencv_imgio_aubio_layer.zip
```
