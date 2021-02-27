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
* aubio fails to install (https://github.com/aubio/aubio/issues/320)
$ pip install numpy
$ python -c 'import numpy as np; print(np.get_include())'
/var/lang/lib/python3.8/site-packages/numpy/core/include
$ export CFLAGS=-I/var/lang/lib/python3.8/site-packages/numpy/core/include
$ pip install -t python/lib/python3.8/site-packages/ -r requirements.txt

terminal1
zip -r -9 moviepy-aubio-layer.zip python
```
