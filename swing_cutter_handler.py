import sys
import os.path
import math
import re
import boto3
import cv2 as cv
import numpy as np
import json
import imageio
from os import environ
from os import listdir
from pygifsicle import optimize

# Load env vars
if 'ACCESS_KEY_ID' in environ:
    ACCESS_KEY_ID = environ['ACCESS_KEY_ID']
    SECRET_ACCESS_KEY = environ['SECRET_ACCESS_KEY']
    TARGET_BUCKET = environ['TARGET_BUCKET']
    META_FOLDER = environ['META_FOLDER']
    POST_STRIKE_FRAMES = environ['POST_STRIKE_FRAMES']
    PRE_STRIKE_FRAMES = environ['PRE_STRIKE_FRAMES']
    S3_BUCKET_YOLO = environ['S3_BUCKET_YOLO']
    S3_BUCKET_KEY_YOLO = environ['S3_BUCKET_KEY_YOLO']
    MEDIA_CONVERT_ROLE = environ['MEDIA_CONVERT_ROLE']
else:
    import settings
    ACCESS_KEY_ID = settings.ACCESS_KEY_ID
    SECRET_ACCESS_KEY = settings.SECRET_ACCESS_KEY
    TARGET_BUCKET = settings.TARGET_BUCKET
    META_FOLDER = settings.META_FOLDER
    POST_STRIKE_FRAMES = settings.POST_STRIKE_FRAMES
    PRE_STRIKE_FRAMES = settings.PRE_STRIKE_FRAMES
    MEDIA_CONVERT_ROLE = settings.MEDIA_CONVERT_ROLE

# need absolute paths on lambda python runtime
downloadPath = '/tmp'
write_path = '/tmp'

# Initialize run parameters
drawPreds = False  # False
writeFullVideo = False  # False
writeFrameNum = False  # True
writeClips = True  # True
writeS3 = True  # True

# Initialize model parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "/tmp/yolov3.weights"

print('before dl modelWeights...')

s3 = boto3.client('s3')
if not os.path.exists(modelWeights):
    s3.download_file(
        S3_BUCKET_YOLO,
        S3_BUCKET_KEY_YOLO,
        modelWeights,
    )

print('before mediaconvert client...')

mediaconvert_client = boto3.client(
    'mediaconvert', endpoint_url='https://vasjpylpa.mediaconvert-fips.us-east-1.amazonaws.com')

print(listdir("/tmp"))

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# initialize trajectory configs
data = {
    'preStrikeFrames': int(PRE_STRIKE_FRAMES),
    'postStrikeFrames': int(POST_STRIKE_FRAMES),
    'minDist': 350,
    'videoHeight': None,
    'videoWidth': None,
    'clipLenSecs': 0,
    'clipNum': 0,
    'uploadId': None,
    'userId': None,

    # tracking contact
    'lastContactDist': None,
    'lastContactFrame': None,
    'ballRight': None,

    # tracking cropping
    'minX': None,
    'maxX': None,
    'minY': None,
    'maxY': None,
    'currentCropMarkers': [],

    'currentFrames': [],
    'frameNum': 0,
    'count': 0,
    'files': [],
}


def lambda_handler(event, context):
    dataRetrieved = event['Records'][0]['s3']
    bucketName = dataRetrieved['bucket']['name']
    fileKey = dataRetrieved['object']['key']
    print(fileKey)
    filePaths = fileKey.split('/')
    print(filePaths)
    fullFileName = filePaths[len(filePaths)-1]
    extIdx = fullFileName.rfind('.')
    srcFileName = fullFileName[0:extIdx]
    fileExt = fullFileName[extIdx+1:]

    # download metadata
    metaSubPath = "/".join(filePaths[1:len(filePaths)-1])
    metaKey = '{}/{}/{}.txt'.format(META_FOLDER, metaSubPath, srcFileName)
    metaDownloadPath = '{}/{}.{}'.format(downloadPath, srcFileName, "txt")
    print(metaKey)
    s3.download_file(
        bucketName,
        metaKey,
        metaDownloadPath,
    )
    with open(metaDownloadPath) as meta_file:
        meta = json.load(meta_file)
        data['userId'] = meta["userId"]
        data['uploadId'] = meta["uploadId"]
        clipLen = int(meta["endSec"]) - int(meta["startSec"])
        data["clipLenSecs"] = clipLen
        data["clipNum"] = int(meta["number"])

    # download file
    fileDownloadPath = '{}/{}.{}'.format(downloadPath, srcFileName, fileExt)
    print(fileDownloadPath)
    s3.download_file(
        bucketName,
        fileKey,
        fileDownloadPath,
    )

    outputs = detectSwings(fileDownloadPath, srcFileName)
    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body":  {
            "uploadId": data['uploadId'],
            "userId": data['userId'],
            "clipNum": data["clipNum"],
            "bucket": TARGET_BUCKET,
            "outputs": outputs,
        },
    }


def uploadFileS3(sourcePath, targetKey):
    print('uploading '+sourcePath+' to '+targetKey)
    s3_session = boto3.session.Session(
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    ).resource('s3')
    s3_session.meta.client.upload_file(
        sourcePath, TARGET_BUCKET, targetKey)


def awsTranscode(sourceKey, targetKey):
    print('transcoding '+sourceKey+' to '+targetKey)

    jobObject = {
        "Role": MEDIA_CONVERT_ROLE,
        "Settings": {
            "TimecodeConfig": {
                "Source": "ZEROBASED"
            },
            "OutputGroups": [
                {
                    "Name": "File Group",
                    "Outputs": [
                        {
                            "ContainerSettings": {
                                "Container": "MP4",
                                "Mp4Settings": {
                                    "CslgAtom": "INCLUDE",
                                    "CttsVersion": 0,
                                    "FreeSpaceBox": "EXCLUDE",
                                    "MoovPlacement": "PROGRESSIVE_DOWNLOAD",
                                    "AudioDuration": "DEFAULT_CODEC_DURATION"
                                }
                            },
                            "VideoDescription": {
                                "ScalingBehavior": "DEFAULT",
                                "TimecodeInsertion": "DISABLED",
                                "AntiAlias": "ENABLED",
                                "Sharpness": 50,
                                "CodecSettings": {
                                    "Codec": "H_264",
                                    "H264Settings": {
                                        "InterlaceMode": "PROGRESSIVE",
                                        # "ScanTypeConversionMode": "InterlaceMode",
                                        "NumberReferenceFrames": 3,
                                        "Syntax": "DEFAULT",
                                        "Softness": 0,
                                        "GopClosedCadence": 1,
                                        "GopSize": 90,
                                        "Slices": 1,
                                        "GopBReference": "DISABLED",
                                        "SlowPal": "DISABLED",
                                        "EntropyEncoding": "CABAC",
                                        "Bitrate": 4500000,
                                        "FramerateControl": "INITIALIZE_FROM_SOURCE",
                                        "RateControlMode": "CBR",
                                        "CodecProfile": "MAIN",
                                        "Telecine": "NONE",
                                        "MinIInterval": 0,
                                        "AdaptiveQuantization": "AUTO",
                                        "CodecLevel": "AUTO",
                                        "FieldEncoding": "PAFF",
                                        "SceneChangeDetect": "ENABLED",
                                        "QualityTuningLevel": "SINGLE_PASS",
                                        "FramerateConversionAlgorithm": "DUPLICATE_DROP",
                                        "UnregisteredSeiTimecode": "DISABLED",
                                        "GopSizeUnits": "FRAMES",
                                        "ParControl": "INITIALIZE_FROM_SOURCE",
                                        "NumberBFramesBetweenReferenceFrames": 2,
                                        "RepeatPps": "DISABLED",
                                        "DynamicSubGop": "STATIC"
                                    }
                                },
                                "AfdSignaling": "NONE",
                                "DropFrameTimecode": "ENABLED",
                                "RespondToAfd": "NONE",
                                "ColorMetadata": "INSERT"
                            }
                        }
                    ],
                    "OutputGroupSettings": {
                        "Type": "FILE_GROUP_SETTINGS",
                        "FileGroupSettings": {
                            "Destination": "s3://"+TARGET_BUCKET+"/"+targetKey.replace(".mp4", "")
                        }
                    }
                }
            ],
            "AdAvailOffset": 0,
            "Inputs": [
                {
                    "VideoSelector": {
                        "ColorSpace": "FOLLOW",
                        "Rotate": "DEGREE_0",
                        "AlphaBehavior": "DISCARD"
                    },
                    "FilterEnable": "AUTO",
                    "PsiControl": "USE_PSI",
                    "FilterStrength": 0,
                    "DeblockFilter": "DISABLED",
                    "DenoiseFilter": "DISABLED",
                    "InputScanType": "AUTO",
                    "TimecodeSource": "ZEROBASED",
                    "FileInput": "https://"+TARGET_BUCKET+".s3.amazonaws.com/"+sourceKey
                }
            ]
        },
        "AccelerationSettings": {
            "Mode": "DISABLED"
        },
        "StatusUpdateInterval": "SECONDS_60",
        "Priority": 0
    }

    mediaconvert_client.create_job(**jobObject)


def writeFrames(name, cap):
    frames = data['currentFrames']
    number = data['count']

    cropData = calcCrop()
    minX, maxX, minY, maxY = cropData[0], cropData[1], cropData[2], cropData[3]
    print('crop: {} {} {} {}'.format(minX, maxX, minY, maxY))

    fileName = f'{name}_swing_{number}'

    videoFile = f'{fileName}.mp4'
    videoPath = f'{write_path}/{videoFile}'

    gifFile = f'{fileName}.gif'
    gifPath = f'{write_path}/{gifFile}'

    jpgFile = f'{fileName}.jpg'
    jpgPath = f'{write_path}/{jpgFile}'

    txtFile = f'{fileName}.txt'
    txtPath = f'{write_path}/{txtFile}'

    vidWriter = cv.VideoWriter(
        videoPath,
        cv.VideoWriter_fourcc(*'mp4v'),  # codec
        30,  # fps
        (maxX-minX, maxY-minY)
    )
    gifImages = []

    for i, frame in enumerate(frames):
        # write video
        vidWriter.write(frame[minY:maxY, minX:maxX, :])
        # write gif
        img = cv.cvtColor(
            frame[minY:maxY, minX:maxX, :], cv.COLOR_BGR2RGB)
        gifImages.append(img)
        # write jpg
        if i == data['preStrikeFrames']:
            imageio.imsave(jpgPath, cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    imageio.mimsave(gifPath, gifImages, fps=35)
    # optimize(gifPath) # not working compressing gifs
    vidWriter.release()

    # create meta file
    timestamp = int(data['clipNum']) * data['clipLenSecs'] + \
        round((data['lastContactFrame']-data['preStrikeFrames'])/30)
    swingFrames = data['preStrikeFrames'] + data['postStrikeFrames']
    clipNum = data['clipNum']
    uploadKey = data['uploadId']

    swingData = {
        "timestamp": timestamp,
        "frames": swingFrames,
        "swing": number,
        "clip": clipNum,
        "uploadKey": uploadKey,
    }

    with open(txtPath, 'w') as outfile:
        json.dump(swingData, outfile)

    return {
        'video': {
            'path': videoPath,
            'file': videoFile
        },
        'gif': {
            'path': gifPath,
            'file': gifFile
        },
        'jpg': {
            'path': jpgPath,
            'file': jpgFile
        },
        'txt': {
            'path': txtPath,
            'file': txtFile,
        }
    }


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def calcCrop():
    minX, maxX, minY, maxY = None, None, None, None
    print(data['currentCropMarkers'])
    for marker in data['currentCropMarkers']:
        if len(marker) > 0:
            if minX is None or marker[0] < minX:
                minX = marker[0]
            if maxX is None or marker[1] > maxX:
                maxX = marker[1]
            if minY is None or marker[2] < minY:
                minY = marker[2]
            if maxY is None or marker[3] > maxY:
                maxY = marker[3]

    # add crop margin buffer
    cropMargin = 5
    minX -= cropMargin
    if minX < 0:
        minX = 0
    maxX += cropMargin
    if maxX > data['videoWidth']:
        maxX = data['videoWidth']
    minY -= cropMargin
    if minY < 0:
        minY = 0
    maxY += cropMargin
    if maxY > data['videoHeight']:
        maxY = data['videoHeight']

    # keep aspect ratio
    width = maxX - minX
    height = maxY - minY
    resizeWidthRatio = (
        height * data['videoWidth']) / (width * data['videoHeight'])
    cropWidth = int(resizeWidthRatio * width)
    widthAdj = int((cropWidth - width) / 2)

    print('cropping from right: {}'.format(data['ballRight']))
    print('crop width adj: {}'.format(widthAdj))

    # only crop from width and side ball is coming from
    # (height is generally correct from person detection)
    if data['ballRight']:
        maxX += widthAdj
        if maxX > data['videoWidth']:
            maxX = data['videoWidth']
    else:
        minX -= widthAdj
        if minX < 0:
            minX = 0

    return [minX, maxX, minY, maxY]


def drawPred(frame, classId, conf, left, top, right, bottom):

    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    # Get the label for the class name and its confidence
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(
        label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(
        1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top),
               cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


def postprocess(frame, outs, data, filename, cap):

    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    frameNum = data['frameNum']

    if writeFrameNum:
        cv.putText(frame, f'frame: {frameNum}', (50, 50), cv.FONT_HERSHEY_COMPLEX, .8,
                   (255, 50, 0), 2, lineType=cv.LINE_AA)

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.

    rac = None
    ball = None
    person = None
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        if drawPreds:
            drawPred(frame, classIds[i], confidences[i],
                     left, top, left + width, top + height)
        if classes[classIds[i]] == "sports ball":
            ball = [left, top]
            # trackCropMarker(left, top, width, height)
        if classes[classIds[i]] == "tennis racket":
            rac = [left, top]
            trackCropMarker(left, top, width, height)
        if classes[classIds[i]] == "person":
            person = [left, top]
            trackCropMarker(left, top, width, height)

    # # if both racquet and ball are in frame # has more false negatives than person + ball
    # if rac and ball:
    #     dist = ((abs(rac[0]-ball[0])**2 + abs(rac[1]-ball[1])**2)**0.5)
    #     print(f'frame: {frameNum} dist: {dist}')
    #     if dist < data['minDist']:
    #         # contact detected
    #         if not data['lastContactDist'] or dist <= data['lastContactDist']:
    #             data['lastContactDist'] = dist
    #             data['lastContactFrame'] = frameNum
    #             data['ballRight'] = True if ball[0]-rac[0] > 0 else False

    # if both person and ball are in frame
    if person and ball:
        dist = ((abs(person[0]-ball[0])**2 + abs(person[1]-ball[1])**2)**0.5)
        print(f'frame: {frameNum} dist: {dist}')
        if dist < data['minDist']:
            # contact detected
            if not data['lastContactDist'] or dist <= data['lastContactDist']:
                data['lastContactDist'] = dist
                data['lastContactFrame'] = frameNum
                data['ballRight'] = True if ball[0]-person[0] > 0 else False

    # cache swing data
    data['currentFrames'].append(frame)
    if data['maxX'] is None:
        data['currentCropMarkers'].append([])
    else:
        data['currentCropMarkers'].append(
            [data['minX'], data['maxX'], data['minY'], data['maxY']])
        data['minX'], data['maxX'], data['minY'], data['maxY'] = None, None, None, None

    # final swing frame detected
    if data['lastContactFrame'] and (frameNum - data['lastContactFrame']) > data['postStrikeFrames']:
        lastContactFrame = data['lastContactFrame']
        print(f'final swing frame detected: {lastContactFrame}!')
        data['count'] += 1

        if writeClips:
            outFile = writeFrames(filename, cap)
            data['files'].append(outFile)

        # reset swing data
        data['lastContactFrame'] = None
        data['lastContactDist'] = None
        data['currentFrames'] = []

    # limit caches
    if len(data['currentFrames']) > (data['preStrikeFrames'] + data['postStrikeFrames']):
        data['currentFrames'].pop(0)
    if len(data['currentCropMarkers']) > (data['preStrikeFrames'] + data['postStrikeFrames']):
        data['currentCropMarkers'].pop(0)


def trackCropMarker(left, top, width, height):
    if data['minX'] is None or left < data['minX']:
        data['minX'] = left
    if data['maxX'] is None or (left + width) > data['maxX']:
        data['maxX'] = left + width
    if data['minY'] is None or top < data['minY']:
        data['minY'] = top
    if data['maxY'] is None or (top + height) > data['maxY']:
        data['maxY'] = top + height


def detectSwings(filePath, fileName):

    if not os.path.isfile(filePath):
        raise ValueError('The path specified is not valid.')

    cap = cv.VideoCapture(filePath)

    if not cap.isOpened():
        raise ValueError('Unable to process video.')

    data['videoWidth'] = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    data['videoHeight'] = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print('vid width {} height {}'.format(
        data['videoWidth'], data['videoHeight']))

    if writeFullVideo:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fullVidWriter = cv.VideoWriter('./out/full_video.mp4', fourcc, 30, (round(
            cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        data['frameNum'] += 1

        if not hasFrame:
            print("Done processing video...")
            # cv.waitKey(3000)
            cap.release()
            break

        # Create a 4D blob from a frame.
        blob = cv.dnn.blobFromImage(
            frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outs = net.forward(getOutputsNames(net))
        # Remove the bounding boxes with low confidence
        postprocess(frame, outs, data, fileName, cap)
        if writeFullVideo:
            fullVidWriter.write(frame)
    if writeFullVideo:
        fullVidWriter.release()

    print(listdir("/tmp"))
    print(listdir("/"))

    # write to s3
    outputs = []
    if writeS3:
        for swing in data['files']:
            videoKey = '{}/{}/{}'.format(data['userId'],
                                         data['uploadId'], swing['video']['file'])
            uploadFileS3(swing['video']['path'], videoKey)
            awsTranscode(videoKey, videoKey)
            os.remove(swing['video']['path'])

            # upload gif
            gifKey = '{}/{}/{}'.format(data['userId'],
                                       data['uploadId'], swing['gif']['file'])
            uploadFileS3(swing['gif']['path'], gifKey)
            os.remove(swing['gif']['path'])

            # upload jpg
            jpgKey = '{}/{}/{}'.format(data['userId'],
                                       data['uploadId'], swing['jpg']['file'])
            uploadFileS3(swing['jpg']['path'], jpgKey)
            os.remove(swing['jpg']['path'])

            # upload txt
            txtKey = '{}/{}/{}'.format(data['userId'],
                                       data['uploadId'], swing['txt']['file'])
            uploadFileS3(swing['txt']['path'], txtKey)
            os.remove(swing['txt']['path'])

            print("finishing up...")
            print(listdir("/tmp"))

            outputs.append({
                'video': videoKey,
                'gif': gifKey,
                'jpg': jpgKey,
                'txt': txtKey,
            })

    return outputs
