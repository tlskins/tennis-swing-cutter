import boto3
from swing_cutter import cut_swings
import os.path
import json
from os import environ

# env vars
if 'ACCESS_KEY_ID' in environ:
    ACCESS_KEY_ID = environ['ACCESS_KEY_ID']
    SECRET_ACCESS_KEY = environ['SECRET_ACCESS_KEY']
    TARGET_BUCKET = environ['TARGET_BUCKET']
    META_FOLDER = environ['META_FOLDER']
    MEDIA_CONVERT_ROLE = environ['MEDIA_CONVERT_ROLE']
else:
    import settings
    ACCESS_KEY_ID = settings.ACCESS_KEY_ID
    SECRET_ACCESS_KEY = settings.SECRET_ACCESS_KEY
    TARGET_BUCKET = settings.TARGET_BUCKET
    META_FOLDER = settings.META_FOLDER
    MEDIA_CONVERT_ROLE = settings.MEDIA_CONVERT_ROLE

DL_PATH = '/tmp'
WR_PATH = '/tmp'
FPS = 30

def lambda_handler(event, context):
    data_ret = event['Records'][0]['s3']
    bucket_nm = data_ret['bucket']['name']
    file_key = data_ret['object']['key']
    file_paths = file_key.split('/')
    full_file_nm = file_paths[len(file_paths)-1]
    ext_idx = full_file_nm.rfind('.')
    src_file_nm = full_file_nm[0:ext_idx]
    file_ext = full_file_nm[ext_idx+1:]

    # download metadata
    meta_sub_path = "/".join(file_paths[1:len(file_paths)-1])
    meta_key = '{}/{}/{}.txt'.format(META_FOLDER, meta_sub_path, src_file_nm)
    meta_dl_path = '{}/{}.{}'.format(DL_PATH, src_file_nm, "txt")
    s3 = boto3.client('s3')
    s3.download_file(
        bucket_nm,
        meta_key,
        meta_dl_path,
    )
    with open(meta_dl_path) as meta_file:
        meta = json.load(meta_file)
        user_id = meta["userId"]
        upload_id = meta["uploadId"]
        clip_len = int(meta["endSec"]) - int(meta["startSec"])
        clip_num = int(meta["number"])
        sound_frames = meta["soundFrames"]

    # download source video
    file_dl_path = '{}/{}.{}'.format(DL_PATH, src_file_nm, file_ext)
    s3.download_file(
        bucket_nm,
        file_key,
        file_dl_path,
    )

    # cut swing videos
    swing_data = cut_swings(file_dl_path, WR_PATH, src_file_nm, sound_frames)

    # upload videos and metadata
    outputs = []
    for swing in swing_data:
        # upload video
        video_key = '{}/{}/{}'.format(user_id, upload_id, '{}.mp4'.format(swing['swing_name']))
        upload_file(swing['video_path'], video_key)
        transcode_video(video_key, video_key)
        os.remove(swing['video_path'])

        # upload gif
        gif_key = '{}/{}/{}'.format(user_id, upload_id, '{}.gif'.format(swing['swing_name']))
        upload_file(swing['gif_path'], gif_key)
        os.remove(swing['gif_path'])

        # upload jpg
        jpg_key = '{}/{}/{}'.format(user_id, upload_id, '{}.jpg'.format(swing['swing_name']))
        upload_file(swing['jpg_path'], jpg_key)
        os.remove(swing['jpg_path'])

        # write json meta data
        txt_path = '{}/{}.txt'.format(WR_PATH, swing['swing_name'])
        with open(txt_path, 'w') as txt_file:
            json.dump({
                'timestamp': clip_num * clip_len + round(swing['start_frame'] / FPS),
                'frames': swing['end_frame']-swing['start_frame'],
                'swing': swing['swing_num'],
                'clip': clip_num,
                'uploadKey': upload_id,
            }, txt_file)

        # upload metadata
        txt_key = '{}/{}/{}'.format(user_id, upload_id, '{}.txt'.format(swing['swing_name']))
        upload_file(txt_path, txt_key)
        os.remove(txt_path)

        outputs.append({
            'video': video_key,
            'gif': gif_key,
            'jpg': jpg_key,
            'txt': txt_key,
        })

    return {
        "statusCode": 200,
        "headers": {"content-type": "application/json"},
        "body":  {
            "uploadId": upload_id,
            "userId": user_id,
            "clipNum": clip_num,
            "bucket": TARGET_BUCKET,
            "outputs": outputs,
        },
    }

def upload_file(src_path, target_key):
    s3_session = boto3.session.Session(
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=SECRET_ACCESS_KEY
    ).resource('s3')
    s3_session.meta.client.upload_file(
        src_path, TARGET_BUCKET, target_key)

def transcode_video(src_key, target_key):
    mediaconvert_client = boto3.client(
    'mediaconvert', endpoint_url='https://vasjpylpa.mediaconvert-fips.us-east-1.amazonaws.com')
    job_obj = {
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
                            "Destination": "s3://"+TARGET_BUCKET+"/"+target_key.replace(".mp4", "")
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
                    "FileInput": "https://"+TARGET_BUCKET+".s3.amazonaws.com/"+src_key
                }
            ]
        },
        "AccelerationSettings": {
            "Mode": "DISABLED"
        },
        "StatusUpdateInterval": "SECONDS_60",
        "Priority": 0
    }

    mediaconvert_client.create_job(**job_obj)
