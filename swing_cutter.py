import numpy as np
import cv2
import datetime
import numpy as np
import sys

np.random.seed(42)

MEDIAN_FRAMES = 90
SWING_GROUP_BUFFER = 30
FPS = 30
PRE_STRIKE_FRAMES = 30
POST_STRIKE_FRAMES = 30
CUT_WIDTH = 1066
CUT_HEIGHT = 600
# yolo
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
YOLO_WEIGHTS_PATH = './yolov4-tiny.weights'
YOLO_CONFIGS_PATH = './yolov4-tiny.cfg'
COCO_NAMES_PATH = './coco.names'
DL_PATH = '/tmp'
WR_PATH = '/tmp'

def cut_swings(src_video_path, write_path, src_nm):
    t1 = datetime.datetime.now()
    video_stream = cv2.VideoCapture(src_video_path)
    total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print('calc median frames...')
    med_frame, gray_med_frame, hsv_med_frame = calc_median_frame(video_stream)
    print('init yolo...')
    model, class_names = init_yolo()

    # get frames with swings
    frames = []
    frame_num = 0
    while frame_num < total_frames-1:
        frame_num += 1

        (grabbed, frame) = video_stream.read()
        if not grabbed:
            continue

        if detect_swing(frame, hsv_med_frame):
            frames.append(frame_num)

    # group swing frames
    swing_frames = []
    last_frame = frames[0]
    curr_frames = [last_frame]
    for i, frame_num in enumerate(frames[1:]):
        if frame_num - last_frame > SWING_GROUP_BUFFER:
            swing_frames.append(curr_frames[:])
            curr_frames = []
        curr_frames.append(frame_num)
        last_frame = frame_num

    # write swings
    swing_num = 1
    last_cut_frame = 0
    outputs = []
    for i, frames in enumerate(swing_frames):
        if len(frames) < 2 or frames[0] <= last_cut_frame:
            continue

        # get middle frame as contact frame
        contact_idx = round(len(frames) / 2)
        contact_frame_num = frames[contact_idx]
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, contact_frame_num)
        ret, contact_frame = video_stream.read()

        # make sure contour is a person
        contour_box = max_contour_box(contact_frame, gray_med_frame)
        person_box = detect_person(contact_frame, model, class_names)
        if person_box is None or contour_box is None or not boxes_intersect(person_box, contour_box):
            continue

        # initialize for output writer
        st_frame_num = contact_frame_num - PRE_STRIKE_FRAMES + 1
        end_frame_num = contact_frame_num + POST_STRIKE_FRAMES
        minX, minY, maxX, maxY = calc_crop(person_box, frame_w, frame_h)
        out_filename = '{}/{}_swing_{}.mp4'.format(write_path, src_nm, swing_num)
        writer = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*"MP4V"), FPS, (maxX-minX, maxY-minY))

        frame_num = st_frame_num
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        while frame_num <= end_frame_num:
            ret, frame = video_stream.read()
            frame = frame[minY:maxY, minX:maxX, :]
            writer.write(frame)
            frame_num += 1

        last_cut_frame = frame_num
        outputs.append({
            "path": out_filename,
            "swing": swing_num,
            "start_frame": st_frame_num,
            "end_frame": end_frame_num,
        })
        swing_num += 1
        writer.release()

    t2 = datetime.datetime.now()
    print('processing time {} seconds'.format((t2-t1).seconds))
    video_stream.release()
    return outputs


def init_yolo():
    class_names = []
    with open(COCO_NAMES_PATH, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet(YOLO_WEIGHTS_PATH, YOLO_CONFIGS_PATH)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255)

    return model, class_names

def calc_median_frame(video_stream):
    frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * \
        np.random.uniform(size=MEDIAN_FRAMES)
    frames = []
    for fid in frameIds:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_stream.read()
        frames.append(frame)
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
    med_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    gray_med_frame = cv2.cvtColor(med_frame, cv2.COLOR_BGR2GRAY)
    hsv_med_frame = cv2.cvtColor(med_frame, cv2.COLOR_BGR2HSV)

    return med_frame, gray_med_frame, hsv_med_frame

def detect_person(frame, model, class_names):
    classes, scores, boxes = model.detect(
        frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    max_box = None
    max_area = 0
    for (classid, _, box) in zip(classes, scores, boxes):
        name = class_names[classid[0]]
        area = box[2]*box[3]
        if name == 'person' and area > max_area:
            max_box = box
            max_area = area

    if max_box is None:
        return None

    return [max_box[0], max_box[1], max_box[0]+max_box[2], max_box[1]+max_box[3]]


def calc_crop(contact_box, frame_w, frame_h):
    centerX = contact_box[0] + round((contact_box[2]-contact_box[0]) / 2)
    centerY = contact_box[1] + round((contact_box[3]-contact_box[1]) / 2)
    minX = centerX - round(CUT_WIDTH/2)
    maxX = centerX + round(CUT_WIDTH/2)
    minY = centerY - round(CUT_HEIGHT/2)
    maxY = centerY + round(CUT_HEIGHT/2)

    if minX < 0:
        maxX += 0 - minX
        minX = 0
    elif maxX > frame_w:
        minX -= maxX - frame_w
        maxX = frame_w

    if minY < 0:
        maxY += 0 - minY
        minY = 0
    elif maxY > frame_h:
        minY -= maxY - frame_h
        maxY = frame_h

    return [minX, minY, maxX, maxY]


def detect_swing(frame, hsv_med_frame):
    hframe = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dframe = cv2.absdiff(hframe, hsv_med_frame)
    _, _, gframe = cv2.split(dframe)  # to grayscale
    blurred = cv2.GaussianBlur(gframe, (11, 11), 0)
    _, tframe = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2 .CHAIN_APPROX_SIMPLE)

    # get largest area contour
    max_cnt = None
    max_area = 0
    for cnt in cnts:
        _, _, w, h = cv2.boundingRect(cnt)
        if w*h > max_area:
            max_area = w*h
            max_cnt = cnt

    if max_cnt is None:
        return False

    (_, _), (MA, _), _ = cv2.fitEllipse(max_cnt)
    return True if MA >= 100 else False


def max_contour_box(frame, gray_med_frame):
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(gframe, gray_med_frame)
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    _, tframe = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_cnt = [0, 0, 0, 0]
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h > max_cnt[2]*max_cnt[3]:
            max_cnt = [x, y, w, h]

    return [max_cnt[0], max_cnt[1], max_cnt[0]+max_cnt[2], max_cnt[1]+max_cnt[3]]


def boxes_intersect(box1, box2):
    return (box1[0] < box2[2] and box2[0] < box1[2] and box1[1] < box2[3] and box2[1] < box1[3])

if __name__ == "__main__":
    srv_video_path = sys.argv[1]
    file_paths = srv_video_path.split('/')
    full_file_nm = file_paths[len(file_paths)-1]
    ext_idx = full_file_nm.rfind('.')
    src_file_nm = full_file_nm[0:ext_idx]

    outputs = cut_swings(srv_video_path, './out', src_file_nm)
    print(outputs)
