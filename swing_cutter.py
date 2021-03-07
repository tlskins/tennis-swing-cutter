import numpy as np
import cv2
import datetime
import numpy as np
import sys
import imageio
from itertools import filterfalse

SWING_GROUP_BUFFER = 45
# leeway between body swing contour frames to still count as a single swing
SWING_GROUP_LEEWAY = 3
FPS = 30
PRE_STRIKE_FRAMES = 30
POST_STRIKE_FRAMES = 30
IDEAL_SWING_FRAMES = 55
IDEAL_CONTACT_RATIO = 0.65
CUT_WIDTH = 1066
CUT_HEIGHT = 600
# yolo
CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
YOLO_WEIGHTS_PATH = './yolov4-tiny.weights'
YOLO_CONFIGS_PATH = './yolov4-tiny.cfg'
COCO_NAMES_PATH = './coco.names'


def cut_swings(src_video_path, write_path, src_nm, sound_frames):
    print('calc video stats...')
    t1 = datetime.datetime.now()
    video_stream = cv2.VideoCapture(src_video_path)
    total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_w = round(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = round(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print('calc video stats: {} seconds'.format((datetime.datetime.now()-t1).seconds))

    print('calc median frames...')
    t1 = datetime.datetime.now()
    hsv_med_frame = calc_median_frame(video_stream, total_frames)
    print('calc median frames: {} seconds'.format((datetime.datetime.now()-t1).seconds))
    
    print('init yolo...')
    t1 = datetime.datetime.now()
    model, class_names = init_yolo()
    print('init yolo: {} seconds'.format((datetime.datetime.now()-t1).seconds))

    print('getting frames with person swings...')
    t1 = datetime.datetime.now()
    swing_frames = []
    swing_body_boxes = []
    frame_num = 0
    while frame_num < total_frames-1:
        frame_num += 1

        (grabbed, frame) = video_stream.read()
        if not grabbed:
            continue

        swing_body_box = detect_swing_body(frame, hsv_med_frame)
        if swing_body_box is None:
            continue

        swing_frames.append(frame_num)
        swing_body_boxes.append(swing_body_box)
    print(swing_frames)
    print('getting frames with person swings: {} seconds'.format((datetime.datetime.now()-t1).seconds))

    print('grouping swing frames...')
    t1 = datetime.datetime.now()
    swing_groups = []
    body_group = {
        'sound_frames': [],
        'st_frame': swing_frames[0],
        'end_frame': swing_frames[0],
    }
    sound_frame_idx = 0
    curr_sound_frame = sound_frames[sound_frame_idx]
    for swing_frame_idx, frame_num in enumerate(swing_frames[1:]):
        # finalize group
        if frame_num - body_group['end_frame'] > SWING_GROUP_LEEWAY:
            group_frames = body_group['end_frame'] - body_group['st_frame']
            if group_frames > 1 and len(body_group['sound_frames']) > 0:
                # calculate contact frame and swing score
                frames_score = abs(
                    group_frames - IDEAL_SWING_FRAMES) / IDEAL_SWING_FRAMES
                contact_score = 100
                contact_frame_num = body_group['sound_frames'][0]
                for s_frame in body_group['sound_frames']:
                    contact_ratio = (
                        s_frame - body_group['st_frame']) / group_frames
                    score = abs(contact_ratio - IDEAL_CONTACT_RATIO)
                    if score < contact_score:
                        contact_score = score
                        contact_frame_num = s_frame

                body_group['score'] = frames_score + 0.7 * contact_score
                body_group['contact_frame_num'] = contact_frame_num

                # only keep if person detected too
                video_stream.set(cv2.CAP_PROP_POS_FRAMES, body_group['contact_frame_num'])
                _, contact_frame = video_stream.read()
                person_box = detect_person(contact_frame, model, class_names)
                swing_body_box = detect_swing_body(contact_frame, hsv_med_frame)

                # keep swing group as possible swing
                if person_box is not None and swing_body_box is not None and boxes_intersect(person_box, swing_body_box):
                    body_group['person_box'] = person_box
                    swing_groups.append(body_group)

            body_group = {
                'sound_frames': [],
                'st_frame': swing_frames[0],
                'end_frame': swing_frames[0],
            }

        # add to group details
        body_group['end_frame'] = frame_num
        if curr_sound_frame == frame_num:
            body_group['sound_frames'].append(frame_num)

        # increment counters
        while curr_sound_frame != -1 and frame_num >= curr_sound_frame:
            sound_frame_idx += 1
            curr_sound_frame = sound_frames[sound_frame_idx] if sound_frame_idx < len(
                sound_frames) else -1
    print(swing_groups)
    print('grouping swing frames: {} seconds'.format((datetime.datetime.now()-t1).seconds))

    print('sorting and selecting swings...')
    t1 = datetime.datetime.now()
    def sort_swing_score(group):
        return group['score']
    swing_groups.sort(reverse=True, key=sort_swing_score)

    final_swing_groups = []
    while len(swing_groups) > 0:
        sel = swing_groups.pop(0)
        final_swing_groups.append(sel)
        swing_groups[:] = [g for g in swing_groups if not abs(
            sel['contact_frame_num'] - g['contact_frame_num']) < SWING_GROUP_BUFFER]

    def sort_swing_frame(group):
        return group['contact_frame_num']
    final_swing_groups.sort(key=sort_swing_frame)
    print('sorting and selecting swings: {} seconds'.format((datetime.datetime.now()-t1).seconds))

    print('writing swings...')
    t1 = datetime.datetime.now()
    swing_num = 1
    outputs = []
    for swing_group in final_swing_groups:
        # initialize for output writers
        st_frame_num = swing_group['contact_frame_num'] - PRE_STRIKE_FRAMES + 1
        end_frame_num = swing_group['contact_frame_num'] + POST_STRIKE_FRAMES
        minX, minY, maxX, maxY = calc_crop(swing_group['person_box'], frame_w, frame_h)
        swing_name = '{}_swing_{}'.format(src_nm, swing_num)

        # gif_imgs = []
        # gif_path = '{}/{}.gif'.format(write_path, swing_name)
        video_path = '{}/{}.mp4'.format(write_path, swing_name)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(
            *"MP4V"), FPS, (maxX-minX, maxY-minY))

        frame_num = st_frame_num
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        while frame_num <= end_frame_num:
            _, frame = video_stream.read()
            crop_frame = frame[minY:maxY, minX:maxX, :]
            writer.write(crop_frame)
            # gif_imgs.append(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
            frame_num += 1

        # save jpg and gifs
        jpg_path = '{}/{}.jpg'.format(write_path, swing_name)
        imageio.imsave(jpg_path, cv2.cvtColor(
            contact_frame, cv2.COLOR_BGR2RGB))
        # imageio.mimsave(gif_path, gif_imgs, fps=35)

        outputs.append({
            "video_path": video_path,
            "gif_path": "",
            "jpg_path": jpg_path,
            "swing_name": swing_name,
            "swing_num": swing_num,
            "start_frame": st_frame_num,
            "end_frame": end_frame_num,
        })
        swing_num += 1
        writer.release()
    video_stream.release()
    print('writing swings: {} seconds'.format((datetime.datetime.now()-t1).seconds))

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


def calc_median_frame(video_stream, total_frames):
    frame_num = 5 * FPS
    frame_nums = []
    while frame_num <= total_frames - 5 * FPS:
        frame_nums.append(frame_num)
        frame_num += round(0.5 * FPS)

    frames = []
    for fnum in frame_nums:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fnum)
        _, frame = video_stream.read()
        frames.append(frame)

    video_stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
    med_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    # gray_med_frame = cv2.cvtColor(med_frame, cv2.COLOR_BGR2GRAY)
    hsv_med_frame = cv2.cvtColor(med_frame, cv2.COLOR_BGR2HSV)

    return hsv_med_frame


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


def detect_swing_body(frame, hsv_med_frame):
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
    max_box = None
    for cnt in cnts:
        box = cv2.boundingRect(cnt)
        x, y, w, h = box
        if w*h > max_area:
            max_area = w*h
            max_cnt = cnt
            max_box = [x, y, x+w, y+h]

    if max_cnt is None:
        return False

    (_, _), (MA, _), _ = cv2.fitEllipse(max_cnt)
    return max_box if MA >= 100 else None


def boxes_intersect(box1, box2):
    return (box1[0] < box2[2] and box2[0] < box1[2] and box1[1] < box2[3] and box2[1] < box1[3])


if __name__ == "__main__":
    srv_video_path = sys.argv[1]
    file_paths = srv_video_path.split('/')
    full_file_nm = file_paths[len(file_paths)-1]
    ext_idx = full_file_nm.rfind('.')
    src_file_nm = full_file_nm[0:ext_idx]

    sound_frames = [2, 44, 87, 94, 95, 107, 122, 157, 165, 197, 231, 232, 248, 252, 285, 286, 287, 298, 325, 343, 350, 366, 431, 433, 434, 435, 438, 461, 482, 541, 544, 545, 546, 548, 612, 613, 614, 620, 640, 655, 666, 678, 680, 681, 683, 686, 688, 689, 690, 691, 697, 698, 699, 700, 705,
                    706, 707, 708, 709, 712, 750, 751, 756, 800, 816, 922, 934, 1023, 1088, 1097, 1098, 1099, 1101, 1150, 1170, 1171, 1172, 1192, 1204, 1207, 1208, 1212, 1228, 1229, 1231, 1234, 1236, 1246, 1250, 1252, 1265, 1275, 1328, 1445, 1462, 1469, 1545, 1591, 1611, 1648, 1690, 1714, 1789, 1790, 1800]

    outputs = cut_swings(srv_video_path, './out', src_file_nm, sound_frames)
    print(outputs)
