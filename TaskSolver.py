'''
COMP9517 20T1 Group Project : Part2
user can draw a rectangular bounding box within the whole window
The system report
    a) pedestrians who enter the bounding box
    b) pedestrians who move out of the bounding box

How to run:
1). run script ./weights/download_yolov3_weights.sh to download pre-trained weights (around 200MB)
2). Place all images in following path -> Group_Component/seqeunce/*.jpg
3). Run python Task2.py

After run:
1). program will ask you to draw the bounding box
2). When finish draw, press 'Enter' to start program
'''

###
from deep_sort import DeepSort
###


import os
import cv2
import glob
import math
import argparse
import numpy as np
from collections import defaultdict

from tqdm import tqdm
from models import *  # set ONNX_EXPORT in models.py
from sort import *

drawing = False
running_state = False
display = start_image = None
x1 = y1 = x2 = y2 = -1

# Text Setting
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50) # position
fontScale = 0.8 # fontScale
thickness = 2 # Line thickness of 2 px
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)

def initialize_model(cfg='cfg/yolov3-spp.cfg',weights='weights/yolov3-spp-ultralytics.pt', img_size=512, device=None):
    if not device:
        device = torch_utils.select_device(device='' if torch.cuda.is_available() else 'CPU')
    img_size = img_size
    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    return model

def letterbox(img, new_shape=(416, 416), color=(114, 114, 114),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

# human detection
def detect(image_path, model, conf_thres = 0.3, device=None):
    if not device:
        device = torch_utils.select_device(device='' if torch.cuda.is_available() else 'CPU')
    img0 = cv2.imread(image_path)
    img = letterbox(img0, new_shape=512)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
   
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    t1 = torch_utils.time_synchronized()
    pred = model(img)[0]
    t2 = torch_utils.time_synchronized()

    # Apply NMS
    pred = non_max_suppression(pred, 0.3, 0.6, multi_label=False)
    detections = []
    det = pred[0]
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    
    # get only person class
    det = det[det[:,-1] == 0]

    bbox_xywh= []
    cls_conf = []
    
    # convert xyxy to xywh (x,y) is center
    for *xyxy, conf, cls in det:
        if conf >= conf_thres:
            xx1,yy1,xx2,yy2 = float(xyxy[0]),float(xyxy[1]),float(xyxy[2]),float(xyxy[3])
            bbox_xywh.append([(xx1+xx2)/2,(yy1+yy2)/2,xx2-xx1,yy2-yy1])
            cls_conf.append(float(conf))
    
    bbox_xywh = np.array(bbox_xywh)
    cls_conf = np.array(cls_conf)
        
    return bbox_xywh, cls_conf, img0
        
    
def draw_rectangle(event, x, y, flags, param):
    global x1, y1, x2, y2, drawing, display, running_state
    if not running_state:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            display = start_image.copy()
            x1, y1 = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                display = start_image.copy()
                cv2.rectangle(display, (x1, y1), (x, y), (0, 0, 255), thickness=2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x2, y2 = x, y
            cv2.rectangle(display, (x1, y1), (x, y), (0, 0, 255), thickness=2)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def checkOverlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other:
    if l1.x > r2.x or l2.x > r1.x:
        return False
    # If one rectangle is above left side of other:
    if l1.y > r2.y or l2.y > r1.y:
        return False
    return True

def random_color():
    return tuple(np.random.choice(range(256), size=3).tolist())

def get_euclidean(pt1,pt2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(pt1,pt2)]))

def get_coords(corner,type):
    # corner = [ (x1,y1) , (x1,y1),...... ]
    if type == 'min':
        min_x = 9999
        min_y = 9999
        for coords in range(len(corner)):
            if corner[coords][0] < min_x:
                min_x = corner[coords][0]
            if corner[coords][1] < min_y:
                min_y = corner[coords][1]
        return (min_x,min_y)
    else :
        min_x = -1
        min_y = -1
        for coords in range(len(corner)):
            if corner[coords][0] > min_x:
                min_x = corner[coords][0]
            if corner[coords][1] > min_y:
                min_y = corner[coords][1]
        return (min_x,min_y)
    
def task1(model, mot_tracker, image_path, proc_images, conf_thres):
    videos = []
    detected_object_ids = set()
    trajectories = defaultdict(list)
    colors = dict()

    for image_path in tqdm(proc_images):
        bbox_xywh, cls_conf, display = detect(image_path, model, device=device, conf_thres=conf_thres)
        tracked_objects = []
        
        if len(cls_conf) > 0:
            tracked_objects = mot_tracker.update(bbox_xywh, cls_conf, display)

        for detection in tracked_objects:
            obj_id = str(int(detection[4]))
            detected_object_ids.add(obj_id)

            if obj_id not in colors:
                colors[obj_id] = random_color()

            x_1, y_1, x_2, y_2 = int(detection[0]), int(detection[1]) , int(detection[2]), int(detection[3])
            cv2.rectangle(display, (x_1, y_1), (x_2, y_2), colors[obj_id], thickness=2)

            rectangle_midpoint = (round((x_2 + x_1) / 2), round((y_2 + y_1) / 2))
            if rectangle_midpoint not in trajectories[obj_id]:
                trajectories[obj_id].append(rectangle_midpoint)
            
        for obj, trajectory in trajectories.items():
            if len(trajectory) > 1:
                for i in range(0, len(trajectory) - 1):
                    cv2.line(display, trajectory[i], trajectory[i + 1], colors[obj], thickness=1)

        cv2.putText(display, "Unique pedestrians detected: %d" % (len(detected_object_ids)),
                    (30, 30), font, fontScale, BLACK, thickness, cv2.LINE_AA)
        cv2.imshow('Task1', display)
        
        videos.append(display)
        #wait no more than 60fps
        cv2.waitKey(int(1000/60))

    return videos

def task2(model, mot_tracker, image_path, proc_images, conf_thres):
    # Get bounding box from user
    global x1, y1, x2, y2, display, start_image, running_state

    videos = []

    start_image = cv2.imread(proc_images[0])
    welcome_text =  "Draw a rectangle and press 'Enter' When you're ready"
    (text_width, text_height) = cv2.getTextSize(welcome_text, font, fontScale=1, thickness=thickness)[0]
    text_offset_x = org[0]
    text_offset_y = org[1]

    box_coords = ((text_offset_x - 20 , text_offset_y -20), (text_offset_x  + text_width - 150, text_offset_y + text_height - 20))
    cv2.rectangle(start_image, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
    cv2.putText(start_image, welcome_text,
                org, font, fontScale, BLACK, thickness, cv2.LINE_AA)
    
    display = start_image.copy()

    cv2.namedWindow('Task2')
    cv2.setMouseCallback('Task2', draw_rectangle)

    while(1):
        cv2.imshow('Task2', display)
        if (cv2.waitKey(20) & 0xFF == 13):
            if (x1 == -1 or x2 == -1 or y1 == -1 or y2 == -1):
                print('Please draw a rectangle')
                continue
            if (abs(x1-x2) < 100 or abs(y1-y2) < 100):
                print('Your rectangle too small')
                continue
            running_state = True
            break    

    # not exist: not record, True: enter bounding box, False: move out bounding box
    tracking_bounding_box = dict()
    n_in = n_out = 0
    for image_path in tqdm(proc_images):
       
        bbox_xywh,cls_conf, display= detect(image_path, model, device=device, conf_thres=conf_thres)
        tracked_objects = []
        
        if len(cls_conf) > 0:
            tracked_objects = mot_tracker.update(bbox_xywh, cls_conf,display)
        
        # draw user's bounding box
        cv2.rectangle(display, (x1, y1), (x2, y2), RED , thickness=2)

        for detection in tracked_objects:
            obj_id = str(int(detection[4]))
            x_1, y_1, x_2, y_2 = int(detection[0]), int(detection[1]) , int(detection[2]),int(detection[3])
            color = GREEN
            # count pedestrians in/out of the bounding 
            if checkOverlap(Point(x1, y1), Point(x2, y2), Point(x_1, y_1), Point(x_2, y_2)):
                if obj_id not in tracking_bounding_box:
                    n_in += 1
                    tracking_bounding_box[obj_id] = True
                    color = RED
            else:
                if (obj_id in tracking_bounding_box) and (tracking_bounding_box[obj_id] == True):
                    # tracking_bounding_box[obj_id] = False
                    del tracking_bounding_box[obj_id]
                    n_out += 1
                    color = RED
            cv2.rectangle(display, (x_1, y_1), (x_2, y_2), color, thickness=2)
            cv2.putText(display, obj_id, (x_1, y_1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # print('In: %d\nOut: %d \r' % (n_in, n_out))
        cv2.putText(display, "Enter: %d" % (n_in),
                    (30, 30), font, fontScale, BLACK, thickness, cv2.LINE_AA)

        cv2.putText(display, "Move Out: %d" % (n_out),
                    (30, 60), font, fontScale, BLACK, thickness, cv2.LINE_AA)

        cv2.imshow('Task2', display)

        videos.append(display)
        #wait no more than 60fps
        cv2.waitKey(int(1000/60))

    return videos

def task3():
    videos = []
    colors = dict()
    for image_path in tqdm(proc_images):
        group_id = 0
        detected_object_ids = set()   
        group_count = 0
        groups = dict() 
        midpoints = []
        bbox_xywh, cls_conf, display = detect(image_path, model, device=device, conf_thres=conf_thres)
        tracked_objects = []
        distances = dict()
        if len(cls_conf) > 0:
            tracked_objects = mot_tracker.update(bbox_xywh, cls_conf, display)
        for detection in tracked_objects:
            obj_id = str(int(detection[4]))
            detected_object_ids.add(obj_id)
            x_1, y_1, x_2, y_2 = int(detection[0]), int(detection[1]) , int(detection[2]), int(detection[3])
            rectangle_midpoint = (round((x_2 + x_1) / 2), round((y_2 + y_1) / 2))
            midpoints.append([obj_id,rectangle_midpoint,(x_1,y_1),(x_2,y_2)])
        for i in range(len(midpoints)):
            if i != len(midpoints) - 1:
                for j in range(i+1,len(midpoints)):
                    dist = get_euclidean(midpoints[i][1],midpoints[j][1])
                    distances[midpoints[i][0],midpoints[j][0]] = dist

        for key,value in distances.items():
            if value <= 50:
                a,b = key
                # check if a is in a group
                a_in_group = False
                a_groupid = -1
                b_in_group = False
                b_groupid = -1
                # base case
                if group_id == 0:
                    group_id += 1
                    groups[group_id] = [a,b]
                else :
                    for k,v in groups.items():
                        if a in groups[k]: # check if a is in a group
                            a_in_group = True
                            a_groupid = k
                        if b in groups[k]: #check if b is in a group
                            b_in_group = True
                            b_groupid = k
                    if a_in_group and b_in_group and a_groupid != b_groupid: # check if both in different groups
                        # merge
                        new_group = groups[a_groupid] + groups[b_groupid]
                        group_id += 1
                        groups[group_id] = new_group
                        del groups[a_groupid]
                        del groups[b_groupid]
                    elif a_in_group and a_groupid != b_groupid: # if a in a group then add b to it
                        groups[a_groupid].append(b)
                    elif b_in_group and a_groupid != b_groupid: # if b is in a group add a to it
                        groups[b_groupid].append(a)
                    elif a_in_group == False and b_in_group == False: # a and b are not in a group
                        group_id += 1
                        groups[group_id] = [a,b]
        try:

            for key,values in groups.items():
                coords = []
                group_count += len(groups[key])
                for obj in values:
                    for i in midpoints: 
                        if i[0] == obj: # if object id in the group is the same as in midpoints get the coordinates
                            coords.append([i[2],i[3]])
                left_upper = []
                right_bottom = []
                for i in range(len(coords)):
                    left_upper.append(coords[i][0])
                    right_bottom.append(coords[i][1])
                left_upper_coords = get_coords(left_upper,'min')
                right_bottom_coords = get_coords(right_bottom,'max')
                color = (255, 0, 0) 
                cv2.rectangle(display, left_upper_coords, right_bottom_coords, color,thickness=2)
        except KeyError:
            pass
        cv2.putText(display, "Pedestrians not in groups: %d" % (abs(len(detected_object_ids) - group_count)),
                    (30, 30), font, fontScale, BLACK, thickness, cv2.LINE_AA)
        cv2.putText(display, "Pedestrians in groups: %d" % (group_count),
                    (30, 60), font, fontScale, BLACK, thickness, cv2.LINE_AA)
        cv2.imshow('Task3', display)
        videos.append(display)
        cv2.waitKey(int(1000/60))
        
    return videos

def main(args):
    # config
    task, cfg, weights, img_size, source, output, conf_thres = args.task, args.cfg, args.weights, args.img_size, args.source, args.output, args.conf_thres
    device = torch_utils.select_device(device='' if torch.cuda.is_available() else 'CPU')

    # initialize the model
    model = initialize_model(cfg=cfg,weights=weights,img_size=img_size,device=device)

    # initialize mot tracker
    mot_tracker = DeepSort('./deep_sort/ckpt.t7')

    # Path Setting
    image_path = source
    proc_images = sorted(glob.glob(os.path.join(image_path, '*.jpg')))

    if task == 1:
        videos = task1(model, mot_tracker, image_path, proc_images, conf_thres)
    elif task == 2:
        videos = task2(model, mot_tracker, image_path, proc_images, conf_thres)
    elif task == 3:
        videos = task3(model, mot_tracker, image_path, proc_images, conf_thres)

    cv2.destroyAllWindows()

    height, width, layer = videos[0].shape

    if not os.path.exists(output):
        os.makedirs(output)
    FPS = 15
    out = cv2.VideoWriter(os.path.join(output, f'output_task{task}.avi'),cv2.VideoWriter_fourcc(*'DIVX'), FPS, ( width,height ))

    for i in range(len(videos)):
        out.write(videos[i])
    out.release()


# Model  Setting

device = torch_utils.select_device(device='' if torch.cuda.is_available() else 'CPU')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, default=1,
                        help='the task to run (1-3)')
    parser.add_argument('--cfg', type=str,
                        default='cfg/yolov3-tiny3-1cls.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str,
                        default='weights/myweights.pt', help='weights path')
    parser.add_argument('--source', type=str,
                        default='Group_Component/sequence', help='source') # input folder
    parser.add_argument('--output', type=str, default='output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    args = parser.parse_args()
    print(args)

    if args.task not in range(1, 4):
        raise Exception("Invalid task provided")

    with torch.no_grad():
        main(args)
