import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch

def face_detection(video_path):

    device = torch.device('cpu')
    mtcnn = MTCNN(device=device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    N = 0
    video_list = []
    while(cap.isOpened()):
        ret, frame = cap.read()
        N += 1

        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_list.append(frame)
        else:
            break

        # if N/fps>60: # only get the first 60s video
        #     break

    cap.release()

    face_list = []
    for t, frame in enumerate(video_list):
        if t==0:
            boxes, _, = mtcnn.detect(frame) # we only detect face bbox in the first frame, keep it in the following frames.
        if t==0:
            box_len = np.max([boxes[0,2]-boxes[0,0], boxes[0,3]-boxes[0,1]])
            box_half_len = np.round(box_len/2*1.1).astype('int')
        box_mid_y = np.round((boxes[0,3]+boxes[0,1])/2).astype('int')
        box_mid_x = np.round((boxes[0,2]+boxes[0,0])/2).astype('int')
        cropped_face = frame[box_mid_y-box_half_len:box_mid_y+box_half_len, box_mid_x-box_half_len:box_mid_x+box_half_len]
        cropped_face = cv2.resize(cropped_face, (128, 128))
        face_list.append(cropped_face)
        
        print('face detection %2d'%(100*(t+1)/len(video_list)), '%', end='\r', flush=True)

    face_list = np.array(face_list) # (T, H, W, C)
    face_list = np.transpose(face_list, (3,0,1,2)) # (C, T, H, W)
    face_list = np.array(face_list)[np.newaxis]

    return face_list, fps