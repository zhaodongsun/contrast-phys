import cv2
import numpy as np
import torch
from PhysNetModel import PhysNet
from utils_sig import *
import matplotlib.pyplot as plt
from face_detection import face_detection


face_list, fps = face_detection(video_path='./my_video.avi')

print('\nrPPG estimation')
device = torch.device('cpu')

with torch.no_grad():
    face_list = torch.tensor(face_list.astype('float32')).to(device)
    model = PhysNet(S=2).to(device).eval()
    model.load_state_dict(torch.load('./model_weights.pt', map_location=device))
    rppg = model(face_list)[:,-1, :]
    rppg = rppg[0].detach().cpu().numpy()
    rppg = butter_bandpass(rppg, lowcut=0.6, highcut=4, fs=fps)

hr, psd_y, psd_x = hr_fft(rppg, fs=fps)

fig, (ax1, ax2) = plt.subplots(2, figsize=(20,10))

ax1.plot(np.arange(len(rppg))/fps, rppg)
ax1.set_xlabel('time (sec)')
ax1.grid('on')
ax1.set_title('rPPG waveform')

ax2.plot(psd_x, psd_y)
ax2.set_xlabel('heart rate (bpm)')
ax2.set_xlim([40,200])
ax2.grid('on')
ax2.set_title('PSD')

plt.savefig('./results.png')

print('heart rate: %.2f bpm'%hr)