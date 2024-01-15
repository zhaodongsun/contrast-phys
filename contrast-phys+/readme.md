# Contrast-Phys+



This is the official code repository of our paper "Contrast-Phys+: Unsupervised and Weakly-supervised Video-based Remote Physiological Measurement via Spatiotemporal Contrast" for weakly-supervised rPPG training.


## Prerequisite

The same requirement as Contrast-Phys.

## Dataset Preprocessing
We follow the same data preprocessing as Contrast-Phys. Since Contrast-Phys+ may use labels for training, you should add bvp for your labeled video files in the training set.

For each .h5 file in the training set
```
  X.h5
  ├── imgs  # face videos with cropped faces, shape [N, 128, 128, C]. N is the temporal length. C is the color channel, for RGB videos, C=3, for NIR videos, C=1.
  ├── bvp   # ground truth PPG signal, shape [N]. You may include it for your labeled videos.
```
 
For each .h5 file in the test set, you should include both `imgs` and `bvp` for performance evaluation.


## Training and testing

### Training
Please make sure your dataset is processed as described above. You only need to modify the code in a few places in `train.py` and `utils_data.py` to start your training. I have marked `TODO` and made comments in `train.py` and `utils_data.py` where you might need to modify. After modifying the code, you can directly run

```
python train.py
```
When you first run `train.py`, the training recording including model weights and some metrics will be saved at `./results/1`

### Testing

After training, you can test the model on the test set. Please make sure .h5 files in test set have `bvp`. You can directly run
```
python test.py with train_exp_num=1
```
The predicted rPPG signals and ground truth PPG signals are saved in `./results/1/1`. You can filter the rPPG signals by `butter_bandpass` function with lowcut=0.6 and highcut=4 and get heart rates by `hr_fft` function in `utils_sig.py`. To get the ground truth heart rates, you should first filter ground truth PPG signals by `butter_bandpass` function with lowcut=0.6 and highcut=4 and get ground truth heart rates by `hr_fft` function.
