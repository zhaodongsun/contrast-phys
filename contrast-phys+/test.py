import numpy as np
import h5py
import torch
from PhysNetModel import PhysNet
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json

ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    e = 29 # the model checkpoint at epoch e
    train_exp_name = 'default'
    train_exp_num = 1 # the training experiment number
    train_exp_dir = './results/%s/%d'%(train_exp_name, train_exp_num) # training experiment directory
    time_interval = 30 # evaluate 30s clips

    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    else:
        device = torch.device('cpu')

@ex.automain
def my_main(_run, e, train_exp_dir, device, time_interval):

    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    model = PhysNet(config_train['S'], config_train['in_ch']).to(device).eval()
    model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt'%(e), map_location=device)) # load weights to the model

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3,0,1,2))
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        rppg = model(img_batch)[:,-1, :]
        rppg = rppg[0].detach().cpu().numpy()
        return rppg

    for h5_path in test_list:
        h5_path = str(h5_path)

        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            bvp = f['bvp']
            # bvppeak = f['bvp_peak']
            fs = config_train['fs']

            duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
            num_blocks = int(duration // time_interval)

            rppg_list = []
            bvp_list = []
            # bvppeak_list = []

            for b in range(num_blocks):
                rppg_clip = dl_model(imgs[b*time_interval*fs:(b+1)*time_interval*fs])
                rppg_list.append(rppg_clip)

                bvp_list.append(bvp[b*time_interval*fs:(b+1)*time_interval*fs])
                # bvppeak_list.append(bvppeak[b*time_interval*fs:(b+1)*time_interval*fs])

            rppg_list = np.array(rppg_list)
            bvp_list = np.array(bvp_list)
            # bvppeak_list = np.array(bvppeak_list)
            # results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'bvppeak_list':bvppeak_list}
            results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
            np.save(pred_exp_dir+'/'+h5_path.split('/')[-1][:-3], results)