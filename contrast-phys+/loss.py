import torch
import torch.nn as nn
tr = torch
import torch.nn.functional as F
import numpy as np
import torch.fft

class ContrastLoss(nn.Module):
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super(ContrastLoss, self).__init__()
        self.ST_sampling = ST_sampling(delta_t, K, Fs, high_pass, low_pass) # spatiotemporal sampler
        self.T_sampling = T_sampling(delta_t, K, Fs, high_pass, low_pass) # temporal sampler for GT signals
        self.distance_func = nn.MSELoss(reduction = 'mean') # mean squared error for comparing two PSDs

    def compare_samples(self, list_a, list_b, exclude_same=False):
        if exclude_same:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    if i != j:
                        total_distance += self.distance_func(list_a[i], list_b[j])
                        M += 1
        else:
            total_distance = 0.
            M = 0
            for i in range(len(list_a)):
                for j in range(len(list_b)):
                    total_distance += self.distance_func(list_a[i], list_b[j])
                    M += 1
        return total_distance / M

    def forward(self, model_output, GT_sig, label_flag):

        samples = self.ST_sampling(model_output)
        samples_GT = self.T_sampling(GT_sig, model_output.shape)

        # positive loss
        pos_loss = (self.compare_samples(samples[0], samples[0], exclude_same=True) + self.compare_samples(samples[1], samples[1], exclude_same=True)) / 2

        # negative loss           
        neg_loss = -self.compare_samples(samples[0], samples[1])

        
        if torch.sum(label_flag) == 0:
            pos_loss_GT = torch.zeros_like(pos_loss)
            neg_loss_GT = torch.zeros_like(neg_loss)
        else:
            # positive loss related to GT
            pos_loss_GT = (label_flag[0] * self.compare_samples(samples[0], samples_GT[0]) + label_flag[1] * self.compare_samples(samples[1], samples_GT[1])) / torch.sum(label_flag)
            # negative loss related to GT
            neg_loss_GT = - (label_flag[0] * self.compare_samples(samples[1], samples_GT[0]) + label_flag[1] * self.compare_samples(samples[0], samples_GT[1])) / torch.sum(label_flag)

        # overall contrastive loss
        loss = pos_loss + neg_loss + pos_loss_GT + neg_loss_GT

        # return overall loss, positive loss, and negative loss
        return loss, pos_loss, neg_loss, pos_loss_GT, neg_loss_GT


class ST_sampling(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input): # input: (2, N, T)
        samples = []
        for b in range(input.shape[0]): # loop over videos (totally 2 videos)
            samples_per_video = []
            for c in range(input.shape[1]): # loop for sampling over spatial dimension
                for i in range(self.K): # loop for sampling K samples with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    x = self.norm_psd(input[b, c, offset:offset + self.delta_t])
                    samples_per_video.append(x)
            samples.append(samples_per_video)
        return samples

class T_sampling(nn.Module):
    # temporal sampling on GT signals.
    
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input, ST_block_shape): # input: shape (2, T), ST_block_shape: (2, N, T)
        samples = []
        for b in range(input.shape[0]): # loop over GT signals (totally 2 GT signals)
            samples_per_sig = []
            for c in range(ST_block_shape[1]):
                for i in range(self.K): # loop for sampling K samples with time length delta_t along temporal dimension
                    offset = torch.randint(0, input.shape[-1] - self.delta_t + 1, (1,), device=input.device) # randomly sample along temporal dimension
                    x = self.norm_psd(input[b, offset:offset + self.delta_t])
                    samples_per_sig.append(x)
            samples.append(samples_per_sig)
        return samples


class CalculateNormPSD(nn.Module):
    # we reuse the code in Gideon2021 to get the normalized power spectral density
    # Gideon, John, and Simon Stent. "The way to my heart is through contrastive learning: Remote photoplethysmography from unlabelled video." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
    
    def __init__(self, Fs, high_pass, low_pass):
        super().__init__()
        self.Fs = Fs
        self.high_pass = high_pass
        self.low_pass = low_pass

    def forward(self, x, zero_pad=0):
        x = x - torch.mean(x, dim=-1, keepdim=True)
        if zero_pad > 0:
            L = x.shape[-1]
            x = F.pad(x, (int(zero_pad/2*L), int(zero_pad/2*L)), 'constant', 0)

        # Get PSD
        x = torch.view_as_real(torch.fft.rfft(x, dim=-1, norm='forward'))
        x = tr.add(x[:, 0] ** 2, x[:, 1] ** 2)

        # Filter PSD for relevant parts
        Fn = self.Fs / 2
        freqs = torch.linspace(0, Fn, x.shape[0])
        use_freqs = torch.logical_and(freqs >= self.high_pass / 60, freqs <= self.low_pass / 60)
        x = x[use_freqs]

        # Normalize PSD
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return x