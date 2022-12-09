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

    def forward(self, model_output):
        samples = self.ST_sampling(model_output)

        # positive loss
        pos_loss = (self.compare_samples(samples[0], samples[0], exclude_same=True) + self.compare_samples(samples[1], samples[1], exclude_same=True)) / 2

        # negative loss           
        neg_loss = -self.compare_samples(samples[0], samples[1])

        # overall contrastive loss
        loss = pos_loss + neg_loss

        # two sets of rPPG samples
        # samples = self.ST_sampling(model_output) # a list with length 2 including rPPG samples from the first video and rPPG samples from the second video
        # samples_ = self.ST_sampling(model_output)

        # We list combinations for both pos. loss (pull rPPG samples from the same video) and neg. loss (repel rPPG samples from two different videos).
        # positive loss
        # pos_loss = (self.compare_samples(samples[0], samples_[0]) + self.compare_samples(samples[1], samples_[1])
        #     + self.compare_samples(samples_[0], samples_[0], exclude_same=True) + self.compare_samples(samples_[1], samples_[1], exclude_same=True)
        #     + self.compare_samples(samples[0], samples[0], exclude_same=True) + self.compare_samples(samples[1], samples[1], exclude_same=True)) / 6
        # # negative loss           
        # neg_loss = -(self.compare_samples(samples[0], samples[1]) + self.compare_samples(samples_[0], samples_[1])
        #     + self.compare_samples(samples[0], samples_[1]) + self.compare_samples(samples_[0], samples[1])) / 4

        # # overall contrastive loss
        # loss = pos_loss + neg_loss

        # return overall loss, positive loss, and negative loss
        return loss, pos_loss, neg_loss


class ST_sampling(nn.Module):
    # spatiotemporal sampling on ST-rPPG block.
    
    def __init__(self, delta_t, K, Fs, high_pass, low_pass):
        super().__init__()
        self.delta_t = delta_t # time length of each rPPG sample
        self.K = K # the number of rPPG samples at each spatial position
        self.norm_psd = CalculateNormPSD(Fs, high_pass, low_pass)

    def forward(self, input): # input: (2, M, T)
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