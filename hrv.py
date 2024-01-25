import numpy as np
from scipy.signal import find_peaks
from astropy.timeseries import LombScargle
import neurokit2 as nk


def get_ibi(sig, sig_fps):
    ppg_clean = nk.ppg_clean(sig.copy(), sampling_rate=sig_fps, method='elgendi')
    info = nk.ppg_findpeaks(ppg_clean,sampling_rate=sig_fps)
    peak_loc = info["PPG_Peaks"]
    
    IPI = (peak_loc[1:]-peak_loc[0:-1])/sig_fps
    T = np.zeros(len(IPI));
    for i in range(len(IPI)):
        T[i] = np.sum(IPI[:i])

    IBI = np.array([T, IPI])
    return IBI

def freq_features(ibi):
    
    def find_highest_peak(sig):
        peak_loc = find_peaks(sig)[0]
        peak_amp = sig[peak_loc]
        highest_peak_loc = np.argmax(peak_amp)
        peak_loc = peak_loc[highest_peak_loc]
        return peak_loc
    
    freq,power = LombScargle(np.cumsum(ibi)/1000, ibi).autopower(minimum_frequency=0.01,maximum_frequency=0.51)
    idx_lf = np.where((freq>0.04)*(freq<0.15))
    idx_hf = np.where((freq>0.15)*(freq<0.40))
    
    freq_lf = freq[idx_lf]
    power_lf = np.sum(power[idx_lf])
    peak_loc_lf= 0 #find_highest_peak(power[idx_lf])
    
    freq_hf = freq[idx_hf]
    power_hf = np.sum(power[idx_hf])
    peak_loc_hf= find_highest_peak(power[idx_hf])
    
    f_list = []
    f_list.append(freq_lf[peak_loc_lf]) # lf_f
    f_list.append(freq_hf[peak_loc_hf]) # hf_f
    f_list.append(power_lf / (power_lf+power_hf)) # lfnu
    f_list.append(power_hf / (power_lf+power_hf)) # hfnu
    f_list.append(power_lf / power_hf) # lf_hf_ratio
    return f_list