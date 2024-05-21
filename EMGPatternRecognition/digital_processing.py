import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
#import pyyawt





def bp_filter(x, low_f, high_f, samplerate, plot=False):
   # x = x - np.mean(x)
    '''
    def bp_filter(x, low_f, high_f, samplerate, plot=False):
    '''
    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')

    x_filt = signal.filtfilt(b, a, x)

    if plot:
        t = np.arange(0, len(x) / samplerate, 1 / samplerate)
        plt.plot(t, x)
        plt.plot(t, x_filt, 'k')
        plt.autoscale(tight=True)
        plt.xlabel('Time')
        plt.ylabel('Amplitude (mV)')
        plt.show()

    return x_filt

def bp_filter_ndimSignalCOlume(x, low_f, high_f, samplerate):
   # x = x - np.mean(x)
    '''
    @matrix version of filter
    def bp_filter(x , low_f, high_f, samplerate =sample frequency, plot=False):
    x = np.array(x) shape (n_samples, n_channels)
    return : x but filter version
    process
    first transpose matrix  (n_channels, n_samples)
    second filter each row append new matrix
    after finish filter transpose new matrix (n_channels, n_samples) =>  (n_samples, n_channels)
    '''
    print(x.shape)
    low_cutoff_bp = low_f / (samplerate / 2)
    high_cutoff_bp = high_f / (samplerate / 2)

    [b, a] = signal.butter(5, [low_cutoff_bp, high_cutoff_bp], btype='bandpass')


    #start filtering


    x_filt_matrix=[]
    x = x.T
    print(x.shape)
    for i in x:# get each row this represent after transpose channels
       x_filt_vector = signal.filtfilt(b, a, i)
       x_filt_matrix.append(x_filt_vector)


    x_filt_matrix = np.array(x_filt_matrix).T

    print(pd.DataFrame(x_filt_matrix))



    return x_filt_matrix


def plot_signal(x, samplerate, chname):
    t = np.arange(0, len(x) / samplerate, 1 / samplerate)
    plt.plot(t, x)
    plt.autoscale(tight=True)
    plt.xlabel('Time')
    plt.ylabel('Amplitude (mV)')
    plt.title(chname)
    plt.show()


