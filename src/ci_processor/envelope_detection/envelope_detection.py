# -*- coding: utf-8 -*-
"""
List of functions for finding the envelope from raw CI output signals. Two methods 
can be used:
    1. windows method: The signal is divided into windows of sampletime 
    derived from measured pulses per second. For each window, the maximum height 
    is determined as the pulse height.
    2. cross correlation with pulse template method: A pulse template is drawn 
    from parameters in the clinical software. envelope is detected by crosscorrelating 
    the template with the raw signal.
    
Authors:
    Marieke ten Hoor <m.m.w.ten.hoor@student.rug.nl>
    Floris Rotteveel <f.rotteveel@rug.nl>
    Etienne Gaudrain <etienne.gaudrain@cnrs.fr>
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy import integrate
import pandas as pd

def find_nearest(array, value):
    """
    Find index of nearest match to input value in an array.

    Parameters
    ----------
    array : numpy array or list
        array to find index
    value : float,int
        value to match to array.

    Returns
    -------
    idx : idx
        index of nearest match to value inside array.
    """
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_time_arr(npdict, start_cue, tstart=None,tend=None):
    """
    Creates a time array from tstart to tend of length npdict with sampletime fs_scope.

    Parameters
    ----------
    npdict : dict 
        dict of measured signal with keys __info__ and seg1. seg1 contains n 
        dicts of size m (m = number of channels measured). n is number of repeated 
        measurements with different cue values (dF0 for example)
    start_cue: str
        string of the cue_val that will be used to create the time array.
    tstart : float, int
        start time of interest (s)
    tend : float, int
        end time of interest (s)

    Returns
    -------
    time_arr : numpy array
        array with time stampts of length of signal of first channel of first cue value.
    istart : idx
        index of tstart inside time_arr
    iend : idx
        index of tend inside time_arr
    """
    
    fs_scope = npdict['__info__']['fs_scope']
    time_arr = np.arange(0,len(npdict['seg1'][start_cue][1]))/fs_scope
    
    #optional outputs
    if tstart != None and tend != None:
        istart = find_nearest(time_arr,tstart)
        iend = find_nearest(time_arr,tend);
        
    elif tstart != None and tend == None:
        istart = find_nearest(time_arr,tstart)
        iend = None
    elif tstart == None and tend != None:
        iend = find_nearest(time_arr,tend);
        istart = None
    else:
        istart=  None
        iend = None
        
    return time_arr,istart,iend

def find_pulse_peaks(npdict,time_arr, istart,iend,minheight=0.05):
    """
    pulsefinder specific for this application.

    Parameters
    ----------
    npdict : dict
        dict of measured signal with keys __info__ and seg1. See function create_time_arr 
        for details.
    time_arr : numpy array
        array with time stampts creeated in create_time_arr
    istart : idx
        index of tstart inside time_arr
    iend : idx
        index of tend inside time_arr

    Returns
    -------
    pdict : dict
        dict with pulse magnitudes (mpulse) and pulse times (tpulse) for each channel 
        and cue value.
    """
    
    fs_scope = npdict['__info__']['fs_scope']
    pps = npdict['__info__']['ch_rate']
    minheight = 0.05 #level just above noise. found by trial&error.
    dist = fs_scope/pps*0.8
    
    pdict = {}
    for cue_val in list(npdict['seg1'].keys()):
        pdict[cue_val] = {}    
        for ch in list(npdict['seg1'][cue_val].keys()):
            pdict[cue_val][ch] = {}
            
            sig = npdict['seg1'][cue_val][ch][istart:iend]
            ipeaks,props = signal.find_peaks(sig,height=minheight,distance=dist)
            tpulse = time_arr[ipeaks]
            mpulse = sig[ipeaks]
            
            pdict[cue_val][ch]['mpulse'] = mpulse
            pdict[cue_val][ch]['tpulse'] = tpulse
    return pdict

def find_first_peak(npdict,cue_val,time_arr,ch1=1,minheight=0.05):
    """
    Find first peak in a signal.

    Parameters
    ----------
    npdict : dict
        dict of measured signals with keys __info__ and seg1. See function create_time_arr 
        for details.
    cue_val : str
        name of cue value, e.g. 'F0_9' for the signal with altered pitch of +9 semitones.
    time_arr : numpy array
        generated in create_time_arr
    ch1 : int
        number of first measured channel. ussually ch1 = 1

    Returns
    -------
    ipeak1 : idx
        index of first peak in signal of first channel.
    """
    
    fs_scope = npdict['__info__']['fs_scope']
    pps = npdict['__info__']['ch_rate']
    dist = fs_scope/pps*0.8
    sig = npdict['seg1'][cue_val][ch1]
    ipeaks,props = signal.find_peaks(sig,height=minheight,distance=dist)
    ipeak1max = ipeaks[0] #location of max of first peak
    sigbf1 = sig[:ipeak1max]
    res = [idx for idx in range(0, len(sigbf1) - 1) if sigbf1[idx] >
       0 and sigbf1[idx + 1] < 0 or sigbf1[idx] < 0 and sigbf1[idx + 1] > 0] #find zero-crossings
    ipeak1 = max(res) #last zero crossing before maximum of first peak (=center of first pulse)
    
    return ipeak1

def find_first_peak_staircase(npdict,cue_val,time_arr,fs_scope,pulsewidth):
    from statistics import mode
    import copy
    
    tdct = copy.deepcopy(npdict)    
    N = int(fs_scope*pulsewidth*4)
    threshold = 0.3
    for ch in list(npdict['seg1'][cue_val].keys()):
        sig  = copy.deepcopy(tdct['seg1'][cue_val][ch])
        sig[sig<0] = -1*sig[sig<0] #full wave rect.
        sig[sig<threshold] = 0
        sig = np.convolve(sig,np.ones(N)/N,mode='valid') #convolve to get rid of noisyness
        tdct['seg1'][cue_val][ch] = sig
    df = pd.DataFrame.from_dict(tdct['seg1'][cue_val])
    maxch = df.idxmax(axis=1)
    dif_maxch = np.diff(maxch)
    idx, peaks = signal.find_peaks(-dif_maxch,threshold=10)
    dif_idx = np.diff(idx)
    
    md = mode(dif_idx)
    wheremode = np.where(dif_idx == md)[0]
    ipeak1 = idx[wheremode[int(len(wheremode)/2)]]
    
    ch = maxch[ipeak1+5]
    looksize = 100
    sig = copy.deepcopy(npdict['seg1'][cue_val][ch][ipeak1-looksize:ipeak1+looksize])
    sig[np.logical_and(sig<threshold,sig>-threshold)] = 0
    zero_crossing = np.argmax(np.diff(np.sign(sig)))
    ipeak1 = ipeak1 + zero_crossing - looksize - N
    
    # ch1 = maxch[ipeak1+5]
    # ch2 = maxch[ipeak1+5]
    # looksize = 500
    # sig1 = copy.deepcopy(npdict['seg1'][cue_val][ch1][ipeak1-looksize:ipeak1+looksize])
    # sig2 = copy.deepcopy(npdict['seg1'][cue_val][ch2][ipeak1-looksize:ipeak1+looksize])
    # sig = sig1+sig2
    # sig[np.logical_and(sig<threshold,sig>-threshold)] = 0
    # zero_crossing = np.argmax(np.diff(np.sign(sig)))
    # ipeak1 = ipeak1 + zero_crossing - looksize
    
    
    # ipeak1= ipeak1 - int(np.round(fs_scope*pulsewidth)) #subtract pulsewidth to place window start at middle of the peak
    return ipeak1

def create_windows(time_arr,ch_rate_p,ipeak1):
    """
    Create time windows in a signal with each windows starting just before the 
    pulses of first channel with window size being channel rate.

    Parameters
    ----------
    time_arr : numpy array
        Generated in create_time_arr.
    ch_rate_p : float
        actual channel rate. Retrieved from npdict.__info__
    ipeak1 : idx
        index of first peak of first channel

    Returns
    -------
    iwindows : array of indices
        indices in time_arr of windows.
    windows : array of floats
        array of window times from time_arr.
    """
    
    tpulse_p = 1/ch_rate_p
    tp1 = time_arr[ipeak1]
    sl = np.arange(time_arr[0],time_arr[-1],tpulse_p)-tp1 #list of multiples of tpulse_p 
    shift= min(sl[sl>=0]) #smallest positive value
    windows = np.arange(time_arr[0]-shift,time_arr[-1],tpulse_p) #create windows of pulse len
    windows = windows[windows>=time_arr[0]]
    if windows[0] != time_arr[0]: #add first timestamp of time_arr
        windows = np.insert(windows,0,time_arr[0])
    windows = np.append(windows,time_arr[-1]) #add last time stamp in the end 
    iwindows = np.zeros(np.size(windows))
    for wnd_num in range(1,len(windows)):
        iwindows[wnd_num] = find_nearest(time_arr,windows[wnd_num])
    iwindows = np.rint(iwindows) #round to whole
    iwindows = iwindows.astype(int) #convert to int
    if iwindows[0] == iwindows[1]: #delete first or last index in case on of them is the same as the one after / before it.
        iwindows = np.delete(iwindows,0)
        windows = np.delete(windows,0)
    if iwindows[-2] == iwindows[-1]:
        iwindows = np.delete(iwindows,-1)
        windows = np.delete(windows,-1)
    return iwindows,windows
    
def create_envelope(sig, time_arr, iwindows, windows):
    """
    create envelope of maximum value per window.

    Parameters
    ----------
    sig : numpy array of float
        signal of specific cue_val and channel from npdict.
    time_arr: np array
        Generated in create_time_arr and shortened to only contain times from tstart to tend
    iwindows : array of indices
        array of window indices from time_array.
    windows : array of float
        array of window times from time_array.

    Returns
    -------
    env : array of 3xn floats
        triple vector with: window timestamps, exact pulsetimes, and pulse magnitudes per window.
    """
    sig_windowed = [sig[(iwindows[num]):(iwindows[num+1])] for num in range(0,len(iwindows)-1)] #divide signal into windows of channel rate
    
    peak_heights = []
    peak_idx = []
    for nth, sub_sig in enumerate(sig_windowed):
        sub_sig = np.where(sub_sig > 0, sub_sig, 0) #rectification
        peak_heights.append(max(sub_sig))
        idx_sig = np.argmax(sub_sig) + iwindows[nth]
        peak_idx.append(idx_sig)
    env = np.c_[windows[:-1], time_arr[peak_idx], peak_heights] #place peaks at start of each window
    return env

def envelope_allch(npdict,time_arr,cue_val,iwindows,windows):
    """
    For loop for running create_window for all channels of a cue value in npdict.

    Parameters
    ----------
    npdict : dict
        dict of measured signal with keys __info__ and seg1. See function create_time_arr 
        for details.
    time_arr: np array
        Generated in create_time_arr and shortened to only contain times from tstart to tend
    cue_val : string
        name of cue value, e.g. 'F0_9' for the signal with altered pitch of +9 semitones.
    iwindows : array of indices
        Generated by create_windows
    windows : array of floats
        Generated by create_windows

    Returns
    -------
    env_all : numpy array
        numpy array of vertically concatenated envelopes.
    """
    chlist = list(npdict['seg1'][cue_val].keys())
    env_all = pd.DataFrame(np.zeros((len(chlist),len(iwindows)-1)),index=chlist)
    env_pulsetimes = pd.DataFrame(np.zeros((len(chlist),len(iwindows)-1)),index=chlist)
    env_windowstamps = pd.DataFrame(np.zeros((len(chlist),len(iwindows)-1)),index=chlist)
    for ch in chlist:
        sig = npdict['seg1'][cue_val][ch]
        env = create_envelope(sig, time_arr, iwindows,windows)
        env = np.transpose(env)
        env_windowstamps.loc[ch,:] = env[0,:]
        env_pulsetimes.loc[ch,:] = env[1,:]
        env_all.loc[ch,:] = env[2,:] 
    return env_windowstamps,env_pulsetimes,env_all

def create_pulse_template(fs_scope,pulsewidth):
    """
    Create a simplified pulse shape based on pulsewidth from AB Soundwave and the 
    measured channel rate.


    Parameters
    ----------
    fs_scope : int
        sampling frequency of oscilloscope.
    pulsewidth : float
        pulse width as noted in AB SoundWave program. Pulsewidth indicates the 
        duration of one phase of the biphasic waveform.  

    Returns
    -------
    list
        pulseshape of template and number of samples of one pulsewidth.

    """
    pw_samples = int(np.round(pulsewidth*fs_scope))
    
    pulseshape = np.zeros(6*pw_samples) #generate single squarewave
    pulseshape[1*pw_samples:3*pw_samples] = -1
    pulseshape[3*pw_samples:5*pw_samples] = 1
    
    return [pulseshape, pw_samples]

def crosscorr_pulse_template(sig,pulseshape,pps_samples,thresfactor):
    """
    Perform cross-correlation of the signal with the pulseshape template. 

    Parameters
    ----------
    sig : np.array size nx1
        CI output signal of 1 channel
    pulseshape : np.array
        pulse template
    pps_samples : float
        pulses per second converted to samples
    thresfactor : float
        threshold factor multiplied with maximum pulse shape found. 0.005 seems 
        to work well to select only true pulses without deleting to many small pulses

    Returns
    -------
    crosscorr : np.array size nx1
        cross-correlation result
    ipeaks : np.array
        indices of found peaks
    """

    crosscorr = np.correlate(sig,pulseshape,mode='same')
    crosscorr[crosscorr<0] = 0
    
    #find maxima indices in lag.
    mindist = pps_samples*0.8
    thres = max(crosscorr)*thresfactor
    ipeaks,props = signal.find_peaks(crosscorr,distance=mindist,height=thres)
    xcorr_heights = props['peak_heights'] #only keep indices, not tuple type
    
    return crosscorr,ipeaks, xcorr_heights

def insert_missing_pulse_idx(ipeaks,pps_samples,iwindows):
    """
    
    NB APPROACH WHERE NUMBER OF MISSING PULSES IS CALCULATED BASED ON DIFPEAKS DOES NOT WORK OF N-OF-M SINCE IT ASSUMES THAT EACH WINDOW WILL HAVE A PEAK
    
    Insert missing pulses at even spacing between found pulses in cross-correlation.

    Parameters
    ----------
    ipeaks : np.array
        indices of found peaks.
    pps_samples : float
        pulses per second converted to samples

    Returns
    -------
    ipeaks : np.array
        new, corrected indices of found peaks.
    """
    
    difpeaks = np.diff(ipeaks)       
    missing_pulses = difpeaks/(pps_samples) - 1 #calculate number of missing pulses between indices of ipeaks
    newlen = len(ipeaks)+sum(np.round(missing_pulses)) #number of pulses that should be in the signal
    ipeaks_lin = np.linspace(ipeaks[0],ipeaks[-1],num=int(newlen))
    ipeaks_lin = np.round(ipeaks_lin)
    for idx, linpulse in enumerate(ipeaks_lin):
        linpar = np.arange(linpulse-5,linpulse+5)
        if bool(set(linpar) & set(ipeaks)):
            ipeaks_lin[idx] = np.nan
    double = np.append(ipeaks_lin,ipeaks) #appended linspaced values and detected pulses
    double = np.sort(double)
    double = double[np.logical_not(np.isnan(double))] #delete nans
    ipeaks = double.astype(int)
    
    ipeaks = ipeaks[ipeaks >= iwindows[1]] #remove all indices before first full window
    while ipeaks[0] >= iwindows[2]: #add indices up to first full window
        newidx = ipeaks[0] - (iwindows[2] - iwindows[1])
        ipeaks = np.insert(ipeaks,0,newidx)
    
    ipeaks = ipeaks[ipeaks < iwindows[-2]] #remove all indices after last full window
    while ipeaks[-1] < iwindows[-3]: #append indices up to last full window
        newidx = ipeaks[-1] + (iwindows[-2] - iwindows[-3])
        ipeaks = np.append(ipeaks,newidx)
    return ipeaks

def create_envelope_crosscorr(sig,time_arr,ipeaks,pw_samples):
    """
    Find envelope from peaks detected with template cross-correlation.

    Parameters
    ----------
    sig : np.array size nx1
        CI output signal of 1 channel
    time_arr : np array
        Generated in create_time_arr and shortened to only contain times from tstart to tend
    ipeaks : np.array
        indices of found peaks
    pw_samples : int
        number of samples for one phase of pulse

    Returns
    -------
    pulsetimes : np.array
        time points of pulses
    envelope : np.array
        height of pulses
    """
    
    pulsetimes = time_arr[ipeaks]
    envelope = np.zeros(np.shape(pulsetimes))
    for idx, sig_idx in enumerate(ipeaks):
        pulse_window = np.array(range(sig_idx-2*pw_samples,sig_idx+2*pw_samples)) #window around pulsetime
        pulse_window = pulse_window[pulse_window >= 0]             
        envelope[idx] = max(sig[pulse_window]) #max inside window = pulse height
    
    return pulsetimes, envelope

def envelope_allch_crosscorr(npdict,time_arr,iwindows,pulse_template,pps_samples,cue_val,chlist,thresfactor):
    """
    Perform envelope detection with cross-correlation using a template for all channels of a CI output.

    Parameters
    ----------
    npdict : dict
        dict of measured signals with keys __info__ and seg1. See function create_time_arr 
        for details.
    time_arr : np array
        Generated in create_time_arr and shortened to only contain times from tstart to tend
    iwindows : array of indices
        Generated by create_windows
    pulse_template : list 
        pulsetemplate with [0]:pulseshape of template and number of samples of one pulsewidth
        and [1]: number of samples for one phase of pulse
    pps_samples : float
        pulses per second converted to samples
    cue_val : str
        Name of cue val to evaluate from npdict
    chlist : list
        list of channels
    thresfactor : float
        threshold factor multiplied with maximum pulse shape found. 0.005 seems 
        to work well to select only true pulses without deleting to many small pulses.

    Returns
    -------
    df_pulsetimes : pd.DataFrame
        dataframe with pulsetimes for all channels of a desired cue_val
    df_env : pd.DataFrame
        dataframe with pulse magnitudes for all channels of a desired cue_val
    """
    
    pulseshape = pulse_template[0]
    pw_samples = pulse_template[1]
    for ch in chlist:
        sig = npdict['seg1'][cue_val][ch]
        
        crosscorr, ipeaks,xcorr_heights = crosscorr_pulse_template(sig,pulseshape,pps_samples,thresfactor)    
        pulse_heights = xcorr_heights/(2*pw_samples)
        ipeaks_insert = insert_missing_pulse_idx(ipeaks,pps_samples,iwindows)
        
        envelope = np.zeros(np.shape(ipeaks_insert)) #place pulseheights at ipeaks that were not inserted.
        for idxidx, val in enumerate(ipeaks):
            x = np.where(ipeaks_insert == ipeaks[idxidx])
            if x[0]:
                envelope[x[0]] = pulse_heights[idxidx]
        pulsetimes = time_arr[ipeaks_insert]
        
        if 'df_env' not in locals():
            df_env = pd.DataFrame(np.zeros((len(chlist),len(ipeaks_insert))),index=chlist)
            df_pulsetimes = pd.DataFrame(np.zeros((len(chlist),len(ipeaks_insert))),index=chlist)
        
        df_pulsetimes.loc[ch,:] = pulsetimes
        df_env.loc[ch,:] = envelope
    return df_pulsetimes, df_env

def envelope_allch_windowed_templ(npdict,time_arr,iwindows,pulse_template,pps_samples,cue_val,thresfactor):
    
    chlist = list(npdict['seg1'][cue_val].keys())
    env_all = pd.DataFrame(np.zeros((len(chlist),len(iwindows)-1)),index=chlist)
    env_pulsetimes = pd.DataFrame(np.zeros((len(chlist),len(iwindows)-1)),index=chlist)
    env_windowstamps = pd.DataFrame(np.zeros((len(chlist),len(iwindows)-1)),index=chlist)
    
    for ch in chlist:
        sig = npdict['seg1'][cue_val][ch]
        env = create_envelope_windowed_templ(sig, time_arr, iwindows,pulse_template,thresfactor)

        env = np.transpose(env)
        env_windowstamps.loc[ch,:] = env[0,:]
        env_pulsetimes.loc[ch,:] = env[1,:]
        env_all.loc[ch,:] = env[2,:] 
    return env_windowstamps,env_pulsetimes,env_all
        
        
def create_envelope_windowed_templ(sig,time_arr,iwindows,pulse_template,thresfactor):
    pulseshape = pulse_template[0]
    pw_samples = pulse_template[1]
    
    sig_windowed = [sig[(iwindows[num]):(iwindows[num+1])] for num in range(0,len(iwindows)-1)] #divide signal into windows of channel rate
    
    peak_heights = []
    peak_idx = []
    for nth, sub_sig in enumerate(sig_windowed):
        crosscorr, ipeak, height = crosscorr_window(sub_sig,pulseshape)
        pulse_height =  height/(2*pw_samples)
        if pulse_height < thresfactor:
            pulse_height = 0
        peak_heights.append(pulse_height)
        idx_sig = ipeak + iwindows[nth]
        peak_idx.append(idx_sig)
    
    #if pulse was split in two by windows, merge
    ipeaks_d = np.diff(peak_idx)
    for idx in np.where(ipeaks_d < 50)[0]: #pulses too close to each other
        p1 = peak_heights[idx]
        p2 = peak_heights[idx+1]
        if p1 >= p2:
            peak_heights[idx] += p2
            peak_heights[idx+1] = 0
        elif p1 < p2:
            peak_heights[idx] = 0
            peak_heights[idx+1] += p1
            
    env = np.c_[time_arr[iwindows[:-1]], time_arr[peak_idx], peak_heights] #place peaks at start of each window
    return env
    
def crosscorr_window(sub_sig,pulseshape):
    crosscorr = np.correlate(sub_sig,pulseshape,mode='same')
    crosscorr[crosscorr<0] = 0
    
    ipeak = np.argmax(crosscorr)
    xcorr_height = crosscorr[ipeak]
    return crosscorr, ipeak, xcorr_height
    
    
def plot_windows(npdict,cue_val,iwindows,xlim1,xlim2):
    """
    Plot window placement. Meant for development

    Parameters
    ----------
    npdict : dict
    dict of measured signal with keys __info__ and seg1. See function create_time_arr 
    for details.
    cue_val : string
        name of cue value, e.g. 'F0_9' for the signal with altered pitch of +9 semitones.
    iwindows : array of indices
        Generated by create_windows
    xlim1 : idx limit
    xlim2 : idx limit 2

    Returns
    -------
    None.
    """
    
    plt.figure()
    for ch in range(1,9):
        plt.plot(npdict['seg1'][cue_val][ch])
    
    plt.vlines(iwindows,ymin=-10,ymax=10,color='grey')
    plt.xlim(xlim1,xlim2) 
    return

def plot_envelope(tsig,sig,tenv,env,xlim1,xlim2,ch):
    plt.figure()
    plt.plot(tenv,env)
    plt.plot(tsig,sig)
    plt.xlim(xlim1,xlim2)
    plt.title('Channel:' + str(ch))
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.legend(['Envelope','Signal'])
    plt.show()