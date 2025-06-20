# -*- coding: utf-8 -*-
"""
Functions to design and prepare sound stimuli for recording with Yokogawa DL750 Scopecorders.
"""
import sounddevice as sd
import soundfile as sf
from glob import glob
import os
from matplotlib import pyplot as plt

def organize_sound_list(path_sounds,models,channels,FS_SCOPE):
    """
    Organizes a list of sound files from a specified directory and calculates 
    the record length for each oscilloscope model.
    Parameters
    ----------
    path_sounds : str
        Path to the directory containing the sound files.
    models : list
        List of strings of models to calculate the record length for.
    channels : list
        List of oscilloscope channel numbers corresponding to each model.
    FS_SCOPE : float
        Sampling frequency of both scopes.
    Returns
    -------
    sound_list : list of dict
        A list of dictionaries where each dictionary contains:
        - 'name' (str): The name of the sound file (without extension).
        - 'x' (numpy.ndarray): The audio data of the sound file.
        - 'fs' (int): The sampling frequency of the sound file.
    nrec_list : list of list
        A nested list where each sublist corresponds to a model and contains 
        the calculated record lengths for the sound files.
    Notes
    -----
    - The function determines the most common file type in the directory 
        (either `.flac` or `.wav`) and processes only files of that type.
    - The `calculate_record_length` function is used to compute the record 
        length for each model based on the audio data, channels, and scope 
        sampling frequency.
    """
    from tmctl import calculate_record_length
    file_list = os.listdir(path_sounds)
    
    # Find what filetype is most common in folder (.flac or .wav)
    print('Finding list of sounds...')
    filetypes = []
    for f in file_list:
        filetypes.append(f.split('.')[-1]) # .flac or .wav
    filetype = '.' + max(set(filetypes), key = filetypes.count)
    
    # Create dictionary with all sounds
    soundfiles = [f for f in file_list if f.endswith(filetype)] # All .flac or .wav files
    sound_list = []

    nrec_list = []
    for idx, _ in enumerate(models):
        nrec_list.append([]) # Create empty list for each model

    for file in soundfiles:
        key = file.split('.')[0] # Remove filetype
        d = {}
        d['name'] = key
        path_sound = os.path.join(path_sounds, file)
        sound_new, fs = sf.read(path_sound)

        if sound_new.ndim == 2:
            sound_new = sound_new.mean(axis=1)
        
        d['x'] = sound_new
        d['fs'] = fs
        sound_list.append(d)
        
        for idx,model in enumerate(models):
            ch = channels[idx] # Get channels for each model
            out1= calculate_record_length(len(sound_new)/fs, len(ch), FS_SCOPE, model=model)
            nrec_list[idx].append(out1) # Get record length for each model
    return sound_list, nrec_list

def create_sequences(sound_list,nrec_list, models = ['M1']):
    """
    Splits a list of sounds into sequences based on the maximum number of acquisitions 
    allowed for specified models and recording lengths.
    Parameters
    ----------
    sound_list : list
        A list of sound data to be divided into sequences.
    nrec_list : list
        A list of recording lengths (in samples) corresponding to each model in `models`.
    models : list of str, optional
        A list of model identifiers (e.g., 'M1', 'S') for which the maximum number of 
        acquisitions is defined. Default is ['M1'].
    Returns
    -------
    list of list
        A list of sequences, where each sequence is a list of sounds. Each sequence 
        contains up to the maximum number of acquisitions allowed for the specified 
        model and recording length.
    Notes
    -----
    - The maximum number of acquisitions for each model and recording length is defined 
        in the `max_acq` dictionary, based on the DL750 Scopecorder User's Manual 
        (IM 701210-06E) Appendix 2.
    """
    #Create dictionary with max number of acquisitions
    max_acq = dict.fromkeys(['M1', 'S'])    # Dictionary with max. number of acquisition of history 
                                            # memory for /M1 and /S (10M). Retrieved from DL750 Scopecorder 
                                            # User's Manual (IM 701210-06E) Appendix 2
    max_acq['M1'] = {1000: 2000, 2500: 1454, 5000: 976, 10000: 728, 25000: 369,
                    50000: 185, 100000: 92, 250000: 38, 500000: 18, 1000000: 8,
                    2500000: 2, 5000000: 2, 10000000: 1, 25000000: 1, 50000000:1,
                    100000000: 1, 250000000: 1}
    max_acq['S'] = {1000: 2000, 2500: 483, 5000: 324, 10000: 241, 25000: 121,
                    50000: 60, 100000: 29, 250000: 11, 500000: 4, 1000000: 3,
                    2500000: 1, 5000000: 1, 10000000: 1, 25000000: 1, 50000000:1,
                    100000000: 0, 250000000: 0}
    
    #Get number of acquisitions
    n_acq_all = []
    for idx,model in enumerate(models):
        nrec_m = nrec_list[idx]
        n_acq_all.append(max_acq[model][nrec_m])
    
    if len(n_acq_all) > 1:
        n_acq = min(n_acq_all)
    else:
        n_acq = n_acq_all[0]

    #Design sequences    
    n_sounds = len(sound_list) # Number of sounds in folder
    n_segs_full = n_sounds//n_acq # Number full of segments
    n_sounds_left = n_sounds%n_acq 
    
    sequences = []
    for seg in range(n_segs_full):
        seg_sounds = sound_list[seg*n_acq:(seg+1)*n_acq]# Create segment with n_acq sounds
        sequences.append(seg_sounds)# Add segment to list of sequences
    if n_sounds_left > 0: #last segment
        seg_sounds = sound_list[-n_sounds_left:]# Create segment with remaining sounds
        sequences.append(seg_sounds)# Add segment to list of sequences
    
    return sequences #list of lists

def create_stimulus(sound,fs,FS_SCOPE,FS_SCARD,nrec_min,WAIT_TIME,G_STIM,G_PRECOND,T_PRECOND):
    """
    Generate a stimulus signal with preconditioning and trigger.
    Parameters
    ----------
    sound : numpy.ndarray
        The input sound signal as a 1D array.
    fs : float
        Sampling rate of the input sound signal (Hz).
    FS_SCOPE : float
        Sampling rate of the oscilloscope (Hz).
    FS_SCARD : float
        Sampling rate of the soundcard (Hz).
    nrec_min : int
        Minimum number of recording samples required.
    WAIT_TIME : float
        Pause duration between recordings (seconds).
    G_STIM : float
        Gain applied to the stimulus signal (dB).
    G_PRECOND : float
        Gain applied to the preconditioning signal (dB).
    T_PRECOND : float
        Duration of the preconditioning signal (seconds).
    Returns
    -------
    numpy.ndarray
        A 2D array where the first column is the trigger signal and the second column
        is the stimulus signal, both resampled to the soundcard's sampling rate.
    """
    import numpy as np
    from scipy import interpolate

    if sound.ndim > 1:
        sound = sound.mean(axis=1)

    # Add pause to wait for trigger
    n = 3 
    n_time = n*(1/FS_SCARD) 
    wait_trig = np.zeros(int(2*n_time*fs))
    sound = np.r_[wait_trig, sound]
    
    # Calculate nrec and add zeros to fill nrec
    nrec_samples = int((nrec_min/FS_SCOPE)*fs)     # Minimal number of samples to fill nrec
    nrec_empty = int(nrec_samples - len(sound))  # Number of nrec samples where no sound is played
    if nrec_empty > 0:
        sound = np.r_[sound, np.zeros(nrec_empty)]
    
    # Add extra pause in between recordings
    wait_samples = int(WAIT_TIME*fs) 
    sound = np.r_[sound, np.zeros(wait_samples)]
    
    # Interpolate sound signal to go from original fs to soundcard fs
    t_stimulus = len(sound)/fs
    t_arr_sound = np.linspace(0, t_stimulus, len(sound))
    t_arr_scard = np.linspace(0, t_stimulus, int(t_stimulus*FS_SCARD))
    interp = interpolate.interp1d(t_arr_sound, sound, bounds_error=False, fill_value=0)
    stimulus1 = interp(t_arr_scard)
    
    # Add preconditioning to stabilize the AGC
    stimulus_att = stimulus1*10**(G_STIM/20)
    precond_sound_att = stimulus1*10**(G_PRECOND/20) # Attenuate preconditioning sound
    precond_sound_rep = np.tile(precond_sound_att, int(np.ceil(T_PRECOND * FS_SCARD / len(precond_sound_att))))[:int(T_PRECOND * FS_SCARD)]
    stimulus = np.r_[precond_sound_rep, stimulus_att]
    
    # Prepare trigger
    trig = np.zeros_like(stimulus1)
    n = 3
    m = 1 # Magnitude trigger
    trig[n:2*n] = np.linspace(0,m,n)
    trig[2*n:3*n] = m
    trig[3*n:4*n] = np.linspace(m,0,n)
    trig[4*n-1:5*n-1] = np.linspace(0,-m,n)
    trig[5*n-1:6*n-1] = -m
    trig[6*n-1:7*n-1] = np.linspace(-m,0,n)
    trigger = np.r_[np.zeros_like(precond_sound_rep), trig] # To prevent echo -> add pause in between
    
    X = np.vstack((trigger, stimulus)).T
    return X

def prepare(path_sounds, FS_SCARD, FS_SCOPE, CH_RATE, models, channels,G_STIM, G_PRECOND, T_PRECOND, WAIT_TIME, GAP):
    '''
    Prepares sound stimuli for recording with Yokogawa DL750 Scopecorders.

    Parameters
    ----------
    path_sounds : str
        Path to the folder containing sound files (.wav or .flac).
    FS_SCARD : float
        Sample frequency of the sound card (MOTU driver).
    FS_SCOPE : float
        Sample frequency of the oscilloscope.
    CH_RATE : float
        Pulse frequency of the CI processor (in pulses per second).
    models : list of str
        List of Yokogawa DL750 models (e.g., ['M1', 'S']).
    channels : list of list of int
        List of channel configurations for each model.
    G_STIM : float
        Gain applied to the stimulus signal (in dB).
    G_PRECOND : float
        Gain applied to the preconditioning signal (in dB).
    T_PRECOND : float
        Duration of the preconditioning signal (in seconds).
    WAIT_TIME : float
        Pause duration between recordings (in seconds).
    GAP : float
        Gap duration between stimuli (in seconds).

    Returns
    -------
    stimuli : list of dict
        List of dictionaries containing prepared stimuli for each sequence.
        Each dictionary has keys:
            - 'names': List of sound names in the sequence.
            - 'Z': Array with trigger in the first column and sound data in the second column.
    nrec_max : list of int
        Maximum record lengths for each model.
    sound_list : list of dict
        List of dictionaries containing sound data. Each dictionary has keys:
            - 'name': Name of the sound file (without extension).
            - 'x': Sound data as a NumPy array.
            - 'fs': Sampling frequency of the sound.
    '''
    import numpy as np
    from scipy import interpolate

    sound_list, nrec_list = organize_sound_list(path_sounds,models,channels,FS_SCOPE)

    # -----------------PREPARE SOUNDS---------------------
    nrec_max = []
    for idx, _ in enumerate(nrec_list):
        nrec_max.append(max(nrec_list[idx])) # Get max. record length for each model
    sequences = create_sequences(sound_list,nrec_max,models=models)
    nrec_min = min(nrec_max) #TODO Check if this always works out well. Get minimum record length of all models
    
    stimuli = []
    for idx,seq in enumerate(sequences):
        print(f'Preparing sequence {idx}')
        stimuli.append({}) # Create empty dictionary for each segment
        stimuli[idx]['names'] = []
        Z = np.array([[],[]]).T
        for iidx, d in enumerate(seq):
            name = d['name']
            stimuli[idx]['names'].append(name) # Add name of sound to list of sounds in segment
            sound = d['x']
            fs = d['fs']
            
            X = create_stimulus(sound,fs,FS_SCOPE,FS_SCARD,nrec_min,WAIT_TIME,G_STIM,G_PRECOND,T_PRECOND)
            Z = np.r_[Z,X]
        
        stimuli[idx]['Z'] = Z # All cue_list of one segment
        print('Prepared sounds: '+ str(stimuli[idx]['names']))   
    return stimuli, nrec_max, sound_list