# -*- coding: utf-8 -*-
"""
Created on 2025-04-14 
Script for recording CI implant-in-a-box output for Cochlear using two oscilloscopes.
This script depends on the tmctl package, which you will need to install from git.web.rug.nl/dBSPL/pytmctl

Authors: 
    Marieke ten Hoor <m.m.w.ten.hoor@student.rug.nl>
    Floris Rotteveel <f.rotteveel@rug.nl>
    Etienne Gaudrain <etienne.gaudrain@cnrs.fr>
"""

import os
import numpy as np
from pathlib import Path
from tmctl import TMCTL

from acquisition.dual_scopes import DualScopes
from acquisition.prepare_signals import prepare
from acquisition.sound_player import SoundPlayer
from ci_processor.npdict import npdict_to_zip
    
def record_Cochlear(path_sounds,destination=None):
    """
    Main function to record pulses from a Cochlear processor using two Yokogawa DL750 scopecorder 
    oscilloscopes and a sound player. This function prepares sound stimuli, configures oscilloscopes 
    and a soundcard, plays the stimuli, acquires data from the oscilloscopes, and saves the recorded 
    data into a compressed zip file.
    Parameters
    ----------
    path_sounds : str
        path to the directory containing the sound files to be used.
    destination : str, optional
        path to save the recorded data. If None, the data will be saved in the same 
        directory as `path_sounds`. If a directory is provided, the data will be saved 
        as a zip file in that directory. If a zip file path is provided, the data will 
        be saved directly to that file.
    Returns
    -------
    None
        The function saves the recorded data to a zip file and does not return any value.
    Notes
    -----
    - The AGC of the processor can be preconditioned at a certain level for a certain duration.
        This is done by playing the target sound for a certain time before sending the trigger.

    Raises
    ------
    ValueError
        If the `destination` is not a directory or a valid zip file path.
    """
    ## ----- SETTINGS ----- ##
    FS_SCARD = 192000       # Sample frequency of MOTU Driver
    FS_SCOPE = 500e3        # Sample frequency oscilloscopes
    VOLT_MAG = 4          # Based on expected magnitude pulse
    TRIG_POS = 10           # Trigger position (%)
    CH_RATE = 900           # Channel rate 
    WAIT_TIME = 2           # Time (sec) between recordings 
    LEVEL = 65              # Loudness in dBA of recording
    LEVEL_PRECOND = 65      # Preconditioning the AGC with sounds of same loudness
    T_PRECOND = 1.0         # Preconditioning length (sec)
    LEVEL_CALIBRATION = 75  # Soundcard calibrated at
    GAP = 0

    G_STIM = LEVEL - LEVEL_CALIBRATION  # Gain for stimulus
    G_PRECOND = LEVEL_PRECOND - LEVEL_CALIBRATION  # Gain for preconditioning

    ## ----- OSCILLOSCOPE SETTINGS ----- ##    
    electrodes_both = [np.arange(1,17), np.arange(17,23)]
    scope_channels_both = [electrodes_both[0],electrodes_both[1]-len(electrodes_both[0])]
    scope_models = ['M1', 'S']

    ## ----- PREPARE SOUNDS ----- ##
    stimuli, nrecs, _ = prepare_signals.prepare(path_sounds, FS_SCARD, FS_SCOPE, CH_RATE, scope_models, scope_channels_both,
                                                                        G_STIM, G_PRECOND, T_PRECOND, WAIT_TIME, GAP)    
    
    ## ----- PREPARE OSCILLOSCOPE AND MOTU ----- ##
    scope1_settings = {'wire': TMCTL.TM_CTL_ETHER, 'adr': '192.168.0.2,anonymous,', 'scope_channels': scope_channels_both[0], 'electrodes':electrodes_both[0], 'nrec':nrecs[0]}
    scope2_settings = {'wire': TMCTL.TM_CTL_USB, 'adr': "1", 'scope_channels': scope_channels_both[1],'electrodes':electrodes_both[1], 'nrec': nrecs[1]}
    settings = [scope1_settings, scope2_settings]
    scopes = DualScopes(settings)
    
    motu = SoundPlayer(dev_name='MOTU')

    ## ----- PLAY SOUND AND ACQUISITION ----- ##
    data = {}
    data['rec'] = {}

    ### NEW: INIT SAVE PATH
    if destination is None:
        destination = path_sounds
    if os.path.isdir(destination):
        fname = 'Cochlear_recording_' + time.strftime("%Y%m%d_%H%M") + '.zip'
        path_save = os.path.join(destination, fname)
    elif Path(destination).suffix == '.zip':
        path_save = destination
    else:
        raise ValueError("Destination must be a directory or a zip file")
    
    ## ----- PLAY SOUND AND ACQUISITION ----- ##
    for idx, seq in enumerate(stimuli):
        print('Playing sequence:', idx)
        n_triggers = len(seq['names'])
        max_retries = 4
        success = False

        for attempt in range(max_retries):
            print(f"Starting attempt {attempt + 1} for sequence {idx}")
            scopes.send(":HIST:CLE")
            msg = scopes.init_both_scopes(FS_SCOPE, VOLT_MAG, TRIG_POS, n_triggers, clock="EXTERNAL")

            time.sleep(5)
            scopes.send("START")
            time.sleep(1)

            motu.play(seq['Z'], FS_SCARD, mapping=[2, 1])

            try:
                scopes.wait_acq()  # adjust timeout if needed
                scopes.receive(":WAV:RECord?")
                success = True
                break  # success, exit retry loop
            except TimeoutError:
                print(f"[!] Timeout: Scope did not trigger on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print("Retrying sequence...")
                    continue
                else:
                    print("Maximum retries reached. Skipping this sequence.")

        if not success:
            print(f"[!] Skipping sequence {idx} due to repeated failure.")
            continue  # skip to next sequence if all retries failed

        for nth, name in enumerate(seq['names'][::-1]):
            scopes.send(":WAV:RECord %d" % -nth)
            print('Acquiring data: ' + name)
            data['rec'][name] = scopes.get_acq_full(mode='WORD')

        # Add basic info if not already added
        if "__info__" not in data:
            data["__info__"] = {
                'path': path_sounds,
                'fs_scard': int(FS_SCARD),
                'fs_scope': int(FS_SCOPE),
                'ch_rate': CH_RATE,
                'command_str': msg
            }

        # ✅ SAVE AFTER EACH SEQUENCE
        npdict_to_zip(data, path_save)
        print(f"[✓] Intermediate save after sequence {idx} to: {path_save}")

if __name__ == '__main__':
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    #rec_path = os.path.join(Path(dir_path).parent, 'stimuli', 'start_sound')
    #rec_path = r"C:\Users\Anita\Downloads\Bachelor's project\Audio clips\Environmental sounds\Natural soundscapes & water sounds"

    #record_Cochlear(rec_path)

    base_dir = r"C:\Users\Anita\Downloads\Bachelor's project\Audio clips\Recordings_Cochlear"

    for root, dirs, files in os.walk(base_dir):
        # Skip if folder has no audio files
        if not any(file.lower().endswith(('.wav', '.flac')) for file in files):
            continue

        # Skip if already processed
        if any(file.endswith(".zip") for file in files):
            print(f"[!] Skipping folder '{root}' — zip file already exists.")
            continue

        print(f"\n=== Starting recording for folder: {root} ===")
        try:
            record_Cochlear(root)
        except Exception as e:
            print(f"[!] Error processing {root}: {type(e).__name__}: {e}")
        print(f"=== Finished folder: {root} ===\n")
        time.sleep(3)
