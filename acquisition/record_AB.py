# -*- coding: utf-8 -*-
import os
import numpy as np

from pathlib import Path
from tmctl import TMCTL
from dual_scopes import DualScopes
import prepare_signals
from npdict import npdict_to_zip
import time
from sound_player import SoundPlayer
from elpi_recording.dual_scopes import DualScopes
from elpi_recording.prepare_signals import prepare
from elpi_recording.sound_player import SoundPlayer
from ci_processor.vectorizer.npdict import npdict_to_zip

def record_AB(path_sounds, destination=None):
    FS_SCARD = 192000
    FS_SCOPE = 500e3
    VOLT_MAG = 4  # Adjust as needed for AB system
    TRIG_POS = 10
    CH_RATE = 900
    WAIT_TIME = 0
    LEVEL = 65
    LEVEL_PRECOND = 65
    T_PRECOND = 1.0
    LEVEL_CALIBRATION = 75
    GAP = 0

    G_STIM = LEVEL - LEVEL_CALIBRATION
    G_PRECOND = LEVEL_PRECOND - LEVEL_CALIBRATION

    electrodes = [np.arange(1, 17)]
    scope_channels = electrodes
    scope_models = ['M1']

    stimuli, nrecs, _ = prepare_signals.prepare(path_sounds, FS_SCARD, FS_SCOPE, CH_RATE, scope_models, scope_channels,
                                                G_STIM, G_PRECOND, T_PRECOND, WAIT_TIME, GAP)

    scope1_settings = {
        'wire': TMCTL.TM_CTL_ETHER,
        'adr': '192.168.0.2,anonymous,',
        'scope_channels': scope_channels[0],
        'electrodes': electrodes[0],
        'nrec': nrecs[0]
    }
    settings = [scope1_settings]
    scopes = DualScopes(settings)
    motu = SoundPlayer(dev_name='MOTU')

    data = {'rec': {}}

    if destination is None:
        destination = path_sounds
    if os.path.isdir(destination):
        fname = 'AB_recording_' + time.strftime("%Y%m%d_%H%M") + '.zip'
        path_save = os.path.join(destination, fname)
    elif Path(destination).suffix == '.zip':
        path_save = destination
    else:
        raise ValueError("Destination must be a directory or a zip file")

    for idx, seq in enumerate(stimuli):
        print(f'Playing sequence: {idx}')
        n_triggers = len(seq['names'])
        max_retries = 3
        success = False

        for attempt in range(max_retries):
            print(f"Starting attempt {attempt + 1} for sequence {idx}")
            scopes.send(":HIST:CLE")
            msg = scopes.init_both_scopes(FS_SCOPE, VOLT_MAG, TRIG_POS, n_triggers, clock="EXTERNAL")

            time.sleep(2)
            scopes.send("START")
            time.sleep(1)

            motu.play(seq['Z'], FS_SCARD, mapping=[2, 1])

            try:
                scopes.wait_acq()
                scopes.receive(":WAV:RECord?")
                success = True
                break
            except TimeoutError:
                print(f"[!] Timeout: Scope did not trigger on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print("Retrying sequence...")
                    continue
                else:
                    print("Maximum retries reached. Skipping this sequence.")

        if not success:
            print(f"[!] Skipping sequence {idx} due to repeated failure.")
            continue

        for nth, name in enumerate(seq['names'][::-1]):
            scopes.send(":WAV:RECord %d" % -nth)
            print('Acquiring data: ' + name)
            data['rec'][name] = scopes.get_acq_full(mode='WORD')

        if "__info__" not in data:
            data["__info__"] = {
                'path': path_sounds,
                'fs_scard': int(FS_SCARD),
                'fs_scope': int(FS_SCOPE),
                'ch_rate': CH_RATE,
                'command_str': msg
            }

        npdict_to_zip(data, path_save)
        print(f"[✓] Intermediate save after sequence {idx} to: {path_save}")
    del scopes, motu
    print("Recording complete.")

if __name__ == '__main__':
    base_dir = r"C:\Users\Anita\Downloads\Bachelor's project\Audio clips\Recordings_AB"

    for root, dirs, files in os.walk(base_dir):
        if not any(file.lower().endswith(('.wav', '.flac')) for file in files):
            continue
        if any(file.endswith(".zip") for file in files):
            print(f"[!] Skipping folder '{root}' — zip file already exists.")
            continue

        print(f"\n=== Starting recording for folder: {root} ===")
        try:
            record_AB(root)
        except Exception as e:
            print(f"[!] Error processing {root}: {type(e).__name__}: {e}")
        print(f"=== Finished folder: {root} ===")
        time.sleep(3)