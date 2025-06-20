# -*- coding: utf-8 -*-
import zarr
import numpy as np
import json # Import json for dict serialization to attributes

def save_to_zarr(z_root, rec_key, pulse_times, pulse_amplitudes,
                 fs_scope, system_type,
                 wav_duration=None, wav_file_name=None, waveform_data=None,
                 compressor=None,
                 pulse_prms_metadata=None): # New parameter for pulse_prms as metadata
    """
    Saves vectorized data (pulse times, amplitudes) and metadata
    to a Zarr group, creating a hierarchical structure.

    Parameters
    ----------
    z_root : zarr.hierarchy.Group
        The root Zarr group where data will be saved.
    rec_key : str
        The hierarchical path for the current recording group within z_root.
        e.g., 'Cochlear/Speech/Emotional/t1_sad_s1_u06'.
    pulse_times : np.ndarray
        Flattened array of pulse times for all channels.
    pulse_amplitudes : np.ndarray
        Flattened array of pulse amplitudes for all channels.
    fs_scope : int
        Sampling rate of the oscilloscope data.
    system_type : str
        Type of CI system ('Cochlear' or 'AB').
    wav_duration : float, optional
        Duration of the original WAV file in seconds.
    wav_file_name : str, optional
        Name of the original WAV file.
    waveform_data : np.ndarray, optional
        Raw audio waveform data.
    compressor : numcodecs.blosc.Blosc, optional
        Zarr compressor to use for arrays. If None, zarr default will be used.
    pulse_prms_metadata : dict, optional
        Dictionary of pulse parameters (metadata) to be saved as Zarr attributes.
    """
    # Create the hierarchical group for the recording
    # Use overwrite=True for development, consider False in production to prevent accidental overwrites
    z_group = z_root.require_group(rec_key)

    # Save pulse data as Zarr arrays
    # Ensure they are 1D arrays of floats as expected by `plot_electrodogram` (my updated one)
    # and the flattening logic in `main_pipeline.py`.
    if pulse_times is not None and pulse_times.size > 0:
        z_group.create_dataset('pulse_times', data=pulse_times.astype('f4'),
                               chunks=True, compressor=compressor, overwrite=True)
    else:
        z_group.create_dataset('pulse_times', shape=(0,), dtype='f4',
                               chunks=True, compressor=compressor, overwrite=True)

    if pulse_amplitudes is not None and pulse_amplitudes.size > 0:
        z_group.create_dataset('pulse_amplitudes', data=pulse_amplitudes.astype('f4'),
                               chunks=True, compressor=compressor, overwrite=True)
    else:
        z_group.create_dataset('pulse_amplitudes', shape=(0,), dtype='f4',
                               chunks=True, compressor=compressor, overwrite=True)

    # Save metadata as attributes
    z_group.attrs['fs_scope'] = int(fs_scope) # Sampling rate of scope
    z_group.attrs['system_type'] = system_type

    if wav_duration is not None:
        z_group.attrs['wav_duration'] = float(wav_duration)
    if wav_file_name is not None:
        z_group.attrs['wav_file_name'] = str(wav_file_name)
    
    # Save pulse_prms as attributes (if it's a dict)
    if pulse_prms_metadata is not None and isinstance(pulse_prms_metadata, dict):
        # Flatten the pulse_prms dict into attributes
        # Or you can save the whole dict as a JSON string attribute if it's deeply nested
        z_group.attrs['pulse_parameters'] = json.dumps(pulse_prms_metadata) # Save as JSON string
        # Alternatively, iterate and save individual items if they are simple:
        # for k, v in pulse_prms_metadata.items():
        #     z_group.attrs[f"pulse_prm_{k}"] = v # Careful with types here for Zarr attrs


    # Save raw waveform data if provided (optional)
    if waveform_data is not None and waveform_data.size > 0:
        z_group.create_dataset('waveform', data=waveform_data.astype('f4'),
                               chunks=True, compressor=compressor, overwrite=True)
    else:
        z_group.create_dataset('waveform', shape=(0,), dtype='f4',
                               chunks=True, compressor=compressor, overwrite=True)

    print(f"Saved recording '{rec_key}' to Zarr.")