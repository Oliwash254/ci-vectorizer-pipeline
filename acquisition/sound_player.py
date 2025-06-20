# -*- coding: utf-8 -*-
import sounddevice as sd
import numpy as np

class SoundPlayer(): 
    
    """
    A class to handle acoustic playback using a specified sound device.

    Attributes
    dev_id : int or None
        The device ID of the selected sound device. None if no device is selected.

    Methods
    __init__(dev_name=None)
        Initializes the acoustic_player with an optional device name.
    play(x, fs, level=0, t_zeros=None)
        Plays a short sound stimulus with specified parameters.
    """
    def __init__(self,dev_name=None):
        self.dev_id = None
        if sd._initialized:
            sd._terminate()
            sd._initialize()
        if dev_name:
            hostapis = sd.query_hostapis()
            
            devs = sd.query_devices()
    
            for d in devs:
                if dev_name in d['name'] and d['max_output_channels']>0 and hostapis[d['hostapi']]['name']=="ASIO":
                    print("Found device:")
                    print("\n".join(['\t%s: %s' % (k, str(v)) for k,v in d.items()]))
                    self.dev_id = d['index']
            
            if self.dev_id is None:
                raise Exception("Desired sound device not found.")
        else:
            self.dev_id = None
    
    def play(self,x,fs,mapping=[1,2],blocking = False):
        """
        Play a sound stimulus

        Parameters
        ----------
        x : array
            The signal itself.
        fs : int
            sampling frequency

        Returns
        -------
        None.

        """
        sd.play(x,fs, mapping=mapping, device=self.dev_id, blocking=blocking, blocksize = 2048)
        sd.wait()