# -*- coding: utf-8 -*-
import time

"""
A class to manage two Yokogawa DL750 scopecorder oscilloscopes for recording CI implant-in-a-box output for both Advanced Bionics and Cochlear processors.

"""
from tmctl import tm, TMCTL, TMCTL_CONFIG, calculate_record_length

class DualScopes:
    def __init__(self,settings):
        """
        Initializes the class with the provided settings.
        Parameters
        ----------
        settings : list
            A list of dicts with one dict per oscilloscope. Each dict provides 
            configuration values such as 'wire', 'adr', and 'n_channels'.

        Attributes
        ----------
        S : list
            A list of objects created using the `tm` class, initialized with 
            the provided settings.
        settings : list
            The input settings used to configure the `tm` objects.
        """
        self.S = []
        self.settings = settings
        for iscope,scope_set in enumerate(settings):
            n_channels = len(scope_set['scope_channels'])
            self.S.append(tm(scope_set['wire'],scope_set['adr'],n_channels=n_channels,verbose=True))
    
    def init_scope(self,scope,nrec, srate, scope_channels, volt_mag, trig_pos, n_triggers, clock="EXTERNAL"):
        """
        Initializes the oscilloscope with the specified parameters.
        Parameters
        ----------
        scope : object
            oscilloscope object to be initialized.
        nrec : int
            recording length
        srate : int
            sampling rate in Hz. Ignored if `clock` is set to "EXTERNAL".
        scope_channels : list of int
            list of channels to initialize on the oscilloscope.
        volt_mag : float
            voltage magnitude for the channels.
        trig_pos : float
            trigger position in percentage of the record length.
        n_triggers : int
            number of triggers to configure.
        clock : str, optional
            clock source for the oscilloscope
        Returns
        -------
        list of str
            A list of commands sent to the oscilloscope for initialization.
        Notes
        -----
        - When `clock` is set to "EXTERNAL", the sampling rate (`srate`) is not required.
        - The oscilloscope is configured with a normal acquisition mode and infinite acquisition count.
        """

        scope.init_channels(channels = scope_channels, volt_mag=[volt_mag]*len(scope_channels),
                            position=0, offset=0, variable=False)
        
        if n_triggers == 1:
            trigmode = "SINGLE"
        else:
            trigmode = "NSINGLE"
        
        if clock == "EXTERNAL": #EXT MODE does not need SRATE
            msg = [":ACQ:MODE NORMAL", ":ACQ:CLOCK %s" % clock, ":ACQ:COUNT INF",
                   ":ACQ:RLEN %d" % nrec,":TRIG:TYPE SIMPLE", ":TRIG:POS %.1f" % trig_pos,
                   ":TRIG:SIMPLE:SOURCE EXT",":TRIG:SIMPLE:SLOPE RISE",":TRIG:MODE %s" % trigmode,
                   ":TRIG:SCO %d" % n_triggers] 
        else:
            msg = [":ACQ:MODE NORMAL", ":ACQ:CLOCK %s" % clock, ":ACQ:COUNT INF",
                   ":ACQ:RLEN %d" % nrec,":TIM:SRATE %d" % srate,":TRIG:TYPE SIMPLE",
                   ":TRIG:POS %.1f" % trig_pos, ":TRIG:SIMPLE:SOURCE EXT",":TRIG:SIMPLE:SLOPE RISE",
                   ":TRIG:MODE %s" % trigmode, ":TRIG:SCO %d" % n_triggers] 
        scope.send(msg)
        return msg

    def init_both_scopes(self, srate, volt_mag, trig_pos, n_triggers, clock="EXTERNAL"):
        """
        Initializes both oscilloscope devices with the specified settings.
        Parameters
        ----------
        srate : float
            sampling rate.
        volt_mag : float
            voltage magnitude.
        trig_pos : float
            trigger position.
        n_triggers : int
            number of triggers to configure.
        clock : str, optional
            clock source for the oscilloscope, can be "EXTERNAL" (default) or "INTERNAL".
        Returns
        -------
        list
            A list of messages returned from the initialization of each oscilloscope.
        """
        msg = []
        for idx,scope in enumerate(self.S):
            scope_channels = self.settings[idx]['scope_channels']
            nrec = self.settings[idx]['nrec']
            msg1 = self.init_scope(scope,nrec, srate, scope_channels, volt_mag, trig_pos, n_triggers, clock)
            msg.append(msg1)
        return msg
    
    def send(self,msg):
        """
        Sends a message to all scopes in the `self.S` collection.
        Parameters
        ----------
        msg : Any
            message to be sent to each scope in the `self.S` collection.
        Raises
        ------
        AttributeError
            If an object in `self.S` does not have a `send` method.
        """

        for scope in (self.S):
            scope.send(msg)
    
    def receive(self,msg=None,mode='short',remove_header=True):
        """
        Iterates over all scopes in the `self.S` attribute and calls their `receive` method
        with the provided parameters.
        Parameters
        ----------
        msg : optional
            The message to be received. The type and structure of the message depend on the implementation.
        mode : str, optional
            The mode in which the message is processed. Default is 'short'.
        remove_header : bool, optional
            Whether to remove the header from the message during processing. Default is True.
        """
        r = []
        for scope in self.S:
            r.append(scope.receive(msg=msg,mode=mode,remove_header=remove_header))
        return r
        
    def wait_acq(self, timeout=15):
        """
        Waits for the acquisition process to complete for all scopes, with a timeout.

        Parameters
        ----------
        timeout : int or float
            Maximum time in seconds to wait for acquisition before raising TimeoutError.
        """

        #for scope in self.S:
            #scope.wait_acq()

        t0 = time.time()
        while True:
            all_done = True
            for idx, scope in enumerate(self.S):
                try:
                    status = scope.receive(":STATUS:CONDITION?", mode='short', remove_header=True).strip()
                except Exception as e:
                    raise RuntimeError(f"[!] Failed to get status from scope {idx}: {e}")
                
                if status != "0":
                    all_done = False
                    break  # At least one scope still acquiring, wait a bit more

            if all_done:
                break

            if time.time() - t0 > timeout:
                raise TimeoutError("Acquisition did not complete in time.")
            time.sleep(0.5)


        #t0 = time.time()
        #while True:
        #    for idx,scope in enumerate(self.S):
                #cond1 = self.S[idx].receive(":STATUS:CONDITION?", mode='short', remove_header=True)
                # Status 0 means "stopped"
                #if cond1.strip() == "0":
                   # break
                #if time.time() - t0 > timeout:
               #     raise TimeoutError("Acquisition did not complete in time.")
               # time.sleep(0.5)

    def get_acq_full(self, mode='WORD'):
        """
        Retrieve acquisition data for all scopes and combines into single dict with all electrodes.
        Parameters
        ----------
        mode : str, optional
            The acquisition mode, by default 'WORD'.
        Returns
        -------
        d : dict
            A dictionary with acquisition data per electrode.
        """
        d = {}
        for idx, scope in enumerate(self.S):
            channels = self.settings[idx]['scope_channels']
            electrodes = self.settings[idx]['electrodes']
            if len(channels) == len(electrodes):
                for ich,ch in enumerate(channels):
                    d[electrodes[ich]] = scope.get_acq(ch,mode=mode)
            else:
                raise ValueError("Number of channels and electrodes do not match")
        return d