import numpy as np
import scipy.signal as sps
import scipy.stats as sp
from scipy.optimize import minimize
def oversample_recording_matrix(X, fs, n=4):
    n = int(n)
    t = np.arange(X.shape[1]) / fs
    new_t = np.arange(0, t[-1], 1 / (fs * n))

    new_X = np.empty((X.shape[0], new_t.size))
    for i in range(X.shape[0]):
        new_X[i, :] = np.interp(new_t, t, X[i, :])

    return new_X, fs * n





class ConstantPulseVectorizer:
    """
    A vectorizer that applies to implants using a fixed pulse shape.

    Attributes
    ----------
    X : numpy.ndarray
        Recording matrix (1 row per channel).
    fs : float
        Sampling frequency.
    anodic_first : bool, default=False
            Whether the first phase is positive. This is only used if `thr` is left to None, to determine the sign of the threshold.
    """

    anodic_first = False

    def __init__(self, X, fs, anodic_first=False, channel_labels=None):
        """
        Parameters
        ----------
        X : numpy.ndarray
            An :math:`n \\times m` array, with n the number of channels, and m the number of samples per channel.
        fs : float
            The scope's sampling frequency.
        anodic_first : bool, default=False
            Whether the first phase is positive. This is only used if `thr` is left to None, to determine the sign of the threshold.
        channel_labels : list, optional
            If not provided, uses a list of 1 to `X.shape[0]`.
        """
        self.X = X.copy()
        self.fs = fs

        if channel_labels is None:
            self.channel_labels = list(range(1, self.X.shape[0]+1))
        else:
            self.channel_labels = channel_labels


    def average_pulse_shape(self, X=None, fs=None, pulse_times=None, thr=None, dur=120e-6, t_offset=-4e-6, slope=None, return_pulses=False):
        """
        Computes the average of found pulses. This will be used to make a pulse template. As a result, it does not matter too
        much if small pulses are missed since they do not contribute much in the average.

        The function uses the class attribute :attr:`anodic_first`, so make sure it initialized properly.

        Parameters
        ----------
        X : numpy.ndarray, optional
            The array containing data of all channels, with 1 row per channel. If omitted, `self.X` is used.
        fs :  float, optional
            The sampling frequency (of the scope). If omitted, `self.fs` is used.
        pulse_times : list, optional
            If provided, the list of pulse time arrays, e.g. as produced by :meth:`vectorize`.
        thr : float, optional
            If None (default), the peak value divided by 100 is used as threshold.
            Othwerwise, the value of the threshold. Keep in mind that for cathodic first pulses, this should be negative.
        dur : float, default=120e-6
            The max duration of a pulse.
        t_offset : float, default=-4e-6
            The time offset in seconds. Negative values will include time before the threshold.
        slope : {'rising', 'falling'}, optional
            Whether detection happens on rising or falling edge. Default is 'falling' is `anodic_first` is False,
            otherwise, 'rising'.
        return_pulses : bool, default=False
            Also returns the individual pulses that were detected.

        Returns
        -------
        avg_pulse : numpy.ndarray
            The average of all detected pulses. It is of length int(dur*fs).
        pulses : list
            If `return_pulses` is True, this is the list of individual pulses detected.
        """

        if X is None:
            X = self.X
        if fs is None:
            fs = self.fs

        if pulse_times is None:
            # Detect pulses based on threshold

            if slope is None:
                if self.anodic_first:
                    slope = 'rising'
                else:
                    slope = 'falling'
            j_offset = int(np.round(self.fs*t_offset))

            if thr is None:
                if slope == 'falling':
                    thr = -np.max(X)/100
                elif slope == 'rising':
                    thr = np.max(X)/100

            j = np.arange(X.shape[1]-1)
            n_dur = int(dur*fs)

            # For each channel
            pulses = []

            for i in range(X.shape[0]):
                if slope=='rising':
                    j_p = j[(X[i,0:-1] < thr) & (X[i,1:] >= thr)]
                elif slope=='falling':
                    j_p = j[(X[i,0:-1] > thr) & (X[i,1:] <= thr)]

                for jj in j_p:
                    jj = jj+j_offset
                    if jj>0 and jj+n_dur<=X.shape[1]:
                        pulse = X[i, jj:(jj+n_dur)]
                        pulses.append(pulse.copy())
        else:
            # Uses provided pulse times

            j_offset = int(np.round(self.fs*t_offset))
            n_dur = int(self.fs*dur)

            pulses = []
            for i in range(len(pulse_times)):
                for tj in pulse_times[i]:
                    j = int(np.round(tj*self.fs))
                    j1 = max(0, (j+j_offset))
                    j2 = min(X.shape[1], (j+j_offset+n_dur))
                    if j2-j1==n_dur:
                        pulse = X[i, j1:j2]
                        pulses.append(pulse.copy())

        m_pulse = np.mean(pulses, axis=0)
        m_pulse = m_pulse / np.max(np.abs(m_pulse))

        if return_pulses:
            return m_pulse, pulses
        else:
            return m_pulse

    def pulse_shape(self, t, t0, p1, ipg, p2, tau=0, a=1):
        """
        Pulse shape without RC component. This is a cathodic first pulse, i.e. the first phase is negative.

        Parameters
        ----------
        t : numpy.ndarray
            Time vector (in seconds).
        t0 : float
            Initial time offset (in seconds).
        p1 : float
            First phase duration.
        ipg : float
            Inter-phase gap.
        p2 : float
            Second phase duration.
        tau : float, default=0
            RC time constant. Will be ignored, and is just to make the
            call compatible with :meth:`pulse_shape_RC`.
        a : float, default=1
            Amplitude.

        Returns
        -------
        p : numpy.ndarray
            The amplitudes for the times given in `t`. The pulse is normalized to 1.
        """

        v = np.zeros_like(t)
        v[(t<t0) | (t>=t0+p1+ipg+p2)] = 0
        v[(t>t0) & (t<=t0+p1)] = -1
        v[(t>t0+p1) & (t<=t0+p1+ipg)] = 0
        v[(t>t0+p1+ipg) & (t<=t0+p1+ipg+p2)] = 1

        return v*a

    def pulse_shape_RC(self, t, t0, p1, ipg, p2, tau, a=1):
        """
        Pulse shape without RC component. This is a cathodic first pulse, i.e. the first phase is negative.

        Parameters
        ----------
        t : numpy.ndarray
            Time vector (in seconds).
        t0 : float
            Initial time offset (in seconds).
        p1 : float
            First phase duration.
        ipg : float
            Inter-phase gap.
        p2 : float
            Second phase duration.
        tau : float
            The RC exponential decay time constant. If 0, it falls back to :meth:`pulse_shape`.
        a : float, default=1
            Amplitude.

        Returns
        -------
        p : numpy.ndarray
            The amplitudes for the times given in `t`. The pulse is normalized to 1.
        """
        if tau==0:
            return self.pulse_shape(t, t0, p1, ipg, p2, 0, a)

        v = np.zeros_like(t)

        s = (t>=t0) & (t<=t0+p1)
        v[s] = np.exp(-(t[s]-t0)/tau) - 1
        ve = v[s][-1]

        s = (t>=t0+p1) & (t<=t0+p1+ipg)
        if np.any(s):
            v[s] = ve * np.exp(-(t[s]-t0-p1)/tau)
            ve = v[s][-1]

        s = (t>=t0+p1+ipg) & (t<=t0+p1+ipg+p2)

        v[s] = 1 - (1-ve) * np.exp(-(t[s]-t0-p1-ipg)/tau)
        ve = v[s][-1]

        s = t>=t0+p1+ipg+p2
        v[s] = ve * np.exp(-(t[s]-t0-p1-ipg-p2)/tau)

        return v*a

    def fit_pulse_shape(self, t, v, no_ipg=False, sym_pulse=False, oversample=True, realign_t0=True, extra_constraints=None,ntau=1):
        """
        Fits a pulse shape template to a recorded pulse shape. Typically, this is applied to the average pulse shape obtained from :meth:`average_pulse_shape`.

        Parameters
        ----------
        t : numpy.ndarray
            Time vector for the average pulse, in seconds.
        v : numpy.ndarray
            Recorded pulse shape (e.g. as returned by :meth:`average_pulse_shape`).
        no_ipg : bool, default=False
            Adds constraints that IPG is null.
        sym_pulse : bool, default=True
            Adds constraints that phases are same duration.
        oversample : bool, default=True
            Whether to oversample the pulse before fitting. The oversampling is done through linear interpolation, multiplying by 10 the number of samples.
        realign_t0 : bool, default=True
            Will set t0 to 0 in the output.
        extra_contraints : list, optional
            A list of functions applying constraints to the parameter vector. These functions take a parameter vector as input, and produce a new one with constraints applied.

        Returns
        -------
        params : numpy.ndarray
            Parameters of the pulse shape [t0, p1, ipg, p2, tau].
        """

        def cnstr_pos(x):
            # Phase durations and tau are all positive
            for i in range(1, len(x)):
                x[i] = max(0, x[i])
            return x

        def cnstr_sym(x):
            phase = np.mean([x[1], x[3]])
            x[1] = phase
            x[3] = phase
            return x

        def cnstr_no_ipg(x):
            x[2] = 0
            return x

        def apply_cnstrs(x, cnstrs):
            for cn in cnstrs:
                x = cn(x)
            return x

        lst_cnstrs = [cnstr_pos]
        if no_ipg:
            lst_cnstrs.append(cnstr_no_ipg)
        if sym_pulse:
            lst_cnstrs.append(cnstr_sym)
        if extra_constraints is not None:
            lst_cnstrs.extend(extra_constraints)

        if len(v.shape)>1 and v.shape[0]>1:
            prms = []
            for i in range(v.shape[0]):
                prms.append( ConstantPulseVectorizer.fit_pulse_shape(self, t, v[i,:], no_ipg=no_ipg, sym_pulse=sym_pulse, oversample=oversample, realign_t0=realign_t0, extra_constraints=None,ntau=ntau) )
            m_prms = apply_cnstrs(np.median(np.array(prms), axis=0), lst_cnstrs)
            return m_prms, prms

        if oversample:
            M = 10
            tt = np.linspace(0, t[-1], t.size * M)
            v = np.interp(tt, t, v)
            t = tt

        # Normalize
        v = v/np.max(np.abs(v))

        # Estimation of phase duration and ipg
        av = np.abs(v)
        i1, = np.nonzero((av[0:-1]<.5) & (av[1:]>=.5))
        if i1.size<2:
            raise Exception("Not enough phases in the provided pulse data.")
        i2, = np.nonzero((av[0:-1]>.5) & (av[1:]<=.5))
        if i2.size<2:
            raise Exception("Not enough phases in the provided pulse data.")
        t0 = t[i1[0]]
        p1 = t[i2[0]] - t[i1[0]]

        if no_ipg:
            ipg = 0
        else:
            ipg = t[i1[1]] - t[i2[0]]

        if sym_pulse:
            p2 = p1
        else:
            p2 = t[i2[1]] - t[i1[1]]

        if ntau == 1:
            tau = 1e-6
            x0 = [t0, p1, ipg, p2, tau, 1]
        elif ntau == 2:
            tau1 = 1e-6
            tau2 = 1e-6
            x0 = [t0, p1, ipg, p2, tau1, tau2, 1]

        def sqe(x, cnstrs=[]):
            x = apply_cnstrs(x, cnstrs)
            return np.sum((self.pulse_shape_RC(t, *x) - v)**2)

        res = minimize(sqe, x0, args=(lst_cnstrs,),  method='Nelder-Mead')

        x = apply_cnstrs(res.x, lst_cnstrs)

        if realign_t0:
            x[0] = 0

        return x[0:-1]

    def get_pulse_duration(self, pulse_prms, decay_duration=7):
        """
        Calculates the duration of a pulse template including exponential decay from the RC component.

        Parameters
        ----------
        pulse_prms : numpy.ndarray
            Parameters as returned by `fit_pulse_shape()`.
        decay_duration : float, default=7
            The multiplier of τ (`pulse_prms[4]`) to account for the time of the exponential decay.

        Returns
        -------
        duration : float
        """

        return np.sum(pulse_prms[1:4])+pulse_prms[4]*decay_duration

    def get_pulse_lag(self, t_pulse, pulse_prms):
        """
        Calculates the lag of the pulse to realign the cross-correlation.

        Parameters
        ----------
        t_pulse : numpy.ndarray
            The time vector of the pulse that was used in the cross-correlation.
        pulse_prms : numpy.ndarray, list
            The pulse parameters as returned by `fit_pulse_shape()`.

        Returns
        -------
        lag : float
            The lag value.
        """

        return t_pulse[-1]/2 - (pulse_prms[0]+pulse_prms[1]+pulse_prms[2])

    def remove_DC(self, X=None, fs=None, win_dur=.5):
        """
        Removes from original recording by using averaging over a Hann window of `win_dur` duration (in seconds).

        Parameters
        ----------
        X : numpy.ndarray, optional
            If provided, the matrix of recordings with 1 row per channel. Otherwise, using `self.X`.
        fs : float, optional
            The sampling frequency. If omitted, using `self.fs`.
        win_dur : float, default=.5
            Window duration (in seconds).

        Returns
        -------
        Y : numpy.ndarray
            A new recording matrix with the DC components removed.
        """

        if X is None:
            X = self.X
        if fs is None:
            fs = self.fs

        Y = np.zeros(X.shape)
        step = int(win_dur*fs/2)
        w = sps.windows.hann(step*2+1)
        nw = X.shape[1]//step+1

        for i in range(X.shape[0]):
            j1 = -step
            for k in range(nw):
                j2 = j1+2*step

                j1_ = max(0, j1)
                j2_ = min(X.shape[1], j2)

                y = X[i, j1_:j2_] * w[(j1_-j1):(j2_-j1)]
                Y[i, j1_:j2_] += np.mean(y) * w[(j1_-j1):(j2_-j1)]

                j1 += step
            Y[i,:] = X[i,:] - Y[i,:]

        return Y

    def find_pulses_corr(self, pulse_prms, X=None, fs=None, thr=None, force_sequential=True):
        """
        Locates pulses using the pulse shape correlation method.

        Parameters
        ----------
        pulse_prms : list
            Pulse parameters, as returned by :meth:`fit_pulse_shape`.
        X : numpy.ndarray, optional
            The recording signal matrix. If omitted, `self.X` is used.
        fs : float, optional
            The sampling frequency. If omitted, `self.fs` is used.
        thr : float, optional
            The threshold value. If None, then the value is determined from the histogram.
        force_sequential : bool, default=True
            Whether to ensure that pulses follow each other (like in CIS).

        Returns
        -------
        peaks : list
            For each channel, a list of peak indices.
        peak_info : list
            Details of the identified peaks (including their amplitude) (a `dict` per channel).
        p : numpy.ndarray
            Pulse shape used in the correlation.
        t_pulse : numpy.ndarray
            The time vector associated with the pulse shape.
        thr : float
            The computed threshold that was used (if thr was None).
        """

        if X is None:
            X = self.X
        if fs is None:
            fs = self.fs

        t_pulse = np.arange(2*self.get_pulse_duration(pulse_prms)*fs)/fs
        p = self.pulse_shape_RC(t_pulse, *pulse_prms)

        Y = np.empty(X.shape)

        for i in range(X.shape[0]):
            Y[i,:] = np.maximum(sps.correlate(X[i,:], p, mode='same'), 0)

        if thr is None:
            c, b = np.histogram(Y.flatten('C'), 100)
            i = np.argmax(np.diff(c)>=0)
            thr = b[i]
            self.find_pulses_corr_thr = thr
            #print(f"Threshold is {thr:.2f}")

        di_peaks = []
        peaks = []
        peak_info = []
        for i in range(Y.shape[0]):
            peaks_i, peak_info_i = sps.find_peaks(Y[i,:], height=thr, width=5) #, distance=int(2*fs/1e4))
            peaks.append(peaks_i)
            peak_info.append(peak_info_i)
            di_peaks.extend(np.diff(peaks_i)/self.fs)

        if force_sequential:
            c, b = np.histogram(di_peaks, 200)
            c_max_i = np.argmax(c)
            print("During peak finding, most common pulse interval: %.2f ms" % (b[c_max_i]*1e3))

            # Filter pulses with shorter interval than the most common interval, if they exist
            if c_max_i>0:
                # There are pulses that are closer to each other than the most common period.
                # We need to refilter the peaks to exclude these. When that happens we keep the
                # higher pulse.

                print("Some pulses are closer to each other than the common period and will be removed.")

                diff_thr = b[c_max_i]/2

                for i in range(len(peaks)):
                    dp = np.diff(peaks[i])
                    ks = np.where(dp < diff_thr)
                    for k in ks:
                        if Y[i,peaks[i][k]] > Y[i,peaks[i][k+1]]:
                            peaks[i][k+1] = np.nan
                        else:
                            peaks[i][k] = np.nan
                    peaks[i] = peaks[i][~np.isnan(peaks[i])]

        return peaks, peak_info, p, t_pulse, thr

    def compute_amplitudes(self, peak_info, p):
        """
        Computes the detected pulse magnitudes.

        Parameters
        ----------
        peak_info : list
            As returned by :meth:`find_pulses`.
        p : numpy.ndarray
            The pulse shape used in the cross-correlation.

        Returns
        -------
        a : list
            A list of np.ndarrays (per channel).
        """

        pulse_ssq = np.sum(p**2)
        a = []
        for i in range(len(peak_info)):
            a.append( peak_info[i]['peak_heights'] / pulse_ssq )
        return a


    def vectorize(self, X=None, fs=None, pulse_window=None, no_ipg=False, sym_pulse=False, iterations=1, pulse_prms=None, remove_DC=False, fit_on_n_pulses=30):
        """
        Vectorizes the loaded analog recording. The output is given in the *sparse* format
        (see the :mod:`conversion` module for a description of the format).

        Parameters
        ----------
        X : numpy.ndarray, optional
            If provided, a 2D array of analog recording. If omitted, `self.X` is used intead.
        fs : float, optional
            If provided, the sampling frequency. I omitted, `self.fs` is used instead.
        pulse_window : float, optional
            The estimated duration of a window that would contain only one pulse, i.e. a bit
            longer than total pulse duration to account for capacitive discharge. This is the `dur`
            parameter of the :meth:`average_pulse_shape` function.
        no_ipg : bool, default=False
            Whether to use a pulse shape without inter-phase gap.
        sym_pulse : bool, default=True
            Whether the pulse shape has two equal duration phases.
        iterations : int, default=2
            The number of iterations. If `pulse_prms` is not provided, in the first pass
            (or if `iterations=1`), the pulses are detected with a basic threshold.
        pulse_prms : list, optional
            If provided, the pulse parameters defining the pulse shape. If None, or omitted,
            the pulse shape is first estimated. Note that `iterations`=2 is without effect if
            `pulse_prms` is provided.
        remove_DC : bool, default=False
            Whether to remove the DC component. On short stimuli, this can create some issues
            if the pulses are not perfectly balanced.
        fit_on_n_pulses : int, default=30
            The number of detected pulses the fitting of the pulse shape is done on.

        Returns
        -------
        pulse_times : list
            A list of np.ndarrays of pulse times (per channel).
        pulse_amplitudes : list
            The amplitude of each pulse with the same structure.
        pulse_prms : list
            The pulse shape parameters.
        """
        if X is None:
            X = self.X
        if fs is None:
            fs = self.fs

        # 1. Remove DC
        if remove_DC:
            Xc = self.remove_DC(X, fs)
        else:
            Xc = X

        if pulse_prms is not None:
            iterations = 1

        t = np.arange(Xc.shape[1])/fs

        pulse_times = None
        t_offset = -4e-6
        if pulse_window is None:
            dur = 120e-6
        else:
            dur = pulse_window

        for it in range(iterations):

            # 2. Average detected pulses
            if pulse_prms is None or it>0:
                m_pulse, pulses = self.average_pulse_shape(Xc, pulse_times=pulse_times, dur=dur, t_offset=t_offset, return_pulses=True)
                t_m_pulse = np.arange(m_pulse.size)/fs
                pulses = np.array(pulses)
                if pulses.shape[0] < fit_on_n_pulses:
                    raise Exception("Not enough pulses in the recording to fit the shape properly. Reduce `fit_on_n_pulses` or increase the length of the recording.")
                sel_pulses = pulses[(-np.mean(pulses**2, axis=1)).argsort()[:fit_on_n_pulses],:]

                self.avg_pulse = m_pulse.copy()
                self.avg_pulse_t = t_m_pulse.copy()

                # 2. Fit template to get parameters
                pulse_prms, _ = self.fit_pulse_shape(t_m_pulse, sel_pulses, no_ipg=no_ipg, sym_pulse=sym_pulse)

                self.phase_durations = (pulse_prms[1], pulse_prms[3])
                self.inter_phase_gap = pulse_prms[2]
                if len(pulse_prms) == 5:
                    self.tau = pulse_prms[4]
                else:
                    self.tau = [pulse_prms[4], pulse_prms[5]]

            # Update dur and t_offset for next iteration
            dur = self.get_pulse_duration(pulse_prms)
            t_offset = -dur / 2
            dur *= 1.25

            # 3. Find peaks
            peaks, peak_info, p, t_pulse, _ = self.find_pulses_corr(pulse_prms, X=Xc, fs=fs)

            # 5. Compute pulse times
            pulse_lag = self.get_pulse_lag(t_pulse, pulse_prms)
            pulse_times = []
            for i in range(len(peaks)):
                pulse_times.append( t[peaks[i]] - pulse_lag )

        # 6. Compute pulse amplitudes
        pulse_amplitudes = self.compute_amplitudes(peak_info, p)

        self.pulse_shape = p
        self.pulse_shape_t = t_pulse

        return pulse_times, pulse_amplitudes, pulse_prms


    def estimate_pulse_rate(self, pulse_times, pulse_rate_hint=None):
        """
        Estimate the pulse rate (across channels) from the data.

        Parameters
        ----------
        pulse_times : numpy.ndarray
            As returned by :meth:`vectorize`.
        pulse_rate_hint : float, optional
            An estimated value for the pulse rate, e.g. the value provided by the clinical software.
            To work, the difference between the hint and the actual rate, expressed as a ratio of the
            actual rate, should be smaller than 1/number of channels.
            If omitted, it will be estimated from the `pulse_times`.

        """
        t = np.concatenate(pulse_times).sort()

        dt = np.diff(t)

        if pulse_rate_hint is None:
            c, b = np.histogram(dt[dt>0], 30)
            c_max_i = np.argmax(c)
            cTe = np.mean(dt[(dt>=b[c_max_i]) & (dt<=b[c_max_i+1])])
        else:
            cTe = 1/pulse_rate_hint

        i = np.r_[0, np.cumsum(np.round(dt / cTe))]
        r = sp.stats.linregress(i, t)

        cT = r.slope

        return 1/cT
    
    def pulse_shape_compare(self, t, ch, X, pulse_times, pulse_amplitudes, pulse_prms):
        """
        Compare analog signal to reconstructed pulse shape and compute goodness of fit.

        Parameters
        ----------
        t : array_like
            Time vector.
        ch : int
            Channel index.
        X : array_like
            Analog signals.
        pulse_times : array_like
            Pulse times per channel.
        pulse_amplitudes : array_like
            Pulse amplitudes per channel.
        pulse_prms : array_like
            Pulse shape parameters.

        Returns
        -------
        fitval : float
            Mean squared error fraction between analog and reconstructed signals.
        """

        from matplotlib import pyplot as plt
        idx = ch-1
        plt.figure()
        plt.plot(t, X[idx], label='Analog')
        plt.vlines(pulse_times[idx], 0, pulse_amplitudes[idx],colors='r',label='Vectorized')
        g = np.zeros(len(t))
        for i in range(len(pulse_times[idx])):
            ps2 = self.pulse_shape_RC(t,pulse_times[idx][i]-pulse_prms[1]-pulse_prms[2],*pulse_prms[1:])
            g+= ps2*pulse_amplitudes[idx][i]
        plt.plot(t,g,color='g',label='Pulse shape')
        plt.legend()
        plt.show()

        #Compute goodness of fit
        diffr = []
        for i in range(len(pulse_times[idx])):
            ts = [pulse_times[idx][i]-2*pulse_prms[1], pulse_times[idx][i]+2*pulse_prms[3]]
            s =  np.where((t>=ts[0]) & (t<=ts[1]))
            d = X[idx][s] - g[s]
            diffr.append(np.sum(d**2)/np.sum(g[s]**2)) #fraction of sum of squares between difference and expected
        fitval = np.mean(diffr)
        return fitval
