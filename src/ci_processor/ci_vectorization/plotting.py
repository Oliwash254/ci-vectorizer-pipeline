# plotting.py
# Plotting functions

import numpy as np
import matplotlib.pyplot as plt

def plot_analog_electrodogram(X, fs, channel_labels=None, fig=None, ax=None, s='auto'):
    """
    Plots the raw analog signal for each channel, offset vertically.
    This function visualizes multi-channel time-series data.

    Parameters
    ----------
    X : numpy.ndarray
        An n_channels x n_samples array of analog recording data.
    fs : float
        Sampling frequency of the data.
    channel_labels : list, optional
        Labels for each channel. If None, uses default integer labels (1, 2, ...).
    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, a new figure is created.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, new axes are created.
    s : float or 'auto', optional
        Scaling factor for the amplitude of each channel's waveform.
        If 'auto', scales based on the maximum absolute value of the signal.
    """

    if s == 'auto':
        m = np.max(np.abs(X))
        s = .49 / m # Scale to fit within ~0.5 vertical unit

    if fig is None and ax is None:
        fig_ = plt.figure()
        ax_ = fig_.add_subplot(111) # Create figure and axes
    elif ax is None: # fig is provided, but ax is not
        ax_ = fig.add_subplot(111)
        fig_ = fig # Use provided figure
    else: # Both fig and ax are provided
        ax_ = ax
        fig_ = fig

    if channel_labels is None:
        channel_labels = [str(i + 1) for i in range(X.shape[0])] # Convert to string labels

    t = np.arange(X.shape[1]) / fs
    s_t = 1
    t_u = 's'
    if t[-1] < 1: # If total time is less than 1 second, use milliseconds
        s_t = 1e3
        t_u = 'ms'

    for i in range(X.shape[0]):
        # Plot horizontal line for channel baseline
        ax_.axhline(i, color='gray', lw=.5, linestyle=':') # Changed to gray, dotted for better contrast
        # Plot the scaled and offset waveform
        ax_.add_line(
            plt.Line2D(t * s_t, s * X[i, :] + i, color='k')
        )

    ax_.set_yticks(np.arange(X.shape[0])) # Set y-ticks at channel positions
    ax_.set_yticklabels(channel_labels) # Set y-tick labels
    ax_.set_xlabel(f"Time ({t_u})")
    ax_.set_ylabel("Channel")
    ax_.autoscale_view()
    ax_.set_ylim(-0.5, X.shape[0] - 0.5) # Set y-limits to encompass all channels without too much padding
    ax_.invert_yaxis() # Often, higher channel numbers are plotted lower, or vice-versa, depending on convention.
                       # If channels are 1-N, you might want 1 at top, N at bottom. If N-1, N at top, 1 at bottom.

    if fig_ is not None:
        fig_.tight_layout()
        # The on_resize event is mostly useful for interactive plotting.
        # If plots are always saved and closed, this isn't strictly necessary.
        # However, it doesn't harm to keep it if there's a chance of interactive display.
        # def on_resize(event):
        #     fig_.tight_layout()
        #     fig_.canvas.draw()
        # fig_.canvas.mpl_connect('resize_event', on_resize)

    return fig_