# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_electrodogram(ax, pulse_times, pulse_channels, pulse_amplitudes=None,
                       t_min=None, t_max=None, channel_range=None,
                       fs=None, title=None, labels=True, cbar=False, # Changed cbar default to False
                       cmap_name='viridis', cmap_levels=100,
                       reverse_channels=False):
    """
    Plots an electrodogram with vertical lines representing pulses,
    mimicking the style of the client's example.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    pulse_times : array_like
        Times of the pulses (in seconds).
    pulse_channels : array_like
        Channels of the pulses (1-indexed).
    pulse_amplitudes : array_like, optional
        Amplitudes of the pulses. If None, amplitudes are assumed to be 1.
    t_min : float, optional
        Minimum time for the x-axis. If None, defaults to min(pulse_times).
    t_max : float, optional
        Maximum time for the x-axis. If None, defaults to max(pulse_times).
    channel_range : tuple, optional
        (min_channel, max_channel) for the y-axis. If None, defaults to
        (min(pulse_channels)-0.5, max(pulse_channels)+0.5).
    fs : int, optional
        Sampling frequency for converting time axis to samples. Not used for plotting.
    title : str, optional
        Title of the plot.
    labels : bool, optional
        If True, add x and y labels.
    cbar : bool, optional
        If True, add a colorbar for amplitudes. Default is False to match client example.
    cmap_name : str, optional
        Name of the colormap to use. Default is 'viridis'.
    cmap_levels : int, optional
        Number of levels for the colormap. Default is 100.
    reverse_channels : bool, optional
        If True, reverse the y-axis (channels) order (e.g., 22 at bottom, 1 at top).
    """

    # Convert inputs to numpy arrays for consistent handling
    pulse_times = np.asarray(pulse_times, dtype=float)
    pulse_channels = np.asarray(pulse_channels, dtype=int)
    pulse_amplitudes = np.asarray(pulse_amplitudes, dtype=float) if pulse_amplitudes is not None else np.ones_like(pulse_times, dtype=float)

    if pulse_times.size == 0:
        ax.set_title(title if title else "Electrodogram (No Pulses Found)")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel')
        return # Nothing to plot if no pulses

    if t_min is None:
        t_min = pulse_times.min() if pulse_times.size > 0 else 0.0
    if t_max is None:
        t_max = pulse_times.max() if pulse_times.size > 0 else 1.0 # Default max time if no pulses

    if channel_range is None:
        min_ch = pulse_channels.min() if pulse_channels.size > 0 else 1
        max_ch = pulse_channels.max() if pulse_channels.size > 0 else 16 # Default to 16 channels if no pulses
        channel_range = (min_ch - 0.5, max_ch + 0.5)

    # Normalize amplitudes for colormap
    min_amp = pulse_amplitudes.min()
    max_amp = pulse_amplitudes.max()
    
    if max_amp == min_amp: # Avoid division by zero if all amplitudes are the same
        norm_amplitudes = np.zeros_like(pulse_amplitudes, dtype=float) # Use 0 for min color if all same
    else:
        norm_amplitudes = (pulse_amplitudes - min_amp) / (max_amp - min_amp)

    cmap = cm.get_cmap(cmap_name, cmap_levels)

    # Plot vertical lines for each pulse
    for i in range(len(pulse_times)):
        ax.plot([pulse_times[i], pulse_times[i]],
                [pulse_channels[i] - 0.4, pulse_channels[i] + 0.4], # Extend +/- 0.4 units around channel center
                color=cmap(norm_amplitudes[i]),
                linewidth=0.5, # Adjusted linewidth to match client example
                alpha=1.0) # Adjusted alpha to match client example

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(channel_range)

    # Add gridlines similar to client example
    # Horizontal gridlines for channels
    for i in np.arange(channel_range[0]+0.5, channel_range[1]+0.5, 1):
        ax.axhline(i, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)

    # Vertical gridlines for time (e.g., every 0.5 seconds)
    if t_max > t_min:
        xticks = np.arange(np.floor(t_min * 2) / 2, np.ceil(t_max * 2) / 2 + 0.001, step=0.5)
        for x in xticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)


    if reverse_channels:
        ax.invert_yaxis() # Invert y-axis for Cochlear plots

    if title:
        ax.set_title(title)
    if labels:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel number') # Changed label to match example
        
        # Set integer ticks for channels and potentially custom labels for reversed order
        if reverse_channels:
            # Map default channel numbers to reversed labels
            all_ch = np.arange(int(channel_range[0]+0.5), int(channel_range[1]+0.5) + 1)
            ax.set_yticks(all_ch)
            ax.set_yticklabels(list(reversed(all_ch)))
        else:
            ax.set_yticks(np.arange(int(channel_range[0]+0.5), int(channel_range[1]+0.5) + 1))


    if cbar: # Only create colorbar if cbar is True
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_amp, vmax=max_amp))
        sm.set_array([])
        cbar_obj = plt.colorbar(sm, ax=ax)
        cbar_obj.set_label('Amplitude')
    
    # Grid already added above, no need for general plt.grid(True)
    # plt.grid(True, linestyle=':', alpha=0.6) # This line is now redundant
