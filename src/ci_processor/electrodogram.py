# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_electrodogram(ax, pulse_times, pulse_channels, pulse_amplitudes=None,
                       t_min=None, t_max=None, channel_range=None,
                       fs=None, title=None, labels=True, cbar=False,
                       cmap_name='viridis', cmap_levels=100,
                       reverse_channels=False):
    """
    Plots an electrodogram with vertical lines representing pulses.
    Pulses now start at the channel line and extend upwards based on amplitude,
    mimicking the client's example plot style for both AB and Cochlear.

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
    # Ensure pulse_amplitudes is an array, default to ones if None
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
        channel_range = (min_ch - 0.5, max_ch + 0.5) # Add 0.5 buffer above/below

    # Determine global max amplitude for scaling pulse height
    # This max determines how tall the tallest pulse will be relative to a channel unit.
    global_max_amplitude = np.max(pulse_amplitudes) if pulse_amplitudes.size > 0 else 1.0
    if global_max_amplitude == 0: # Avoid division by zero if all amplitudes are 0
        global_max_amplitude = 1.0

    # Scale for vertical line length:
    # Based on client's `/(max_pulse*2)`, a pulse with `max_pulse` amplitude extends 0.5 units upwards.
    # So, for any amplitude, its height will be (amplitude / global_max_amplitude) * 0.5
    # Let's cap the max height at 0.8 units to ensure pulses don't overlap too much with the next channel's line
    # or go beyond the nominal channel boundaries, while still reflecting amplitude differences clearly.
    max_scaled_height_unit = 0.8 # Max height a pulse can extend upwards into the next channel slot.


    # Normalize amplitudes for color mapping (0 to 1) for the chosen colormap
    min_amp_for_cmap = pulse_amplitudes.min()
    max_amp_for_cmap = pulse_amplitudes.max()
    
    if max_amp_for_cmap == min_amp_for_cmap: # Avoid division by zero if all amplitudes are the same
        norm_amplitudes_for_cmap = np.zeros_like(pulse_amplitudes, dtype=float) # Use 0 for min color if all same
    else:
        norm_amplitudes_for_cmap = (pulse_amplitudes - min_amp_for_cmap) / (max_amp_for_cmap - min_amp_for_cmap)

    cmap = cm.get_cmap(cmap_name, cmap_levels)

    # Plot vertical lines for each pulse
    for i in range(len(pulse_times)):
        # Calculate pulse height based on its amplitude relative to the global max
        current_pulse_height = (pulse_amplitudes[i] / global_max_amplitude) * max_scaled_height_unit
        
        ax.plot([pulse_times[i], pulse_times[i]],
                [pulse_channels[i], pulse_channels[i] + current_pulse_height], # Pulses start at channel line, extend upwards
                color=cmap(norm_amplitudes_for_cmap[i]),
                linewidth=0.5, # Adjusted linewidth to match client example
                alpha=1.0) # Adjusted alpha to match client example

    ax.set_xlim(t_min, t_max)
    ax.set_ylim(channel_range)

    # Add gridlines similar to client example
    # Horizontal gridlines — for channels
    for i in np.arange(channel_range[0]+0.5, channel_range[1]+0.5, 1): # Draws lines at 1.0, 2.0, ...
        ax.axhline(i, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)

    # Vertical gridlines — for time (e.g., every 0.5 seconds)
    if t_max > t_min:
        # Determine appropriate x-ticks for gridlines
        xticks = np.arange(0, t_max + 0.001, step=0.5) # Start from 0, step 0.5s
        for x in xticks:
            ax.axvline(x, color='gray', linestyle='--', linewidth=0.3, alpha=0.4)


    if reverse_channels:
        ax.invert_yaxis() # Invert y-axis for Cochlear plots (highest channel number at visual bottom)

    if title:
        ax.set_title(title)
    if labels:
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel number') # Changed label to match example
        
        # Set integer ticks for channels and potentially custom labels for reversed order
        if reverse_channels:
            # Get actual channel numbers and reverse labels if y-axis is inverted
            all_ch_numbers = np.arange(int(channel_range[0]+0.5), int(channel_range[1]+0.5) + 1)
            ax.set_yticks(all_ch_numbers) # Set ticks at integer channel positions
            ax.set_yticklabels(list(reversed(all_ch_numbers))) # Apply reversed labels
        else:
            # For non-reversed, simple integer ticks and labels
            ax.set_yticks(np.arange(int(channel_range[0]+0.5), int(channel_range[1]+0.5) + 1))


    if cbar: # Only create colorbar if cbar is True
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_amp_for_cmap, vmax=max_amp_for_cmap))
        sm.set_array([])
        cbar_obj = plt.colorbar(sm, ax=ax)
        cbar_obj.set_label('Amplitude')

