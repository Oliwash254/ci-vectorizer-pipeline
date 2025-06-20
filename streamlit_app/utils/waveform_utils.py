import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def plot_electrodogram(
    pulse_times,
    pulse_amplitudes,
    channel_labels=None,
    duration=None,
    reverse_channels=False,
    normalize_amplitudes=False,
    max_channels=22,
    linewidth=1,
    figsize=(10, 6),
):
    """
    Plot an electrodogram from pulse times and amplitudes.

    Parameters
    ----------
    pulse_times : list of np.array
        List where each entry corresponds to a channel and contains pulse times (in seconds).
    pulse_amplitudes : list of np.array
        List where each entry corresponds to a channel and contains pulse amplitudes.
    channel_labels : list of int, optional
        Labels for channels (e.g., [1, 2, 3, ...]).
    duration : float, optional
        Maximum duration (in seconds) to show on x-axis.
    reverse_channels : bool, default=False
        If True, reverse channel order (for Cochlear: show channel 22 at bottom).
    normalize_amplitudes : bool, default=False
        If True, normalize amplitudes across all channels.
    max_channels : int, default=22
        Maximum number of channels expected.
    linewidth : float, default=1
        Line width for pulses.
    figsize : tuple, default=(10, 6)
        Size of the matplotlib figure.
    """
    if channel_labels is None:
        channel_labels = list(range(1, len(pulse_times) + 1))

    if reverse_channels:
        pulse_times = pulse_times[::-1]
        pulse_amplitudes = pulse_amplitudes[::-1]
        channel_labels = channel_labels[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    max_amp = max(
        np.max(np.abs(a)) if len(a) > 0 else 0 for a in pulse_amplitudes
    ) if normalize_amplitudes else 1

    for ch_idx, (pt, pa) in enumerate(zip(pulse_times, pulse_amplitudes)):
        ch = channel_labels[ch_idx]
        if len(pt) == 0:
            continue

        x = np.column_stack([pt, pt]).flatten()
        y = np.column_stack([
            np.full_like(pt, ch - 0.4),
            np.full_like(pt, ch + 0.4)
        ]).flatten()

        if normalize_amplitudes:
            amp_scaled = np.repeat(pa / max_amp * 0.8, 2)
            y = y * amp_scaled

        ax.plot(x, y, linewidth=linewidth, alpha=0.9)

    ax.set_yticks(channel_labels)
    ax.set_ylabel("Electrode Channel")
    ax.set_xlabel("Time (s)")
    ax.set_title("Electrodogram")
    ax.set_ylim((min(channel_labels)-1, max(channel_labels)+1))
    if duration is not None:
        ax.set_xlim((0, duration))
    ax.grid(True)
    st.pyplot(fig)


def render_electrodogram_and_download(pulse_times, pulse_amplitudes, filename="electrodogram.png", **kwargs):
    """
    Wrapper that renders and also exposes a download button.

    Parameters
    ----------
    pulse_times : list of np.array
        List of pulse time arrays.
    pulse_amplitudes : list of np.array
        List of pulse amplitude arrays.
    filename : str
        Download filename for the image.
    **kwargs : dict
        Additional keyword args passed to plot_electrodogram.
    """
    fig = render_electrodogram_to_figure(pulse_times, pulse_amplitudes, **kwargs)
    st.pyplot(fig)

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">Download Electrodogram as PNG</a>'
    st.markdown(href, unsafe_allow_html=True)


def render_electrodogram_to_figure(pulse_times, pulse_amplitudes, channel_labels=None,
                                   duration=None, reverse_channels=False,
                                   normalize_amplitudes=False, max_channels=22,
                                   linewidth=1, figsize=(10, 6)):
    """
    Returns the matplotlib figure object for the electrodogram.
    """
    if channel_labels is None:
        channel_labels = list(range(1, len(pulse_times) + 1))

    if reverse_channels:
        pulse_times = pulse_times[::-1]
        pulse_amplitudes = pulse_amplitudes[::-1]
        channel_labels = channel_labels[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    max_amp = max(
        np.max(np.abs(a)) if len(a) > 0 else 0 for a in pulse_amplitudes
    ) if normalize_amplitudes else 1

    for ch_idx, (pt, pa) in enumerate(zip(pulse_times, pulse_amplitudes)):
        ch = channel_labels[ch_idx]
        if len(pt) == 0:
            continue

        x = np.column_stack([pt, pt]).flatten()
        y = np.column_stack([
            np.full_like(pt, ch - 0.4),
            np.full_like(pt, ch + 0.4)
        ]).flatten()

        if normalize_amplitudes:
            amp_scaled = np.repeat(pa / max_amp * 0.8, 2)
            y = y * amp_scaled

        ax.plot(x, y, linewidth=linewidth, alpha=0.9)

    ax.set_yticks(channel_labels)
    ax.set_ylabel("Electrode Channel")
    ax.set_xlabel("Time (s)")
    ax.set_title("Electrodogram")
    ax.set_ylim((min(channel_labels)-1, max(channel_labels)+1))
    if duration is not None:
        ax.set_xlim((0, duration))
    ax.grid(True)

    return fig
