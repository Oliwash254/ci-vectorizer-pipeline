import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def plot_analog_electrodogram(X, fs, channel_labels=None, fig=None, ax=None, s='auto'):
    """
    Plot a stacked electrodogram with each channel offset vertically.
    """
    if s == 'auto':
        m = np.max(np.abs(X))
        s = .49 / m if m > 0 else 1

    if fig is None and ax is None:
        fig_, ax_ = plt.subplots(figsize=(10, 6))
    else:
        fig_ = fig if fig is not None else plt.figure()
        ax_ = ax if ax is not None else fig_.add_subplot()

    if channel_labels is None:
        channel_labels = list(range(1, X.shape[0] + 1))

    t = np.arange(X.shape[1]) / fs
    s_t = 1
    t_u = 's'
    if t[-1] < 1:
        s_t = 1e3
        t_u = 'ms'

    for i in range(X.shape[0]):
        ax_.axhline(i, color='k', lw=0.1)
        ax_.add_line(
            plt.Line2D(t * s_t, s * X[i, :] + i, color='k')
        )

    ax_.set_yticks(np.arange(X.shape[0]))
    ax_.set_yticklabels(channel_labels)
    ax_.set_xlabel(f"Time ({t_u})")
    ax_.set_ylabel("Channels")
    ax_.autoscale_view()

    if fig_ is not None:
        fig_.tight_layout()

        def on_resize(event):
            fig_.tight_layout()
            fig_.canvas.draw()

        fig_.canvas.mpl_connect('resize_event', on_resize)

    return fig_

def plot_multichannel_waveform(waveform, fs, title="Waveform", channel_names=None):
    """
    Plot a multichannel waveform using Matplotlib and display it via Streamlit.
    """
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]  # Convert mono to 2D

    n_channels, n_samples = waveform.shape
    duration = n_samples / fs
    time = np.linspace(0, duration, n_samples)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(n_channels):
        label = channel_names[i] if channel_names else f"Ch {i+1}"
        ax.plot(time, waveform[i], label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize="small")
    st.pyplot(fig)
