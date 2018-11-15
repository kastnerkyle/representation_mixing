from ..datasets.audio import stft
from .plot import get_viridis
import numpy as np


def specgram(arr, fftsize=512, step=16, mean_normalize=True, real=False,
             compute_onesided=True, min_value=-100, max_value=np.inf, axis=0):
    arr = np.array(arr)
    if len(arr.shape) != 1:
        raise ValueError("arr must be a 1D np array or list")

    if axis != 0:
        raise ValueError("Must have axis=0")

    Pxx = 20. * np.log10(np.abs(stft(arr, fftsize=fftsize, step=step, mean_normalize=mean_normalize, real=real, compute_onesided=compute_onesided)))
    return np.clip(Pxx, min_value, max_value)


def specplot(arr, mplaxis, time_ratio=4, cmap="viridis"):
    """
    assumes arr comes in with time on axis 0, frequency on axis 1
    """
    import matplotlib.pyplot as plt
    if cmap == "viridis":
        cmap = get_viridis()
    # Transpose so time is X axis, and invert y axis so
    # frequency is low at bottom
    mag = arr.T[::-1, :]
    mplaxis.matshow(mag, cmap=cmap)
    x1 = mag.shape[0]
    y1 = mag.shape[1]

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        b = [x_range, y_range]
        mi = np.argmax(b)
        mx = b[mi]
        mn = b[1] if mi == 0 else b[0]
        ratio = time_ratio / 1. if mi == 0 else 1. / time_ratio
        if x_range <= y_range:
            return ratio * mx / float(mn)
        else:
            return ratio * mn / float(mx)
    asp = autoaspect(x1, y1)
    mplaxis.set_aspect(asp)
    mplaxis.xaxis.tick_bottom()
