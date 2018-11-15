# Setup WaveNet vocoder hparams
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

from hparams import hparams
wn_preset = "20180510_mixture_lj_checkpoint_step000320000_ema.json"
wn_checkpoint_path = "20180510_mixture_lj_checkpoint_step000320000_ema.pth"
with open(wn_preset) as f:
    hparams.parse_json(f.read())

# Setup WaveNet vocoder
from train import build_model
from synthesis import wavegen
import torch
from scipy.io import wavfile

from functools import partial
import numpy as np
import os
import sys
import audio
from tqdm import tqdm

from nnmnkwii import preprocessing as P
from hparams import hparams
from os.path import exists
import librosa

from wavenet_vocoder_core.util import is_mulaw_quantize, is_mulaw, is_raw

if len(sys.argv) < 2:
    raise ValueError("Must pass directory of wav files as only argument")

in_path = sys.argv[1]
assert os.path.exists(in_path)

def _process_utterance(wav_path, out_dir):
    fname = wav_path.split(os.sep)[-1].split(".")[0]
    audio_filename = '{}_resolved.npy'.format(fname)
    mel_filename = '{}_mel.npy'.format(fname)
    apth = os.path.join(out_dir, audio_filename)
    mpth = os.path.join(out_dir, mel_filename)
    if os.path.exists(apth) and os.path.exists(mpth):
        print("File {} already processed".format(wav_path))
        return

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    if hparams.rescaling:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max

    # Mu-law quantize
    if is_mulaw_quantize(hparams.input_type):
        # [0, quantize_channels)
        out = P.mulaw_quantize(wav, hparams.quantize_channels)

        # Trim silences
        start, end = audio.start_and_end_indices(out, hparams.silence_threshold)
        wav = wav[start:end]
        out = out[start:end]
        constant_values = P.mulaw_quantize(0, hparams.quantize_channels)
        out_dtype = np.int16
    elif is_mulaw(hparams.input_type):
        # [-1, 1]
        out = P.mulaw(wav, hparams.quantize_channels)
        constant_values = P.mulaw(0.0, hparams.quantize_channels)
        out_dtype = np.float32
    else:
        # [-1, 1]
        out = wav
        constant_values = 0.0
        out_dtype = np.float32

    # Compute a mel-scale spectrogram from the trimmed wav:
    # (N, D)
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32).T
    # lws pads zeros internally before performing stft
    # this is needed to adjust time resolution between audio and mel-spectrogram
    l, r = audio.lws_pad_lr(wav, hparams.fft_size, audio.get_hop_size())

    # zero pad for quantized signal
    out = np.pad(out, (l, r), mode="constant", constant_values=constant_values)
    N = mel_spectrogram.shape[0]
    assert len(out) >= N * audio.get_hop_size()

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop_size so that we can use
    # transposed convolution to upsample
    out = out[:N * audio.get_hop_size()]
    assert len(out) % audio.get_hop_size() == 0

    timesteps = len(out)

    # Write the spectrograms to disk:
    np.save(apth,
            out.astype(out_dtype), allow_pickle=False)
    np.save(mpth,
            mel_spectrogram.astype(np.float32), allow_pickle=False)


def soundsc(X, gain_scale=.9, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    gain_scale : float
        Gain multipler, default .9 (90% of maximum representation)

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-32767, 32767) scaled version of X as int16, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = gain_scale * X
    X = X * 2 ** 15
    return X.astype('int16')


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

print("Load checkpoint from {}".format(wn_checkpoint_path))
if use_cuda:
    checkpoint = torch.load(wn_checkpoint_path)
else:
    checkpoint = torch.load(wn_checkpoint_path, map_location="cpu")

if in_path[-1] == str(os.sep):
    in_path = in_path[:-1]

model = build_model().to(device)
model.load_state_dict(checkpoint["state_dict"])

wav_paths = [in_path + os.sep + "{}".format(fi) for fi in os.listdir(in_path) if ".wav" in fi]
out_dir = in_path + "_mel"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

for wp in wav_paths:
    print("Saving mels for {}".format(wp))
    _process_utterance(wp, out_dir)

mel_dir = out_dir
wav_out_dir = mel_dir + "_wavenet_render"
if not os.path.exists(wav_out_dir):
    os.mkdir(wav_out_dir)
sample_rate = 22050
mel_paths = [mel_dir + os.sep + "{}".format(fi) for fi in os.listdir(mel_dir) if "mel" in fi]
for mel_path in mel_paths:
    c = np.load(mel_path)
    if c.shape[1] != hparams.num_mels:
        np.swapaxes(c, 0, 1)
    waveform = wavegen(model, c=c, fast=True, tqdm=tqdm)
    fname = mel_path.split(os.sep)[-1].split(".")[0]
    fpath = wav_out_dir + str(os.sep) + '{}.wav'.format(fname)
    wavfile.write(fpath, sample_rate, waveform)
    print("Saved HD audio {}".format(fpath))
