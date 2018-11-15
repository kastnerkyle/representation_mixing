from __future__ import print_function
# Author: Kyle Kastner
# License: BSD 3-clause
# Thanks to Jose (@sotelo) for tons of guidance and debug help
# Credit also to Junyoung (@jych) and Shawn (@shawntan) for help/utility funcs
import os
import re
import tarfile
from collections import Counter, OrderedDict
import sys
import pickle
import numpy as np
import fnmatch
from scipy import linalg
from scipy.io import wavfile
from scipy import fftpack
from functools import wraps
import exceptions
import subprocess
import copy
import shutil
import xml
import xml.etree.cElementTree as ElementTree
import HTMLParser
import functools
import operator
import gzip
import struct
import array
from .audio import stft

from ..core import download
from ..core import get_logger

logger = get_logger()

def get_tfbldr_dataset_dir(dirname=None):
    lookup_dir = os.getenv("TFBLDR_DATASETS", os.path.join(
        os.path.expanduser("~"), "tfbldr_datasets"))
    if not os.path.exists(lookup_dir):
        logger.info("TFBLDR_DATASETS directory {} not found, creating".format(lookup_dir))
        os.mkdir(lookup_dir)
    if dirname is None:
        return lookup_dir

    subdir = os.path.join(lookup_dir, dirname)
    if not os.path.exists(subdir):
        logger.info("TFBLDR_DATASETS subdirectory {} not found, creating".format(subdir))
        os.mkdir(subdir)
    return subdir


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=True, verbose=True):
    """
    Print and execute command on system
    """
    all_lines = []
    for line in execute(cmd, shell=shell):
        if verbose:
            print(line, end="")
        all_lines.append(line.strip())
    return all_lines


# https://mrcoles.com/blog/3-decorator-examples-and-awesome-python/
def rsync_fetch(fetch_func, machine_to_fetch_from, *args, **kwargs):
    """
    assumes the filename in IOError is a subdir, will rsync one level above that

    be sure not to call it as
    rsync_fetch(fetch_func, machine_name)
    not
    rsync_fetch(fetch_func(), machine_name)
    """
    try:
        r = fetch_func(*args, **kwargs)
    except Exception as e:
        if isinstance(e, IOError):
            full_path = e.filename
            filedir = str(os.sep).join(full_path.split(os.sep)[:-1])
            if not os.path.exists(filedir):
                if filedir[-1] != "/":
                    fd = filedir + "/"
                else:
                    fd = filedir
                os.makedirs(fd)

            if filedir[-1] != "/":
                fd = filedir + "/"
            else:
                fd = filedir

            if not os.path.exists(full_path):
                sdir = str(machine_to_fetch_from) + ":" + fd
                cmd = "rsync -avhp --progress %s %s" % (sdir, fd)
                pe(cmd, shell=True)
        else:
            print("unknown error {}".format(e))
        r = fetch_func(*args, **kwargs)
    return r


"""
- all points:
>> [[x1, y1, e1], ..., [xn, yn, en]]
- indexed values
>> [h1, ... hn]
"""


def distance(p1, p2, axis=None):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=axis))


def clear_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = distance(p1, p2) + distance(p2, p3)
        if dist > 1500:
            to_remove.add(i)
    npts = []
    for i in range(len(pts)):
        if i not in to_remove:
            npts += [pts[i]]
    return np.array(npts)


def separate(pts):
    seps = []
    for i in range(0, len(pts) - 1):
        if distance(pts[i], pts[i+1]) > 600:
            seps += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seps, seps + [len(pts)])]


def iamondb_extract(partial_path):
    """
    Lightly modified from https://github.com/Grzego/handwriting-generation/blob/master/preprocess.py
    """
    data = []
    charset = set()

    file_no = 0
    pth = os.path.join(partial_path, "original")
    for root, dirs, files in os.walk(pth):
        # sort the dirs to iterate the same way every time
        # https://stackoverflow.com/questions/18282370/os-walk-iterates-in-what-order
        dirs.sort()
        for file in files:
            file_name, extension = os.path.splitext(file)
            if extension == '.xml':
                file_no += 1
                print('[{:5d}] File {} -- '.format(file_no, os.path.join(root, file)), end='')
                xml = ElementTree.parse(os.path.join(root, file)).getroot()
                transcription = xml.findall('Transcription')
                if not transcription:
                    print('skipped')
                    continue
                #texts = [html.unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                texts = [HTMLParser.HTMLParser().unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                points = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]
                strokes = []
                mid_points = []
                for ps in points:
                    pts = np.array([[int(p.get('x')), int(p.get('y')), 0] for p in ps])
                    pts[-1, 2] = 1

                    pts = clear_middle(pts)
                    if len(pts) == 0:
                        continue

                    seps = separate(pts)
                    for pss in seps:
                        if len(seps) > 1 and len(pss) == 1:
                            continue
                        pss[-1, 2] = 1

                        xmax, ymax = max(pss, key=lambda x: x[0])[0], max(pss, key=lambda x: x[1])[1]
                        xmin, ymin = min(pss, key=lambda x: x[0])[0], min(pss, key=lambda x: x[1])[1]

                        strokes += [pss]
                        mid_points += [[(xmax + xmin) / 2., (ymax + ymin) / 2.]]
                distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                             for p1, p2 in zip(mid_points, mid_points[1:])]
                splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)
                lines = []
                for b, e in zip([0] + splits, splits + [len(strokes)]):
                    lines += [[p for pts in strokes[b:e] for p in pts]]
                print('lines = {:4d}; texts = {:4d}'.format(len(lines), len(texts)))
                charset |= set(''.join(texts))
                data += [(texts, lines)]
    print('data = {}; charset = ({}) {}'.format(len(data), len(charset), ''.join(sorted(charset))))

    translation = {'<NULL>': 0}
    for c in ''.join(sorted(charset)):
        translation[c] = len(translation)

    def translate(txt):
        return list(map(lambda x: translation[x], txt))

    dataset = []
    labels = []
    for texts, lines in data:
        for text, line in zip(texts, lines):
            line = np.array(line, dtype=np.float32)
            line[:, 0] = line[:, 0] - np.min(line[:, 0])
            line[:, 1] = line[:, 1] - np.mean(line[:, 1])

            dataset += [line]
            labels += [translate(text)]

    whole_data = np.concatenate(dataset, axis=0)

    std_y = np.std(whole_data[:, 1])
    norm_data = []
    for line in dataset:
        line[:, :2] /= std_y
        norm_data += [line]
    dataset = norm_data

    print('datset = {}; labels = {}'.format(len(dataset), len(labels)))

    save_path = os.path.join(partial_path, 'preprocessed_data')
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass
    np.save(os.path.join(save_path, 'dataset'), np.array(dataset))
    np.save(os.path.join(save_path, 'labels'), np.array(labels))
    with open(os.path.join(save_path, 'translation.pkl'), 'wb') as file:
        pickle.dump(translation, file)
    print("Preprocessing finished and cached at {}".format(save_path))


def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    #partial_path = get_dataset_dir("iamondb")
    partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "iamondb"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(combined_data_path):
        files = "original-xml-part.tar.gz"
        url = "http://www.iam.unibe.ch/fki/databases/"
        url += "iam-on-line-handwriting-database/"
        url += "download-the-iam-on-line-handwriting-database"
        err = "Path %s does not exist!" % combined_data_path
        err += " Download the %s files from %s" % (files, url)
        err += " and place them in the directory %s" % partial_path
        print("WARNING: {}".format(err))
    return partial_path


def fetch_iamondb():
    partial_path = check_fetch_iamondb()
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(untarred_data_path):
        print("Now untarring {}".format(combined_data_path))
        tar = tarfile.open(combined_data_path, "r:gz")
        tar.extractall(partial_path)
        tar.close()

    saved_dataset_path = os.path.join(partial_path, 'preprocessed_data')

    if not os.path.exists(saved_dataset_path):
        iamondb_extract(partial_path)

    dataset_path = os.path.join(saved_dataset_path, "dataset.npy")
    labels_path = os.path.join(saved_dataset_path, "labels.npy")
    translation_path = os.path.join(saved_dataset_path, "translation.pkl")

    dataset = np.load(dataset_path)
    dataset = [np.array(d) for d in dataset]

    temp = []
    for d in dataset:
        # dataset stores actual pen points, but we will train on differences between consecutive points
        offs = d[1:, :2] - d[:-1, :2]
        ends = d[1:, 2]
        temp += [np.concatenate([[[0., 0., 1.]], np.concatenate([offs, ends[:, None]], axis=1)], axis=0)]
    # because lines are of different length, we store them in python array (not numpy)
    dataset = temp
    labels = np.load(labels_path)
    labels = [np.array(l) for l in labels]
    with open(translation_path, 'rb') as f:
        translation = pickle.load(f)
    # be sure of consisten ordering
    new_translation = OrderedDict()
    for k in sorted(translation.keys()):
        new_translation[k] = translation[k]
    translation = new_translation
    dataset_storage = {}
    dataset_storage["data"] = dataset
    dataset_storage["target"] = labels
    inverse_translation = {v: k for k, v in translation.items()}
    dataset_storage["target_phrases"] = ["".join([inverse_translation[ci] for ci in labels[i]]) for i in range(len(labels))]
    dataset_storage["vocabulary_size"] = len(translation)
    dataset_storage["vocabulary"] = translation
    return dataset_storage


def fetch_ljspeech(path="/Tmp/kastner/lj_speech/LJSpeech-1.0/"):
    if not path.endswith(os.sep):
        path = path + os.sep

    if not os.path.exists(path + "wavs"):
        e = IOError("No wav files found in {}, under {}".format(path, path + "wavs"), None, path + "wavs")
        raise e
    if not os.path.exists(path + "txts"):
        e = IOError("No txt files found in {}, under {}".format(path, path + "txts"), None, path + "txts")
        raise e
    if not os.path.exists(path + "phones"):
        e = IOError("No phone files found in {}, under {}".format(path, path + "phones"), None, path + "phones")
        raise e
    if not os.path.exists(path + "gentle_json"):
        e = IOError("No phone files found in {}, under {}".format(path, path + "phones"), None, path + "phones")
        raise e
    wavfiles = [path + "wavs/" + ff for ff in os.listdir(path + "wavs/")]
    txtfiles = [path + "txts/" + ff for ff in os.listdir(path + "txts/")]
    phonefiles = [path + "phones/" + ff for ff in os.listdir(path + "phones/")]
    jsonfiles = [path + "gentle_json/" + ff for ff in os.listdir(path + "gentle_json/")]
    d = {}
    d["wavfiles"] = wavfiles
    d["txtfiles"] = txtfiles
    d["phonefiles"] = phonefiles
    d["jsonfiles"] = jsonfiles
    return d


"""
def check_fetch_ljspeech(conditioning_type):
    ''' Check for ljspeech

        This dataset cannot be downloaded or preprocessed automatically!
    '''
    if conditioning_type == "hybrid":
        partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "lj_speech_hybrid_speakers"
    else:
        raise ValueError("Unknown conditioning_type={} specified".format(conditioning_type))
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(partial_path + os.sep + "norm_info") or not os.path.exists(partial_path + os.sep + "numpy_features"):
        err = "lj_speech_hybrid_speakers files not found. These files need special preprocessing! Do that, and place norm_info and numpy_features in {}"
        print("WARNING: {}".format(err.format(partial_path)))
    return partial_path

def fetch_ljspeech(conditioning_type="hybrid"):
    '''
    only returns file paths, and metadata/conversion routines
    '''
    partial_path = check_fetch_ljspeech(conditioning_type)
    features_path = os.path.join(partial_path, "numpy_features")
    norm_path = os.path.join(partial_path, "norm_info")
    if not os.path.exists(features_path) or not os.path.exists(norm_path):
        e = IOError("No feature files found in {}, under {}".format(partial_path, features_path), None, features_path)
        raise e

    feature_files = [features_path + os.sep + f for f in os.listdir(features_path)]
    if len(feature_files) == 0:
        e = IOError("No feature files found in {}, under {}".format(partial_path, features_path), None, features_path)
        raise e

    ljspeech_hybridset = [' ', '!', ',', '-', '.', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    translation = OrderedDict()
    for n, k in enumerate(ljspeech_hybridset):
        translation[k] = n

    dataset_storage = {}
    dataset_storage["file_paths"] = feature_files
    dataset_storage["vocabulary_size"] = len(ljspeech_hybridset)
    dataset_storage["vocabulary"] = translation
    return dataset_storage
"""


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def parse_idx(fd):
    """
    Parse an IDX file, and return it as a numpy array.
    From https://github.com/datapythonista/mnist

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


def check_fetch_mnist():
    mnist_dir = get_tfbldr_dataset_dir("mnist")
    base = "http://yann.lecun.com/exdb/mnist/"
    zips = [base + "train-images-idx3-ubyte.gz",
            base + "train-labels-idx1-ubyte.gz",
            base + "t10k-images-idx3-ubyte.gz",
            base + "t10k-labels-idx1-ubyte.gz"]

    for z in zips:
        fname = z.split("/")[-1]
        full_path = os.path.join(mnist_dir, fname)
        if not os.path.exists(full_path):
            logger.info("{} not found, downloading...".format(full_path))
            download(z, full_path)
    return mnist_dir


def fetch_mnist():
    """
    Flattened or image-shaped 28x28 mnist digits with float pixel values in [0 - 255]

    n_samples : 70000
    n_feature : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : float32 array, shape (70000, 784)
        summary["target"] : int32 array, shape (70000,)
        summary["images"] : float32 array, shape (70000, 28, 28, 1)
        summary["train_indices"] : int32 array, shape (50000,)
        summary["valid_indices"] : int32 array, shape (10000,)
        summary["test_indices"] : int32 array, shape (10000,)

    """
    data_path = check_fetch_mnist()
    train_image_gz = "train-images-idx3-ubyte.gz"
    train_label_gz = "train-labels-idx1-ubyte.gz"
    test_image_gz = "t10k-images-idx3-ubyte.gz"
    test_label_gz = "t10k-labels-idx1-ubyte.gz"

    out = []
    for path in [train_image_gz, train_label_gz, test_image_gz, test_label_gz]:
        f = gzip.open(os.path.join(data_path, path), 'rb')
        out.append(parse_idx(f))
        f.close()
    train_indices = np.arange(0, 50000)
    valid_indices = np.arange(50000, 60000)
    test_indices = np.arange(60000, 70000)
    data = np.concatenate((out[0], out[2]),
                          axis=0).astype(np.float32)
    target = np.concatenate((out[1], out[3]),
                            axis=0).astype(np.int32)
    return {"data": copy.deepcopy(data.reshape((data.shape[0], -1))),
            "target": target,
            "images": data[..., None],
            "train_indices": train_indices.astype(np.int32),
            "valid_indices": valid_indices.astype(np.int32),
            "test_indices": test_indices.astype(np.int32)}


def check_fetch_fashion_mnist():
    fashion_mnist_dir = get_tfbldr_dataset_dir("fashion_mnist")

    base = "https://raw.githubusercontent.com/kastnerkyle/fashion-mnist/master/data/fashion/"
    zips = [base + "train-images-idx3-ubyte.gz",
            base + "train-labels-idx1-ubyte.gz",
            base + "t10k-images-idx3-ubyte.gz",
            base + "t10k-labels-idx1-ubyte.gz"]

    for z in zips:
        fname = z.split("/")[-1]
        full_path = os.path.join(fashion_mnist_dir, fname)
        if not os.path.exists(full_path):
            logger.info("{} not found, downloading...".format(full_path))
            download(z, full_path, bypass_certificate_check=True)
    return fashion_mnist_dir


def fetch_fashion_mnist():
    """
    Flattened or image-shaped 28x28 fashion mnist digits with float pixel values in [0 - 255]

    n_samples : 70000
    n_feature : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : float32 array, shape (70000, 784)
        summary["target"] : int32 array, shape (70000,)
        summary["images"] : float32 array, shape (70000, 28, 28, 1)
        summary["train_indices"] : int32 array, shape (50000,)
        summary["valid_indices"] : int32 array, shape (10000,)
        summary["test_indices"] : int32 array, shape (10000,)

    """
    data_path = check_fetch_fashion_mnist()
    train_image_gz = "train-images-idx3-ubyte.gz"
    train_label_gz = "train-labels-idx1-ubyte.gz"
    test_image_gz = "t10k-images-idx3-ubyte.gz"
    test_label_gz = "t10k-labels-idx1-ubyte.gz"

    out = []
    for path in [train_image_gz, train_label_gz, test_image_gz, test_label_gz]:
        f = gzip.open(os.path.join(data_path, path), 'rb')
        out.append(parse_idx(f))
        f.close()
    train_indices = np.arange(0, 50000)
    valid_indices = np.arange(50000, 60000)
    test_indices = np.arange(60000, 70000)
    data = np.concatenate((out[0], out[2]),
                          axis=0).astype(np.float32)
    target = np.concatenate((out[1], out[3]),
                            axis=0).astype(np.int32)
    return {"data": copy.deepcopy(data.reshape((data.shape[0], -1))),
            "target": target,
            "images": data[..., None],
            "train_indices": train_indices.astype(np.int32),
            "valid_indices": valid_indices.astype(np.int32),
            "test_indices": test_indices.astype(np.int32)}


def check_fetch_fruitspeech():
    """ Check for fruitspeech data

    Recorded by Hakon Sandsmark
    """
    url = "https://raw.githubusercontent.com/kastnerkyle/fruitspeech_dataset/master/fruitspeech.tar.gz"
    partial_path = get_tfbldr_dataset_dir("fruitspeech")
    full_path = os.path.join(partial_path, "fruitspeech.tar.gz")
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(full_path):
        download(url, full_path, progress_update_percentage=1)
    audio_path = os.path.join(partial_path, "fruitspeech")
    if not os.path.exists(audio_path):
        tar = tarfile.open(full_path)
        os.chdir(partial_path)
        tar.extractall()
        tar.close()
    return audio_path


def fetch_fruitspeech(fftsize=512, step=16, mean_normalize=True,
                      real=False, compute_onesided=False):
    audio_path = check_fetch_fruitspeech()
    files = sorted([audio_path + os.sep + f for f in os.listdir(audio_path) if f.endswith(".wav")])
    specgrams = []
    raws = []
    words = []
    for wav_path in files:
        fs, d = wavfile.read(wav_path)
        d = d.astype("float32")
        Pxx = 20. * np.log10(np.abs(stft(d, fftsize=fftsize, step=step, mean_normalize=mean_normalize, real=real, compute_onesided=compute_onesided)))
        word = wav_path.split(os.sep)[-1].split("_")[0]
        specgrams.append(Pxx)
        raws.append(d)
        words.append(word)
    out = {}
    out["specgrams"] = specgrams
    out["data"] = raws
    out["target"] = words
    return out


def make_sinewaves(n_timesteps, n_waves, base_freq=3, offset=True,
                   use_cos=False,
                   harmonic=False,
                   harmonic_multipliers=[1.7, .62],
                   square=False,
                   square_thresh=0):
    """
    Generate sinewaves offset in phase, with optional harmonics or as square wave
    """
    n_full = n_timesteps
    n_offsets = n_waves
    d1 = float(base_freq) * np.arange(n_full) / (2 * np.pi)
    d2 = float(base_freq) * np.arange(n_offsets) / (2 * np.pi)
    if not offset:
        d2 *= 0.
    wave_type = np.sin if not use_cos else np.cos
    full_sines = wave_type(np.array([d1] * n_offsets).T + d2).astype("float32")
    # Uncomment to add harmonics
    if harmonic:
        for harmonic_m in harmonic_multipliers:
            full_sines += wave_type(np.array([harmonic_m * d1] * n_offsets).T + d2)
    if square:
        full_sines[full_sines <= square_thresh] = 0
        full_sines[full_sines > square_thresh] = 1
    full_sines = full_sines[:, :, None]
    return full_sines


def check_fetch_norvig_words():
    partial_path = get_tfbldr_dataset_dir("norvig_words")
    full_path = partial_path + os.sep + "count_1w.txt"
    if not os.path.exists(full_path):
        logger.info("{} not found, downloading...".format(full_path))
        download("https://norvig.com/ngrams/count_1w.txt", full_path)
    return partial_path


def fetch_norvig_words():
    # https://norvig.com/ngrams/count_1w.txt
    words_path = check_fetch_norvig_words()

    words_file = words_path + os.sep + "count_1w.txt"
    with open(words_file, "r") as f:
        lines = f.readlines()
    words = [l.split("\t")[0] for l in lines] 
    words = sorted(words)[::-1]
    d = {}
    d["data"] = words
    return d
