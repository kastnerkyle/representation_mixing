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
import itertools
import xml
import xml.etree.cElementTree as ElementTree
import HTMLParser
import functools
import operator
import gzip
import struct
import array
import copy

from .audio import stft

from ..core import download
from ..core import get_logger

from itertools import islice, chain
import string

logger = get_logger()

def make_mask(arr):
    mask = np.ones_like(arr[:, :, 0])
    last_step = arr.shape[0] * arr[0, :, 0]
    for mbi in range(arr.shape[1]):
        for step in range(arr.shape[0]):
            if arr[step:, mbi].min() == 0. and arr[step:, mbi].max() == 0.:
                last_step[mbi] = step
                mask[step:, mbi] = 0.
                break
    return mask


# https://stackoverflow.com/questions/845058/how-to-get-line-count-cheaply-in-python
def _make_gen(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)

def rawgencount(filename):
    f = open(filename, 'rb')
    # 2 / 3 compat
    if hasattr(f, "raw"):
        f_gen = _make_gen(f.raw.read)
    else:
        f_gen = _make_gen(f.read)
    return sum(buf.count(b'\n') for buf in f_gen)

ascii_printable = string.printable

class char_textfile_iterator(object):
    def __init__(self, textfile_path, batch_size,
                 seq_length,
                 number_of_lines_in_file=None,
                 one_hot_size=None, random_state=None):
        """ split the file evenly into batch_size chunks,
            contiguous, truncating the last uneven part """

        self.textfile_path = textfile_path
        self.random_state = random_state
        self.batch_size = batch_size
        self.seq_length = seq_length
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.char2ind = {k: v for v, k in enumerate(string.printable)}
        self.ind2char = {v: k for k, v in self.char2ind.items()}

        if number_of_lines_in_file is None:
            n_lines = rawgencount(textfile_path)
            number_of_lines_in_file = n_lines
        self.number_of_lines_in_file = number_of_lines_in_file

        init_gap_ = (number_of_lines_in_file - number_of_lines_in_file % batch_size) // batch_size
        self.init_gap_ = init_gap_

        self.indices_ = [i * self.init_gap_ for i in range(self.batch_size)]
        self._f_handles = [open(textfile_path, 'rb') for i in range(self.batch_size)]
        self.chainslices_ = [chain.from_iterable(islice(self._f_handles[n], ind, ind + self.init_gap_)) for n, ind in enumerate(self.indices_)]

    def next_batch(self):
        batch = np.zeros((self.seq_length, self.batch_size, 1)).astype("int32")
        resets = np.zeros((self.seq_length, self.batch_size))
        for bi in range(self.batch_size):
            for si in range(self.seq_length):
                try:
                    c = self.chainslices_[bi].next()
                    batch[si, bi] = self.char2ind[c]
                except StopIteration:
                    ind = self.random_state.randint(0, self.number_of_lines_in_file - self.init_gap_ - 1)
                    self._f_handles[bi] = open(self.textfile_path, "rb")
                    self.indices_[bi] = ind
                    self.chainslices_[bi] = chain.from_iterable(islice(self._f_handles[bi], ind, ind + self.init_gap_))
                    resets[si, bi] = 1.
                    c = self.chainslices_[bi].next()
                    batch[si, bi] = self.char2ind[c]

        batch = batch.astype("float32")
        return batch, resets


class list_iterator(object):
    def __init__(self, list_of_iteration_args, batch_size,
                 one_hot_size=None, random_state=None):
        """
        one_hot_size
        should be either None, or a list of one hot size desired
        same length as list_of_iteration_args

        list_of_iteration_args = [my_image_data, my_label_data]
        one_hot_size = [None, 10]
        """
        self.list_of_iteration_args = list_of_iteration_args
        self.batch_size = batch_size

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.one_hot_size = one_hot_size
        if one_hot_size is not None:
            assert len(one_hot_size) == len(list_of_iteration_args)

        iteration_args_lengths = []
        iteration_args_dims = []
        for n, ts in enumerate(list_of_iteration_args):
            c = [(li, np.array(tis).shape) for li, tis in enumerate(ts)]
            if len(iteration_args_lengths) > 0:
                if len(c[-1][1]) == 0:
                    raise ValueError("iteration_args arguments should be at least 2D arrays, detected 1D")
                # +1 to handle len vs idx offset
                if c[-1][0] + 1 != iteration_args_lengths[-1]:
                    raise ValueError("iteration_args arguments should have the same iteration length (dimension 0)")
                #if c[-1][1] != iteration_args_dims[-1]:
                #    from IPython import embed; embed(); raise ValueError()

            iteration_args_lengths.append(c[-1][0] + 1)
            iteration_args_dims.append(c[-1][1])
        self.iteration_args_lengths_ = iteration_args_lengths
        self.iteration_args_dims_ = iteration_args_dims

        # set up the matrices to slice one_hot indexes out of
        # todo: setup slice functions? or just keep handling in next_batch
        if one_hot_size is None:
            self._oh_slicers = [None] * len(list_of_iteration_args)
        else:
            self._oh_slicers = []
            for ooh in one_hot_size:
                if ooh is None:
                    self._oh_slicers.append(None)
                else:
                    self._oh_slicers.append(np.eye(ooh, dtype=np.float32))

        # set up the indices selected for the first batch
        self.indices_ = self.random_state.choice(self.iteration_args_lengths_[0],
                                                 size=(batch_size,), replace=False)

    def next_batch(self):
        next_batches = []
        for l in range(len(self.list_of_iteration_args)):
            if self._oh_slicers[l] is None:
                t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l]), dtype=np.float32)
            else:
                t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l])[:-1] + [self._oh_slicers[l].shape[-1]], dtype=np.float32)
            for bi in range(self.batch_size):
                t[bi] = self.list_of_iteration_args[l][self.indices_[bi]]
            next_batches.append(t)
        self.indices_ = self.random_state.choice(self.iteration_args_lengths_[0],
                                                 size=(self.batch_size,), replace=False)
        return next_batches


class ordered_list_iterator(object):
    def __init__(self, list_of_iteration_args, index_list, batch_size,
                 one_hot_size=None, random_state=None):
        """
        one_hot_size
        should be either None, or a list of one hot size desired
        same length as list_of_iteration_args

        index list should tell whether or not some indexes are "grouped"

        list_of_iteration_args = [my_image_data, my_label_data]
        one_hot_size = [None, 10]
        """
        self.list_of_iteration_args = list_of_iteration_args
        self.batch_size = batch_size

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.one_hot_size = one_hot_size
        if one_hot_size is not None:
            assert len(one_hot_size) == len(list_of_iteration_args)

        iteration_args_lengths = []
        iteration_args_dims = []
        for n, ts in enumerate(list_of_iteration_args):
            c = [(li, np.array(tis).shape) for li, tis in enumerate(ts)]
            if len(iteration_args_lengths) > 0:
                assert c[-1][0] == iteration_args_lengths[-1]
                assert c[-1][1] == iteration_args_dims[-1]
            iteration_args_lengths.append(c[-1][0] + 1)
            iteration_args_dims.append(c[-1][1])
        self.iteration_args_lengths_ = iteration_args_lengths
        self.iteration_args_dims_ = iteration_args_dims

        # set up the matrices to slice one_hot indexes out of
        # todo: setup slice functions? or just keep handling in next_batch
        if one_hot_size is None:
            self._oh_slicers = [None] * len(list_of_iteration_args)
        else:
            self._oh_slicers = []
            for ooh in one_hot_size:
                if ooh is None:
                    self._oh_slicers.append(None)
                else:
                    self._oh_slicers.append(np.eye(ooh, dtype=np.float32))

        self.index_list = index_list
        if len(self.index_list) != self.iteration_args_lengths_[0]:
            raise ValueError("index_list must have same length as iterations args, got {} and {}".format(len(self.index_list), self.iteration_args_lengths_[0]))
        self.index_set = sorted(list(set(index_list)))
        self.index_groups = {k: np.array([n for n, i in enumerate(index_list) if i == k]) for k in self.index_set}

        shuf_set = copy.copy(self.index_set)
        self.random_state.shuffle(shuf_set)
        self.all_index_ = np.array([i for i in shuf_set for ii in self.index_groups[i]])
        self.all_indices_ = np.array([ii for i in shuf_set for ii in self.index_groups[i]])
        self.all_indices_ = self.all_indices_[:len(self.all_indices_) - len(self.all_indices_) % self.batch_size]

        self.indices_offset_ = 0
        self.indices_ = self.all_indices_[self.indices_offset_:self.indices_offset_ + self.batch_size]
        self.index_ = self.all_index_[self.indices_offset_:self.indices_offset_ + self.batch_size]
        self.indices_offset_ += self.batch_size


    def next_batch(self):
        next_batches = []
        for l in range(len(self.list_of_iteration_args)):
            if self._oh_slicers[l] is None:
                t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l]), dtype=np.float32)
            else:
                t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l])[:-1] + [self._oh_slicers[l].shape[-1]], dtype=np.float32)
            for bi in range(self.batch_size):
                t[bi] = self.list_of_iteration_args[l][self.indices_[bi]]
            next_batches.append(t)

        this_index = self.index_

        if self.indices_offset_ + self.batch_size >= len(self.all_indices_):
            shuf_set = copy.copy(self.index_set)
            self.random_state.shuffle(shuf_set)
            self.all_index_ = np.array([i for i in shuf_set for ii in self.index_groups[i]])
            self.all_indices_ = np.array([ii for i in shuf_set for ii in self.index_groups[i]])
            self.all_indices_ = self.all_indices_[:len(self.all_indices_) - len(self.all_indices_) % self.batch_size]
            self.indices_offset_ = 0
        self.indices_ = self.all_indices_[self.indices_offset_:self.indices_offset_ + self.batch_size]
        self.index_ = self.all_index_[self.indices_offset_:self.indices_offset_ + self.batch_size]
        self.indices_offset_ += self.batch_size
        return next_batches


class tbptt_list_iterator(object):
    def __init__(self, tbptt_seqs, list_of_other_seqs, batch_size,
                 truncation_length,
                 tbptt_one_hot_size=None, other_one_hot_size=None,
                 random_state=None):
        """
        skips sequences shorter than truncation_len
        also cuts the tail off

        tbptt_one_hot_size
        should be either None, or the one hot size desired

        other_one_hot_size
        should either be None (if not doing one-hot) or a list the same length
        as the respective argument with integer one hot size, or None
        for no one_hot transformation, example:

        list_of_other_seqs = [my_char_data, my_vector_data]
        other_one_hot_size = [127, None]
        """
        self.tbptt_seqs = tbptt_seqs
        self.list_of_other_seqs = list_of_other_seqs
        self.batch_size = batch_size
        self.truncation_length = truncation_length

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.tbptt_one_hot_size = tbptt_one_hot_size

        self.other_one_hot_size = other_one_hot_size
        if other_one_hot_size is not None:
            assert len(other_one_hot_size) == len(list_of_other_seqs)

        tbptt_seqs_length = [n for n, i in enumerate(tbptt_seqs)][-1] + 1
        self.indices_lookup_ = {}
        s = 0
        for n, ts in enumerate(tbptt_seqs):
            if len(ts) >= truncation_length + 1:
                self.indices_lookup_[s] = n
                s += 1

        # this one has things removed
        self.tbptt_seqs_length_ = len(self.indices_lookup_)

        other_seqs_lengths = []
        for other_seqs in list_of_other_seqs:
            r = [n for n, i in enumerate(other_seqs)]
            l = r[-1] + 1
            other_seqs_lengths.append(l)
        self.other_seqs_lengths_ = other_seqs_lengths

        other_seqs_max_lengths = []
        for other_seqs in list_of_other_seqs:
            max_l = -1
            for os in other_seqs:
                max_l = len(os) if len(os) > max_l else max_l
            other_seqs_max_lengths.append(max_l)
        self.other_seqs_max_lengths_ = other_seqs_max_lengths

        # make sure all sequences have the minimum number of elements
        base = self.tbptt_seqs_length_
        for sl in self.other_seqs_lengths_:
            assert sl >= base

        # set up the matrices to slice one_hot indexes out of
        # todo: setup slice functions? or just keep handling in next_batch
        if tbptt_one_hot_size is None:
            self._tbptt_oh_slicer = None
        else:
            self._tbptt_oh_slicer = np.eye(tbptt_one_hot_size)

        if other_one_hot_size is None:
            self._other_oh_slicers = [None] * len(other_seq_lengths)
        else:
            self._other_oh_slicers = []
            for ooh in other_one_hot_size:
                if ooh is None:
                    self._other_oh_slicers.append(None)
                else:
                    self._other_oh_slicers.append(np.eye(ooh, dtype=np.float32))
        # set up the indices selected for the first batch
        self.indices_ = np.array([self.indices_lookup_[si]
                                  for si in self.random_state.choice(self.tbptt_seqs_length_,
                                      size=(batch_size,), replace=False)])
        # set up the batch offset indicators for tracking where we are
        self.batches_ = np.zeros((batch_size,), dtype=np.int32)

    def next_batch(self):
        # whether the result is "fresh" or continuation
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            # cuts off the end of every long sequence! tricky logic
            if self.batches_[i] + self.truncation_length + 1 > self.tbptt_seqs[self.indices_[i]].shape[0]:
                ni = self.indices_lookup_[self.random_state.randint(0, self.tbptt_seqs_length_ - 1)]
                self.indices_[i] = ni
                self.batches_[i] = 0
                reset_states[i] = 0.

        # could slice before one hot to be slightly more efficient but eh
        items = [self.tbptt_seqs[ii] for ii in self.indices_]
        if self._tbptt_oh_slicer is None:
            truncation_items = items
        else:
            truncation_items = [self._tbptt_oh_slicer[ai] for ai in items]

        other_items = []
        for oi in range(len(self.list_of_other_seqs)):
            items = [self.list_of_other_seqs[oi][ii] for ii in self.indices_]
            if self._other_oh_slicers[oi] is None:
                other_items.append(items)
            else:
                other_items.append([self._other_oh_slicers[oi][ai] for ai in items])

        # make storage
        tbptt_arr = np.zeros((self.truncation_length + 1, self.batch_size, truncation_items[0].shape[-1]), dtype=np.float32)
        other_arrs = [np.zeros((self.other_seqs_max_lengths_[ni], self.batch_size, other_arr[0].shape[-1]), dtype=np.float32)
                      for ni, other_arr in enumerate(other_items)]
        for i in range(self.batch_size):
            ns = truncation_items[i][self.batches_[i]:self.batches_[i] + self.truncation_length + 1]
            # dropped sequences shorter than truncation_len already
            tbptt_arr[:, i, :] = ns
            for na, oa in enumerate(other_arrs):
                oa[:len(other_items[na][i]), i, :] = other_items[na][i]
            self.batches_[i] += self.truncation_length
        return [tbptt_arr,] + other_arrs + [reset_states,]

    def next_masked_batch(self):
        r = self.next_batch()
        # reset is the last element
        end_result = []
        for ri in r[:-1]:
            ri_mask = make_mask(ri)
            end_result.append(ri)
            end_result.append(ri_mask)
        end_result.append(r[-1])
        return end_result


class tbptt_file_list_iterator(object):
    def __init__(self, list_of_files,
                 file_seqs_access_fn,
                 batch_size,
                 truncation_length,
                 tbptt_one_hot_size=None,
                 other_one_hot_size=None,
                 random_state=None):
        """
        skips sequences shorter than truncation_len
        also cuts the tail off

        tbptt_one_hot_size
        should be either None, or the one hot size desired

        other_one_hot_size
        should either be None (if not doing one-hot) or a list the same length
        as the other_seqs returned from file_seqs_access_fn with integer one hot size, or None
        for no one_hot transformation, example:

        list_of_other_seqs = [my_char_data, my_vector_data]
        other_one_hot_size = [127, None]
        """
        self.list_of_files = list_of_files
        # gets a file path, returns (tbptt_seq, other_seqs)
        # if one_hot, the respective elements need to be *indices*
        self.file_seqs_access_fn = file_seqs_access_fn
        self.batch_size = batch_size
        self.truncation_length = truncation_length

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.tbptt_one_hot_size = tbptt_one_hot_size
        if self.tbptt_one_hot_size is None:
            self._tbptt_oh_slicer = None
        else:
            self._tbptt_oh_slicer = np.eye(tbptt_one_hot_size)

        self.other_one_hot_size = other_one_hot_size
        if other_one_hot_size is None:
            self._other_oh_slicers = [None] * 20 # if there's more than 20 of these we have a problem
        else:
            self._other_oh_slicers = []
            for ooh in other_one_hot_size:
                if ooh is None:
                    self._other_oh_slicers.append(None)
                else:
                    self._other_oh_slicers.append(np.eye(ooh, dtype=np.float32))

        self.indices_ = self.random_state.choice(len(self.list_of_files), size=(batch_size,), replace=False)
        self.batches_ = np.zeros((batch_size,), dtype=np.float32)
        self.current_fnames_ = None

        fnames = [self.list_of_files[i] for i in self.indices_]
        self.current_fnames_ = fnames
        datas = [self.file_seqs_access_fn(f) for f in fnames]
        tbptt_seqs = [d[0] for d in datas]
        other_seqs = [d[1:] for d in datas]

        self.current_tbptt_seqs_ = []
        self.current_other_seqs_ = []

        for idx in range(len(tbptt_seqs)):
            if not (len(tbptt_seqs[idx]) >= self.truncation_length + 1):
                new_tbptt = tbptt_seqs[idx]
                new_others = other_seqs[idx]
                num_tries = 0
                while not (len(new_tbptt) >= self.truncation_length + 1):
                    #print("idx {}:file {} too short, resample".format(idx, self.indices_[idx]))
                    new_file_idx = self.random_state.randint(0, len(self.list_of_files) - 1)
                    fname = self.list_of_files[new_file_idx]
                    new_data = self.file_seqs_access_fn(fname)
                    new_tbptt = new_data[0]
                    new_others = new_data[1:]
                    num_tries += 1
                    if num_tries >= 20:
                        raise ValueError("Issue in file iterator next_batch, can't get a large enough file after 20 tries!")
                self.indices_[idx] = new_file_idx
                tbptt_seqs[idx] = new_tbptt
                other_seqs[idx] = new_others
            self.current_tbptt_seqs_.append(tbptt_seqs[idx])
            self.current_other_seqs_.append(other_seqs[idx])

    def next_batch(self):
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        # check lengths and if it's too short, resample...
        for i in range(self.batch_size):
            if self.batches_[i] + self.truncation_length + 1 > len(self.current_tbptt_seqs_[i]):
                ni = self.random_state.randint(0, len(self.list_of_files) - 1)
                fname = self.list_of_files[ni]
                new_data = self.file_seqs_access_fn(fname)
                new_tbptt = new_data[0]
                new_others = new_data[1:]
                num_tries = 0
                while not (len(new_tbptt) >= self.truncation_length + 1):
                    ni = self.random_state.randint(0, len(self.list_of_files) - 1)
                    fname = self.list_of_files[ni]
                    new_data = self.file_seqs_access_fn(fname)
                    new_tbptt = new_data[0]
                    new_others = new_data[1:]
                    num_tries += 1
                    if num_tries >= 20:
                        print("Issue in file iterator next_batch, can't get a large enough file after {} tries! Tried {}, name {}".format(num_tries), ni, self.list_of_files[ni])
                self.batches_[i] = 0.
                reset_states[i] = 0.
                self.current_tbptt_seqs_[i] = new_tbptt
                self.current_other_seqs_[i] = new_others

        items = [self.current_tbptt_seqs_[ii] for ii in range(len(self.current_tbptt_seqs_))]
        if self._tbptt_oh_slicer is None:
            truncation_items = items
        else:
            truncation_items = [self._tbptt_oh_slicer[ai] for ai in items]

        other_items = []
        # batch index
        for oi in range(len(self.current_other_seqs_)):
            items = self.current_other_seqs_[oi]
            subitems = []
            for j in range(len(items)):
                if self._other_oh_slicers[j] is None:
                    subitems.append(np.array(items))
                else:
                    subitems.append(np.array([self._other_oh_slicers[j][ai] for ai in items[j]]))
            other_items.append(subitems)

        tbptt_arr = np.zeros((self.truncation_length + 1, self.batch_size, truncation_items[0].shape[-1]))
        other_seqs_max_lengths = [max([len(other_items[i][j]) for i in range(len(other_items))])
                                       for j in range(len(other_items[i]))]
        other_arrs = [np.zeros((other_seqs_max_lengths[ni], self.batch_size, np.array(other_items[0][ni]).shape[-1]), dtype=np.float32)
                      for ni in range(len(other_items[0]))]

        for i in range(self.batch_size):
            ns = truncation_items[i][int(self.batches_[i]):int(self.batches_[i] + self.truncation_length + 1)]
            tbptt_arr[:, i, :] = ns
            for na in range(len(other_arrs)):
                other_arrs[na][:len(other_items[i][na]), i, :] = other_items[i][na]
            self.batches_[i] += self.truncation_length
        return [tbptt_arr,] + other_arrs + [reset_states,]


    def next_masked_batch(self):
        r = self.next_batch()
        # reset is the last element
        end_result = []
        for ri in r[:-1]:
            ri_mask = make_mask(ri)
            end_result.append(ri)
            end_result.append(ri_mask)
        end_result.append(r[-1])
        return end_result
