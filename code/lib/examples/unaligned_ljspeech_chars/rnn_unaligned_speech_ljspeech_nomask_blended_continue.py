from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple

import logging
import shutil
from tfbldr.datasets import rsync_fetch, fetch_ljspeech
from tfbldr.datasets import wavfile_caching_mel_tbptt_iterator
from tfbldr.utils import next_experiment_path
from tfbldr import get_logger
from tfbldr import run_loop
from tfbldr.nodes import Linear
from tfbldr.nodes import Linear
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import BiLSTMLayer
from tfbldr.nodes import SequenceConv1dStack
from tfbldr.nodes import Embedding
from tfbldr.nodes import GaussianAttentionCell
from tfbldr.nodes import DiscreteMixtureOfLogistics
from tfbldr.nodes import DiscreteMixtureOfLogisticsCost
from tfbldr.nodes import AdditiveGaussianNoise
from tfbldr import scan

if len(sys.argv) < 1:
   raise ValueError("Continue script only for continuing training of a previous model")

seq_len = 256
batch_size = 64
window_mixtures = 10
cell_dropout = .925
#noise_scale = 8.
prenet_units = 128
n_filts = 128
n_stacks = 3
enc_units = 128
dec_units = 512
emb_dim = 15
truncation_len = seq_len
cell_dropout_scale = cell_dropout
epsilon = 1E-8
forward_init = "truncated_normal"
rnn_init = "truncated_normal"

basedir = "/Tmp/kastner/lj_speech/LJSpeech-1.0/"
ljspeech = rsync_fetch(fetch_ljspeech, "leto01")

# THESE ARE CANNOT BE PAIRED (SOME MISSING), ITERATOR PAIRS THEM UP BY NAME
wavfiles = ljspeech["wavfiles"]
jsonfiles = ljspeech["jsonfiles"]

model_path = sys.argv[1]
seed = int(abs(hash(model_path))) % (2 ** 32 - 1)

# THESE HAVE TO BE THE SAME TO ENSURE SPLIT IS CORRECT
train_random_state = np.random.RandomState(seed)
valid_random_state = np.random.RandomState(seed)

train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, stop_index=.95, shuffle=True, random_state=train_random_state)
valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, start_index=.95, shuffle=True, random_state=valid_random_state)

"""
for i in range(10000):
    print(i)
    mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
"""

# STRONG CHECK TO ENSURE NO OVERLAP IN TRAIN/VALID
for tai in train_itr.all_indices_:
    assert tai not in valid_itr.all_indices_
for vai in valid_itr.all_indices_:
    assert vai not in train_itr.all_indices_

random_state = np.random.RandomState(1442)
# use the max of the two blended types...
vocabulary_size = max(train_itr.vocabulary_sizes)
output_size = train_itr.n_mel_filters

att_w_init = np.zeros((batch_size, 2 * enc_units))
att_k_init = np.zeros((batch_size, window_mixtures))
att_h_init = np.zeros((batch_size, dec_units))
att_c_init = np.zeros((batch_size, dec_units))
h1_init = np.zeros((batch_size, dec_units))
c1_init = np.zeros((batch_size, dec_units))
h2_init = np.zeros((batch_size, dec_units))
c2_init = np.zeros((batch_size, dec_units))

stateful_args = [att_w_init,
                 att_k_init,
                 att_h_init,
                 att_c_init,
                 h1_init,
                 c1_init,
                 h2_init,
                 c2_init]

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    logger = get_logger()
    logger.info("CONTINUING TRAINING FROM MODEL PATH {}".format(model_path))
    saver.restore(sess, model_path)
    graph = tf.get_default_graph()

    fields = ["mels",
              "mel_mask",
              "in_mels",
              "in_mel_mask",
              "out_mels",
              "out_mel_mask",
              "text",
              "text_mask",
              "mask",
              "mask_mask",
              "bias",
              "cell_dropout",
              "prenet_dropout",
              "bn_flag",
              "pred",
              #"mix", "means", "lins",
              "att_w_init",
              "att_k_init",
              "att_h_init",
              "att_c_init",
              "h1_init",
              "c1_init",
              "h2_init",
              "c2_init",
              "att_w",
              "att_k",
              "att_phi",
              "att_h",
              "att_c",
              "h1",
              "c1",
              "h2",
              "c2",
              "loss",
              "train_step",
              "learning_rate"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    step_count = 0
    def loop(sess, itr, extras, stateful_args):
        """
        global step_count
        global noise_scale
        step_count += 1
        if step_count > 10000:
            step_count = 0
            if noise_scale == 2:
               noise_scale = 1.
            else:
                noise_scale = noise_scale - 2.
            if noise_scale < .5:
                noise_scale = .5
        """
        mels, mel_mask, text, text_mask, mask, mask_mask, reset = itr.next_masked_batch()
        in_m = mels[:-1]
        in_mel_mask = mel_mask[:-1]

        #noise_block = np.clip(random_state.randn(*in_m.shape), -6, 6)
        #in_m = in_m + noise_scale * noise_block

        out_m = mels[1:]
        out_mel_mask = mel_mask[1:]

        att_w_init = stateful_args[0]
        att_k_init = stateful_args[1]
        att_h_init = stateful_args[2]
        att_c_init = stateful_args[3]
        h1_init = stateful_args[4]
        c1_init = stateful_args[5]
        h2_init = stateful_args[6]
        c2_init = stateful_args[7]

        att_w_init *= reset
        att_k_init *= reset
        att_h_init *= reset
        att_c_init *= reset
        h1_init *= reset
        c1_init *= reset
        h2_init *= reset
        c2_init *= reset

        feed = {
                vs.in_mels: in_m,
                vs.in_mel_mask: in_mel_mask,
                vs.out_mels: out_m,
                vs.out_mel_mask: out_mel_mask,
                vs.bn_flag: 0.,
                vs.text: text,
                vs.text_mask: text_mask,
                vs.mask: mask,
                vs.mask_mask: mask_mask,
                vs.att_w_init: att_w_init,
                vs.att_k_init: att_k_init,
                vs.att_h_init: att_h_init,
                vs.att_c_init: att_c_init,
                vs.h1_init: h1_init,
                vs.c1_init: c1_init,
                vs.h2_init: h2_init,
                vs.c2_init: c2_init}
        outs = [vs.att_w, vs.att_k,
                vs.att_h, vs.att_c,
                vs.h1, vs.c1, vs.h2, vs.c2,
                vs.att_phi,
                vs.loss, vs.train_step]

        r = sess.run(outs, feed_dict=feed)

        att_w_np = r[0]
        att_k_np = r[1]
        att_h_np = r[2]
        att_c_np = r[3]
        h1_np = r[4]
        c1_np = r[5]
        h2_np = r[6]
        c2_np = r[7]
        att_phi_np = r[8]
        l = r[-2]
        _ = r[-1]

        # set next inits
        att_w_init = att_w_np[-1]
        att_k_init = att_k_np[-1]
        att_h_init = att_h_np[-1]
        att_c_init = att_c_np[-1]
        h1_init = h1_np[-1]
        c1_init = c1_np[-1]
        h2_init = h2_np[-1]
        c2_init = c2_np[-1]

        stateful_args = [att_w_init,
                         att_k_init,
                         att_h_init,
                         att_c_init,
                         h1_init,
                         c1_init,
                         h2_init,
                         c2_init]
        return l, None, stateful_args

    run_loop(sess,
             loop, train_itr,
             loop, train_itr,
             continue_training=True,
             n_steps=1000000,
             n_train_steps_per=1000,
             train_stateful_args=stateful_args,
             n_valid_steps_per=0,
             valid_stateful_args=stateful_args)
