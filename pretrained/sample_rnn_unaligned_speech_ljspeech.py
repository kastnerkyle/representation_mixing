import argparse
import os
import tensorflow as tf
import numpy as np
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
import time
#from datasets import wavfile_caching_mel_tbptt_iterator
from audio import soundsc
from audio import stft
from audio import iterate_invert_spectrogram
from transform_text import transform_text
from transform_text import inverse_transform_text
from scipy.io import wavfile


parser = argparse.ArgumentParser()
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--sonify', dest='sonify', type=int, default=100)
parser.add_argument('--gl', dest='gl', type=int, default=100)
args = parser.parse_args()

dl_link = "https://www.dropbox.com/s/domlnxdk7wetqr3/representation_mixing_model.tar.gz?raw=1"

model_path = "model_final_blended_maskfix_dirtyattnandhidconnect_softplusstep_1scale_pt925drop/models/model-512000"

if not os.path.exists(model_path.split(os.sep)[0]):
    print("Downloading model from {}".format(dl_link))
    import urllib
    urllib.urlretrieve (dl_link, "representation_mixing_model.tar.gz")
    import tarfile
    tar = tarfile.open("representation_mixing_model.tar.gz", "r:gz")
    tar.extractall()
    tar.close()

random_state = np.random.RandomState(args.seed)
"""
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
"""
#ljspeech = rsync_fetch(fetch_ljspeech, "leto01")
#ljspeech = fetch_ljspeech()
#wavfiles = ljspeech["wavfiles"]
#jsonfiles = ljspeech["jsonfiles"]
batch_size = 64
seq_len = 256
window_mixtures = 10
enc_units = 128
dec_units = 512
emb_dim = 15
sonify_steps = args.sonify
gl_steps = args.gl

sample_rate = 22050
window_size = 512
step = 128
n_mel = 80

itr_random_state = np.random.RandomState(3122)

has_mask = True

# batch sizes from standard validation iterator
# mels = (256, 64, 80)
# mel_mask = (256, 64)
# text = (145, 64, 1)
# text_mask = (145, 64)
# mask = (145, 64)
# mask_mask = (145, 64)
# reset = (64, 1)
mels = np.zeros((256, 64, 80))
mel_mask = np.ones_like(mels[..., 0])
text = np.zeros((145, 64, 1))
text_mask = np.ones_like(text[..., 0])
mask = np.ones((145, 64))
mask_mask = np.ones_like(mask)
reset = np.zeros((64, 1))


# hardcoded per-dimension mean and std for mel data from the training iterator, read from a file
d = np.load("norm-mean-std-txt-cleanenglish_cleanersenglish_phone_cleaners-logmel-wsz512-wst128-leh125-ueh7800-nmel80.npz")
saved_std = d["std"]
saved_mean = d["mean"]


#mels, mel_mask, text, text_mask, mask, mask_mask, reset = itr.next_masked_batch()

n_to_sample = 0
pre_lines = []
post_lines = []
int_lines = []
masks = []

with open("sample_lines.txt", "r") as f:
    lines = f.readlines()
lines = [l.strip() for l in lines]
print("lines to sample:")
for l in lines:
    if "." in l:
        lm1 = l[:-1].replace(".", ",")
        l = lm1 + l[-1]
    pre_lines.append(l)
    tt, mm = transform_text(l)
    ot = inverse_transform_text(tt, mm)
    # cut off eos
    int_lines.append(tt[:-1])
    masks.append(mm[:-1])
    post_lines.append(ot[:-1])
    print(ot[:-1])

if len(int_lines) > batch_size:
    raise ValueError("More lines to test ({}) than batch_size ({}), needs multi-batch support!".format(len(int_lines), len(batch_size)))
longest = max([len(il) for il in int_lines])
text = np.zeros((longest, batch_size, 1))
text_mask = np.zeros((longest, batch_size))
mask = np.zeros((longest, batch_size, 1))
mask_mask = np.zeros((longest, batch_size))
for n, il in enumerate(int_lines):
    text[:len(il), n, 0] = il
    text_mask[:len(il), n] = 1.
    mask[:len(il), n, 0] = masks[n]
    mask_mask[:len(il), n] = 1.
n_to_sample = len(int_lines)

with open("sampled_text_summary.txt", "w") as f:
    assert len(pre_lines) == len(post_lines)
    for jj, (pre, post) in enumerate(zip(pre_lines, post_lines)):
        f.writelines(["{} : {} : {}\n".format(jj, pre, post)])

#with tf.Session(config=config) as sess:
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ["mels",
              "mel_mask",
              "in_mels",
              "in_mel_mask",
              "out_mels",
              "out_mel_mask",
              "text",
              "text_mask",
              "bias",
              "cell_dropout",
              #"prenet_dropout",
              #"attskip_dropout",
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
              "c2"]
    if has_mask:
        fields += ["mask", "mask_mask"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    att_w_init = np.zeros((batch_size, 2 * enc_units))
    att_k_init = np.zeros((batch_size, window_mixtures))
    att_h_init = np.zeros((batch_size, dec_units))
    att_c_init = np.zeros((batch_size, dec_units))
    h1_init = np.zeros((batch_size, dec_units))
    c1_init = np.zeros((batch_size, dec_units))
    h2_init = np.zeros((batch_size, dec_units))
    c2_init = np.zeros((batch_size, dec_units))

    """
    # only predict 1 thing
    endpoint = np.where(text_mask[:, 0] == 1)[0][-1]
    text = text[:endpoint]
    for jj in range(text.shape[1]):
        text[:, jj] = text[:, 0]
    text_mask = 0. * text[:, :, 0] + 1.
    """

    in_mels = 0. * mels[:1]
    in_mel_mask = 0. * mel_mask[:1] + 1.

    preds = []
    att_ws = []
    att_phis = []
    random_state = np.random.RandomState(11)
    #noise_scale = .5
    is_finished_sampling = [False] * n_to_sample
    finished_at = 100000000
    finished_step = [-1] * n_to_sample

    ii = 0

    # add ~.2 sec to the end
    # sample_rate * .2 / fft_step
    # min_part to account for the last window
    min_part = window_size / float(sample_rate)
    extra_steps = max(0, int((sample_rate * (.2 - min_part)) / float(step)))
    while True:
        print("pred step {}".format(ii))
        #noise_block = np.clip(random_state.randn(*in_mels.shape), -6, 6)
        #in_mels = in_mels + noise_scale * noise_block
        #if ii > 0:
        #    in_mels[0] = mels[ii - 1]
        feed = {
                vs.in_mels: in_mels,
                vs.in_mel_mask: in_mel_mask,
                vs.bn_flag: 1.,
                vs.text: text,
                vs.text_mask: text_mask,
                vs.cell_dropout: 1.,
                vs.att_w_init: att_w_init,
                vs.att_k_init: att_k_init,
                vs.att_h_init: att_h_init,
                vs.att_c_init: att_c_init,
                vs.h1_init: h1_init,
                vs.c1_init: c1_init,
                vs.h2_init: h2_init,
                vs.c2_init: c2_init}
        if has_mask:
            feed[vs.mask] = mask
            feed[vs.mask_mask] = mask_mask
        outs = [vs.att_w, vs.att_k,
                vs.att_h, vs.att_c,
                vs.h1, vs.c1, vs.h2, vs.c2,
                vs.att_phi, vs.pred]
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
        pred_np = r[9]

        ii += 1
        max_text = max([text_mask[:, mbi].sum() for mbi in range(n_to_sample)])
        if ii > 30 * max_text:
            # it's gone too far, kill
            finished_step = [int(30 * max_text)] * n_to_sample
            print("Exceeded 30 * max text length of {},  terminating...".format(max_text))
            break

        att_ws.append(att_w_np[0])
        att_phis.append(att_phi_np[0])
        preds.append(pred_np[0])

        # set next inits and input values
        in_mels[0] = pred_np
        att_w_init = att_w_np[-1]
        att_k_init = att_k_np[-1]
        att_h_init = att_h_np[-1]
        att_c_init = att_c_np[-1]
        h1_init = h1_np[-1]
        c1_init = c1_np[-1]
        h2_init = h2_np[-1]
        c2_init = c2_np[-1]

        for mbi in range(n_to_sample):
            last_sym = int(text_mask[:, mbi].sum()) - 1
            if np.argmax(att_phi_np[0, mbi]) >= last_sym or np.argmax(att_phi_np[0, mbi]) == text_mask.shape[0]:
                if is_finished_sampling[mbi] == False:
                    is_finished_sampling[mbi] = True
                    finished_step[mbi] = ii

        if all(is_finished_sampling):
            print("All samples finished at step {}".format(finished_at))
        else:
            # should assign until all are finished
            finished_at = ii

        if ii > (finished_at + extra_steps):
            print("Extra padding {} finished at step {}".format(extra_steps, ii))
            break


def implot(arr, axarr, scale=None, title="", cmap=None):
    # plotting part
    #mag = 20. * np.log10(np.abs(arr))
    mag = arr
    # Transpose so time is X axis, and invert y axis so
    # frequency is low at bottom
    mag = mag.T#[::-1, :]

    if cmap != None:
        axarr.imshow(mag, cmap=cmap, origin="lower")
    else:
        axarr.imshow(mag, origin="lower")

    plt.axis("off")
    x1 = mag.shape[0]
    y1 = mag.shape[1]
    if scale == "specgram":
        y1 = int(y1 * .20)

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(x1, y1)
    axarr.set_aspect(asp)
    plt.title(title)

n_plot = n_to_sample
preds = np.array(preds)
for jj in range(n_plot):
    spectrogram = preds[:, jj] * saved_std + saved_mean
    specname = "sample_{}_spec.png".format(jj)
    f, axarr = plt.subplots(1, 1)
    implot(spectrogram[:finished_step[jj]], axarr, scale="specgram")
    plt.savefig(specname)
    plt.close()

att_phis = np.array(att_phis)
for jj in range(n_plot):
    phi_i = att_phis[:, jj]
    attname = "sample_{}_att.png".format(jj)
    f, axarr = plt.subplots(1, 1)
    implot(phi_i[:finished_step[jj], :len(masks[jj])], axarr, cmap="gray")
    plt.savefig(attname)
    plt.close()

def logmel(waveform):
    z = tf.contrib.signal.stft(waveform, window_size, step)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel,
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=125.,
        upper_edge_hertz=7800.)
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)

def sonify(spectrogram, samples, transform_op_fn, logscaled=True):
    graph = tf.Graph()
    with graph.as_default():

        noise = tf.Variable(tf.random_normal([samples], stddev=1e-6))

        x = transform_op_fn(noise)
        y = spectrogram

        if logscaled:
            x = tf.expm1(x)
            y = tf.expm1(y)

        # tf.nn.normalize arguments changed between versions...
        def normalize(a):
            return a / tf.sqrt(tf.maximum(tf.reduce_sum(a ** 2, axis=0), 1E-12))

        x = normalize(x)
        y = normalize(y)
        tf.losses.mean_squared_error(x, y[-tf.shape(x)[0]:])

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss=tf.losses.get_total_loss(),
            var_list=[noise],
            tol=1e-16,
            method='L-BFGS-B',
            options={
                'maxiter': sonify_steps,
                'disp': True
            })

    # THIS REALLY SHOULDN'T RUN ON GPU BUT SEEMS TO?
    config = tf.ConfigProto(
        device_count={'CPU' : 1, 'GPU' : 0},
        allow_soft_placement=True,
        log_device_placement=False
        )
    with tf.Session(config=config, graph=graph) as session:
        session.run(tf.global_variables_initializer())
        optimizer.minimize(session)
        waveform = session.run(noise)
    return waveform

pre_time = 0
post_time = 0
joint_time = 0
for jj in range(n_plot):
    # use extra steps from earlier
    pjj = preds[:(finished_step[jj] + extra_steps), jj]
    spectrogram = pjj * saved_std + saved_mean
    mel_dump = "sample_{}_mels.npz".format(jj)
    np.savez(mel_dump, mels=spectrogram)
    prename = "sample_{}_pre.wav".format(jj)

    this_time = time.time()

    reconstructed_waveform = sonify(spectrogram, len(spectrogram) * step, logmel)
    end_this_time = time.time()
    wavfile.write(prename, sample_rate, soundsc(reconstructed_waveform))
    elapsed = end_this_time - this_time

    print("Elapsed pre sampling time {} s".format(elapsed))
    pre_time += elapsed
    joint_time += elapsed

    fftsize = 512
    substep = 32
    postname = "sample_{}_post.wav".format(jj)
    this_time = time.time()

    rw_s = np.abs(stft(reconstructed_waveform, fftsize=fftsize, step=substep, real=False,
                       compute_onesided=False))
    rw = iterate_invert_spectrogram(rw_s, fftsize, substep, n_iter=gl_steps, verbose=True)
    end_this_time = time.time()
    wavfile.write(postname, sample_rate, soundsc(rw))
    elapsed = end_this_time - this_time

    print("Elapsed post sampling time {} s".format(elapsed))
    post_time += elapsed
    joint_time += elapsed
print("Combined pre time {} s".format(pre_time))
print("Combined post time {} s".format(post_time))
print("Combined joint time {} s".format(joint_time))
