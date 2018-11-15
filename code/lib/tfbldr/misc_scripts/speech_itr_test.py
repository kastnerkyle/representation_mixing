from tfbldr.datasets import tbptt_file_list_iterator
import os
import numpy as np

files = os.listdir("/Tmp/kastner/lj_speech_hybrid_speakers/numpy_features/")
files = ["/Tmp/kastner/lj_speech_hybrid_speakers/numpy_features/" + f for f in files]
ljspeech_hybridset = [' ', '!', ',', '-', '.', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
hybrid_lookup = {v: k for k, v in enumerate(sorted(ljspeech_hybridset))}
hybrid_inverse_lookup = {v: k for k, v in hybrid_lookup.items()}

def file_access(f):
    d = np.load(f)
    text = d["text"]
    inds = [hybrid_lookup[t] for t in text.ravel()[0]]
    audio = d["audio_features"]
    return (audio, inds)

random_state = np.random.RandomState(1442)
batch_size = 8
truncation_length = 256
itr = tbptt_file_list_iterator(files, file_access,
                               batch_size,
                               truncation_length,
                               other_one_hot_size=[len(ljspeech_hybridset)],
                               random_state=random_state)
for i in range(100000):
    print(i)
    r = itr.next_masked_batch()
