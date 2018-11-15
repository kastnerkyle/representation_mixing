from ..loaders import get_tfbldr_dataset_dir
from ..loaders import pe
from ..loaders import copytree
from ...core import get_logger

logger = get_logger()

import os
import shutil
import time
import multiprocessing
import cPickle as pickle
import functools

try:
    from music21 import converter, interval, pitch, harmony, analysis, spanner, midi, meter
    from music21 import corpus
except ImportError:
    logger.info("Unable to retrieve music21 related utilities")

from .music import music21_to_pitch_duration
from .music import music21_to_chord_duration
from .music import pitch_and_duration_to_quantized
from .music import chord_and_chord_duration_to_quantized
from .analysis import midi_to_notes
from .analysis import notes_to_midi

TIMEOUT_ID = "MULTIPROCESSING_TIMEOUT"

# http://stackoverflow.com/questions/29494001/how-can-i-abort-a-task-in-a-multiprocessing-pool-after-a-timeout
def abortable_worker(func, *args, **kwargs):
    # returns ("MULTIPROCESSING_TIMEOUT",) if timeout
    timeout = kwargs['timeout']
    p = multiprocessing.dummy.Pool(1)
    res = p.apply_async(func, args=args)
    # assumes timeout is an integer >= 1
    for i in range(timeout + 1):
        if i > 0:
            time.sleep(1)
        if res.ready():
            if res.successful():
                try:
                    return res.get(timeout=1)
                except multiprocessing.TimeoutError:
                    logger.info("Aborting due to timeout in get")
                    p.terminate()
                    return (TIMEOUT_ID,)
    logger.info("Aborting due to timeout")
    p.terminate()
    return (TIMEOUT_ID,)


def _music_single_extract(files, data_path, verbose, n):
    if verbose:
        logger.info("Starting file {} of {}".format(n, len(files)))
    f = files[n]
    file_path = os.path.join(data_path, f)

    start_time = time.time()
    p = converter.parse(file_path)
    k = p.analyze("key")
    orig_key = k.name
    orig_mode = k.mode
    if k.mode not in ["minor", "major"]:
        logger.info("Mode neither minor not major in {}, aborting".format(f))
        return (TIMEOUT_ID,)

    if verbose:
        parse_time = time.time()
        r = parse_time - start_time
        logger.info("Parse time {}:{}".format(f, r))

    time_sigs = [str(ts).split(" ")[-1].split(">")[0] for ts in p.recurse().getElementsByClass(meter.TimeSignature)]
    nums = [int(ts.split("/")[0]) for ts in time_sigs]
    num_check = all([n == nums[0] for n in nums])
    denoms = [int(ts.split("/")[1]) for ts in time_sigs]
    denom_check = all([d == denoms[0] for d in denoms])

    if not denom_check:
        # don't know how to handle time-signature base changes right now
        logger.info("Time signature denominator changed in {}, aborting".format(f))
        return (TIMEOUT_ID,)

    if len(time_sigs) < 1:
        time_sigs = ["4/4"]

    """
    https://gist.github.com/aldous-rey/68c6c43450517aa47474
    https://github.com/feynmanliang/bachbot/blob/557abb971b6886f831e0566956ec76ee17aa9649/scripts/datasets.py#L97
    """
    majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("C#",-1),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("F#",6),("G-", 6),("G", 5)])
    minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("C#",-4),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("F#",3),("G-", 3),("G", 2)])

    if k.mode == "major":
        half_steps = majors[k.tonic.name]
    elif k.mode == "minor":
        half_steps = minors[k.tonic.name]
    p = p.transpose(half_steps)

    for ks in p.flat.getKeySignatures():
        ks.sharps = 0

    k = p.analyze("key")
    if verbose:
        transpose_time = time.time()
        r = transpose_time - start_time
        logger.info("Transpose time {}:{}".format(f, r))

    chords, chord_functions, chord_durations = music21_to_chord_duration(p, k)
    pitches, parts_times, parts_delta_times, parts_fermatas = music21_to_pitch_duration(p)
    if verbose:
        pitch_duration_time = time.time()
        r = pitch_duration_time - start_time
        logger.info("music21 to pitch_duration time {}:{}".format(f, r))

    str_key = k.name

    if verbose:
        ttime = time.time()
        r = ttime - start_time
        logger.info("Overall file time {}:{}".format(f, r))
    str_time_sig = time_sigs[0]
    return (pitches, parts_times, parts_delta_times, str_key, orig_key, orig_mode, str_time_sig, f, p.quarterLength, chords, chord_functions, chord_durations, parts_fermatas)


def _music_extract(data_path, pickle_path, ext=".xml",
                   parse_timeout=100,
                   multiprocess_count=4,
                   verbose=False):

    if not os.path.exists(pickle_path):
        logger.info("Pickled file {} not found, creating. This may take a few minutes...".format(pickle_path))
        itime = time.time()

        all_chords = []
        all_chord_functions = []
        all_chord_durations = []
        all_parts_fermatas = []
        all_pitches = []
        all_parts_times = []
        all_parts_delta_times = []
        all_orig_keys = []
        all_orig_modes = []
        all_keys = []
        all_time_sigs = []
        all_filenames = []
        all_quarter_lengths = []

        if "basestring" not in globals():
            basestring = str

        if isinstance(data_path, basestring):
            files = sorted([fi for fi in os.listdir(data_path) if fi.endswith(ext)])
        else:
            files = sorted([ap for ap in data_path if ap.endswith(ext)])

        logger.info("Processing {} files".format(len(files)))
        if multiprocess_count is not None:
            pool = multiprocessing.Pool(multiprocess_count)

            ex = functools.partial(_music_single_extract,
                                   files, data_path,
                                   verbose)
            abortable_ex = functools.partial(abortable_worker, ex, timeout=parse_timeout)
            result = pool.map(abortable_ex, range(len(files)))
            pool.close()
            pool.join()
        else:
            result = []
            for n in range(len(files)):
                r = _music_single_extract(files, data_path,
                                          verbose, n)
                result.append(r)

        for n, r in enumerate(result):
            if r[0] != TIMEOUT_ID:
                (pitches, parts_times, parts_delta_times,
                 key, orig_key, orig_mode,
                 time_signature, fname, quarter_length,
                 chords, chord_functions, chord_durations,
                 fermatas) = r

                all_chords.append(chords)
                all_chord_functions.append(chord_functions)
                all_chord_durations.append(chord_durations)
                all_pitches.append(pitches)
                all_parts_times.append(parts_times)
                all_parts_delta_times.append(parts_delta_times)
                all_parts_fermatas.append(fermatas)
                all_keys.append(key)
                all_orig_keys.append(orig_key)
                all_orig_modes.append(orig_mode)
                all_time_sigs.append(time_signature)
                all_filenames.append(fname)
                all_quarter_lengths.append(quarter_length)
            else:
                logger.info("Result {} timed out".format(n))

        gtime = time.time()
        if verbose:
            r = gtime - itime
            logger.info("Overall time {}".format(r))

        d = {"data_pitches": all_pitches,
             "data_parts_times": all_parts_times,
             "data_parts_delta_times": all_parts_delta_times,
             "data_parts_fermatas": all_parts_fermatas,
             "data_keys": all_keys,
             "data_orig_keys": all_orig_keys,
             "data_orig_modes": all_orig_modes,
             "data_time_sigs": all_time_sigs,
             "data_chords": all_chords,
             "data_chord_functions": all_chord_functions,
             "data_chord_durations": all_chord_durations,
             "data_quarter_lengths": all_quarter_lengths,
             "filenames": all_filenames}

        with open(pickle_path, "wb") as f:
            logger.info("Saving pickle file {}".format(pickle_path))
            pickle.dump(d, f)
        logger.info("Pickle file {} saved".format(pickle_path))
    else:
        logger.info("Loading cached data from {}".format(pickle_path))
        with open(pickle_path, "rb") as f:
            d = pickle.load(f)

    all_pitches = d["data_pitches"]
    all_parts_times = d["data_parts_times"]
    all_parts_delta_times = d["data_parts_delta_times"]
    all_parts_fermatas = d["data_parts_fermatas"]
    all_keys = d["data_keys"]
    all_orig_keys = d["data_orig_keys"]
    all_orig_modes = d["data_orig_modes"]
    all_time_sigs = d["data_time_sigs"]
    all_chords = d["data_chords"]
    all_chord_functions = d["data_chord_functions"]
    all_chord_durations = d["data_chord_durations"]
    all_chord_quarter_lengths = d["data_quarter_lengths"]
    all_filenames = d["filenames"]

    r = {"list_of_data_pitches": all_pitches,
         "list_of_data_times": all_parts_times,
         "list_of_data_time_deltas": all_parts_delta_times,
         "list_of_data_parts_fermatas": all_parts_fermatas,
         "list_of_data_keys": all_keys,
         "list_of_data_orig_keys": all_orig_keys,
         "list_of_data_orig_modes": all_orig_modes,
         "list_of_data_time_sigs": all_time_sigs,
         "list_of_data_chords": all_chords,
         "list_of_data_chord_functions": all_chord_functions,
         "list_of_data_chord_durations": all_chord_durations,
         "list_of_data_chord_quarter_lengths": all_chord_quarter_lengths,
         "list_of_filenames": all_filenames}
    return r


def _common_features_from_music_extract(mu, equal_voice_count=4, verbose=False):
    all_quantized_16th_pitches = []
    all_quantized_16th_pitches_no_hold = []
    all_quantized_16th_fermatas = []
    all_quantized_16th_subbeats = []
    all_quantized_16th_chords = []
    all_quantized_16th_chord_functions = []
    all_pitches = mu["list_of_data_pitches"]
    all_parts_delta_times = mu["list_of_data_time_deltas"]
    all_parts_fermatas = mu["list_of_data_parts_fermatas"]
    all_chords = mu["list_of_data_chords"]
    all_chord_functions = mu["list_of_data_chord_functions"]
    all_chord_durations = mu["list_of_data_chord_durations"]


    invalids = []
    for i in range(len(all_pitches)):
        try:
            qq = pitch_and_duration_to_quantized(all_pitches[i], all_parts_delta_times[i], .25, list_of_metas_voices=[all_parts_fermatas[i]], verbose=verbose)
            qqnh = pitch_and_duration_to_quantized(all_pitches[i], all_parts_delta_times[i], .25, list_of_metas_voices=[all_parts_fermatas[i]], verbose=verbose, hold_symbol=False)
            cc = chord_and_chord_duration_to_quantized(all_chords[i], all_chord_durations[i], .25, list_of_chord_metas=[all_chord_functions[i]], verbose=verbose)

            if qq[0].shape[1] != equal_voice_count:
                #print("Invalid voices {}".format(i))
                invalids.append(i)
            else:
                subbeat_counter = [1, 2, 3, 4] * (len(qq[0]) // 4 + 1)
                subbeat_counter = subbeat_counter[:len(qq[0])]
                # collapse fermatas and make the subbeat counter
                collapsed_fermatas = [1 if sum([qq[1][vi][ti] for vi in range(len(qq[1]))]) > 0 else 0
                 for ti in range(len(qq[0]))]
                all_quantized_16th_pitches.append(qq[0])
                all_quantized_16th_pitches_no_hold.append(qqnh[0])
                all_quantized_16th_fermatas.append(collapsed_fermatas)
                all_quantized_16th_subbeats.append(subbeat_counter)
                all_quantized_16th_chords.append(cc[0])
                all_quantized_16th_chord_functions.append(cc[1])
        except:
            #print("Invalid err {}".format(i))
            invalids.append(i)
    assert len(all_quantized_16th_chords) == len(all_quantized_16th_pitches)
    mu_res = {}
    for k, v in mu.items():
        mu_res[k] = [vi for n, vi in enumerate(v) if n not in invalids]
        assert len(mu_res[k]) == len(all_quantized_16th_pitches)
    mu_res["list_of_data_quantized_16th_pitches"] = all_quantized_16th_pitches
    mu_res["list_of_data_quantized_16th_pitches_no_hold"] = all_quantized_16th_pitches_no_hold
    mu_res["list_of_data_quantized_16th_fermatas"] = all_quantized_16th_fermatas
    mu_res["list_of_data_quantized_16th_subbeats"] = all_quantized_16th_subbeats
    mu_res["list_of_data_quantized_16th_chords"] = all_quantized_16th_chords
    mu_res["list_of_data_quantized_16th_chord_functions"] = all_quantized_16th_chord_functions
    return mu_res


def check_fetch_jsb():
    """ Move files into tfbldr dir, in case python path is nfs """
    all_bach_paths = corpus.getComposer("bach")
    partial_path = get_tfbldr_dataset_dir("jsb")
    for path in all_bach_paths:
        if "riemenschneider" in path:
            continue
        filename = os.path.split(path)[-1]
        local_path = os.path.join(partial_path, filename)
        if not os.path.exists(local_path):
            shutil.copy2(path, local_path)
    return partial_path


def fetch_jsb(keys=["C major", "A minor"],
              equal_voice_count=4,
              verbose=True):
    """
    Bach chorales, transposed to C major or A minor (depending on original key).
    Requires music21.
    """
    data_path = check_fetch_jsb()
    pickle_path = os.path.join(data_path, "__processed_jsb.pkl")
    mu = _music_extract(data_path, pickle_path, ext=".mxl",
                        # debug... set to None
                        #multiprocess_count=None,
                        verbose=verbose)
    mu_res = _common_features_from_music_extract(mu, equal_voice_count=equal_voice_count,
                                                 verbose=verbose)
    return mu_res


def check_fetch_josquin():
    """ Move files into tfbldr dir, in case python path is nfs """
    partial_path = get_tfbldr_dataset_dir("josquin")
    if not os.path.exists(partial_path + os.sep + "jrp-scores"):
        cur = os.getcwd()
        os.chdir(partial_path)
        pe("git clone --recursive https://github.com/josquin-research-project/jrp-scores")
        os.chdir("jrp-scores")
        pe("make webreduced")
        os.chdir(cur)

    jos_sub = partial_path + os.sep + "jrp-scores" + os.sep + "Jos" + os.sep + "kern-reduced"
    return jos_sub


def fetch_josquin(keys=["C major", "A minor"],
                  equal_voice_count=4,
                  verbose=True):
    """
    Josquin transposed to C major or A minor (depending on original key).
    Requires music21.
    """
    data_path = check_fetch_josquin()
    pickle_path = os.path.join(data_path, "__processed_josquin.pkl")
    mu = _music_extract(data_path, pickle_path, ext=".krn",
                        # debug... set to None
                        #multiprocess_count=None,
                        verbose=verbose)
    mu_res = _common_features_from_music_extract(mu, equal_voice_count=equal_voice_count,
                                                 verbose=verbose)
    return mu_res
